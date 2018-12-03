import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda
from misc.utils import softmax_sample
from models.base import VisualSpatialEncoder, GaussianPooling

class LanguageModel(chainer.Chain):
    def __init__(self, vocab_size, seq_length, global_shapes, res6=None):
        self.global_num = global_shapes[0]*global_shapes[1]//(32*32)
        super(LanguageModel, self).__init__(
            sp_encoder = VisualSpatialEncoder(res6=res6, global_num=self.global_num),
            gaussian_p = GaussianPooling(10, global_shapes=global_shapes),
            word_emb = L.EmbedID(vocab_size+2, 512),
            LSTM = MyLSTM(512, 512, 512, 0.5),
            h_emb   = L.Linear(512, 1000),
            s_t_emb = L.Linear(512, 1000),
            
            W_s  = L.Linear(1000, 512),
            W_g = L.Linear(1000 , 512, nobias=True),
            w_h = L.Linear(512, 1),
            
            mlp_1 = L.Linear(1000, 512),
            mlp_2 = L.Linear(512, vocab_size+1),
        )
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.dropout_ratio=0.5
        
    def LSTM_initialize(self):
        self.LSTM.reset_state()
        
    def calc_spatial_features(self, global_features, local_features,l_coord, g_coord):
        return self.sp_encoder(global_features, local_features,l_coord, g_coord)
        
    def forward(self, feats, sp_feats, sp_feats_emb, gaussians, w, i):
        w = self.word_emb(w)
        if i==0:
            s_t, h = self.LSTM(vis=feats, sos=w)
        else:
            s_t, h = self.LSTM(vis=feats, word=w)
        return self.out(sp_feats, sp_feats_emb, h, s_t, gaussians)

    def attention_layer(self, sp_feats, sp_feats_emb, h, s_t, gaussians):
        b = h.shape[0]
        h_emb  = self.W_g(h)
        beta_t = self.w_h(F.dropout(F.tanh(self.W_s(s_t) + h_emb), ratio=self.dropout_ratio))
        h_emb  = F.reshape(F.broadcast_to(F.reshape(h_emb, (-1, 1, 512)), (b, self.global_num+36, 512)), (-1,512))
        
        alpha_ = F.reshape(F.softmax(
                F.concat([F.log(gaussians+1e-15)+F.reshape(#sp_feats, sp_feats_emb
                            self.w_h(
                                F.dropout(F.tanh( 
                                    sp_feats_emb+h_emb
                                ), ratio=self.dropout_ratio)
                            ), (-1, self.global_num+36)), beta_t], axis = 1
                        )), (-1, self.global_num+36+1, 1))
        alpha = F.broadcast_to(alpha_, (b, self.global_num+36+1, 1000))
        c = F.sum(alpha[:,:-1,:] * F.reshape(sp_feats, (-1, self.global_num+36, 1000)), axis = 1)
        c = alpha[:,-1,:] * s_t + c
        
        return alpha_, c
            
    def out(self, sp_feats, sp_feats_emb, h, s_t, gaussians):
        h = F.dropout(F.tanh(self.h_emb(h)), ratio=self.dropout_ratio)
        s_t = F.dropout(F.relu(self.s_t_emb(s_t)), ratio=self.dropout_ratio)
        alpha_, c = self.attention_layer(sp_feats, sp_feats_emb, h, s_t, gaussians)
        del sp_feats, sp_feats_emb, s_t
        h = F.tanh(self.mlp_1(h+c))
        return alpha_, self.mlp_2(F.dropout(h, ratio=self.dropout_ratio))
        
    def __call__(self, vis_feats, sp_feats, sp_feats_emb, coord, seqz, lang_last_ind):
        seqz = seqz.data
        xp = cuda.get_array_module(vis_feats)
        batch_size = vis_feats.shape[0]
        self.LSTM_initialize()
        log_probs = []
        gaussians = self.gaussian_p.get_gaussian_for_attention(coord)
        for i in range(max(lang_last_ind)+1):
            if i==0:
                mask = xp.ones(len(seqz))
                sos = Variable(xp.ones(batch_size,dtype=xp.int32)*(self.vocab_size+1))
                sos = self.word_emb(sos)
                s_t, h = self.LSTM(vis=vis_feats, sos=sos)
            else:
                mask = xp.where(seqz[:, i-1]!=0,1,0)
                w = self.word_emb(Variable(seqz[:, i-1]))
                s_t, h = self.LSTM(vis=vis_feats, word=w)
            _, h = self.out(sp_feats, sp_feats_emb, h, s_t, gaussians)
            logsoft = (F.log_softmax(h)*mask.reshape(batch_size, 1).repeat(h.data.shape[1], axis=1))[np.arange(batch_size), seqz[:,i]]
                
            log_probs.append(logsoft.reshape(1,batch_size)) 
                
        return F.concat(log_probs, axis=0) #length x batch_size 
    
    def sample(self, vis_feats, sp_feats, sp_feats_emb, coord, temperature=1):
        xp = cuda.get_array_module(vis_feats)
        batch_size = vis_feats.shape[0]
        self.LSTM_initialize()
        
        output = xp.zeros((batch_size, self.seq_length), dtype=xp.int32)
        log_probs = [] # length x b*vocab_size
        gaussians = self.gaussian_p.get_gaussian_for_attention(coord)
        mask = xp.ones(batch_size)
        
        with chainer.using_config('train', False):
            for i in range(self.seq_length):
                if i==0:
                    mask_ = xp.ones(batch_size)
                    sos = self.word_emb(Variable(xp.ones(batch_size,dtype=xp.int32)*(self.vocab_size+1)))
                    s_t, h = self.LSTM(vis=vis_feats, sos=sos)
                else:
                    mask_ = xp.where(w!=0,1,0)
                    mask *= mask_
                    if mask.sum()==0:
                        break
                    output[:,i-1] = w
                    w = self.word_emb(Variable(w))
                    s_t, h = self.LSTM(vis=vis_feats, word=w)
                _, h = self.out(sp_feats, sp_feats_emb, h, s_t, gaussians)
                logsoft = F.log_softmax(h)*mask.reshape(batch_size, 1).repeat(h.data.shape[1], axis=1)

                prob_prev = F.exp(logsoft/temperature)
                prob_prev /= F.broadcast_to(F.sum(prob_prev, axis=1, keepdims=True), prob_prev.shape)
                w = softmax_sample(prob_prev)
                log_probs.append(logsoft[np.arange(batch_size), w].reshape(1,batch_size))
        return output, F.concat(log_probs, axis=0)

    def max_sample(self, vis_feats, sp_feats, sp_feats_emb, coord):
        xp = cuda.get_array_module(vis_feats)
        batch_size = vis_feats.shape[0]
        self.LSTM_initialize()
        
        output = xp.zeros((batch_size, self.seq_length), dtype=xp.int32)
        mask = xp.ones(batch_size)
        gaussians = self.gaussian_p.get_gaussian_for_attention(coord)
        for i in range(self.seq_length):
            if i==0:
                sos = self.word_emb(Variable(xp.ones(batch_size,dtype=xp.int32)*(self.vocab_size+1)))
                s_t, h = self.LSTM(vis=vis_feats, sos=sos)
            else:
                mask_ = xp.where(output[:,i-1]!=0,1,0)
                mask *= mask_
                if mask.sum()==0:
                    break
                w = self.word_emb(Variable(output[:,i-1]))
                s_t, h = self.LSTM(vis=vis_feats, word=w)
            _, h = self.out(sp_feats, sp_feats_emb, h, s_t, gaussians)
            output[:,i] = xp.argmax(h.data[:,:-1], axis=1)
            
        result = []
        for out in output:
            for i, w in enumerate(out):
                if w==0:
                    result.append(out[:i])
                    break
                
        return result

class MyLSTM(chainer.Chain):
    
    def __init__(self, vis_size, word_size, rnn_size, dropout_ratio):
         super(MyLSTM, self).__init__(
                vis2h = L.Linear(vis_size, 4*rnn_size),
                sos2h = L.Linear(word_size, 4*rnn_size, nobias = True),
                af_LSTM = L.LSTM(word_size, rnn_size),
                h2h = L.Linear(rnn_size, rnn_size),
                w2h = L.Linear(word_size, rnn_size, nobias = True),
         )
         self.dropout_ratio = dropout_ratio
    
    def reset_state(self):
        self.af_LSTM.reset_state()
        
    def __call__(self, vis = None, sos = None, word = None):
        if sos is not None:
            h = self.vis2h(vis)+ self.sos2h(sos)
            a, i, o, g = F.split_axis(h, 4, axis = 1)
            a = F.tanh(a)
            i = F.sigmoid(i)
            o = F.sigmoid(o)
            g = F.sigmoid(g)
            c = a * i 
            h = F.dropout(o *F.tanh(c), ratio = self.dropout_ratio)
            
            self.af_LSTM.set_state(c, h)
            
        else:
            word_emb = F.dropout(word, ratio = self.dropout_ratio)
            g = F.sigmoid(self.w2h(word_emb) + self.h2h(self.af_LSTM.h))
            h = F.dropout(self.af_LSTM(word_emb), ratio = self.dropout_ratio)
            
        s_t = F.dropout(g * F.tanh(self.af_LSTM.c), ratio=self.dropout_ratio)
    
        return s_t, h

import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda

class VisualEncoder(chainer.Chain):
    def __init__(self, res6=None, res_dim=2048, res6_dim=1000, encoding_size=512,
                 dif_num=5, global_shapes=(224,224)):
        initializer = chainer.initializers.GlorotNormal(scale=math.sqrt(2))
        super(VisualEncoder, self).__init__(
            gaussian_p = GaussianPooling(5, global_shapes=global_shapes),
            cxt_enc  = L.Linear(res_dim, res6_dim),
            ann_enc = L.Linear (res_dim, res6_dim),
            dif_ann_enc = L.Linear(res_dim, res6_dim),
            joint_enc = L.Linear(res6_dim*3+5*(dif_num+1), encoding_size, initialW=initializer)
        )
        self.feat_ind = [2048, 5, 2048, 25]
        self.res_dim = res6_dim
        self.embedding_size = encoding_size
        self.dropout_ratio = 0.25
        if res6 !=None:
            self.cxt_enc = res6.copy()
            self.ann_enc = res6.copy()
            self.dif_ann_enc = res6.copy()
        
    def __call__(self, feats, sp_cxt_feats, coord, init_norm=20):
        cxt = self.cxt_enc(self.gaussian_p(sp_cxt_feats, coord))
        #cxt = self.cxt_enc(F.mean(sp_cxt_feats, axis=1))
        ann = self.ann_enc(feats[:, :sum(self.feat_ind[:1])])
        loc = feats[:, sum(self.feat_ind[:1]):sum(self.feat_ind[:2])]
        diff_ann = self.dif_ann_enc(feats[:, sum(self.feat_ind[:2]):sum(self.feat_ind[:3])])
        diff_loc = feats[:, sum(self.feat_ind[:3]):]
        
        cxt = F.normalize(cxt)*init_norm
        ann = F.normalize(ann)*init_norm
        loc = F.normalize(loc+1e-15)*init_norm
        diff_ann = F.normalize(diff_ann)*init_norm
        diff_loc = F.normalize(diff_loc+1e-15)*init_norm
        
        J = F.concat([cxt, ann, loc, diff_ann, diff_loc], axis=1)
        J = F.dropout(self.joint_enc(J), self.dropout_ratio)
        return J
    
class VisualSpatialEncoder(chainer.Chain):
    def __init__(self, res6=None, res_dim=2048, res6_dim=1000, encoding_size=512, global_num=49):
        initializer = chainer.initializers.GlorotNormal(scale=math.sqrt(2))
        super(VisualSpatialEncoder, self).__init__(
            sp_cxt_enc1  = L.Linear(res_dim+2, res6_dim),
            sp_ann_enc1  = L.Linear(res_dim+2, res6_dim),
            sp_cxt_enc2  = L.Linear(res6_dim, encoding_size, initialW=initializer),
            sp_ann_enc2  = L.Linear(res6_dim, encoding_size, initialW=initializer),
        )
        self.global_num = global_num
        self.local_num = 36
        self.dropout_ratio = 0.5
        self.res_dim = res6_dim
        self.embedding_size = encoding_size
        
        if res6 !=None:
            self.sp_cxt_enc1.W.data[:,:2048] = res6.copy().W.data
            self.sp_cxt_enc1.b.data[:2048] = res6.copy().b.data
            self.sp_ann_enc1.W.data[:,:2048] = res6.copy().W.data
            self.sp_ann_enc1.b.data[:2048] = res6.copy().b.data
            
    def __call__(self, global_features, local_features,l_coord, g_coord):
        global_features = F.dropout(F.relu(self.sp_cxt_enc1(
                F.reshape(F.concat([global_features, g_coord], axis = 2), (-1, 2050))
                           )), ratio=self.dropout_ratio)
        local_features = F.dropout(F.relu(self.sp_ann_enc1(
                F.reshape(F.concat([local_features, l_coord], axis = 2), (-1, 2050))
                           )), ratio=self.dropout_ratio)
        
        im_org_features = F.reshape(F.concat(
            [F.reshape(global_features, (-1, self.global_num, self.res_dim))
             , F.reshape(local_features, (-1, self.local_num, self.res_dim))], axis = 1
        ), (-1, self.res_dim))
        global_features = F.reshape(self.sp_cxt_enc2(global_features), (-1, self.global_num, self.embedding_size))
        local_features =  F.reshape(self.sp_ann_enc2(local_features), (-1, self.local_num, self.embedding_size))
        
        im_emb_features = F.reshape(F.concat([global_features, local_features], axis = 1), (-1, self.embedding_size))
        
        return im_org_features, im_emb_features
    
    
class LanguageEncoderAttn(chainer.Chain):
    def __init__(self,vocab_size):
        super(LanguageEncoderAttn, self).__init__(
            word_emb = L.EmbedID(vocab_size+2, 512),
            LSTM = L.LSTM(512, 512),
            linear1 = L.Linear(512, 512),
            linear2 = L.Linear(512, 1),
            norm = L.BatchNormalization(512, eps=1e-5),
        )
        
    def LSTMForward(self, sents_emb, max_last_ind):
        self.LSTM.reset_state()
        h_list = []
        for i in range(max_last_ind+1):
            h = self.LSTM(sents_emb[:,i])
            h_list.append(h)# length*b*512
        return h_list
    
    def create_word_mask(self, lang_last_ind, xp):
        mask = xp.zeros((len(lang_last_ind), max(lang_last_ind)+1), dtype=xp.float32)
        for i in range(len(lang_last_ind)):
            mask[i,:lang_last_ind[i]+1] = 1
        return mask
    
    def sentence_attention(self, lstm_out, lang_last_ind):
        batch_size = len(lang_last_ind)
        seq_length = max(lang_last_ind)+1
        lstm_out = F.reshape(F.concat(lstm_out, axis = 1), (batch_size*seq_length, -1))
        xp = cuda.get_array_module(lstm_out)
        
        word_mask = self.create_word_mask(lang_last_ind, xp) #b*seq_length
        h = F.dropout(F.relu(self.linear1(lstm_out)), ratio=0.1)
        h = F.reshape(self.linear2(h), (batch_size, seq_length))
        h = h*word_mask+(word_mask*1024-1024) 
        att_softmax = F.softmax(h, axis=1)
        self.attention_result = att_softmax
        lstm_out = F.reshape(lstm_out, (batch_size, seq_length, -1))
        att_mask = F.broadcast_to(F.reshape(att_softmax, (batch_size, seq_length, 1)), lstm_out.shape)  # N x T  x d
        att_mask = att_mask * lstm_out 
        att_mask = F.sum(att_mask, axis = 1)
        return att_mask
    
    def __call__(self, sents, lang_last_ind, attention=True):
        sents_emb = F.dropout(self.word_emb(sents), ratio=0.5)
        sents_emb = self.LSTMForward(sents_emb, max(lang_last_ind))
        sents_emb = self.norm(self.sentence_attention(sents_emb, lang_last_ind))
        return sents_emb
    
class GaussianPooling(chainer.Chain):
    def __init__(self, init_var=3, global_shapes=(224,224)):
        super(GaussianPooling, self).__init__()
        
        with self.init_scope():
            self.x_var = chainer.Parameter(init_var, (1, 1))
            self.y_var = chainer.Parameter(init_var, (1, 1))
        self.global_shapes = global_shapes
        self.local_num = 36

    def __call__(self, global_features, bboxes):
        resize_shapes = [self.global_shapes for _ in range(len(bboxes))]
        gaussians = []
        xp = cuda.get_array_module(self.x_var.data)
        for resize_shape, bbox in zip(resize_shapes, bboxes):
            G = self.get_gaussian(resize_shape, bbox)
            G = G/(xp.sum(G.data) + 1e-15)
            gaussians.append(G)
        gaussians = F.broadcast_to(F.reshape(F.stack(gaussians, axis = 0), (global_features.shape[0], global_features.shape[1], 1)), global_features.shape)
        
        return F.sum(global_features*gaussians, axis = 1)
    
    def get_gaussian_for_attention(self, bboxes):
        resize_shapes = [self.global_shapes for _ in range(len(bboxes))]
        xp = cuda.get_array_module(self.x_var.data)
        gaussians = []
        for resize_shape, bbox in zip(resize_shapes, bboxes):
            G = self.get_gaussian(resize_shape, bbox)
            G = G/(xp.max(G.data)+1e-15)
            gaussians.append(F.hstack([G, Variable(xp.ones(self.local_num, dtype =xp.float32))]))
        gaussians = F.stack(gaussians, axis = 0)    
        return gaussians
    
    def get_gaussian(self, resize_shape, bbox):
        xp = cuda.get_array_module(self.x_var.data)
        W, H = resize_shape[0]//32 , resize_shape[1]//32
        X,Y = xp.meshgrid(xp.linspace(0,W,W),xp.linspace(0,H,H));
        X,Y = X.astype(xp.float32), Y.astype(xp.float32)
        mu_x, mu_y = xp.array([(bbox[0]+bbox[2])*W/2]).astype(xp.float32), xp.array([(bbox[1]+bbox[3])*H/2]).astype(xp.float32)
        sigma_x, sigma_y = F.broadcast_to(self.x_var, X.shape), F.broadcast_to(self.y_var, Y.shape)
        G = F.exp(-((X-mu_x)**2/(2.0*sigma_x**2+1e-15)+(Y-mu_y)**2/(2.0*sigma_y**2+1e-15)))
        G = F.flatten(G)
        return G
    
class MetricNet(chainer.Chain):
    def __init__(self):
        initializer = chainer.initializers.GlorotNormal(scale=math.sqrt(2))
        super(MetricNet, self).__init__(
            fc1 = L.Linear(512+512, 512, initialW=initializer),
            norm1 = L.BatchNormalization(512, eps=1e-5),
            fc2 = L.Linear(512, 512, initialW=initializer),
            norm2 = L.BatchNormalization(512, eps=1e-5),
            fc3 = L.Linear(512, 1, initialW=initializer),
            
            vis_norm = L.BatchNormalization(512, eps=1e-5),
        )
        
    def __call__(self, vis, lang):
        joined = F.concat([self.vis_norm(vis), lang], axis=1)
        joined = F.dropout(F.relu(self.norm1(self.fc1(joined))), ratio=0.2)
        joined = F.dropout(F.relu(self.norm2(self.fc2(joined))), ratio=0.2)
        joined = self.fc3(joined)
        return joined
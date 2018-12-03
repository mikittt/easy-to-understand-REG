import os.path as osp
import json
import numpy as np
import math
import chainer
from chainer import cuda, Variable
import chainer.functions as F

def softmax_sample(p):
    """
    input p: softmax output, chainer.Variable, batchsize * num_class
    output: sampled index
    """
    xp = cuda.get_array_module(p.data)
    rand = xp.random.uniform(size=p.data.shape[0], dtype=p.data.dtype)
    next_state = xp.zeros(p.data.shape[0], dtype="int32")
    xp.ElementwiseKernel(
        'raw T p, raw T rand, I num',
        'I next_state',
        '''
            T cumsum = 0;
            for(I j=0; j < num; j++) {
                cumsum += p[i * num + j];
                if(cumsum > rand[i]) {
                    next_state = j;
                    break;
                }
            }
        ''',
        'sample')(p.data, rand, p.data.shape[1], next_state)
    return next_state

def calc_max_ind(seq):
    length = np.array([seq.shape[1]-1]*seq.shape[0])
    for ind, s in enumerate(seq):
        for i, w in enumerate(s):
            if w==0:
                length[ind] = i
                break
    return length

def beam_search(model, vis_feats, sp_feats, sp_feats_emb, coord, beam_width):
    xp = cuda.get_array_module(vis_feats)
    results = []
    results_with_att = []
    gaussians = model.gaussian_p.get_gaussian_for_attention(coord)
    batch_size = vis_feats.shape[0]
    for b in range(batch_size):
        model.LSTM_initialize()
        candidates = [(model, [model.vocab_size+1], 0, 0, [])]
        feat = vis_feats[b][xp.newaxis,:]
        sp_feat = F.split_axis(sp_feats, batch_size, axis=0)[b]
        sp_feat_emb = F.split_axis(sp_feats_emb, batch_size, axis=0)[b]
        gaussian = gaussians[b][xp.newaxis,:]
        for i in range(model.seq_length):
            next_candidates = []
            for prev_net, tokens, sum_likelihood, likelihood, alphas in candidates:
                if tokens[-1] == 0:
                    next_candidates.append((None, tokens, sum_likelihood, likelihood, alphas))
                    continue
                net = prev_net.copy()
                w = Variable(xp.asarray([tokens[-1]]).astype(np.int32))
                att, h = net.forward(feat, sp_feat, sp_feat_emb, gaussian, w, i)
                token_likelihood = cuda.to_cpu(F.log_softmax(h).data[:,:-1])[0]
                order = token_likelihood.argsort()[:-beam_width-1:-1]
                next_candidates.extend([(net, tokens + [j], sum_likelihood+token_likelihood[j], (likelihood * len(tokens) + token_likelihood[j])/(len(tokens) + 1), alphas+[cuda.to_cpu(att.data)]) for j in order])
            candidates = sorted(next_candidates, key=lambda x: -x[3])[:beam_width]
            if all([candidate[1][-1] == 0 for candidate in candidates]):
                break
        result = [{'sent':[int(w) for w in candidate[1][1:-1]],'ppl':float(math.exp(-candidate[3]))} for candidate in candidates]
        result_with_att = [{'sent':[int(w) for w in candidate[1][1:-1]],'ppl':float(math.exp(-candidate[3])), 'att':candidate[4]} for candidate in candidates]
        results.append(result)
        results_with_att.append(result_with_att)
    return results, results_with_att
   
def calc_coordinate_feature(coord, local_shape, global_shapes=(224,224)):
    global_shape = [global_shapes]*len(local_shape)
    local_features = []
    global_features = []
    for num, shape in enumerate([local_shape, global_shape]):
        for target in range(len(local_shape)):
            s = np.array(shape[target])//32
            if num == 0 :
                X,Y = np.meshgrid(np.linspace(coord[target][0],coord[target][2], s[0]+1),np.linspace(coord[target][1],coord[target][3], s[1]+1))
                a = [[(X[i,j]+X[i+1,j+1])/2, (Y[i,j]+Y[i+1, j+1])/2] for i in range(s[1]) for j in range(s[0])]
                local_features.append(a)
            else:
                X,Y = np.meshgrid(np.linspace(0,1, s[0]+1),np.linspace(0,1, s[1]+1))
                a = [[(X[i,j]+X[i+1,j+1])/2, (Y[i,j]+Y[i+1, j+1])/2] for i in range(s[1]) for j in range(s[0])]
                global_features.append(a)
    return np.array(local_features, dtype = np.float32), np.array(global_features, dtype = np.float32)


def load_vcab_init(dictionary, save_path, 
                   glove_path):
    if not os.path.exists(save_path):
        initial_emb = np.zeros((len(dictionary)+2, 300), dtype = np.float32)
        word2emb = {}
        with open(glove_path, 'r') as f:
            entries = f.readlines()
            for entry in entries:
                vals = entry.split(' ')
                word = vals[0]
                vals = list(map(float, vals[1:]))
                word2emb[word] = np.array(vals)
            for word in list(dictionary.keys()):
                if word not in word2emb:
                    continue
                initial_emb[int(dictionary[word]), :300] = word2emb[word]
            np.save(save_path, initial_emb)
    else:
        initial_emb = np.load(save_path)
    return initial_emb

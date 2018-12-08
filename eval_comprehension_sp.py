import config
import os
import os.path as osp
import numpy as np

import chainer
from chainer import Variable, cuda, serializers
import chainer.functions as F

from misc.DataLoader import DataLoader
from misc.utils import calc_max_ind, calc_coordinate_feature
from models.base import VisualEncoder, LanguageEncoderAttn
from models.Reinforcer import ListenerReward
from models.LanguageModel import LanguageModel
from misc.eval_utils import compute_margin_loss, computeLosses

def eval_all(params):
    target_save_dir = osp.join(params['save_dir'],'prepro', params['dataset']+'_'+params['splitBy'])
    model_dir = osp.join(params['save_dir'],'model', params['dataset']+'_'+params['splitBy'])
    result_dir = osp.join('result', params['dataset']+'_'+params['splitBy'])
    
    if not osp.isdir(result_dir):
        os.makedirs(result_dir)
    
    if params['old']:
        params['data_json'] = 'old'+params['data_json']
        params['data_h5'] = 'old'+params['data_h5']
        params['image_feats_h5'] = 'old'+params['image_feats']
        params['ann_feats_h5'] = 'old'+params['ann_feats']
        params['ann_feats_input'] = 'old'+params['ann_feats_input']
        params['shapes'] = 'old'+params['shapes']
        params['id'] = 'old'+params['id']
        
    if params['dataset'] in ['refcoco', 'refcoco+', 'refcocog']:
        global_shapes = (224, 224)
    elif params['dataset'] == 'refgta':
        global_shapes = (480, 288)
    loader = DataLoader(params)
    
    featsOpt = {'sp_ann':osp.join(target_save_dir, params['sp_ann_feats']),
                'ann_input':osp.join(target_save_dir, params['ann_feats']),
                'img':osp.join(target_save_dir, params['image_feats']),
               'shapes':osp.join(target_save_dir, params['ann_shapes'])}
    loader.loadFeats(featsOpt) 
    chainer.config.train = False
    chainer.config.enable_backprop = False
    
    gpu_id = params['gpu_id']
    cuda.get_device(gpu_id).use()
    xp = cuda.cupy
    

    ve = VisualEncoder(global_shapes=global_shapes).to_gpu(gpu_id)
    rl_crit = ListenerReward(len(loader.ix_to_word), global_shapes=global_shapes).to_gpu(gpu_id)
    lm = LanguageModel(len(loader.ix_to_word), loader.seq_length, global_shapes).to_gpu(gpu_id)
    
    serializers.load_hdf5(osp.join(model_dir, params['id']+".h5"), rl_crit)
    
    serializers.load_hdf5(osp.join(model_dir, params['id']+params['id2']+"ve.h5"), ve)
    serializers.load_hdf5(osp.join(model_dir, params['id']+params['id2']+"lm.h5"), lm)
    
    accuracy = 0
    loss_evals  = 0
    while True:
        data = loader.getImageBatch(params['split'], params)
        image_id = data['image_id']
        img_ann_ids = data['img_ann_ids']
        sent_ids = data['sent_ids']
        gd_ixs = data['gd_ixs']
        feats = Variable(xp.array(data['feats'], dtype=xp.float32))
        sp_cxt_feats = Variable(xp.array(data['sp_cxt_feats'], dtype=xp.float32))
        sp_ann_feats = Variable(xp.array(data['sp_ann_feats'], dtype=xp.float32))
        local_shapes = data['local_shapes']
        seqz = data['seqz']
        lang_last_ind = calc_max_ind(seqz)
        for i, sent_id in enumerate(sent_ids):
            gd_ix = gd_ixs[i]
            labels = xp.zeros(len(img_ann_ids), dtype=xp.int32)
            labels[gd_ix] = 1
            labels = Variable(labels)

            sent_seqz = np.concatenate([[seqz[i]] for _ in range(len(img_ann_ids))],axis=0)
            one_last_ind =  np.array([lang_last_ind[i]]*len(img_ann_ids))
            sent_seqz = Variable(xp.array(sent_seqz, dtype=xp.int32))
                
            coord = cuda.to_cpu(feats[:, sum(ve.feat_ind[:1]):sum(ve.feat_ind[:2])].data)
            local_sp_coord, global_sp_coord = calc_coordinate_feature(coord, local_shapes, 
                                                                              global_shapes=global_shapes)
            local_sp_coord, global_sp_coord = xp.array(local_sp_coord, dtype=xp.float32), xp.array(global_sp_coord, dtype=xp.float32)
            vis_enc_feats = ve(feats, sp_cxt_feats, coord)
            sp_feats, sp_feats_emb = lm.calc_spatial_features(sp_cxt_feats, sp_ann_feats, 
                                                                      local_sp_coord, global_sp_coord)
            vis_feats = vis_enc_feats
            logprobs  = lm(vis_feats, sp_feats, sp_feats_emb, 
                                   coord, sent_seqz, one_last_ind).data
            
            lm_scores = -cuda.to_cpu(computeLosses(logprobs, one_last_ind))
            
            score = cuda.to_cpu(F.sigmoid(rl_crit.calc_score(feats, sp_cxt_feats, 
                                                                 coord, sent_seqz, one_last_ind)).data)[:,0]
            
            if params['mode']==0:
                _, pos_sc, max_neg_sc = compute_margin_loss(lm_scores, gd_ix, 0)
            elif params['mode']==1:
                _, pos_sc, max_neg_sc = compute_margin_loss(score, gd_ix, 0)
            elif params['mode']==2:
                scores = score + params['lamda'] * lm_scores
                _, pos_sc, max_neg_sc = compute_margin_loss(scores, gd_ix, 0)
            if pos_sc > max_neg_sc:
                accuracy += 1
            loss_evals += 1
            print('{}-th: evaluating [{}]  ... image[{}/{}] sent[{}], acc={}'.format(loss_evals, params['split'], data['bounds']['it_pos_now'], data['bounds']['it_max'], i, accuracy*100.0/loss_evals))
        
        if data['bounds']['wrapped']:
            print('validation finished!')
            f = open(result_dir+'/'+params['id']+params['id2']+str(params['mode'])+str(params['lamda'])+'comp.txt', 'w') 
            f.write(str(accuracy*100.0/loss_evals)) 
            f.close() 
            break
                    
if __name__ == '__main__':

    args = config.parse_opt()
    params = vars(args) # convert to ordinary dict
    eval_all(params)
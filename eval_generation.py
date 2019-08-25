import config
import os
import os.path as osp
import numpy as np
import json

import chainer
from chainer import Variable, cuda, serializers

from misc.DataLoader import DataLoader
from misc.utils import calc_max_ind, beam_search, calc_coordinate_feature
from models.base import VisualEncoder, LanguageEncoderAttn
from models.LanguageModel import LanguageModel
from misc.eval_utils import compute_margin_loss, computeLosses, language_eval

def eval_all(params):
    target_save_dir = osp.join(params['save_dir'],'prepro', params['dataset']+'_'+params['splitBy'])
    model_dir = osp.join(params['save_dir'],'model', params['dataset']+'_'+params['splitBy'])
    result_dir = osp.join('result', params['dataset']+'_'+params['splitBy'])
    
    if not osp.isdir(result_dir):
        os.makedirs(result_dir)
    
    if params['old']:
        params['data_json'] = 'old'+params['data_json']
        params['data_h5'] = 'old'+params['data_h5']
        params['image_feats_h5'] = 'old'+params['image_feats_h5']
        params['ann_feats_h5'] = 'old'+params['ann_feats_h5']
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
    lm = LanguageModel(len(loader.ix_to_word), loader.seq_length, global_shapes).to_gpu(gpu_id)
    
    serializers.load_hdf5(osp.join(model_dir, params['id']+params['id2']+"ve.h5"), ve)
    serializers.load_hdf5(osp.join(model_dir, params['id']+params['id2']+"lm.h5"), lm)
    
    predictions = []
    beam_all_results = []
    while True:
        data = loader.getTestBatch(params['split'], params)
        ref_ids = data['ref_ids']
        lang_last_ind = calc_max_ind(data['seqz'])
        feats = Variable(xp.array(data['feats'], dtype=xp.float32))
        sp_cxt_feats = Variable(xp.array(data['sp_cxt_feats'], dtype=xp.float32))
        sp_ann_feats = Variable(xp.array(data['sp_ann_feats'], dtype=xp.float32))
        local_shapes = data['local_shapes']
        coord = data['feats'][:, sum(ve.feat_ind[:1]):sum(ve.feat_ind[:2])]
        local_sp_coord, global_sp_coord = calc_coordinate_feature(coord, local_shapes,
                                                         global_shapes=global_shapes)
        local_sp_coord, global_sp_coord = xp.array(local_sp_coord, dtype=xp.float32), xp.array(global_sp_coord, dtype=xp.float32)

        vis_enc_feats = ve(feats, sp_cxt_feats, coord)
        vis_feats = vis_enc_feats
        sp_feats, sp_feats_emb = lm.calc_spatial_features(sp_cxt_feats, sp_ann_feats, 
                                                            local_sp_coord, global_sp_coord)
        if params['beam_width']==1:
            results = lm.max_sample(vis_feats)
        else:
            beam_results, _ = beam_search(lm, vis_feats, sp_feats, sp_feats_emb, coord, params['beam_width'])

            results = [result[0]['sent'] for result in beam_results]
            ppls = [result[0]['ppl'] for result in beam_results]
            
        for i, result in enumerate(results):
            gen_sentence= ' '.join([loader.ix_to_word[str(w)] for w in result])
            if params['beam_width']==1:
                print(gen_sentence)
            else:
                print(gen_sentence, ', ppl : ', ppls[i])
            entry = {'ref_id':ref_ids[i], 'sent':gen_sentence}
            predictions.append(entry)
            if params['beam_width']>1:
                beam_all_results.append({'ref_id':ref_ids[i], 'beam':beam_results[i]})
        print('evaluating validation performance... {}/{}'.format(data['bounds']['it_pos_now'], data['bounds']['it_max']))
        
        if data['bounds']['wrapped']:
            print('validation finished!')
            break
    lang_stats = language_eval(predictions, params['split'], params)
    print(lang_stats)
    
    print('sentence mean length: ', np.mean([len(pred['sent'].split()) for pred in predictions]))
    with open(result_dir+'/'+params['id']+params['id2']+str(params['beam_width'])+params['split']+'raw.json','w') as f:
        json.dump(predictions, f)
    with open(result_dir+'/'+params['id']+params['id2']+str(params['beam_width'])+params['split']+'.json','w') as f:
        json.dump(lang_stats, f)
    with open(result_dir+'/'+params['id']+params['id2']+str(params['beam_width'])+params['split']+'all_beam.json','w') as f:
        json.dump(beam_all_results, f)
        
                    
if __name__ == '__main__':

    args = config.parse_opt()
    params = vars(args) # convert to ordinary dict
    eval_all(params)
import config
import os
import os.path as osp
import math
import json
import time

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda, serializers, optimizers

from misc.DataLoader import DataLoader
from misc.utils import calc_max_ind, calc_coordinate_feature
from models.base import VisualEncoder, LanguageEncoderAttn
from models.Reinforcer import ListenerReward
from models.LanguageModel import LanguageModel
from misc.eval_utils import compute_margin_loss, computeLosses, calc_rank_loss, calc_rank_acc
from misc.crit import lm_crits

def train_all(params):
    target_save_dir = osp.join(params['save_dir'], 'prepro', params['dataset']+'_'+params['splitBy'])
    graph_dir = osp.join('log_graph', params['dataset']+'_'+params['splitBy'])
    model_dir = osp.join(params['save_dir'], 'model', params['dataset']+'_'+params['splitBy'])  

    if params['old']:
        params['data_json'] = 'old'+params['data_json']
        params['data_h5'] = 'old'+params['data_h5']
        params['image_feats'] = 'old'+params['image_feats']
        params['sp_ann_feats'] = 'old'+params['sp_ann_feats']
        params['ann_feats'] = 'old'+params['ann_feats']
        params['ann_shapes'] = 'old'+params['ann_shapes']
        params['id'] = 'old'+params['id']
        params['word_emb_path'] = 'old'+params['word_emb_path']
        
    if params['dataset'] in ['refcoco', 'refcoco+', 'refcocog']:
        global_shapes = (224, 224)
    elif params['dataset'] == 'refgta':
        global_shapes = (480, 288)
        
    loader = DataLoader(params)
    
    # model setting
    batch_size = params['batch_size']
    gpu_id = params['gpu_id']
    cuda.get_device(gpu_id).use()
    xp = cuda.cupy
    featsOpt = {'sp_ann':osp.join(target_save_dir, params['sp_ann_feats']),
                'ann_input':osp.join(target_save_dir, params['ann_feats']),
                'img':osp.join(target_save_dir, params['image_feats']),
               'shapes':osp.join(target_save_dir, params['ann_shapes'])}
    loader.loadFeats(featsOpt, mmap_mode=False) 
    loader.shuffle('train')
    
    ve = VisualEncoder(res6=L.ResNet152Layers().fc6, global_shapes=global_shapes).to_gpu(gpu_id)
    rl_crit = ListenerReward(len(loader.ix_to_word), global_shapes=global_shapes).to_gpu(gpu_id)
    lm = LanguageModel(len(loader.ix_to_word), loader.seq_length, 
                       global_shapes, res6=L.ResNet152Layers().fc6).to_gpu(gpu_id)
    
    serializers.load_hdf5(osp.join(model_dir, params['id']+".h5"), rl_crit)
    
    ve_optim = optimizers.Adam(alpha=4e-5, beta1=0.8)
    lm_optim = optimizers.Adam(alpha=4e-4, beta1=0.8)
    
    ve_optim.setup(ve)
    lm_optim.setup(lm)
    
    ve_optim.add_hook(chainer.optimizer.GradientClipping(params['grad_clip']))
    lm_optim.add_hook(chainer.optimizer.GradientClipping(params['grad_clip']))
    
    ## non-finetune layer
    ve.joint_enc.W.update_rule.hyperparam.alpha = 4e-4
    ve.joint_enc.b.update_rule.hyperparam.alpha = 4e-4
    lm.gaussian_p.x_var.update_rule.hyperparam.alpha = 1e-2
    lm.gaussian_p.y_var.update_rule.hyperparam.alpha = 1e-2
    ve.gaussian_p.x_var.update_rule.hyperparam.alpha = 1e-2
    ve.gaussian_p.y_var.update_rule.hyperparam.alpha = 1e-2
    
    iteration=0
    epoch=0
    lam = params['rank_lam']
    val_loss_history = []
    val_loss_lm_s_history = []
    val_loss_lm_l_history = []
    val_loss_l_history = []
    val_acc_history = []
    val_rank_acc_history = []
    min_val_loss = 100
    while True:
        chainer.config.train = True
        chainer.config.enable_backprop = True
        ve.zerograds()
        lm.zerograds()
        rl_crit.zerograds()
        
        start = time.time()
        
        data = loader.getBatch('train', params)
        
        ref_ann_ids = data['ref_ann_ids']
            
        pos_feats = Variable(xp.array(data['feats'], dtype=xp.float32))
        pos_sp_cxt_feats = Variable(xp.array(data['sp_cxt_feats'], dtype=xp.float32))
        pos_sp_ann_feats = Variable(xp.array(data['sp_ann_feats'], dtype=xp.float32))
        
        neg_feats = Variable(xp.array(data['neg_feats'], dtype=xp.float32))
        neg_pos_sp_cxt_feats = Variable(xp.array(data['neg_sp_cxt_feats'], dtype=xp.float32))
        neg_pos_sp_ann_feats = Variable(xp.array(data['neg_sp_ann_feats'], dtype=xp.float32))
        local_shapes = np.concatenate([data['local_shapes'], 
                                       data['neg_local_shapes'], 
                                       data['local_shapes']], axis=0)
        
        feats = F.concat([pos_feats, neg_feats, pos_feats], axis=0)
        sp_cxt_feats = F.concat([pos_sp_cxt_feats, neg_pos_sp_cxt_feats, pos_sp_cxt_feats], axis=0)
        sp_ann_feats = F.concat([pos_sp_ann_feats, neg_pos_sp_ann_feats, pos_sp_ann_feats], axis=0)
        seqz  = np.concatenate([data['seqz'],data['seqz'], data['neg_seqz']], axis=0)
        lang_last_ind = calc_max_ind(seqz)
        seqz = Variable(xp.array(seqz, dtype=xp.int32))
    
        coord = cuda.to_cpu(feats[:, sum(ve.feat_ind[:1]):sum(ve.feat_ind[:2])].data)
        local_sp_coord, global_sp_coord = calc_coordinate_feature(coord, local_shapes, 
                                                                  global_shapes=global_shapes)
        local_sp_coord, global_sp_coord = xp.array(local_sp_coord, dtype=xp.float32), xp.array(global_sp_coord, dtype=xp.float32)
        
        # encode vis feature
        vis_feats = ve(feats, sp_cxt_feats, coord)
        sp_feats, sp_feats_emb = lm.calc_spatial_features(sp_cxt_feats, sp_ann_feats,
                                                          local_sp_coord, global_sp_coord)
        
        logprobs = lm(vis_feats, sp_feats, sp_feats_emb, 
                      coord, seqz, lang_last_ind)
        
        
        # lang loss
        pairP, vis_unpairP, lang_unpairP  = F.split_axis(logprobs, 3, axis = 1)
        pair_num, _, lang_unpair_num = np.split(lang_last_ind, 3)
        num_labels = {'T':xp.array(pair_num),'F':xp.array(lang_unpair_num)}
        lm_flows   = {'T':pairP, 'visF':[pairP, vis_unpairP], 'langF':[pairP, lang_unpairP]}
        lm_loss    = lm_crits(lm_flows, num_labels, params['lm_margin'], 
                             vlamda=params['vis_rank_weight'], llamda=params['lang_rank_weight'])
        
        # RL loss (pos,pos)
        rl_vis_feats = F.split_axis(vis_feats, 3, axis=0)[0]
        rl_coord     = np.split(coord, 3, axis=0)[0]
        rl_sp_vis_feats = F.split_axis(sp_feats, 3, axis=0)[0]
        rl_sp_vis_emb   = F.split_axis(sp_feats_emb, 3, axis=0)[0]
        sampled_seq, sample_log_probs = lm.sample(rl_vis_feats, rl_sp_vis_feats, rl_sp_vis_emb, rl_coord)
        sampled_lang_last_ind = calc_max_ind(sampled_seq)
        rl_loss = rl_crit(pos_feats, pos_sp_cxt_feats, rl_coord, sampled_seq, sample_log_probs, sampled_lang_last_ind)
        
        loss = lm_loss + rl_loss
        print(lm_loss, rl_loss)
        
        if params['dataset']=='refgta' and params['ranking'] and iteration>8000:
            lam += 0.4/8000
            score = F.sum(pairP, axis=0)/(xp.array(pair_num+1))
            rank_loss = calc_rank_loss(score, data['rank'], margin=0.01)*lam
            loss +=rank_loss
        loss.backward()
        
        ve_optim.update()
        lm_optim.update()
        
        if data['bounds']['wrapped']:
            print('one epoch finished!')
            loader.shuffle('train')
            
                
        if iteration % params['losses_log_every']==0:
            acc = xp.where(rl_crit.reward>0.5, 1, 0).mean()
            print('{} iter : train loss {}, acc : {} reward_mean : {}'.format(iteration,loss.data, acc, rl_crit.reward.mean()))
        
            
        if (iteration % params['save_checkpoint_every'] == 0 and iteration >0):
            chainer.config.train = False
            chainer.config.enable_backprop = False
            loader.resetImageIterator('val')
            loss_sum = 0
            loss_generation = 0
            loss_lm_margin = 0
            loss_evals = 0
            accuracy = 0
            rank_acc = 0
            rank_num = 0
            while True:
                data     = loader.getImageBatch('val', params)
                image_id = data['image_id']
                img_ann_ids = data['img_ann_ids']
                sent_ids = data['sent_ids']
                gd_ixs   = data['gd_ixs']
                feats    = Variable(xp.array(data['feats'], dtype=xp.float32))
                sp_cxt_feats = Variable(xp.array(data['sp_cxt_feats'], dtype=xp.float32))
                sp_ann_feats = Variable(xp.array(data['sp_ann_feats'], dtype=xp.float32))
                local_shapes = data['local_shapes']
                seqz = data['seqz']
                lang_last_ind = calc_max_ind(seqz)
                scores = []
                for i, sent_id in enumerate(sent_ids):
                    
                    gd_ix  = gd_ixs[i]
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
                    
                    gd_ix = gd_ixs[i]
                    lm_generation_loss = lm_crits({'T':logprobs[:, gd_ix, xp.newaxis]}, 
                                                  {'T':one_last_ind[gd_ix,np.newaxis]}, 
                                                  params['lm_margin'], 
                                                  vlamda=0, llamda=0).data
                    lm_scores = -computeLosses(logprobs, one_last_ind)  
                    lm_margin_loss, pos_sc, max_neg_sc  = compute_margin_loss(lm_scores, gd_ix, params['lm_margin'])
                    scores.append(lm_scores[gd_ix])
                    
                    loss_generation += lm_generation_loss
                    loss_lm_margin += lm_margin_loss
                    loss_sum += lm_generation_loss + lm_margin_loss
                    loss_evals += 1
                    if pos_sc > max_neg_sc:
                        accuracy +=1
                if params['dataset']=='refgta':
                    rank_a, rank_n = calc_rank_acc(scores, data['rank'])
                    rank_acc += rank_a
                    rank_num += rank_n 
                print('{} iter | {}/{} validating acc : {}'.format(iteration, 
                                                                   data['bounds']['it_pos_now'], 
                                                                   data['bounds']['it_max'], 
                                                                   accuracy/loss_evals))
                
                if data['bounds']['wrapped']:
                    print('validation finished!')
                    fin_val_loss = cuda.to_cpu(loss_sum/loss_evals)
                    loss_generation = cuda.to_cpu(loss_generation/loss_evals)
                    loss_lm_margin = cuda.to_cpu(loss_lm_margin/loss_evals)
                    fin_val_acc = accuracy/loss_evals
                    break
            val_loss_history.append(fin_val_loss)
            val_loss_lm_s_history.append(loss_generation)
            val_loss_lm_l_history.append(loss_lm_margin)
            val_acc_history.append(fin_val_acc)
            if min_val_loss>fin_val_loss:
                print('val loss {} -> {} improved!'.format(min_val_loss, val_loss_history[-1]))
                min_val_loss = fin_val_loss
                serializers.save_hdf5(osp.join(model_dir, params['id']+params['id2']+"ve.h5"), ve)
                serializers.save_hdf5(osp.join(model_dir, params['id']+params['id2']+"lm.h5"), lm)
                
            ## graph
            plt.title("accuracy")
            plt.plot(np.arange(len( val_acc_history)),  val_acc_history, label="val_accuracy")
            plt.legend()
            plt.savefig(os.path.join(graph_dir, params['id'] +params['id2']+ "_joint_acc.png"))
            plt.close()

            plt.title("loss")
            plt.plot(np.arange(len(val_loss_history)), val_loss_history, label="all_loss")
            plt.plot(np.arange(len(val_loss_history)), val_loss_lm_s_history, label="generation_loss")
            plt.legend()
            plt.savefig(os.path.join(graph_dir, params['id'] +params['id2']+ "_joint_loss.png"))
            plt.close()
            
            plt.title("loss")
            plt.plot(np.arange(len(val_loss_history)), val_loss_lm_l_history, label="lm_comp_loss")
            plt.legend()
            plt.savefig(os.path.join(graph_dir, params['id'] +params['id2']+ "_joint_comp_loss.png"))
            plt.close()
            
            
            if params['dataset'] == 'refgta':
                val_rank_acc_history.append(rank_acc/rank_num)
                plt.title("rank loss")
                plt.plot(np.arange(len(val_rank_acc_history)), val_rank_acc_history, label="rank_acc")
                plt.legend()
                plt.savefig(os.path.join(graph_dir, params['id'] +params['id2']+ "_joint_rank_acc.png"))
                plt.close()
            
        if iteration > params['learning_rate_decay_start'] and params['learning_rate_decay_start'] >= 0:
            frac = (iteration - params['learning_rate_decay_start']) / params['learning_rate_decay_every']
            decay_factor = math.pow(0.1, frac)
            ve_optim.alpha *= decay_factor
            lm_optim.alpha *= decay_factor
            
        iteration+=1
                
                    
if __name__ == '__main__':

    args = config.parse_opt()
    params = vars(args) # convert to ordinary dict
    train_all(params)
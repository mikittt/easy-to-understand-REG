import numpy as np
import chainer.functions as F
from chainer import Variable
from chainer import cuda
import sys
import os.path as osp

def compute_margin_loss(scores, gd_ix, margin):
	scores = scores.copy()
	pos_sc = scores[gd_ix].copy()
	scores[gd_ix] = -1e5
	max_neg_sc = scores.max()
	loss = max([0, margin + max_neg_sc - pos_sc])
	return loss, pos_sc, max_neg_sc

def computeLosses(logprobs, lang_num):
    xp = cuda.get_array_module(logprobs)
    lang_loss = -xp.sum(logprobs, axis=0)/(xp.array(lang_num+1))
    return lang_loss
    
def calc_rank_loss(score, ranks, margin=0.05):
    xp   = cuda.get_array_module(score)
    cnt  = 0
    cnt_ = 0
    loss = 0
    for i, rank in enumerate(ranks):
        # rank : [[],[0,2],[]]
        for j, one in enumerate(rank):
            # one : [0,2]
            for o in one:
                loss += F.relu(margin+score[cnt_+o]-score[cnt_+j])
                cnt += 1
        cnt_+=len(rank)
    if cnt>0:
        return loss/cnt
    else:
        return Variable(np.array(0))
    
def calc_rank_acc(score, ranks):
    xp   = cuda.get_array_module(score)
    cnt  = 0
    cnt_ = 0
    acc  = 0
    for i, rank in enumerate(ranks):
        for j, one in enumerate(rank):
            for o in one:
                if score[cnt_+j]>score[cnt_+o]:
                    print(score[cnt_+j],score[cnt_+o])
                    acc+=1
                cnt+=1
                
        cnt_+=len(rank)
    return acc, cnt

def language_eval(pred, split, params):
    sys.path.insert(0, osp.join('pyutils', 'refer2'))
    if 'refcoco' in params['dataset']:
        image_root = params['coco_image_root']
    else:
        image_root = params['gta_image_root']
    from refer import REFER
    refer = REFER(params['data_root'], image_root, params['dataset'], params['splitBy'], old_version=False)
		
    sys.path.insert(0, osp.join('pyutils', 'refer2', 'evaluation'))
    from refEvaluation import RefEvaluation
    eval_cider_r = params['dataset']=='refgta'
    refEval = RefEvaluation(refer, pred, eval_cider_r=eval_cider_r)
    refEval.evaluate()
    overall = {}
    for metric, score in refEval.eval.items():
        overall[metric] = score
    print (overall)
    from crossEvaluation import CrossEvaluation
    ceval = CrossEvaluation(refer, pred)
    ceval.cross_evaluate()
    ceval.make_ref_to_evals()
    ref_to_evals = ceval.ref_to_evals 
    ceval.Xscore('CIDEr')
    return overall

    
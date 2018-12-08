import config
import sys
import os.path as osp
import json
import numpy as np

from tqdm import tqdm

import chainer
from chainer import Variable, cuda, serializers
import chainer.functions as F

import cplex ## install by conda install -c ibmdecisionoptimization cplex

from models.Reinforcer import ListenerReward
from misc.utils import calc_max_ind
from misc.DataLoader import DataLoader


def main(params):
    target_save_dir = osp.join(params['save_dir'],'prepro', params['dataset']+'_'+params['splitBy'])
    model_dir = osp.join(params['save_dir'], 'model', params['dataset']+'_'+params['splitBy'])
    
    if params['old']:
        params['data_json'] = 'old'+params['data_json']
        params['data_h5'] = 'old'+params['data_h5']
        params['image_feats'] = 'old'+params['image_feats']
        params['ann_feats'] = 'old'+params['ann_feats']
        params['id'] = 'old'+params['id']
        
    if params['dataset'] in ['refcoco', 'refcoco+', 'refcocog']:
        global_shapes = (224, 224)
        image_root = params['coco_image_root']
    elif params['dataset'] == 'refgta':
        global_shapes = (480, 288)
        image_root = params['gta_image_root']
        
    with open(target_save_dir+params["split"]+'_'+params['id']+params['id2']+str(params['beam_width'])+'.json') as f:
        data =  json.load(f)
    ref_to_beams = {item['ref_id']: item['beam'] for item in data}
    
    # add ref_id to each beam
    for ref_id, beams in ref_to_beams.items():
        for beam in beams:  
            beam['ref_id'] = ref_id  # make up ref_id in beam
            
    loader = DataLoader(params)
    featsOpt = {'sp_ann':osp.join(target_save_dir, params['sp_ann_feats']),
                'ann_input':osp.join(target_save_dir, params['ann_feats']),
                'img':osp.join(target_save_dir, params['image_feats']),
               'shapes':osp.join(target_save_dir, params['ann_shapes'])}
    loader.loadFeats(featsOpt) 
    loader.shuffle('train')
    loader.loadFeats(featsOpt) 
    chainer.config.train = False
    chainer.config.enable_backprop = False
    
    gpu_id = params['gpu_id']
    cuda.get_device(gpu_id).use()
    xp = cuda.cupy
    
    rl_crit = ListenerReward(len(loader.ix_to_word), global_shapes=global_shapes).to_gpu(gpu_id)
    serializers.load_hdf5(osp.join(model_dir, params['id']+".h5"), rl_crit)
    #serializers.load_hdf5(osp.join(model_dir, "attn_rank.h5"), rl_crit)
    img_to_ref_ids, img_to_ref_confusion = calc_confusion(loader, data, ref_to_beams, rl_crit, params, xp)
    
    sys.path.insert(0, osp.join('pyutils', 'refer2'))
    sys.path.insert(0, osp.join('pyutils', 'refer2', 'evaluation'))
    from refer import REFER
    from refEvaluation import RefEvaluation
    from crossEvaluation import CrossEvaluation
    refer = REFER(params['data_root'], image_root, params['dataset'], params['splitBy'], old_version=params['old'])
    
    if params['dataset'] == 'refcoco':
        lambda1 = 5  
        lambda2 = 5
    elif params['dataset'] == 'refcoco+':
        lambda1 = 5
        lambda2 = 5
    elif params['dataset'] == 'refcocog':
        lambda1 = 5
        lambda2 = 5
    elif params['dataset'] == 'refgta':
        lambda1 = 5
        lambda2 = 5
    else:
        error('No such dataset option for ', params['dataset'])
        

    # compute unary potential, img_to_ref_unary
    # let's firstly try one image
    Res = []
    for image_id in img_to_ref_confusion:
        # ref_ids and confusion matrices for this image
        img_ref_ids = img_to_ref_ids[image_id]
        ref_to_confusion = img_to_ref_confusion[image_id]
        # compute unary potential for each ref_id
        for ref_id in img_ref_ids:
            confusion = ref_to_confusion[ref_id]  # (beam_size, #img_ref_ids)
            beams = ref_to_beams[ref_id]  # [{ppl, sent, logp}] of beam_size
            compute_unary(ref_id, beams, confusion, img_ref_ids, lambda1, lambda2)

        # here's more preparation
        ref_beam_to_ix, ix_to_ref_beam, all_beams = make_index(img_ref_ids, ref_to_beams)

        # compute pairwise potentials
        pairwise_ref_beam_ids = compute_pairwise(img_ref_ids, ref_to_beams)

        # call cplex
        res = bilp(img_ref_ids, ref_to_beams, all_beams, pairwise_ref_beam_ids, ref_beam_to_ix, loader)
        Res += res
    # evaluate
    eval_cider_r = params['dataset']=='refgta'
    refEval = RefEvaluation(refer, Res, eval_cider_r=eval_cider_r)
    refEval.evaluate()
    overall = {}
    for metric, score in refEval.eval.items():
        overall[metric] = score
    print (overall)

    if params['write_result'] > 0:
        refToEval = refEval.refToEval
        for res in Res:
            ref_id, sent = res['ref_id'], res['sent']
            refToEval[ref_id]['sent'] = sent
        with open('' + params['id'] +params['id2']+ '_out.json', 'w') as outfile:
            json.dump({'overall': overall, 'refToEval': refToEval}, outfile)

    # CrossEvaluation takes as input [{ref_id, sent}]
    ceval = CrossEvaluation(refer, Res)
    ceval.cross_evaluate()
    ceval.make_ref_to_evals()
    ref_to_evals = ceval.ref_to_evals  # ref_to_evals = {ref_id: {ref_id: {method: score}}}

    # compute cross score
    xcider = ceval.Xscore('CIDEr')

def calc_confusion(loader, data, ref_to_beams, rl_crit, params, xp):
    img_to_ref_ids = {}
    for item in data:
        ref_id = item['ref_id']
        image_id = loader.Refs[ref_id]['image_id']
        if image_id not in img_to_ref_ids:
            img_to_ref_ids[image_id] = []
        img_to_ref_ids[image_id].append(ref_id)
        
    img_to_ref_confusion = {}
    for image_id in tqdm(img_to_ref_ids):
        img_to_ref_confusion[image_id] = {}
        img_ref_ids = img_to_ref_ids[image_id]
        img_ann_ids = [loader.Refs[ref_id]['ann_id'] for ref_id in img_ref_ids]
        sp_cxt_feats, _, feats, local_shapes = loader.fetch_feats(img_ann_ids, 1, params)
        
        feats = Variable(xp.array(feats, dtype=xp.float32))
        sp_cxt_feats = Variable(xp.array(sp_cxt_feats, dtype=xp.float32))
        for ref_id in img_ref_ids:
            sents = [one['sent'] for one in ref_to_beams[ref_id]]
            lang_last_ind = np.array([len(sent) for sent in sents])
            seqz = loader.encode_sequence(sents)
            lang_last_ind = calc_max_ind(seqz)
            out_score = []
            for one_ind in range(len(seqz)):
                one_seq = Variable(xp.array([seqz[one_ind] for _ in range(feats.shape[0])], dtype=xp.int32))
                one_last_ind = [lang_last_ind[one_ind] for _ in range(feats.shape[0])]
                coord = cuda.to_cpu(feats[:, sum(rl_crit.ve.feat_ind[:1]):sum(rl_crit.ve.feat_ind[:2])].data)
                score = cuda.to_cpu(F.sigmoid(rl_crit.calc_score(feats, sp_cxt_feats, 
                                                                 coord, one_seq, one_last_ind)).data)[:,0]
                out_score.append(score)
            print(np.array(out_score).shape)            
            img_to_ref_confusion[image_id][ref_id] = np.array(out_score)
    return img_to_ref_ids, img_to_ref_confusion

def rerank(ref_id, beams, confusion, img_ref_ids, lambda1=0, lambda2=0):
    """
    input:
    - ref_id        : ref_id
    - beams         : [{ppl, sent, logp}] of beam_size
    - confusion     : (beam_size, #img_ref_ids) array 
    - img_ref_ids   : ref_ids within this image
    output:
    - beams         : reranked [{ppl, sent, loop}]
    """
    assert len(beams) == len(confusion)

    rerank_sc = []
    rix = img_ref_ids.index(ref_id)

    for b in range(len(beams)):
        # score 1: ppl
        ppl = -beams[b]['ppl']
        # score 2: self correlatoin, cossim(ref, beam_sent)
        cossim = list(confusion)
        self_sc = cossim[b][rix]
        # score 3: max cross correlation, -max_cossim(other ref, beam_sent)
        cossim[b][rix] = -1e5
        cross_sc = -max(cossim[b])
        # add to rerank_sc
        rerank_sc.append( ppl + lambda1 * self_sc + lambda2 * cross_sc )

        # print ppl, lambda1 * self_sc, lambda2 * cross_sc

    rerank_sc = np.array(rerank_sc)
    sort_ixs  = np.argsort(rerank_sc).tolist()
    sort_ixs = sort_ixs[::-1]  # from big to small
    reranked_beams = []
    for ix in sort_ixs:
        reranked_beams.append(beams[ix])

    return reranked_beams


def bilp(ref_ids, ref_to_beams, all_beams, pairwise_ref_beam_ids, ref_beam_to_ix, loader):
    """
    input: within one image we are given
    - ref_ids      			: img_ref_ids
    - ref_to_beams 			: ref_id -> beams, where beams = [{ref_id, ppl, logp, sent, unary}]
    - all_beams    			: all beams in ref_ids order
    - pairwise_ref_beam_ids : pairwise potential on duplicate ref_beam_ids 
    - ref_beam_to_ix        : ref_beam_id -> ix in all beams, used for constraining (ix1, ix2, ix12)
    """
    potential = []
    _r = []
    _c = []
    _s = []
    _sense = ''
    _rhs = []
    constraints = 0

    # set up potential of beam unaries in ref_ids order
    for ref_id in ref_ids:
        beams = ref_to_beams[ref_id]
        for b in beams:
            potential.append(b['unary'])

    # set up constraints, we can only pick one sent from 10 beams for each ref_id
    blk = 0
    for rix, ref_id in enumerate(ref_ids):
        num_beams = len(ref_to_beams[ref_id])
        _r += [rix] * num_beams
        _c += (blk+np.arange(num_beams)).tolist()
        _s += [1] * num_beams
        _sense += 'E'
        _rhs += [1]
        blk += num_beams  # start_ix for next ref_id in potential
    constraints += len(ref_ids)

    # set up pairwise terms, we penalize duplicates by -10
    for ref_beam_id1, ref_beam_id2 in pairwise_ref_beam_ids:
        ix1 = ref_beam_to_ix[ref_beam_id1]
        ix2 = ref_beam_to_ix[ref_beam_id2]
        # 1) add pairwise potential
        potential += [-10]  # penalize -10
        # 2) x_ab - x_a <= 0
        _r += [constraints]*2
        _c += [len(potential)-1, ix1]
        _s += [1, -1]
        _rhs += [0]
        _sense += 'L'
        constraints += 1
        # 3) x_ab - x_b <= 0
        _r += [constraints]*2
        _c += [len(potential)-1, ix2]
        _s += [1, -1]
        _rhs += [0]
        _sense += 'L'
        constraints += 1
        # 4) x_a + x_b - x_ab <= 1
        _r += [constraints]*3
        _c += [ix1, ix2, len(potential)-1]
        _s += [1, 1, -1]
        _rhs += [1]
        _sense += 'L'
        constraints += 1

    # solve
    prob = cplex.Cplex()
    # objective function
    prob.objective.set_sense(prob.objective.sense.maximize)
    # variable constraints
    lb = np.zeros(len(potential))
    ub = np.ones(len(potential))
    ctype = 'I' * len(potential)
    prob.variables.add(obj=potential, lb=lb, ub=ub, types=ctype)
    # linear constraints
    prob.linear_constraints.add(rhs=_rhs, senses=_sense)
    prob.linear_constraints.set_coefficients(zip(_r, _c, _s))
    # solve
    try:
        prob.solve()
        values = prob.solution.get_values()
        # convert to integer
        values = [int(round(v)) for v in values]
        result = []
        for ix, v in enumerate(values[:len(all_beams)]):
            if int(round(v)) == 1:
                target = all_beams[ix]
                target['sent'] = ' '.join([loader.ix_to_word[str(w)] for w in target['sent']])
                result += [all_beams[ix]]
    except:
        result = []
        for ref_id in ref_ids:
            target = ref_to_beams[ref_id][0]
            target['sent'] = ' '.join([loader.ix_to_word[str(w)] for w in target['sent']])
            result += [target]
        print(result)
    return result


def compute_unary(ref_id, beams, confusion, img_ref_ids, lambda1, lambda2):
    """
    input:
    - ref_id        : ref_id
    - beams         : [{ppl, sent, logp}] of beam_size
    - confusion     : (beam_size, #img_ref_ids) array 
    - img_ref_ids   : ref_ids within this image
    output:
    - beams         : [{ppl, sent, logp, ref_id, unary}]
    """
    assert len(beams) == len(confusion)

    rix = img_ref_ids.index(ref_id)
    for b in range(len(beams)):
        # score 1: -ppl
        ppl_sc = -beams[b]['ppl']
        # score 2: self correlatoin, cossim(ref, beam_sent)
        cossim = np.array(confusion, copy=True)  
        self_sc = cossim[b][rix]
        # score 3: max cross correlation, -max_cossim(other ref, beam_sent)
        cossim[b][rix] = -1e5
        cross_sc = -max(cossim[b])
        # unary potential
        beams[b]['unary'] = ppl_sc + lambda1 * self_sc + lambda2 * cross_sc


def make_index(img_ref_ids, ref_to_beams):
    """
    input:
    - img_ref_ids 	 : list of ref_id in one image
    - ref_to_beams   : ref_id -> [{ppl, sent, logp, (ref_id)}]
    output:
    - ix_to_ref_beam : ix in all_beams -> ref_beam_id, here ref_beam_id means ref_id_beam_ix
    - ref_beam_to_ix : ref_beam_id -> ix in all_beams, here ref_beam_id means ref_id_beam_ix
    - all_beams      : all beams in img_ref_ids order
    """
    all_beams = []
    ix_to_ref_beam = {}
    ref_beam_to_ix = {}
    ix = 0
    for ref_id in img_ref_ids:
        beams = ref_to_beams[ref_id]
        for b, beam in enumerate(beams):
            ref_beam_to_ix[str(ref_id)+'_'+str(b)] = ix  # ref_beam_id = ref_id_beam_ix
            ix_to_ref_beam[ix] = str(ref_id)+'_'+str(b)
            all_beams += [beam]
            ix += 1
    return ref_beam_to_ix, ix_to_ref_beam, all_beams


def compute_pairwise(img_ref_ids, ref_to_beams):
    """
    input:
    - img_ref_ids : list of ref_ids in this image
    - ref_to_beams: ref_id -> beams, where beams = [{ppl, sent, logp, (ref_id), (unary) }]
    return:
    - pairwise_ref_beam_ids: [(ref_beam_id, ref_beam_id), (...), ...], here ref_beam_id means ref_id_beam_ix 
    We will use ref_beam_to_ix to index the unary potential later on.
    """
    def check_duplicate(beams1, beams2):
        duplicate = []
        for b1, beam1 in enumerate(beams1):
            for b2, beam2 in enumerate(beams2):
                if beam1['sent'] == beam2['sent']:
                    duplicate += [(b1, b2)]
        return duplicate

    pairwise = []
    for ref_id1 in img_ref_ids:
        beams1 = ref_to_beams[ref_id1]
        for ref_id2 in img_ref_ids:
            if ref_id2 != ref_id1:
                beams2 = ref_to_beams[ref_id2]
                duplicate = check_duplicate(beams1, beams2)
                for b1, b2 in duplicate:
                    pairwise += [ (str(ref_id1)+'_'+str(b1), str(ref_id2)+'_'+str(b2)) ]

    print ('Among %s img_ref_ids, %s pairs are found duplicate.' % (len(img_ref_ids), len(pairwise)))
    return pairwise


def check_pairwise(pairwise, ref_beam_to_ix, all_beams):
    print ('Among %s img_refs' % (len(all_beams)/10))
    for ref_beam_id1, ref_beam_id2 in pairwise:
        ix1 = ref_beam_to_ix[ref_beam_id1]
        ix2 = ref_beam_to_ix[ref_beam_id2]
        beam1 = all_beams[ix1]
        beam2 = all_beams[ix2]
        print([(beam1['ref_id'], beam1['sent']), (beam2['ref_id'], beam2['sent'])])

if __name__ == '__main__':

    args = config.parse_opt()
    params = vars(args) # convert to ordinary dict
    main(params)
    
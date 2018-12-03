import json
import os.path as osp
import random
import numpy as np
import h5py
import chainer.functions as F

class DataLoader:
    def __init__(self, opt):
        self.target_save_dir = osp.join(opt['save_dir'], 'prepro', opt['dataset']+'_'+opt['splitBy'])
        self.dataset = opt['dataset']
        if self.dataset in ['refcoco','refcoco+','refcocog']:
            self.global_num=49
        else:
            self.global_num=135
        print('DataLoader loading data.json')
        with open(osp.join(self.target_save_dir,opt["data_json"])) as f:
            self.info = json.load(f)
        self.ix_to_word = self.info['ix_to_word']
        self.word_to_ix = self.info['word_to_ix']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)
        self.ix_to_cat = self.info['ix_to_cat']
        print('object category size is ',len(self.ix_to_cat))
        self.images = self.info['images']
        self.anns = self.info['anns']
        self.refs = self.info['refs']
        
        # open hdf5 file
        print('DataLoader loading data.h5')
        self.data_h5 = h5py.File(osp.join(self.target_save_dir, opt['data_h5']), 'r')
        self.data_zseq = self.data_h5['/zseq_labels'].value
        self.data_seqz = self.data_h5['/seqz_labels'].value
        self.seq_length = self.data_seqz.shape[1]
        print('max sequence length in data is ' , self.seq_length)
        
        # construct Refs, Images, Anns, Sentences and annToRef
        self.Refs, self.Images, self.Anns, self.Sentences, self.annToRef = {}, {}, {}, {}, {}
        for ref in self.info['refs']:
            self.Refs[ref['ref_id']] = ref
            self.annToRef[ref['ann_id']] = ref
        for image in self.info['images']:
            self.Images[image['image_id']] = image
        for ann in self.info['anns']:
            self.Anns[ann['ann_id']] = ann
        for sent in self.info['sentences']:
            self.Sentences[sent['sent_id']] = sent
        
        self.sentToRef = {}
        for ref in self.refs:
            for sent_id in ref['sent_ids']:
                self.sentToRef[sent_id] = ref
        
        if opt['dataset']=='refgta':
            self.sentToInfo = self.info['sents_info']
        # ref iterators for each split
        self.split_ix = {}
        self.iterators = {}
        for ref_id in self.Refs:
            split = self.Refs[ref_id]['split']
            if split not in self.split_ix:
                self.split_ix[split] = []
                self.iterators[split] = 0
            self.split_ix[split].append(ref_id)
        for split in self.split_ix:
            print('assigned {} refs to split {}'.format(len(self.split_ix[split]), split))
            
        # sent iterators for each split
        self.sent_split_ix = {}
        self.sent_iterators = {}
        for sent_id in self.Sentences:
            split = self.sentToRef[sent_id]['split']
            if split not in  self.sent_split_ix:
                self.sent_split_ix[split] = []
                self.sent_iterators[split] = 0
            self.sent_split_ix[split].append(sent_id)
        for split in self.sent_split_ix:
            print('assigned {} sents to split {}'.format(len(self.sent_split_ix[split]), split))
            
        #image iterators for each split
        self.img_split_ix = {}
        self.img_iterators = {}
        for image_id in self.Images:
            split_names = []
            for ref_id in self.Images[image_id]['ref_ids']:
                if self.Refs[ref_id]['split'] not in split_names:
                    split_names.append(self.Refs[ref_id]['split'])
            for split in split_names:
                if split not in self.img_split_ix:
                    self.img_split_ix[split] = []
                    self.img_iterators[split] = 0
                self.img_split_ix[split].append(image_id)
        for split in self.img_split_ix:
            print('assigned {} images to split {}'.format(len(self.img_split_ix[split]), split))
    
    def shuffle(self, split):
        random.shuffle(self.split_ix[split])
        
    def resetIterator(self, split):
        self.iterators[split] = 0
    
    def resetSentIterator(self, split):
        self.sent_iterators[split] = 0

    def resetImageIterator(self, split):
        self.img_iterators[split] = 0
        
    def setImageIterator(self, split, ind):
        self.img_iterators[split] = ind
            
    def loadFeats(self, featsOpt, mmap_mode=True):
        print('loading feats...')
        self.feats = {}
        for key in featsOpt:
            if mmap_mode:
                self.feats[key] = np.load(featsOpt[key], mmap_mode='r')
            else:
                self.feats[key] = np.load(featsOpt[key])
        print('Done!')
        
    def decode_sequence(self, seq, lang_last_ind):
        sents = []
        for b in range(seq.shape[0]):
            sent = []
            for i in range(lang_last_ind[b]):
                sent.append(self.ix_to_word[str(seq[b,i])])
            sents.append(sent)
        return sents
    
    def encode_sequence(self, seq):
        sents = np.zeros((len(seq), self.seq_length), dtype=np.int32)
        for b in range(len(seq)):
            sents[b, :len(seq[b])] = seq[b][:self.seq_length]
        return sents
    
    def estimate_time_range(self, times):
        
        SD = np.sqrt(np.var(times)*len(times)/(len(times)-1))
        SE = SD/np.sqrt(len(times))
        mean = np.mean(times)
        min_time, max_time = mean-1*SE, mean+1*SE
        '''
        times = sorted(times)[1:-1]
        min, max = np.min(times), np.max(times)
        '''
        return min_time, max_time
    
    def rank_sent_ids(self, sent_ids):
        acc = []
        time_range = []
        for sent_id in sent_ids:
            sent_info = self.sentToInfo[str(sent_id)]['info']
            acc.append(sum([one_info['if_true'] for one_info in sent_info]))
            time_range.append(self.estimate_time_range(sorted([one_info['time'] for one_info in sent_info])[1:-1]))
        acc = np.array(acc)
        time_range = np.array(time_range)
        
        rank = [[] for _ in range(len(acc))]
        same_acc = {}
        for i in range(len(acc)):
            larger = np.where(acc[i]>acc)[0]
            rank[i].extend(larger)
            larger = len(larger)
            if acc[i]==5:
                if larger not in same_acc:
                    same_acc[larger] = []
                same_acc[larger].append(i)
        for key in same_acc:
            if len(same_acc[key])>1:
                one_pair = np.array(same_acc[key])
                for one in one_pair:
                    rank[one].extend(one_pair[np.where(time_range[one][1]<time_range[one_pair][:,0])])
        return rank
    
    def fetch_sent_ids(self, ref_id, num_sents):
        ref = self.Refs[ref_id]
        ref_sent_num = len(ref['sent_ids'])
        if ref_sent_num<num_sents:
            picked_sent_ids = ref['sent_ids']
            for _ in range(num_sents-ref_sent_num):
                picked_sent_ids.append(ref['sent_ids'][random.randint(0, ref_sent_num-1)])
        else:
            picked_sent_ids = ref['sent_ids']
            random.shuffle(picked_sent_ids)
            picked_sent_ids = picked_sent_ids[:num_sents]
        return picked_sent_ids
    
    def fetch_feats(self, batch_ann_ids, expand_size, opt):
        batch_size = len(batch_ann_ids)
        h5_id_list = []
        img_list = []
        l_feats = []
        for ann_id in batch_ann_ids:
            ann = self.Anns[ann_id]
            h5_id_list.append(ann['h5_id'])
            image = self.Images[ann['image_id']]
            img_list.append(image['h5_id'])
            x, y, w, h = ann['box']
            iw, ih = image['width'], image['height']
            l_feats.append([x/iw, y/ih, (x+w)/iw, (y+h)/ih, w*h/(iw*ih)])
        local_shapes = np.tile(
            self.feats['shapes'][h5_id_list].reshape((batch_size, 1, -1)),(1, expand_size, 1)
        ).reshape((batch_size*expand_size, -1))
        sp_cxt_feats = np.tile(
            self.feats['img'][img_list].reshape((batch_size, 1, self.global_num, -1)), (1, expand_size, 1, 1)
        ).reshape((batch_size*expand_size, self.global_num, -1))
        sp_ann_feats = np.tile(
            self.feats['sp_ann'][h5_id_list].reshape((batch_size, 1, 36, -1)), (1, expand_size, 1, 1)
        ).reshape((batch_size*expand_size, 36, -1)) 
        cxt_feats = np.tile(
            self.feats['img'][img_list].mean(axis=1).reshape((batch_size, 1, -1)), (1, expand_size, 1)
        ).reshape((batch_size*expand_size, -1))
        ann_feats = np.tile(
            self.feats['ann_input'][h5_id_list].reshape((batch_size, 1, -1)), (1, expand_size, 1)
        ).reshape((batch_size*expand_size, -1))
        l_feats = np.tile(
            np.array(l_feats, dtype=np.float32).reshape((batch_size, 1, -1)), (1, expand_size, 1)
        ).reshape((batch_size*expand_size, -1))
        df, dlf = self.fetch_dif_feats(batch_ann_ids, expand_size, opt)
        df_feats = np.tile(df.reshape((batch_size, 1, -1)), (1, expand_size, 1)).reshape((batch_size*expand_size, -1))
        dlf_feats = np.tile(dlf.reshape((batch_size, 1, -1)), (1, expand_size, 1)).reshape((batch_size*expand_size, -1))
        feats = np.concatenate([ann_feats, l_feats, df_feats, dlf_feats], axis=1)
        return sp_cxt_feats, sp_ann_feats, feats, local_shapes
        
    def fetch_seqs(self, batch_sent_ids, opt):
        seq = []
        for sent_id in batch_sent_ids:
            sent = self.Sentences[sent_id]
            if opt['pad_zero'] == 'front':
                seq.append(self.data_zseq[sent['h5_id']])
            else:
                seq.append(self.data_seqz[sent['h5_id']])
        return np.array(seq, dtype=np.int32)
        
    def fetch_neighbour_ids(self, ref_ann_id):
        ref_ann = self.Anns[ref_ann_id]
        x, y, w, h = ref_ann['box']
        rx, ry = x+w/2, y+h/2
        def calc_distance_from_target(ann_id):
            x, y, w, h = self.Anns[ann_id]['box']
            ax, ay = x+w/2, y+h/2
            return (rx-ax)**2 + (ry-ay)**2 
        image = self.Images[ref_ann['image_id']]
        ann_ids = image['ann_ids']
        ann_ids = sorted([[ann_id,calc_distance_from_target(ann_id)] for ann_id in ann_ids], key=lambda x: x[1])
        ann_ids = [ann_id[0] for ann_id in ann_ids]
        st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = [], [], [], []
        for ann_id in ann_ids:
            if ann_id != ref_ann_id:
                if self.Anns[ann_id]['category_id'] == ref_ann['category_id']:
                    st_ann_ids.append(ann_id)
                    if ann_id in self.annToRef:
                        st_ref_ids.append(self.annToRef[ann_id]['ref_id'])
                else:
                    dt_ann_ids.append(ann_id)
                    if ann_id in self.annToRef:
                        dt_ref_ids.append(self.annToRef[ann_id]['ref_id'])
                    
        return st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids
        
    def fetch_dif_feats(self, batch_ann_id, expand_size, opt):
        batch_size = len(batch_ann_id)
        dif_ann_feats = np.zeros((batch_size, 2048), dtype=np.float32)
        dif_lfeats = np.zeros((batch_size, 5*5), dtype=np.float32)
        for i, ann_id in enumerate(batch_ann_id):
            _, st_ann_ids, _, _= self.fetch_neighbour_ids(ann_id)
            if len(st_ann_ids)==0:
                continue
            cand_ann_feats = self.feats['ann_input'][[self.Anns[st_id_]['h5_id'] for st_id_ in st_ann_ids[:5]]]
            ref_ann_feat = self.feats['ann_input'][self.Anns[ann_id]['h5_id']]
            dif_ann_feat = np.mean(cand_ann_feats-ref_ann_feat, axis=0)
            image = self.Images[self.Anns[ann_id]['image_id']]
            rbox = self.Anns[ann_id]['box']
            rcx, rcy, rw, rh = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
            dif_lfeat = []
            for j  in range(min(5, len(st_ann_ids))):
                cbox = self.Anns[st_ann_ids[j]]['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                dif_lfeat.extend([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
            dif_ann_feats[i]=dif_ann_feat
            dif_lfeats[i,:len(dif_lfeat)]=dif_lfeat
        return dif_ann_feats, dif_lfeats
        
    def sample_neg_ids(self, pos_ann_ids, opt):
        neg_ann_ids, neg_sent_ids = [],[]
        for pos_ann_id in pos_ann_ids:
            st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = self.fetch_neighbour_ids(pos_ann_id)
            cand_ann_ids, cand_cossim = [],[]
            for k in range(opt['seq_per_ref']):
                
                if len(st_ann_ids) > 0 and random.random() < opt['sample_ratio']:
                    ix = random.randint(0, len(st_ann_ids)-1)
                    neg_ann_id = st_ann_ids[ix]
                elif len(dt_ann_ids) > 0:
                    ix = random.randint(0, len(dt_ann_ids)-1)
                    neg_ann_id = dt_ann_ids[ix]
                else:
                    ix = random.randint(0, len(self.anns)-1)
                    neg_ann_id = self.anns[ix]['ann_id']
                neg_ann_ids.append(neg_ann_id)
                ### ref
                if len(st_ref_ids) > 0 and random.random() < opt['sample_ratio']:
                    ix = random.randint(0, len(st_ref_ids)-1)
                    neg_ref_id = st_ref_ids[ix]
                elif len(dt_ref_ids) > 0:
                    ix = random.randint(0, len(dt_ref_ids)-1)
                    neg_ref_id = dt_ref_ids[ix]
                else:
                    ix = random.randint(0, len(self.info['refs'])-1)
                    neg_ref_id = self.info['refs'][ix]['ref_id']
                    
                cand_sent_ids = self.Refs[neg_ref_id]['sent_ids']
                ix = random.randint(0, len(cand_sent_ids)-1)
                neg_sent_ids.append(cand_sent_ids[ix])
        return neg_ann_ids, neg_sent_ids
        
    def getBatch(self, split, opt):
        batch_size = opt['batch_size']
        seq_per_ref = opt['seq_per_ref']
        sample_ratio = opt['sample_ratio']
        sample_neg = opt['sample_neg']
        split_ix = self.split_ix[split]
        max_index = len(split_ix)-1
        wrapped = False
        batch_ref_ids = []
        batch_ann_ids = []
        batch_sent_ids = []
        rank =[]
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 1
                wrapped = True
            self.iterators[split] = ri_next
            ref_id = split_ix[ri]
            batch_ref_ids.append(ref_id)
            ann_id = self.Refs[ref_id]['ann_id']
            batch_ann_ids.append(ann_id)
            sent_ids = self.fetch_sent_ids(ref_id, seq_per_ref)
            batch_sent_ids.extend(sent_ids)   
            if self.dataset == 'refgta':
                rank.append(self.rank_sent_ids(sent_ids))

        sp_cxt_feats, sp_ann_feats, feats, local_shapes = self.fetch_feats(batch_ann_ids, seq_per_ref, opt)
        seqz = self.fetch_seqs(batch_sent_ids, {'pad_zero':'end'})
        zseq = self.fetch_seqs(batch_sent_ids, {'pad_zero':'front'})
        neg_ann_ids, neg_sent_ids, neg_feats, neg_seqz, neg_zseq = None, None, None, None, None
        if sample_neg>0:
            neg_ann_ids, neg_sent_ids = self.sample_neg_ids(batch_ann_ids, opt)
            neg_sp_cxt_feats, neg_sp_ann_feats, neg_feats, neg_local_shapes = self.fetch_feats(neg_ann_ids, 1, opt) 
            neg_seqz = self.fetch_seqs(neg_sent_ids, {'pad_zero': 'end'}) 
            neg_zseq = self.fetch_seqs(neg_sent_ids, {'pad_zero':'front'})
        data = {}
        data['ref_ids'] = [batch_ref_ids[i//seq_per_ref] for i in range(len(batch_ref_ids)*seq_per_ref)]
        data['ref_ann_ids'] = [batch_ann_ids[i//seq_per_ref] for i in range(len(batch_ann_ids)*seq_per_ref)]
        data['ref_sent_ids'] = batch_sent_ids
        data['feats'] = feats
        data['sp_cxt_feats'] = sp_cxt_feats
        data['sp_ann_feats'] = sp_ann_feats
        data['local_shapes'] = local_shapes
        data['seqz'] = seqz
        data['zseq'] = zseq
        data['neg_ann_ids'] = neg_ann_ids
        data['neg_sent_ids'] = neg_sent_ids
        data['neg_feats'] = neg_feats
        data['neg_sp_cxt_feats'] = neg_sp_cxt_feats
        data['neg_sp_ann_feats'] = neg_sp_ann_feats
        data['neg_local_shapes'] = neg_local_shapes
        data['neg_seqz'] = neg_seqz
        data['neg_zseq'] = neg_zseq
        
        if self.dataset == 'refgta':
            data['rank'] = np.array(rank)
        data['bounds'] = {'it_pos_now':self.iterators[split],'it_max':len(split_ix),'wrapped':wrapped}
        return data
    
    def getImageBatch(self, split, opt):
        wrapped=False
        img_split_ix = self.img_split_ix[split]
        mi = self.img_iterators[split]
        image_id = img_split_ix[mi]
        mi_next = mi + 1
        if mi_next>len(img_split_ix)-1:
            wrapped=True
        self.img_iterators[split] = mi_next
        
        image = self.Images[image_id]
        ann_ids = image['ann_ids']
        sp_cxt_feats, sp_ann_feats, feats, local_shapes = self.fetch_feats(ann_ids, 1, opt)
        sent_ids = []
        gd_ixs = []
        rank = []
        for ix, ann_id in enumerate(ann_ids):
            if ann_id in self.annToRef:
                ref = self.annToRef[ann_id]
                if ref['split']==split:
                    ref_sent_ids = ref['sent_ids']
                    sent_ids.extend(ref_sent_ids)
                    gd_ixs.extend([ix]*len(ref_sent_ids))
                    if self.dataset=='refgta':
                        rank.append(self.rank_sent_ids(ref_sent_ids))
        seqz = self.fetch_seqs(sent_ids, {'pad_zero':'end'})
        zseq = self.fetch_seqs(sent_ids, {'pad_zero':'front'})
        data = {}
        data['image_id'] = image_id
        data['img_ann_ids'] = ann_ids   
        data['feats'] = feats  
        data['sp_cxt_feats'] = sp_cxt_feats
        data['sp_ann_feats'] = sp_ann_feats
        data['local_shapes'] = local_shapes
        data['sent_ids'] = sent_ids
        data['gd_ixs'] = gd_ixs
        data['seqz'] = seqz
        data['zseq'] = zseq 
        if self.dataset == 'refgta':
            data['rank'] = np.array(rank)
        data['bounds'] = {'it_pos_now':self.img_iterators[split], 'it_max':len(img_split_ix), 'wrapped':wrapped}
        return data
    
    def getTestBatch(self, split, opt):
        batch_size = opt['batch_size']
        seq_per_ref = opt['seq_per_ref']
        split_ix = self.split_ix[split]
        max_index = len(split_ix)-1
        wrapped = False
        batch_ref_ids = []
        batch_ann_ids = []
        batch_sent_ids = []
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next > max_index:
                wrapped = True
            self.iterators[split] = ri_next
            ref_id = split_ix[ri]
            batch_ref_ids.append(ref_id)
            ann_id = self.Refs[ref_id]['ann_id']
            batch_ann_ids.append(ann_id)
            batch_sent_ids.extend(self.fetch_sent_ids(ref_id, seq_per_ref))
        
        sp_cxt_feats, sp_ann_feats, feats, local_shapes = self.fetch_feats(batch_ann_ids, 1, opt)
        seqz = self.fetch_seqs(batch_sent_ids, {'pad_zero':'end'})
        zseq = self.fetch_seqs(batch_sent_ids, {'pad_zero':'front'})
        data = {}
        data['ref_ids'] = batch_ref_ids
        data['ref_ann_ids'] = batch_ann_ids
        data['feats'] = feats
        data['sp_cxt_feats'] = sp_cxt_feats
        data['sp_ann_feats'] = sp_ann_feats
        data['local_shapes'] = local_shapes
        data['seqz'] = seqz
        data['zseq'] = zseq
        data['bounds'] = {'it_pos_now':self.iterators[split],'it_max':len(split_ix),'wrapped':wrapped}
        return data

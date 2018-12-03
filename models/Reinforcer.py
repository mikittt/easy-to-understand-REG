import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from models.base import LanguageEncoderAttn, VisualEncoder, MetricNet
    
class ListenerReward(chainer.Chain):
    def __init__(self, vocab_size, scale=1, global_shapes=(224,224)):
            
        super(ListenerReward, self).__init__(
            ve = VisualEncoder(global_shapes=global_shapes),
            le = LanguageEncoderAttn(vocab_size),
            me = MetricNet()
        )
        self.scale=scale
            
    def calc_score(self, feats, sp_feats, coord, seq, lang_length):
        with chainer.using_config('train', False):
            vis_enc_feats = self.ve(feats, sp_feats, coord)
            lang_enc_feats = self.le(seq, lang_length)
            lr_score = self.me(vis_enc_feats, lang_enc_feats)
        return lr_score
    
    def create_mask(self, lang_last_ind, xp):
        mask = xp.array(np.where(lang_last_ind>0, 1, 0), dtype=xp.float32)
        return mask
    
    def __call__(self, feats, sp_feats, coord, seq, seq_prob, lang_length):#, baseline):
        xp = cuda.get_array_module(feats)
        mask = self.create_mask(lang_length, xp)
        lr_score = F.sigmoid(self.calc_score(feats, sp_feats, coord, seq, lang_length)).data[:,0]
        self.reward = lr_score*self.scale*mask
        loss = -F.mean(F.sum(seq_prob, axis=0)/(xp.array(lang_length+1))*(self.reward-self.reward.mean()))#F.broadcast_to(baseline, self.reward.shape))[:,0])
        return loss
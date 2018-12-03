import os
import numpy as np
import sys
import os.path as osp
sys.path.append('./')

import config
from PIL import Image
from tqdm import tqdm
from misc.DataLoader import DataLoader
from misc.resize import keep_asR_resize
import chainer
from chainer import cuda, Variable
import chainer.links as L

def extract_feature(params):
    if params['dataset'] in ['refcoco', 'refcoco+', 'refcocog']:
        image_root =  '/data/unagi0/mtanaka/MSCOCO/images/train2014'
    elif params['dataset'] == 'refgta':
        image_root = '/data/unagi0/mtanaka/gta5/gtav_cv_mod_data/'
    target_save_dir = osp.join(params['save_dir'],'prepro', params['dataset']+'_'+params['splitBy'])
    
    if params['old']:
        params['data_json'] = 'old'+params['data_json']
        params['data_h5'] = 'old'+params['data_h5']
        params['ann_feats'] = 'old'+params['ann_feats']
        
    loader = DataLoader(params)
        
    # model setting
    batch_size = params['batch_size']
    gpu_id = params['gpu_id']
    cuda.get_device(gpu_id).use()
    xp = cuda.cupy
    
    res = L.ResNet152Layers()
    res.to_gpu(gpu_id)
    chainer.config.train = False
    chainer.config.enable_backprop = False
    
    anns = loader.anns
    images = loader.Images
    perm = np.arange(len(anns))
    ann_feats = []
    shapes = []
    for bs in tqdm(range(0, len(anns), batch_size)):
        batch = []
        for ix in perm[bs:bs+batch_size]:
            ann = anns[ix]
            h5_id = ann['h5_id']
            assert h5_id==ix, 'h5_id not match' 
            img = images[ann['image_id']]
            x1, y1, w, h = ann['box']
            image = Image.open(os.path.join(image_root, img['file_name'])).convert('RGB').crop((x1, y1, x1+w, y1+h))
            image, resize_shape = keep_asR_resize(image)
            shapes.append(resize_shape)
            image = np.array(image).astype(np.float32)[:, :, ::-1]
            image -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
            image = image.transpose((2, 0, 1))
            batch.append(image)
        batch = Variable(xp.array(batch, dtype=xp.float32))
        feature = res(batch, layers=['res5'])
        feature = cuda.to_cpu(feature['res5'].data)
        ann_feats.extend(np.transpose(feature, (0,2,3,1)).reshape(-1, 36, 2048))
    np.save(os.path.join(target_save_dir, params['sp_ann_feats']), ann_feats)
    np.save(os.path.join(target_save_dir, params['ann_shapes']), shapes)



if __name__ == '__main__':

    args = config.parse_opt()
    params = vars(args) # convert to ordinary dict
    extract_feature(params)
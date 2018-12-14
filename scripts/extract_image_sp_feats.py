import os
import numpy as np
import sys
import os.path as osp
sys.path.append('./')

import config
from PIL import Image
import h5py
from tqdm import tqdm
from misc.DataLoader import DataLoader
import chainer
from chainer import cuda, Variable
import chainer.links as L

def extract_feature(params):
    
    if params['dataset'] in ['refcoco', 'refcoco+', 'refcocog']:
        image_root =  params['coco_image_root']
    elif params['dataset'] == 'refgta':
        image_root = params['gta_image_root']
    target_save_dir = osp.join(params['save_dir'],'prepro', params['dataset']+'_'+params['splitBy'])
    
    if params['old']:
        params['data_json'] = 'old'+params['data_json']
        params['data_h5'] = 'old'+params['data_h5']
        params['image_feats'] = 'old'+params['image_feats']
        
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
    
    images = loader.images
    perm = np.arange(len(images))
    image_feats = []
    
    for bs in tqdm(range(0, len(images), batch_size)):
        batch = []
        for ix in perm[bs:bs+batch_size]:
            image = Image.open(os.path.join(image_root, images[ix]['file_name'])).convert('RGB')
            if params['dataset'] in ['refcoco', 'refcoco+', 'refcocog']:
                image = image.resize((224, 224), Image.ANTIALIAS)
            else:
                image = image.resize((480, 288), Image.ANTIALIAS)
            image = np.array(image).astype(np.float32)[:, :, ::-1]
            image -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
            image = image.transpose((2, 0, 1))
            batch.append(image)
        batch = Variable(xp.array(batch, dtype=xp.float32))
        feature = res(batch, layers=['res5'])
        feature = cuda.to_cpu(feature['res5'].data)
        if params['dataset'] in ['refcoco', 'refcoco+', 'refcocog']:
            image_feats.extend(np.transpose(feature, (0,2,3,1)).reshape(-1, 49, 2048))
        else:
            image_feats.extend(np.transpose(feature, (0,2,3,1)).reshape(-1, 135, 2048))
    
    np.save(os.path.join(target_save_dir, params['image_feats']), image_feats)


if __name__ == '__main__':

    args = config.parse_opt()
    params = vars(args) # convert to ordinary dict
    extract_feature(params)
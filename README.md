# Code of [Towards Human-Friendly Referring Expression Generation](https://arxiv.org/abs/1811.12104)
- Results from Our dataset(RefGTA)

![results](https://raw.githubusercontent.com/mikittt/Human_Friendly_REG/master/demo/fig1.png)

- Results from RefCOCOg

<img src="https://raw.githubusercontent.com/mikittt/Human_Friendly_REG/master/demo/fig2.png" width="50%">


## Requirements

- Python 3.6 
- [Chainer](https://github.com/chainer/chainer) 5.0.0
- Cuda 9.0, CuDNN v7
- Numpy, PIL, Matplotlib, H5py, Tqdm  

If you use rerank, please install cplex by ``conda install -c ibmdecisionoptimization cplex``.

Training with refcocog and refgta requires 16 GB and 32 GB gpu respectively for the default setting.
If the memory is insufficient, please reduce the batch size.

Baseline code is [here](https://github.com/mikittt/re-SLR).

## Dataset

Please go to [this directory](https://github.com/mikittt/Human_Friendly_REG/tree/master/pyutils/refer2/).  
Download RefCOCO, RefCOCO+, RefCOCOg and RefGTA(our dataset).


## Preprocessing
Preprocessed data and extracted features are [here](https://drive.google.com/open?id=1j6kmPq3_RROGO8plICmN6kjM1DCrq_-k).

### Preprocess annotation
Run ``prepro.py``.  
```
python prepro.py --dataset refgta --splitBy utokyo
```

### Extract features

- Extract local features  
(We resize images to different sizes depending on their aspect ratio, so please set the batch size to 1 for extracting local spatial features.)

```bash
python scripts/extract_target_emb_feats.py --dataset refgta --splitBy utokyo --batch_size 64

# local spatial features
python scripts/extract_target_sp_feats.py --dataset refgta --splitBy utokyo --batch_size 1
```

- Extract global features
```bash
python scripts/extract_image_sp_feats.py --dataset refgta --splitBy utokyo --batch_size 64
```

## Training

First, train reinforcer.  
If you train with ranking on RefGTA, please add ``-r``.
```
python scripts/train_vlsim.py --dataset refgta --splitBy utokyo --id sp
```

Second, train speaker using reinforcer whose parameters are fixed.  
If you train with ranking on RefGTA, please add ``-r``.
```
python train_sp.py --dataset refgta --splitBy utokyo --id sp
```

## Evaluation

Pretrained model is [here](https://drive.google.com/open?id=1sEhePkoIqlzDcAPNFubfH9OODS6yZYkj).  
Generated sentences are [here](https://drive.google.com/open?id=13YZcylNpa8-tBena0swy2VocBBOnIS0-).

- generation evaluation (batch size 1 only is supported.)
```
python eval_generation_sp.py --dataset refgta --splitBy utokyo --split test --batch_size 1 --id sp
```

- generation evaluation after reranking
```
python rerank_generated_captions.py --dataset refgta --splitBy utokyo --split test --id sp
```

- comprehension evaluation  
(--mode 0:speaker comprehension, 1:reinforcer comprehension, 2:ensemble)
```
python eval_comprehension_sp.py --dataset refgra --splitBy utokyo --split test --mode 0 --id sp
```

### Acknowledgement
Our codes are based on [this repositry](https://github.com/lichengunc/speaker_listener_reinforcer).

### License
MIT License

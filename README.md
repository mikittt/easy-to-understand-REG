# Code of [Towards Human-Friendly Referring Expression Generation](https://arxiv.org/abs/1811.12104)
- Results from Our dataset(RefGTA)

![results](https://raw.githubusercontent.com/mikittt/Human_Friendly_REG/master/demo/fig1.png)

- Results from RefCOCOg

<img src="https://raw.githubusercontent.com/mikittt/Human_Friendly_REG/master/demo/fig2.png" width="50%">

This code is implemented in python3 with [Chainer](https://github.com/chainer/chainer).

## Dataset

Please go to [this directory](https://github.com/mikittt/Human_Friendly_REG/tree/master/pyutils/refer2/).  
This contains RefCOCO, RefCOCO+, RefCOCOg and RefGTA(our dataset).


## Preprocessing
run ``prepro.py`` or download from [here](https://drive.google.com/open?id=1j6kmPq3_RROGO8plICmN6kjM1DCrq_-k).  
(include extracted features)
```
python prepro.py --dataset refgta --splitBy utokyo
```

## Extract features

- extract local features  
(we resize images to different sizes depending on their aspect ratio, so please set batch size to 1 for extracting local spatial features)

```bash
python scripts/extract_target_emb_feats.py --dataset refgta --splitBy utokyo --batch_sizze 64

# local spatial features
python scripts/extract_target_sp_feats.py --dataset refgta --splitBy utokyo --batch_size 1
```

- extract global features
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

pretrained model is [here](https://drive.google.com/open?id=1sEhePkoIqlzDcAPNFubfH9OODS6yZYkj)


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

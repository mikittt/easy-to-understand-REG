# Code of [Towards Human-Friendly Referring Expression Generation](https://arxiv.org/abs/1811.12104)
- Results	from Our dataset(RefGTA)

![results](https://raw.githubusercontent.com/mikittt/Human_Friendly_REG/master/demo/fig1.png)

- Results	from RefCOCOg

<img src="https://raw.githubusercontent.com/mikittt/Human_Friendly_REG/master/demo/fig2.png" width="50%">

This code is implemented in python3 with [Chainer](https://github.com/chainer/chainer).

※ Under construction (dataset only available for now)

## Dataset

Please go to [this directory](https://github.com/mikittt/Human_Friendly_REG/tree/master/pyutils/refer2/)

## Preprocessing

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
```
python scripts/train_vlsim.py --dataset refgta --splitBy utokyo
```

Second, train speaker using reinforcer whose parameters are fixed.
```
python train_sp.py --dataset refgta --splitBy utokyo
```

## Evaluation

- generation evaluation (batch size 1 only is supported.)
```
python eval_generation_sp.py --dataset refgta --splitBy utokyo --split test --batch_size 1
```

- generation evaluation after reranking
```
python rerank_generated_captions.py --dataset refgta --splitBy utokyo --split test
```

- comprehension evaluation (--mode 0:speaker comprehension, 1:reinforcer comprehension, 2:ensemble)
```
python eval_comprehension_sp.py --dataset refgra --splitBy utokyo --split test --mode 0
```

### Acknowledgement
Our codes are based on [this repositry](https://github.com/lichengunc/speaker_listener_reinforcer).

### License
MIT License

## Note

This API is able to load all 4 referring expression datasets, i.e., RefCOCO, RefCOCO+, RefCOCOg and RefGTA.  
This repository is made by modifying [this one](https://github.com/lichengunc/refer2).  


## Setup
Run "make" before using the code.
It will generate ``_mask.c`` and ``_mask.so`` in ``external/`` folder.
These mask-related codes are copied from mscoco [API](https://github.com/pdollar/coco).

## Download
- Annotations (Refcoco/+/g)
Run ```download.sh``` and download annotations in ```../../dataset```. (Please see detail in the original repositries, [refer](https://github.com/lichengunc/refer) and [refer2](https://github.com/lichengunc/refer2))
Note that there are refcoco(+) and refcoco(+)_old folders because more captions were collected for test set to evaluate generation quality more robustly.

- Annotations (RefGTA) 
Please put [refgta data](https://drive.google.com/open?id=19UQsGDb8s9oi-v7bAw41ZqzypwM5ECaQ) in the  ```../../dataset/anns/original/refgta```
- [MSCOCO](http://mscoco.org/dataset/#overview) images for RefCOCO, RefCOCO+ and RefCOCOg
Download in ```../../dataset/coco_image```.
- [GTA images](https://drive.google.com/open?id=1pcdwA--xSAkbsOwjqhhyXMRZH7_sjQXU) for RefGTA 
**※ GTA V images are allowed for use in non-commercial and research uses**
Download in ```../../dataset/gta_image```


## How to use
The "refer.py" is able to load all 4 datasets with different kinds of data split.
```bash
# locate your own data_root, image_root, and choose the dataset_splitBy you want to use
refer = REFER(data_root, image_root, dataset='refcoco',  splitBy='unc')
refer = REFER(data_root, image_root, dataset='refcoco',  splitBy='google')
refer = REFER(data_root, image_root, dataset='refcoco+', splitBy='unc')
refer = REFER(data_root, image_root, dataset='refcocog', splitBy='google')  # testing data haven't been released yet
refer = REFER(data_root, image_root, dataset='refgta', splitBy='utokyo')
```


<!-- refs(dataset).p contains list of refs, where each ref is
{ref_id, ann_id, category_id, file_name, image_id, sent_ids, sentences}
ignore filename

Each sentences is a list of sent
{arw, sent, sent_id, tokens}
 -->

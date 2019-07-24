mkdir ../../dataset
cd ../../dataset
mkdir ann
mkdir ann/original
mkdir coco_image
mkdir gta_image

cd ann/original


## links: https://github.com/lichengunc/refer,https://github.com/lichengunc/refer2
mkdir refclef
wget http://bvision.cs.unc.edu/licheng/referit/data/refclef.zip -P refclef
unzip refclef/refclef.zip -d refclef
rm refclef/refclef.zip

mkdir refcoco
wget http://bvision.cs.unc.edu/licheng/referit/data/new_data/refcoco.zip -P refcoco
unzip refcoco/refcoco.zip -d refcoco
rm refcoco/refcoco.zip

mkdir refcoco+
wget http://bvision.cs.unc.edu/licheng/referit/data/new_data/refcoco+.zip -P refcoco+
unzip refcoco+/refcoco+.zip -d refcoco+
rm refcoco+/refcoco+.zip

mkdir refcocog
wget http://bvision.cs.unc.edu/licheng/referit/data/refcocog.zip -P refcocog
unzip refcocog/refcocog.zip -d refcocog
rm refcocog/refcocog.zip

mkdir refcoco_old
wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip -P refcoco_old
unzip refcoco_old/refcloco.zip -d refcoco_old
rm refcoco_old/refcloco.zip

mkdir refcoco+_old
wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip -P refcoco+_old
unzip refcoco+_old/refcoco+.zip -d refcoco+_old
rm refcoco+_old/refcoco+.zip

mkdir refgta
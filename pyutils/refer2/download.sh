mkdir ../../dataset
cd ../../dataset
mkdir ann
mkdir ann/original
mkdir coco_image
mkdir gta_image

cd ann/original

mkdir ref2
cd ref2

## links: https://github.com/lichengunc/refer,https://github.com/lichengunc/refer2
wget http://bvision.cs.unc.edu/licheng/referit/data/refclef.zip 
unzip refclef.zip 
rm refclef.zip

wget http://bvision.cs.unc.edu/licheng/referit/data/new_data/refcoco.zip 
unzip refcoco.zip 
rm refcoco.zip

wget http://bvision.cs.unc.edu/licheng/referit/data/new_data/refcoco+.zip 
unzip refcoco+.zip 
rm refcoco+.zip

wget http://bvision.cs.unc.edu/licheng/referit/data/refcocog.zip 
unzip refcocog.zip 
rm refcocog.zip

mkdir refgta


cd ../
mkdir ref
cd ref


wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip 
unzip refcoco.zip 
rm refcoco.zip

wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip 
unzip refcoco+.zip 
rm refcoco+.zip

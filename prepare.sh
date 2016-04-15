echo "Downloading VGGNet pre-trained on ILSVC12 (copy from https://gist.github.com/ksimonyan/211839e770f7b538e2d8)"
wget -O ./models/vggnet/VGG_ILSVRC_16_layers.caffemodel "https://www.dropbox.com/s/yqkm2tgqonditgs/VGG_ILSVRC_16_layers.caffemodel?dl=1"

echo "Downloading Deepbit 32bit model pre-trained on cifar10"
wget -O ./models/deepbit/DeepBit32_final_iter_1.caffemodel "https://www.dropbox.com/s/z815s0cjdipwr5b/DeepBit32_final_iter_1.caffemodel?dl=1"

echo "Downloading CIFAR10 Dataset"
wget -O cifar10-dataset.zip "https://www.dropbox.com/s/f7q3bbgvat2q1u2/cifar10-dataset.zip?dl=1" 
unzip cifar10-dataset.zip -d ./data/cifar10

matlab -nojvm -nodesktop -r "run aug_img_faster.m; quit;"
#matlab -nojvm -nodesktop -r "run aug_img.m; quit;"

echo "Convert CIFAR10 to leveldb"
./data/cifar10/create_imagenet.sh 


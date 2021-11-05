mkdir data
cd data
wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz -O urban8k.tgz
tar -xzf urban8k.tgz
echo "urbansound8k unzipped"
rm urban8k.tgz

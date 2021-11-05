# print OS type
cat /etc/os-release

pip install -r requirements.txt
# install libsndfile for librosa lib
apt-get update
apt-get install -y libsndfile1-dev

python train.py
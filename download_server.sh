apt install python3-pip unzip -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install imageio tqdm reedsolo

mkdir -p ~/data
curl -L -o ~/data/GBRASNET.zip https://www.kaggle.com/api/v1/datasets/download/zapak1010/bossbase-bows2
unzip ~/data/GBRASNET.zip -d ~/data/
rm ~/data/GBRASNET.zip

cp -r ~/data/GBRASNET/BOSSbase-1.01 ~/data/GBRASNET/BOSSbase-1.01-div

git clone https://github.com/Mechetel/SimpleSteganalysisCNN.git
bash ~/SimpleSteganalysisCNN/data/split_server.sh

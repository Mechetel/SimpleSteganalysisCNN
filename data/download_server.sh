apt install python3-pip unzip -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install imageio tqdm reedsolo scikit-learn



mkdir -p ~/data
curl -L -o ~/data/GBRASNET.zip https://www.kaggle.com/api/v1/datasets/download/zapak1010/bossbase-bows2
unzip ~/data/GBRASNET.zip -d ~/data/
rm ~/data/GBRASNET.zip
cp -r ~/data/GBRASNET/BOSSbase-1.01 ~/data/GBRASNET/BOSSbase-1.01-div

# Alternative download of BOSSbase from another source
mkdir -p ~/data
curl -L -o ~/data/bossbase.zip https://www.kaggle.com/api/v1/datasets/download/lijiyu/bossbase
unzip ~/data/bossbase.zip -d ~/data/
rm ~/data/bossbase.zip



git clone https://github.com/Mechetel/SimpleSteganalysisCNN.git
bash ~/SimpleSteganalysisCNN/data/split_server.sh

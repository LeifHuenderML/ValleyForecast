conda create --name valley_fever python=3.8
conda activate valley_fever
conda install -c anaconda cudatoolkit -y
conda install pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install pytorch torchvision torchaudio -c pytorch -c nvidia -y
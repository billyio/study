# 既存コンテナ
docker pull gcr.io/kaggle-255202/kaggle_base

# コンテナ起動
# --name test コンテナの名前を指定（今回はtest）
docker run --privileged=true --runtime=nvidia -p 8888:8888 -d -v ~/project:/root/user/project --name test kaggle_base /sbin/init
or
docker start <コンテナ名またはコンテナID>

# コンテナに入る
docker exec -it test /bin/bash
docker exec -it brain-rsna /bin/bash

# jupyter notebook
jupyter notebook --ip=0.0.0.0 --allow-root

# ローカルターミナル
# jupyter起動
gcloud compute ssh "kaggle-cnn-vm" -- -N -f -L 28888:localhost:8888

# ブラウザでlocalhost:28888へアクセス

# gcloud command not found
sudo apt install curl
curl https://sdk.cloud.google.com | bash
exec -l $SHELL #SHELL再起動
which gcloud

# gcsfuse install
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse

gcloud auth application-default login
# 毎回やる
gcsfuse io-kaggle-bucket /path/to/mount
ls /path/to/mount

# ローカル
# gsutil command not found
source ~/google-cloud-sdk/completion.zsh.inc
source ~/google-cloud-sdk/path.zsh.inc
source ~/.zshrc

gcsfuse io-kaggle-bucket /path/to/mount

# よく使うdockerコマンド
https://qiita.com/tera_shin/items/8a43e904bd15990d3129
docker ps -a # コンテナ一覧

# viコマンド
https://qiita.com/TD3P/items/0510bee10bcfd88afeee

# ローカル、ポート番号削除
# プロセスの確認（rails）
$ ps ax | grep rails
7532 .....

# ポートNoが分かっていればこれで確認できる
$ lsof -i:8888 #ポートNo

# http
lsof -i | grep http
7532 .....
上記コマンドで該当のPIDをを見つけ、killすれば解決する

$ kill [該当のPID]

# upload folder, files
gsutil cp -m -r <アップロードするやつ> <アップロード先　gs://io-kaggle-bucket/rsna_appian/input>
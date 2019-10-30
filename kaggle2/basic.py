参照
・colab操作
# https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166
・９０分対策
# https://qiita.com/enmaru/items/2770df602dd7778d4ce6
・GCPとdocker（簡易版）
# https://www.kaggle.com/getting-started/43929
・GCPとdocker（GPU版）
# https://qiita.com/lain21/items/a33a39d465cd08b662f1#gce%E3%82%A4%E3%83%B3%E3%82%B9%E3%82%BF%E3%83%B3%E3%82%B9%E3%81%AE%E4%BD%9C%E6%88%90


https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

!kaggle competitions download -c "コンペ名" -p "ダウンロード先　/content/gdrive/My\ Drive/kaggle/cancer"

# unzip
import os
os.chdir('gdrive/My Drive/kaggle/cancer')  #change dir
!mkdir train  #create a directory named train/
!mkdir test  #create a directory named test/
!unzip -q train.zip -d train/  #unzip data in train/
!unzip -q test.zip -d test/  #unzip data in test/
!unzip sample_submission.csv.zip
!unzip train_labels.csv.zip

path = '/content/drive/My Drive/input/PATH'
df = pd.read_csv(path_data+'/train.csv')

import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
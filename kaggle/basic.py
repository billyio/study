参照
・colab操作
# https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166
・９０分対策
# https://qiita.com/enmaru/items/2770df602dd7778d4ce6

from google.colab import drive
drive.mount('/content/drive')

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
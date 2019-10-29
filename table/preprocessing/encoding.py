df = pd.read_csv('train.csv')
df['カラム名']

# One-hotエンコーディング

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
  original_columns = list(df.columns)
  categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
  df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
  new_columns = [c for c in df.columns if c not in original_columns]
  return df, new_columns

# 例
a, b = one_hot_encoder(a, nan_as_category=True)
a（Onehotされた新しいDataframe）、b（生成される新しいカラム名）


# 多重共線性を取り除くためにdrop_firstをTrueとする
df_ = pd.get_dummies(df, drop_first=True, columns=['カラム名1'])
df_.head()

# バイナリのOneHotEncoder
# 2クラスのデータを2値に変換（OneHotEncoder）
lb = LabelBinarizer()
encoded = lb.fit_transform(df[['Sex']].values)

print('エンコード結果: ', encoded)

# One-hotエンコーディング（OneHotEncoder）
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
# ohe = OneHotEncoder(sparse=False, drop='first')
encoded = ohe.fit_transform(df[['カラム名1', 'カラム名2']].values)

print('カテゴリ: ', ohe.categories_)
print('カテゴリ名: ', ohe.get_feature_names(['カラム名1', 'カラム名2']))

# 列名を取得
label = ohe.get_feature_names(['カラム名1', 'カラム名2'])

# データフレーム化
df_ = pd.DataFrame(encoded, columns=label, dtype=np.int8)

# データフレームを結合
pd.concat([df, df_], axis=1)


# ラベルエンコーディング（LabelEncoder）
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
encoded = le.fit_transform(df['カラム名1'].values)
decoded = le.inverse_transform(encoded)
df['encoded'] = encoded

print('存在するクラス: ', le.classes_)
print('変換先: C, Q, S ->', le.transform(['C', 'Q', 'S']))
print('エンコード結果: ', encoded)
print('元に戻す: ', decoded)

# ラベルエンコーディング（OrdinalEncoder）
# 複数変数
from sklearn.preprocessing import OrdinalEncoder

oe = preprocessing.OrdinalEncoder()
# ['カラム名1', 'カラム名2'] => [0, 1]
encoded = oe.fit_transform(df[['カラム名1', 'カラム名2']].values)
decoded = oe.inverse_transform(encoded)

print('エンコード結果: ', encoded)
print('元に戻す: ', decoded)

# dict
# 辞書の作成
d = dict(k1=1, k2=2, k3=3)
print(d)
{'k1': 1, 'k2': 2, 'k3': 3}

# カウントエンコーディング
# ラベルの出現回数で置き換えます。名義尺度を比例尺度に置き換えているとも言える
import collections

counter = collections.Counter(df['カラム名1'].values)
count_dict = dict(counter.most_common())
encoded = df['カラム名1'].map(lambda x: count_dict[x]).values
df['encoded'] = encoded

print('エンコード結果: ', encoded)

# ラベルカウントエンコーディング
# ラベルの出現回数の順序に置き換えます。 名義尺度を順序尺度に置き換えている
import collections

counter = collections.Counter(df['カラム名1'].values)
count_dict = dict(counter.most_common())
label_count_dict = {key:i for i, key in enumerate(count_dict.keys(), start=1)}
encoded = df['カラム名1'].map(lambda x: label_count_dict[x]).values
df['encoded'] = encoded

print('エンコード結果: ', encoded)

# Target Mean Encoding
# ターゲット変数の平均値で置き換えます。名義尺度を比例尺度に置き換えている
# 二値、もしくは回帰問題の場合に使える手法

# 例1
target_dict = df[['Embarked','Survived']].groupby(['Embarked'])['Survived'].mean().to_dict()
encoded = df['Embarked'].map(lambda x: target_dict[x]).values
df['encoded'] = encoded

print('エンコード結果: ', encoded)

# 例2
df = pd.DataFrame({ 'category': ['A','A','B','B','B','C','C','C','C','D'],
                    'label': [1,0,1,0,0,1,0,1,1,1]
                  })

label_mean = df.groupby('category').label.mean()
df = df.assign(target_enc=df['category'].map(label_mean).copy())

df = df.assign(target_enc2=df.groupby('category')['label'].transform('mean').copy())

  category  label
0        A      1
1        A      0
2        B      1
3        B      0
4        B      0
5        C      1
6        C      0
7        C      1
8        C      1
9        D      1 

=>  category  label  target_enc  target_enc2
0        A      1    0.500000     0.500000
1        A      0    0.500000     0.500000
2        B      1    0.333333     0.333333
3        B      0    0.333333     0.333333
4        B      0    0.333333     0.333333
5        C      1    0.750000     0.750000
6        C      0    0.750000     0.750000
7        C      1    0.750000     0.750000
8        C      1    0.750000     0.750000
9        D      1    1.000000     1.000000




# ハッシュエンコーディング
One-hot encodingでは、カテゴリ数に応じて列数が増えることや、新しい値が出現する度に列数を増やす必要があることが問題点として挙げられる
これを解決するために、ハッシュ関数を用いて固定の配列に変換するのがHash encoding
ハッシュ関数とは、ある値（キー）を別の値（ハッシュ）にマッピングする操作
このマッピングを予め設定しておくことで、一意な変換が可能
しかし、長さ（ハッシュ値の数）を固定するので、カテゴリ数の方が大きい場合は複数のカテゴリが同じハッシュ値にマッピングされ得ます（これを衝突という）
そこで、複数のハッシュ関数を用意して、最も精度の良いものを選ぶそう

もっとも、衝突は必ずしも悪いことではありません。
例えば、複数形（catとcats）や表現の違い（JapanとJP）は同じことを指しているので、同じハッシュ値になった方が良い表現だと言えます。
One-hotでは複数の列に割り振られてしまうものを一つに纏められるのがHash encodingの強みでもあります。

import category_encoders as ce

encoder = ce.HashingEncoder(cols=['Embarked', 'Sex'], n_components=4)
encoder.fit(df['Embarked'], df['Survived'])
encoded = encoder.transform(df['Embarked'])
encoded

# frequency encoding
各値の出現頻度で表現する方法

#Frequency Encoding Sample
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({
    "category" : ["A","A","A","B","B","C"],
})

#各カテゴリーの出現回数を計算
grouped = df.groupby("category").size().reset_index(name='category_counts')

#元のデータセットにカテゴリーをキーとして結合
df = df.merge(grouped, how = "left", on = "category")
df["frequency"] = df["category_counts"]/df["category_counts"].count()

df.head(10)


# カテゴリ変数（文字列）を数値変換

# 例
import pandas as pd
df = pd.DataFrame({'列１': ['b', 'b', 'a', 'c', 'b'], '列２': ['あ', 'い', 'い', None, 'え']})

for column in df.columns:
    labels, uniques = pd.factorize(df[column])
    df[column] = labels

print(df)
#    列１  列２
# 0   0   0
# 1   0   1
# 2   1   1
# 3   2  -1
# 4   0   2
print(labels)
# [0 0 1 2 0]
# [0 1 1 2 3]
print(uniques)
# Index(['b', 'a', 'c'], dtype='object')
# Index(['あ', 'い', 'え'], dtype='object')


None =>　−1


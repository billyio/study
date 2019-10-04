df.info()	
df.shape
df.count()	各カラムの有効データ数を表示
df.dtypes	各カラムのデータ型を表示
len(df) - df.count()	すべてのカラムに対して、欠損値の数を計算

# dtypeを指定して抽出
df.select_dtypes(include='int')
# 複数のdtypes
df.select_dtypes(include=[int, float, 'datetime'])

# 結合
# 縦方向の結合
pd.concat([df1, df2])
# 横方向の結合
pd.concat([df1, df4], axis=1)
# 横方向　かつ　共通な行名を残す
# join_axes で残したいラベルの名前を指定
pd.concat([df1, df4], axis=1, join='inner')
# join_axes で残したいラベルのindexを指定
pd.concat([df1, df4], axis=1, join_axes=[df4.index])

# test1, test2の共通列名'key'で結合
pd.merge(test1, test2, on='key')
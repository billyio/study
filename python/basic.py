# dict
# 辞書の作成
d = dict(k1=1, k2=2, k3=3)
print(d)
{'k1': 1, 'k2': 2, 'k3': 3}

# counter

# most_common()
# (要素, 出現回数)という形のタプルを出現回数順に並べたリストを返す。

print(c.most_common())
# [('a', 4), ('c', 2), ('b', 1)]

numpy.round(x)	四捨五入
numpy.trunc(x)	切り捨て
numpy.floor(x)	切り捨て
numpy.ceil(x)	切り上げ
numpy.fix(x)	零に近い方で整数をとる


groupby

df.groupby('列名１').mean()
df.groupby(['列名１', '列名２']).mean() # 複数ラベルの組み合わせ


drop
第一引数labelsと第二引数axisで指定
行の場合はaxis=0（デフォルト）
引数inplaceをTrueにすると元のDataFrameが変更

sort

sort_values()
引数ascending デフォルトは昇順。引数ascending=False で降順

# filter
条件に一致する要素のみ抽出する

例
filter(lambda n:n%2==1, items)　#lambda関数がTrueを返す要素（奇数）要素のみ抽出する
[1, 3]  # 表示
items　#itemsは不変
[1, 2, 3] # 表示

# reset_index
データを削除したり並べ替えたあとにインデックスを振り直す
http://ailaby.com/reset_index/
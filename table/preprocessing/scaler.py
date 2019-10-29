・MinMaxScaler：データが0~1に入るよう変換
・StandardSclare：平均0、分散1になるように変換
・RobustSclaer：中央値と四分位数で変換。外れ値を無視できる変換方法

# 標準化
データの平均値と分散を変換

変換操作は以下の式で表される。
Y = (X−μ) / σ

from sklearn.preprocessing import StandardScaler
StandardScaler(copy=True, with_mean=True, with_std=True)

copy: デフォルトはTrue。Trueの場合、元のデータは変換されない
with_mean: デフォルト値はTrue。
  Trueの場合、平均値を0。
  Falseの場合、Y=X/σ。分散は1になるが、平均は同じままとは限らない。
with_std: デフォルト値はTrue. 
  Trueの場合、分散を0とする。
  Falseの場合、Y=X−μ。分散は変化せず、平均は0となる。

# 例
from sklearn.preprocessing import StandardScaler
x = np.arange(0, 8, 1.).reshape(-1, 2)
sscaler = StandardScaler() # インスタンスの作成
sscaler.fit(x)           # xの平均と分散を計算
y = sscaler.transform(x) # xを変換

# 正規化
データの最大値と最小値を制限する変換
最大値を1, 最小値を0とすることが多い。

変換操作は以下の式で表される。
Y = (X−xmin) / (xmax−xmin)

from sklearn.preprocessing import MinMaxScaler
MinMaxScaler(feature_range=(0, 1), copy=True)

feature_range: デフォルト値は(0, 1)

# 例
from sklearn.preprocessing import MinMaxScaler
x = np.arange(0, 6, 1.).reshape(-1, 2)
mmscaler = MinMaxScaler() # インスタンスの作成
mmscaler.fit(x)           # xの最大・最小を計算
y = mmscaler.transform(x) # xを変換

# 外れ値に頑健な標準化
変換前のデータに極端に大きな値または小さな値が含まれていた場合、標準化を行うと大きく結果が変わってしまう。
これを避けるため、データの四分位点を基準にして標準化を行う

RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True) 

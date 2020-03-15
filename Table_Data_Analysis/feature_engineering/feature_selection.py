特徴量選択

参照
・まとめqiita
# https://qiita.com/shimopino/items/5fee7504c7acf044a521
# https://qiita.com/fhiyo/items/33b295de64f5a6a047c6

メリット
・変数を少なくすることで解釈性を上げる
・計算コストを下げて、学習時間を短縮する
・過適合を避けて汎用性を向上させる
・高次元データによって、パフォーマンスが下がることを防ぐ

特徴量選択の手法は、大別して3つある
・Filter Method
・Wrapper Method
・Emedded Method

Filter Methodは、大別して３つある
・特徴量の値のみ
・特徴量間の相関係数
・統計的評価指標

# 特徴量のみ
・分散がゼロ　=> 全て同じ値 => 削減

from sklearn.feature_selection import VarianceThreshold
X = desc_df.values
select = VarianceThreshold()
X_new = select.fit_transform(X)

np.array(descs)[select.get_support()==False]  # 削減後の特徴量の数を確認

・分散がほぼゼロ　=> データをよく観察して削除するか判断

・特徴量がほかの特徴量と完全に一致 


# 特徴量間の相関係数
メリット
・互いに相関の高い特徴量の片方を削除することで、精度にあまり影響を与えずに特徴量空間の次元を下げる
・線形モデルの解釈性を上げることができる。

ピアソン相関係数（いわゆる普通の相関係数）

threshold = 0.8 # 閾値

feat_corr = set()
corr_matrix = X_train.corr()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            feat_name = corr_matrix.columns[i]
            feat_corr.add(feat_name)

print(len(set(feat_corr)))
# -> 658

X_train.drop(labels=feat_corr, axis='columns', inplace=True)
X_test.drop(labels=feat_corr, axis='columns', inplace=True)

print(len(X_train.columns))
# -> 4025

# 統計的評価指標
流れ
1. 特徴量をある評価指標で測る
2. 高いランクの特徴量だけを残す

あくまで1つの特徴量とターゲットを使用して性能を評価するので、特徴量間の関係などは基本的には分からない

# Mutual Information（相互情報量）
ある特徴量Xと別の特徴量Yの間の同時分布P(X,Y)P(X,Y)と個々の分布の積P(X)P(Y)P(X)P(Y)がでれだけ似ているのかを計算
もし互いに独立であればMutual Infomartionは0になる

∑i,yP(xi,yj)×logp(xi,yj)P(xi)P(yj)

from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile

MI = mutual_info_regression(X_train, y_train)
MI = pd.Series(MI)
MI.index = X_train.columns
MI.sort_values(ascending=False).plot(kind='bar', figsize=(20,10))

# KBest : 抽出する特徴量の"数"を指定
kbest_sel_ = SelectKBest(mutual_info_regression, k=10)
print(len(kbest_sel_.get_support()))

# Percentile : 抽出する特徴量の"割合"を指定
percentile_sel_ = SelectPercentile(mutual_info_regression, percentile=10)
print(len(percentile_sel_.get_support()))

# カイ2乗、フィッシャー係数
特徴量がカテゴリ変数でありターゲットが2値の場合によく使用される手法
例えば2値の分類タスクであるカテゴリの特徴量を評価する際、カテゴリごとの2値の出現回数などをカテゴリごとに比較したり、全体の出現回数などと比較する

Si=∑n[j[(μ[ij]−μ[i]) / ∑n[j]*ρ[ij]**2

chi2 == カイ二乗検定

from sklearn.feature_selection import chi2

# fisher score
fscore = chi2(X_train.fillna(0), y_train)

# Univariate
ターゲットが2値である必要があり、よく連続値の特徴量に対して使用される
２変数に対して行う手法はANOVAなどと呼ばれる

各特徴量と目的変数との相関係数を計算して、その値からF scoreを計算し、p-valueを求める
p-valueの値が小さい順にソートすることで、特徴量の重要度が分かる

前提として
（1）変数とターゲットに線形の関係があること
（2）変数が正規分布している

from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile

univariate = f_regression(X_train, y_train)
sel_ = SelectKBest(f_regression, k=1000).fit(X_train, y_train)


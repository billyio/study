# image_process memo

# 画像の読み込み
cv2.imread(" PATH ")
# 画像を保存（第1引数は画像のファイル名，第2引数は保存したい画像）
cv2.imwrite("output_gray.jpg", output_gray)
# 画像をウィンドウ上に表示（第1引数は文字列型で指定するウィンドウ名、第2引数は表示したい画像）
cv2.imshow("result", output_gray)

# Gray scale
def BGR2GRAY(img):
	b = img[:, :, 0].copy()
	g = img[:, :, 1].copy()
	r = img[:, :, 2].copy()

	# Gray scale
	out = 0.2126 * r + 0.7152 * g + 0.0722 * b
	out = out.astype(np.uint8)

	return out

# binalization（白黒の二極化）
# 閾値　"th"reshold
def binarization(img, th=128):
	img[img < th] = 0
	img[img >= th] = 255
	return img

# Otsu Binalization
def otsu_binarization(img, th=128):
	imax_sigma = 0
	max_t = 0
	# determine threshold
	for _t in range(1, 255):
		v0 = out[np.where(out < _t)]
		m0 = np.mean(v0) if len(v0) > 0 else 0.
		w0 = len(v0) / (H * W)
		v1 = out[np.where(out >= _t)]
		m1 = np.mean(v1) if len(v1) > 0 else 0.
		w1 = len(v1) / (H * W)
		sigma = w0 * w1 * ((m0 - m1) ** 2)
		if sigma > max_sigma:
			max_sigma = sigma
			max_t = _t

	# Binarization
	print("threshold >>", max_t)
	th = max_t
	out[out < th] = 0
	out[out >= th] = 255

	return out

# HSV変換
# Hue(色相)、Saturation(彩度)、Value(明度)
# Hue (0 <= H < 360)
# 赤 黄色  緑  水色  青  紫   赤
# 0  60  120  180 240 300 360
# Saturation ... 低いと灰色さが顕著になり、くすんだ色となる。 ( 0<= S < 1)
# Value ... 高いほど白に近く、低いほど黒に近くなる。 ( 0 <= V < 1)

# RGB -> HSV変換は以下の式で定義される。
# R,G,Bが[0, 1]の範囲にあるとする。
Max = max(R,G,B)
Min = min(R,G,B)

H =  { 0                            (if Min=Max)
       60 x (G-R) / (Max-Min) + 60  (if Min=B)
       60 x (B-G) / (Max-Min) + 180 (if Min=R)
       60 x (R-B) / (Max-Min) + 300 (if Min=G)     
S = Max - Min
V = Max

# BGR -> HSV
def BGR2HSV(_img):
	img = _img.copy() / 255.

	hsv = np.zeros_like(img, dtype=np.float32)

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()
	min_arg = np.argmin(img, axis=2)

	# H
	hsv[..., 0][np.where(max_v == min_v)]= 0
	## if min == B
	ind = np.where(min_arg == 0)
	hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
	## if min == R
	ind = np.where(min_arg == 2)
	hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
	## if min == G
	ind = np.where(min_arg == 1)
	hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300
		
	# S
	hsv[..., 1] = max_v.copy() - min_v.copy()

	# V
	hsv[..., 2] = max_v.copy()
	
	return hsv


def HSV2BGR(_img, hsv):
	img = _img.copy() / 255.

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()

	out = np.zeros_like(img)

	H = hsv[..., 0]
	S = hsv[..., 1]
	V = hsv[..., 2]

	C = S
	H_ = H / 60.
	X = C * (1 - np.abs( H_ % 2 - 1))
	Z = np.zeros_like(H)

	vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

	for i in range(6):
		ind = np.where((i <= H_) & (H_ < (i+1)))
		out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
		out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
		out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

	out[np.where(max_v == min_v)] = 0
	out = np.clip(out, 0, 1)
	out = (out * 255).astype(np.uint8)

	return out

# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.COLOR_BGR2HLS

# 減色処理
# Dicrease color
def dicrease_color(img):
	out = img.copy()

	out = out // 64 * 64 + 32

	return out

# 平均プーリング
# average pooling
# G=8, 8x8にグリッド分割
def average_pooling(img, G=8):
    out = img.copy()

    H, W, C = img.shape
    Nh = int(H / G)
    Nw = int(W / G)

    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                out[G*y:G*(y+1), G*x:G*(x+1), c] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1), c]).astype(np.int)
    
    return out

# 最大値でプーリング
def max_pooling(img, G=8):
    # Max Pooling
    out = img.copy()

    H, W, C = img.shape
    Nh = int(H / G)
    Nw = int(W / G)

    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                out[G*y:G*(y+1), G*x:G*(x+1), c] = np.max(out[G*y:G*(y+1), G*x:G*(x+1), c])

    return out


# 平滑化（スムージング）
# 画像上の濃淡変動を滑らかにする処理のことである．
# 雑音の低減を図る場合や画像をぼかす効果を得たいときに使用
# ・移動平均フィルタリング：　
# 注目画素を中心とする局所領域の平均値をその注目画素に出力する
# フィルタを対象画像のすべての画素に適用する方法である．
# 局所領域のすべての画素に対して同じ重み係数が使用される．

# ・加重平均フィルタリング：　
# 注目する中心画素に対して周辺の画素の寄与は小さいと仮定したフィルタ処理．
# 平滑化の程度を重み係数を変えることによって制御できる．
# 移動平均フィルタと比べ，より緩やかな平滑化を行うことができる．
# 現実の撮影系に存在するボケのカーネル分布は，ガウス分布をもつことが知られており，そのため正規分布型の重み係数が使用されることが多い．
# このようなフィルタを特にガウシアンフィルタ（Ｇａｕｓｓｉａｎ　Ｆｉｌｔｅｒ）と呼ぶ．

# ・メディアンフィルタ：
# 局所領域における濃淡のレベルの中央値（Median）を出力するフィルタである．
# エッジ情報が保存されやすい特徴を持っている．特にスパイク状の雑音（ごま塩雑音とも呼ばれる）を容易に取り除くことが可能である．


# ガウシアンフィルタ
# 画像の平滑化を行うフィルタの一種であり、ノイズ除去にも使用
# 注目画素の周辺画素を、ガウス分布による重み付けで平滑化し、次式で定義される。 
# このような重みはカーネルやフィルタと呼ばれる。
# ただし、画像の端はこのままではフィルタリングできないため、画素が足りない部分は0で埋める。
# これを0パディングと呼ぶ。 かつ、重みは正規化する。(sum g = 1)

# # https://algorithm.joho.info/programming/python/opencv-gaussian-filter-py/

# フィルタ一覧
# https://imagingsolution.net/imaging/filter-algorithm/

# 重み g(x,y,s) = 1/ (s*sqrt(2 * pi)) * exp( - (x^2 + y^2) / (2*s^2))
# 標準偏差s = 1.3による8近傍ガウシアンフィルタは
#             1 2 1
# K =  1/16 [ 2 4 2 ]
#             1 2 1

# Gaussian filter
def gaussian_filter(img, K_size=3, sigma=1.3):
	if len(img.shape) == 3:
		H, W, C = img.shape
	else:
		img = np.expand_dims(img, axis=-1)
		H, W, C = img.shape

		
	## Zero padding
	pad = K_size // 2
	out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
	out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

	## prepare Kernel
	K = np.zeros((K_size, K_size), dtype=np.float)
	for x in range(-pad, -pad + K_size):
		for y in range(-pad, -pad + K_size):
			K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
	K /= (sigma * np.sqrt(2 * np.pi))
	K /= K.sum()

	tmp = out.copy()

	# filtering
	for y in range(H):
		for x in range(W):
			for c in range(C):
				out[pad + y, pad + x] = np.sum(K * tmp[y: y + K_size, x: x + K_size])

	out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out


# Median filter
def median_filter(img, K_size=3):
    H, W, C = img.shape

    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad*2, W + pad*2, C), dtype=np.float)
    out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                # 中央値
                out[pad+y, pad+x, c] = np.median(tmp[y:y+K_size, x:x+K_size, c])
                # 平均値
                # out[pad+y, pad+x, c] = np.mean(tmp[y:y+K_size, x:x+K_size, c])

    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

    return out

# MAX-MIN フィルタ
# エッジ検出のフィルタの一つ
# エッジ検出とは、
# 画像内の線を検出することであり、このような画像内の情報を抜き出す操作を特徴抽出と呼ぶ。 
# エッジ検出では多くの場合、グレースケール画像に対してフィルタリングを行う。
import cv2
import numpy as np

# Read image
img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# Gray scale
gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
gray = gray.astype(np.uint8)

# Max-Min Filter
K_size = 3

## Zero padding
pad = K_size // 2
out = np.zeros((H + pad*2, W + pad*2), dtype=np.float)
out[pad:pad+H, pad:pad+W] = gray.copy().astype(np.float)
tmp = out.copy()

for y in range(H):
    for x in range(W):
        out[pad+y, pad+x] = np.max(tmp[y:y+K_size, x:x+K_size]) - np.min(tmp[y:y+K_size, x:x+K_size])

out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)

# 微分フィルタ
# 輝度の急激な変化が起こっている部分のエッジを取り出すフィルタ
# 画像から輪郭が抽出できる

# ただし、通常の差分演算を行うと、画面に含まれる雑音成分にも反応してしまうため、
# 雑音の低減とノイズの除去の両方の働きを持つフィルタがいくつか提案される
# Sobel(ゾーベル)フィルタ、Prewitt(プレヴィット)フィルタがその一例

# http://apple.ee.uec.ac.jp/COMPROG/handout/comprog11.pdf

# 微分フィルタ
#     (a)縦方向         (b)横方向
#       0 -1  0            0 0 0
# K = [ 0  1  0 ]   K = [ -1 1 0 ]　
#       0  0  0            0 0 0

# Prewittフィルタ
#     (a)縦方向          (b)横方向
#       -1 -1 -1          -1 0 1
# K = [  0  0  0 ]  K = [ -1 0 1 ]
#        1  1  1          -1 0 1

# Sobelフィルタ
#     (a)縦方向       (b)横方向
#        1  2  1           1  0 -1
# K = [  0  0  0 ]   K = [ 2  0 -2 ]
#       -1 -2 -1           1  0 -1

import cv2
import numpy as np

# Read image
img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# Gray scale
gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
gray = gray.astype(np.uint8)

# sobel Filter
K_size = 3

## Zero padding
pad = K_size // 2
out = np.zeros((H + pad*2, W + pad*2), dtype=np.float)
out[pad:pad+H, pad:pad+W] = gray.copy().astype(np.float)
tmp = out.copy()

out_v = out.copy()
out_h = out.copy()

## Sobel vertical
Kv = [[0., -1., 0.],[0., 1., 0.],[0., 0., 0.]]
## Sobel horizontal
Kh = [[0., 0., 0.],[-1., 1., 0.], [0., 0., 0.]]

for y in range(H):
    for x in range(W):
        out_v[pad+y, pad+x] = np.sum(Kv * (tmp[y:y+K_size, x:x+K_size]))
        out_h[pad+y, pad+x] = np.sum(Kh * (tmp[y:y+K_size, x:x+K_size]))

#out_v = np.abs(out_v)
#out_h = np.abs(out_h)
out_v[out_v < 0] = 0
out_h[out_h < 0] = 0
out_v[out_v > 255] = 255
out_h[out_h > 255] = 255

out_v = out_v[pad:pad+H, pad:pad+W].astype(np.uint8)
out_h = out_h[pad:pad+H, pad:pad+W].astype(np.uint8)

# Save result
cv2.imwrite("out_v.jpg", out_v)
cv2.imshow("result", out_v)

cv2.imwrite("out_h.jpg", out_h)
cv2.imshow("result", out_h)

# Prewittフィルタ
import cv2
import numpy as np

# Read image
img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# Gray scale
gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
gray = gray.astype(np.uint8)

# sobel Filter
K_size = 3

## Zero padding
pad = K_size // 2
out = np.zeros((H + pad*2, W + pad*2), dtype=np.float)
out[pad:pad+H, pad:pad+W] = gray.copy().astype(np.float)
tmp = out.copy()

## Sobel vertical
K = [[-1., -1., -1.],[0., 0., 0.], [1., 1., 1.]]
## Sobel horizontal
#K = [[-1., 0., 1.],[-1., 0., 1.],[-1., 0., 1.]]

for y in range(H):
    for x in range(W):
        out[pad+y, pad+x] = np.sum(K * (tmp[y:y+K_size, x:x+K_size]))

out[out < 0] = 0
out[out > 255] = 255

out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)

# Sobelフィルタ
# Prewittフィルタに重みをつけたもの
import cv2
import numpy as np

# Read image
img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# Gray scale
gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
gray = gray.astype(np.uint8)

# sobel Filter
K_size = 3

## Zero padding
pad = K_size // 2
out = np.zeros((H + pad*2, W + pad*2), dtype=np.float)
out[pad:pad+H, pad:pad+W] = gray.copy().astype(np.float)
tmp = out.copy()

## Sobel vertical
K = [[1., 2., 1.],[0., 0., 0.], [-1., -2., -1.]]
## Sobel horizontal
#K = [[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]]

for y in range(H):
    for x in range(W):
        out[pad+y, pad+x] = np.sum(K * (tmp[y:y+K_size, x:x+K_size]))

out[out < 0] = 0
out[out > 255] = 255

out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)

# Laplacianフィルタ
# 輝度の二次微分をとることでエッジ検出を行うフィルタ
# ノイズが強調されやすい

# sobel Filter
K_size = 3

## Zero padding
pad = K_size // 2
out = np.zeros((H + pad*2, W + pad*2), dtype=np.float)
out[pad:pad+H, pad:pad+W] = gray.copy().astype(np.float)
tmp = out.copy()
## Laplacian vertical
K = [[0., 1., 0.],[1., -4., 1.], [0., 1., 0.]]

for y in range(H):
    for x in range(W):
        out[pad+y, pad+x] = np.sum(K * (tmp[y:y+K_size, x:x+K_size]))

out[out < 0] = 0
out[out > 255] = 255

out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)

# Emboss
# 輪郭部分を浮き出しにするフィルタ

## Zero padding
pad = K_size // 2
out = np.zeros((H + pad*2, W + pad*2), dtype=np.float)
out[pad:pad+H, pad:pad+W] = gray.copy().astype(np.float)
tmp = out.copy()

## Emboss vertical
K = [[-2., -1., 0.],[-1., 1., 1.], [0., 1., 2.]]

for y in range(H):
    for x in range(W):
        out[pad+y, pad+x] = np.sum(K * (tmp[y:y+K_size, x:x+K_size]))

out[out < 0] = 0
out[out > 255] = 255

out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)

# LoGフィルタ
# Laplacian of Gaussian
# ガウシアンフィルタで画像を平滑化した後にラプラシアンフィルタで輪郭を取り出すフィルタ
# Laplcianフィルタは二次微分をとるのでノイズが強調されるのを防ぐために、予めGaussianフィルタでノイズを抑える。

# LoGフィルタは次式で定義される。
# LoG(x,y) = (x^2 + y^2 - s^2) / (2 * pi * s^6) * exp(-(x^2+y^2) / (2*s^2))

import cv2
import numpy as np

# Read image
img = cv2.imread("imori_noise.jpg")
H, W, C = img.shape

b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# Gray scale
gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
gray = gray.astype(np.uint8)

# Gaussian Filter
K_size = 5
s = 3

## Zero padding
pad = K_size // 2
out = np.zeros((H + pad*2, W + pad*2), dtype=np.float)
out[pad:pad+H, pad:pad+W] = gray.copy().astype(np.float)
tmp = out.copy()

## Kernel
K = np.zeros((K_size, K_size), dtype=np.float)
for x in range(-pad, -pad+K_size):
    for y in range(-pad, -pad+K_size):
        K[y+pad, x+pad] = (x**2 + y**2 - s**2) * np.exp( -(x**2 + y**2) / (2* (s**2)))
K /= (2 * np.pi * (s**6))
K /= K.sum()

for y in range(H):
    for x in range(W):
        out[pad+y, pad+x] = np.sum(K * tmp[y:y+K_size, x:x+K_size])

out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)

# ヒストグラム正規化
# 濃度階調変換(gray-scale transformation)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("imori_dark.jpg").astype(np.float)
H, W, C = img.shape

# Trans [0, 255]
a, b = 0., 255.

c = img.min()
d = img.max()

out = img.copy()

out = (b-a) / (d - c) * (out - c) + a
out[out < a] = a
out[out > b] = b
out = out.astype(np.uint8)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out_his.png")
plt.show()

# Save result
cv2.imwrite("out.jpg", out)

# ヒストグラム操作
# ヒストグラムの平均値をm0=128、標準偏差をs0=52になるように操作する
# 平均値m、標準偏差s、のヒストグラムを平均値m0, 標準偏差s0に変更するには、
# 次式によって変換する。
# x.out = s0 / s * (x.in - m) + m0


# Read image
img = cv2.imread("imori_dark.jpg").astype(np.float)
H, W, C = img.shape

# Trans [0, 255]
m0 = 128
s0 = 52

m = np.mean(img)
s = np.std(img)

out = img.copy()
out = s0 / s * (out - m) + m0
out[out < 0] = 0
out[out > 255] = 255
out = out.astype(np.uint8)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out_his.png")
plt.show()

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# ヒストグラム平坦化
# 上記の平均値や標準偏差などを必要とせず、ヒストグラム値を均衡にする操作
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

# Histogram flattening
S = H * W * C * 1.

out = img.copy()

sum_h = 0.
z_max = 255.

for i in range(1, 255):
    ind = np.where(img == i)
    sum_h += len(img[ind])
    z_prime = z_max / S * sum_h
    out[ind] = z_prime

out = out.astype(np.uint8)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out_his.png")
plt.show()

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# ガンマ補正
# カメラなどの媒体の経由によって画素値が非線形的に変換された場合の補正
# ディスプレイなどで画像をそのまま表示すると画面が暗くなってしまうため、
# RGBの値を予め大きくすることで、ディスプレイの特性を排除した画像表示を行うことが目的

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("../Question_21_30/imori_gamma.jpg").astype(np.float)

# Gammma correction
c = 1.
g = 2.2

out = img.copy()
out /= 255.
out = (1/c * out) ** (1/g)

out *= 255
out = out.astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)

# 最近傍補間
# 画像を拡大した際に最近傍にある画素をそのまま使う線形補間法
# 単純なアルゴリズムなので、他の補間法と比較して処理速度が速い反面、画質が劣化しやすい

# Read image
img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

# Nearest Neighbor
a = 1.5
aH = int(a * H)
aW = int(a * W)

y = np.arange(aH).repeat(aW).reshape(aW, -1)
x = np.tile(np.arange(aW), (aH, 1))
y = np.round(y / a).astype(np.int)
x = np.round(x / a).astype(np.int)

out = img[y,x]

out = out.astype(np.uint8)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# Bi-linear補間
# 周辺の４画素に距離に応じた重みをつけることで補完する手法
# 計算量が多いだけ処理時間がかかるが、画質の劣化を抑えることができる。

# Read image
img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

# Bi-lenear
a = 1.5
aH = int(a * H)
aW = int(a * W)

y = np.arange(aH).repeat(aW).reshape(aW, -1)
x = np.tile(np.arange(aW), (aH, 1))
y = (y / a)
x = (x / a)

ix = np.floor(x).astype(np.int)
iy = np.floor(y).astype(np.int)

ix = np.minimum(ix, W-2)
iy = np.minimum(iy, H-2)

dx = x - ix
dy = y - iy

dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)


out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

out[out>255] = 255
out = out.astype(np.uint8)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# Bi-cubic補間
# Bi-cubic補間とはBi-linear補間の拡張であり、周辺の16画素から補間を行う

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Bi-cubic
a = 1.5
aH = int(a * H)
aW = int(a * W)

y = np.arange(aH).repeat(aW).reshape(aW, -1)
x = np.tile(np.arange(aW), (aH, 1))
y = (y / a)
x = (x / a)

ix = np.floor(x).astype(np.int)
iy = np.floor(y).astype(np.int)

ix = np.minimum(ix, W-1)
iy = np.minimum(iy, H-1)

dx2 = x - ix
dy2 = y - iy
dx1 = dx2 + 1
dy1 = dy2 + 1
dx3 = 1 - dx2
dy3 = 1 - dy2
dx4 = 1 + dx3
dy4 = 1 + dy3

dxs = [dx1, dx2, dx3, dx4]
dys = [dy1, dy2, dy3, dy4]

def weight(t):
    a = -1.
    at = np.abs(t)
    w = np.zeros_like(t)
    ind = np.where(at <= 1)
    w[ind] = ((a+2) * np.power(at, 3) - (a+3) * np.power(at, 2) + 1)[ind]
    ind = np.where((at > 1) & (at <= 2))
    w[ind] = (a*np.power(at, 3) - 5*a*np.power(at, 2) + 8*a*at - 4*a)[ind]
    return w

w_sum = np.zeros((aH, aW, C), dtype=np.float32)
out = np.zeros((aH, aW, C), dtype=np.float32)

for j in range(-1, 3):
    for i in range(-1, 3):
        ind_x = np.minimum(np.maximum(ix + i, 0), W-1)
        ind_y = np.minimum(np.maximum(iy + j, 0), H-1)

        wx = weight(dxs[i+1])
        wy = weight(dys[j+1])
        wx = np.repeat(np.expand_dims(wx, axis=-1), 3, axis=-1)
        wy = np.repeat(np.expand_dims(wy, axis=-1), 3, axis=-1)

        w_sum += wx * wy
        out += wx * wy * img[ind_y, ind_x]

out /= w_sum
out[out>255] = 255
out = out.astype(np.uint8)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# アフィン変換
# アフィン変換とは3x3の行列を用いて画像の変換を行う操作
# (1)平行移動(Q.28) (2)拡大縮小(Q.29) (3)回転(Q.30) (4)スキュー(Q.31) がある

# 平行移動

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Affine
a = 1.
b = 0.
c = 0.
d = 1.
tx = 30
ty = -30

y = np.arange(H).repeat(W).reshape(W, -1)
x = np.tile(np.arange(W), (H, 1))

out = np.zeros((H+1, W+1, C), dtype=np.float32)

x_new = a * x + b * y + tx
y_new = c * x + d * y + ty

x_new = np.minimum(np.maximum(x_new, 0), W).astype(np.int)
y_new = np.minimum(np.maximum(y_new, 0), H).astype(np.int)

out[y_new, x_new] = img[y, x]
out = out[:H, :W]
out = out.astype(np.uint8)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# 拡大縮小

# Read image
_img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = _img.shape

# Affine
a = 1.3
b = 0.
c = 0.
d = 0.8
tx = 30
ty = -30

img = np.zeros((H+2, W+2, C), dtype=np.float32)
img[1:H+1, 1:W+1] = _img

H_new = np.round(H * d).astype(np.int)
W_new = np.round(W * a).astype(np.int)
out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

x_new = np.tile(np.arange(W_new), (H_new, 1))
y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

adbc = a * d - b * c
x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

out[y_new, x_new] = img[y, x]

out = out[:H_new, :W_new]
out = out.astype(np.uint8)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# 回転（１）

# Read image
_img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = _img.shape


# Affine

A = 30.
theta = - np.pi * A / 180.

a = np.cos(theta)
b = -np.sin(theta)
c = np.sin(theta)
d = np.cos(theta)
tx = 0
ty = 0

img = np.zeros((H+2, W+2, C), dtype=np.float32)
img[1:H+1, 1:W+1] = _img

H_new = np.round(H).astype(np.int)
W_new = np.round(W).astype(np.int)
out = np.zeros((H_new, W_new, C), dtype=np.float32)

x_new = np.tile(np.arange(W_new), (H_new, 1))
y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

adbc = a * d - b * c
x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

out[y_new, x_new] = img[y, x]

out = out.astype(np.uint8)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# 回転（２）

# Read image
_img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = _img.shape

# Affine
A = 30.
theta = - np.pi * A / 180.

a = np.cos(theta)
b = -np.sin(theta)
c = np.sin(theta)
d = np.cos(theta)
tx = 0
ty = 0

img = np.zeros((H+2, W+2, C), dtype=np.float32)
img[1:H+1, 1:W+1] = _img

H_new = np.round(H).astype(np.int)
W_new = np.round(W).astype(np.int)
out = np.zeros((H_new, W_new, C), dtype=np.float32)

x_new = np.tile(np.arange(W_new), (H_new, 1))
y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

adbc = a * d - b * c
x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

dcx = (x.max() + x.min()) // 2 - W // 2
dcy = (y.max() + y.min()) // 2 - H // 2

x -= dcx
y -= dcy

x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

out[y_new, x_new] = img[y, x]
out = out.astype(np.uint8)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# スキュー
# スキュー画像とは、画像を斜め方向に伸ばした画像

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Affine
def affine(img, dx=30, dy=30):
    # get shape
    H, W, C = img.shape

    # Affine hyper parameters
    a = 1.
    b = dx / H
    c = dy / W
    d = 1.
    tx = 0.
    ty = 0.

    # prepare temporary
    _img = np.zeros((H+2, W+2, C), dtype=np.float32)

    # insert image to center of temporary
    _img[1:H+1, 1:W+1] = img

    # prepare affine image temporary
    H_new = np.ceil(dy + H).astype(np.int)
    W_new = np.ceil(dx + W).astype(np.int)
    out = np.zeros((H_new, W_new, C), dtype=np.float32)

    # preprare assigned index
    x_new = np.tile(np.arange(W_new), (H_new, 1))
    y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

    # prepare inverse matrix for affine
    adbc = a * d - b * c
    x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

    x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

    # assign value from original to affine image
    out[y_new, x_new] = _img[y, x]
    out = out.astype(np.uint8)

    return out


# Read image
img = cv2.imread("../Question_31_40/imori.jpg").astype(np.float32)

# Affine
out = affine(img, dx=30, dy=30)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# フーリエ変換
# 二次元離散フーリエ変換(DFT: Discrete Fourier Transformation)とは
# フーリエ変換の画像に対する処理方法である。
# 通常のフーリエ変換はアナログ信号や音声などの連続値かつ一次元を対象に周波数成分を求める計算処理である。
# 一方、ディジタル画像は[0,255]の離散値をとり、かつ画像はHxWの二次元表示であるので、二次元離散フーリエ変換が行われる。

import cv2
import numpy as np
import matplotlib.pyplot as plt


# DFT hyper-parameters
K, L = 128, 128
channel = 3


# DFT
def dft(img):
	H, W, _ = img.shape

	# Prepare DFT coefficient
	G = np.zeros((L, K, channel), dtype=np.complex)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# dft
	for c in range(channel):
		for l in range(L):
			for k in range(K):
				G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)
				#for n in range(N):
				#    for m in range(M):
				#        v += gray[n, m] * np.exp(-2j * np.pi * (m * k / M + n * l / N))
				#G[l, k] = v / np.sqrt(M * N)

	return G

# IDFT
def idft(G):
	# prepare out image
	H, W, _ = G.shape
	out = np.zeros((H, W, channel), dtype=np.float32)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# idft
	for c in range(channel):
		for l in range(H):
			for k in range(W):
				out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

	# clipping
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out



# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# DFT
G = dft(img)

# write poser spectal to image
ps = (np.abs(G) / np.abs(G).max() * 255).astype(np.uint8)
cv2.imwrite("out_ps.jpg", ps)

# IDFT
out = idft(G)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)



"""
fimg = np.fft.fft2(gray)
    
# 第1象限と第3象限, 第2象限と第4象限を入れ替え
fimg =  np.fft.fftshift(fimg)
print(fimg.shape)
# パワースペクトルの計算
mag = 20*np.log(np.abs(fimg))
    
# 入力画像とスペクトル画像をグラフ描画
plt.subplot(121)
plt.imshow(gray, cmap = 'gray')
plt.subplot(122)
plt.imshow(mag, cmap = 'gray')
plt.show()
"""

# フーリエ変換
# ローパスフィルタ
# 画像における高周波成分とは色が変わっている部分（ノイズや輪郭など）を示し、
# 低周波成分とは色があまり変わっていない部分（夕日のグラデーションなど）を表す

import cv2
import numpy as np
import matplotlib.pyplot as plt


# DFT hyper-parameters
K, L = 128, 128
channel = 3

# bgr -> gray
def bgr2gray(img):
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray


# DFT
def dft(img):
	# Prepare DFT coefficient
	G = np.zeros((L, K, channel), dtype=np.complex)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# dft
	for c in range(channel):
		for l in range(L):
			for k in range(K):
				G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)
				#for n in range(N):
				#    for m in range(M):
				#        v += gray[n, m] * np.exp(-2j * np.pi * (m * k / M + n * l / N))
				#G[l, k] = v / np.sqrt(M * N)

	return G

# IDFT
def idft(G):
	# prepare out image
	H, W, _ = G.shape
	out = np.zeros((H, W, channel), dtype=np.float32)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# idft
	for c in range(channel):
		for l in range(H):
			for k in range(W):
				out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

	# clipping
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out


# LPF
def lpf(G, ratio=0.5):
	H, W, _ = G.shape	

	# transfer positions
	_G = np.zeros_like(G)
	_G[:H//2, :W//2] = G[H//2:, W//2:]
	_G[:H//2, W//2:] = G[H//2:, :W//2]
	_G[H//2:, :W//2] = G[:H//2, W//2:]
	_G[H//2:, W//2:] = G[:H//2, :W//2]

	# get distance from center (H / 2, W / 2)
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# make filter
	_x = x - W // 2
	_y = y - H // 2
	r = np.sqrt(_x ** 2 + _y ** 2)
	mask = np.ones((H, W), dtype=np.float32)
	mask[r > (W // 2 * ratio)] = 0

	mask = np.repeat(mask, channel).reshape(H, W, channel)

	# filtering
	_G *= mask

	# reverse original positions
	G[:H//2, :W//2] = _G[H//2:, W//2:]
	G[:H//2, W//2:] = _G[H//2:, :W//2]
	G[H//2:, :W//2] = _G[:H//2, W//2:]
	G[H//2:, W//2:] = _G[:H//2, :W//2]

	return G


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Gray scale
gray = bgr2gray(img)

# DFT
G = dft(img)

# LPF
G = lpf(G)

# IDFT
out = idft(G)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# フーリエ変換
# ハイパスフィルタ

import cv2
import numpy as np
import matplotlib.pyplot as plt


# DFT hyper-parameters
K, L = 128, 128
channel = 3

# bgr -> gray
def bgr2gray(img):
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray


# DFT
def dft(img):
	# Prepare DFT coefficient
	G = np.zeros((L, K, channel), dtype=np.complex)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# dft
	for c in range(channel):
		for l in range(L):
			for k in range(K):
				G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)
				#for n in range(N):
				#    for m in range(M):
				#        v += gray[n, m] * np.exp(-2j * np.pi * (m * k / M + n * l / N))
				#G[l, k] = v / np.sqrt(M * N)

	return G

# IDFT
def idft(G):
	# prepare out image
	H, W, _ = G.shape
	out = np.zeros((H, W, channel), dtype=np.float32)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# idft
	for c in range(channel):
		for l in range(H):
			for k in range(W):
				out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

	# clipping
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out


# HPF
def hpf(G, ratio=0.1):
	H, W, _ = G.shape	

	# transfer positions
	_G = np.zeros_like(G)
	_G[:H//2, :W//2] = G[H//2:, W//2:]
	_G[:H//2, W//2:] = G[H//2:, :W//2]
	_G[H//2:, :W//2] = G[:H//2, W//2:]
	_G[H//2:, W//2:] = G[:H//2, :W//2]

	# get distance from center (H / 2, W / 2)
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# make filter
	_x = x - W // 2
	_y = y - H // 2
	r = np.sqrt(_x ** 2 + _y ** 2)
	mask = np.ones((H, W), dtype=np.float32)
	mask[r < (W // 2 * ratio)] = 0

	mask = np.repeat(mask, channel).reshape(H, W, channel)

	# filtering
	_G *= mask

	# reverse original positions
	G[:H//2, :W//2] = _G[H//2:, W//2:]
	G[:H//2, W//2:] = _G[H//2:, :W//2]
	G[H//2:, :W//2] = _G[:H//2, W//2:]
	G[H//2:, W//2:] = _G[:H//2, :W//2]

	return G


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Gray scale
gray = bgr2gray(img)

# DFT
G = dft(img)

# HPF
G = hpf(G)

# IDFT
out = idft(G)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# フーリエ変換
# ハンドパスフィルタ
# 低周波成分と高周波成分の中間の周波数成分のみを通すハイパスフィルタ

import cv2
import numpy as np
import matplotlib.pyplot as plt


# DFT hyper-parameters
K, L = 128, 128
channel = 3

# bgr -> gray
def bgr2gray(img):
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray


# DFT
def dft(img):
	# Prepare DFT coefficient
	G = np.zeros((L, K, channel), dtype=np.complex)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# dft
	for c in range(channel):
		for l in range(L):
			for k in range(K):
				G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)
				#for n in range(N):
				#    for m in range(M):
				#        v += gray[n, m] * np.exp(-2j * np.pi * (m * k / M + n * l / N))
				#G[l, k] = v / np.sqrt(M * N)

	return G

# IDFT
def idft(G):
	# prepare out image
	H, W, _ = G.shape
	out = np.zeros((H, W, channel), dtype=np.float32)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# idft
	for c in range(channel):
		for l in range(H):
			for k in range(W):
				out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

	# clipping
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out


# HPF
def hpf(G, ratio=0.1):
	H, W, _ = G.shape	

	# transfer positions
	_G = np.zeros_like(G)
	_G[:H//2, :W//2] = G[H//2:, W//2:]
	_G[:H//2, W//2:] = G[H//2:, :W//2]
	_G[H//2:, :W//2] = G[:H//2, W//2:]
	_G[H//2:, W//2:] = G[:H//2, :W//2]

	# get distance from center (H / 2, W / 2)
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# make filter
	_x = x - W // 2
	_y = y - H // 2
	r = np.sqrt(_x ** 2 + _y ** 2)
	mask = np.ones((H, W), dtype=np.float32)
	mask[r < (W // 2 * ratio)] = 0

	mask = np.repeat(mask, channel).reshape(H, W, channel)

	# filtering
	_G *= mask

	# reverse original positions
	G[:H//2, :W//2] = _G[H//2:, W//2:]
	G[:H//2, W//2:] = _G[H//2:, :W//2]
	G[H//2:, :W//2] = _G[:H//2, W//2:]
	G[H//2:, W//2:] = _G[:H//2, :W//2]

	return G


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

# Gray scale
gray = bgr2gray(img)

# DFT
G = dft(img)

# HPF
G = hpf(G)

# IDFT
out = idft(G)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)


# JPEG圧縮

# https://www.marguerite.jp/Nihongo/Labo/Image/JPEG.html

# アルゴリズムは、
# 1. RGB を YCbCrに変換
# 2. YCbCrをDCT
# 3. DCTしたものを量子化
# 4. 量子化したものをIDCT
# 5. IDCTしたYCbCrをRGBに変換

# STEP1　離散コサイン
# imori.jpgをグレースケール化し離散コサイン変換を行い、逆離散コサイン変換を行う

# STEP1　離散コサイン変換は、離散信号を周波数領域へ変換する方法の一つであり、信号圧縮に広く用いられている

# 逆離散コサイン変換(IDCT: Inverse Discrete Cosine Transformation)とは離散コサイン変換の逆（復号）である
# K は復元時にどれだけ解像度を良くするかを決定するパラメータである。 
# K = Tの時は、DCT係数を全部使うのでIDCT後の解像度は最大になるが、Kが１や２などの時は復元に使う情報量（DCT係数）が減るので解像度が下がる。
# これを適度に設定することで、画像の容量を減らすことができる。


import cv2
import numpy as np
import matplotlib.pyplot as plt

# DCT hyoer-parameter 
T = 8
K = 8
channel = 3

# DCT weight
def w(x, y, u, v):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2)
    if v == 0:
        cv /= np.sqrt(2)
    theta = np.pi / (2 * T)
    return (( 2 * cu * cv / T) * np.cos((2*x+1)*u*theta) * np.cos((2*y+1)*v*theta))

# DCT
def dct(img):
    H, W, _ = img.shape

    F = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for v in range(T):
                    for u in range(T):
                        for y in range(T):
                            for x in range(T):
                                F[v+yi, u+xi, c] += img[y+yi, x+xi, c] * w(x,y,u,v)

    return F


# IDCT
def idct(F):
    H, W, _ = F.shape

    out = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for y in range(T):
                    for x in range(T):
                        for v in range(K):
                            for u in range(K):
                                out[y+yi, x+xi, c] += F[v+yi, u+xi, c] * w(x,y,u,v)

    out = np.clip(out, 0, 255)
    out = np.round(out).astype(np.uint8)

    return out



# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# DCT
F = dct(img)

# IDCT
out = idct(F)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# IDCTで用いるDCT係数を8でなく、4にすると画像の劣化が生じる。 
# 入力画像とIDCT画像のPSNRを求めよ。また、IDCTによるビットレートを求めよ。

# PSNR(Peak Signal to Noise Ratio)とは信号対雑音比と呼ばれ、画像がどれだけ劣化したかを示す。
# PSNRが大きいほど、画像が劣化していないことを示す

import cv2
import numpy as np
import matplotlib.pyplot as plt

# DCT hyoer-parameter
T = 8
K = 4
channel = 3

# DCT weight
def w(x, y, u, v):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2)
    if v == 0:
        cv /= np.sqrt(2)
    theta = np.pi / (2 * T)
    return (( 2 * cu * cv / T) * np.cos((2*x+1)*u*theta) * np.cos((2*y+1)*v*theta))

# DCT
def dct(img):
    H, W, _ = img.shape

    F = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for v in range(T):
                    for u in range(T):
                        for y in range(T):
                            for x in range(T):
                                F[v+yi, u+xi, c] += img[y+yi, x+xi, c] * w(x,y,u,v)

    return F


# IDCT
def idct(F):
    H, W, _ = F.shape

    out = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for y in range(T):
                    for x in range(T):
                        for v in range(K):
                            for u in range(K):
                                out[y+yi, x+xi, c] += F[v+yi, u+xi, c] * w(x,y,u,v)

    out = np.clip(out, 0, 255)
    out = np.round(out).astype(np.uint8)

    return out


# MSE
def MSE(img1, img2):
    H, W, _ = img1.shape
    mse = np.sum((img1 - img2) ** 2) / (H * W * channel)
    return mse

# PSNR
def PSNR(mse, vmax=255):
    return 10 * np.log10(vmax * vmax / mse)

# bitrate
def BITRATE():
    return 1. * T * K * K / T / T


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# DCT
F = dct(img)

# IDCT
out = idct(F)

# MSE
mse = MSE(img, out)

# PSNR
psnr = PSNR(mse)

# bitrate
bitrate = BITRATE()

print("MSE:", mse)
print("PSNR:", psnr)
print("bitrate:", bitrate)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# JPEG圧縮　DCT＋量子化

# DCT係数を量子化し、IDCTで復元せよ。また、その時の画像の容量を比べよ。
# DCT係数を量子化することはjpeg画像にする符号化で用いられる手法である。
# 量子化とは、値を予め決定された区分毎に値を大まかに丸め込む作業であり、floorやceil, roundなどが似た計算である。
# JPEG画像ではDCT係数を下記で表される量子化テーブルに則って量子化する。
# 量子化では8x8の係数をQで割り、四捨五入する。その後Qを掛けることで行われる。 IDCTでは係数は全て用いるものとする。

import cv2
import numpy as np
import matplotlib.pyplot as plt

# DCT hyoer-parameter
T = 8
K = 4
channel = 3

# DCT weight
def DCT_w(x, y, u, v):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2)
    if v == 0:
        cv /= np.sqrt(2)
    theta = np.pi / (2 * T)
    return (( 2 * cu * cv / T) * np.cos((2*x+1)*u*theta) * np.cos((2*y+1)*v*theta))

# DCT
def dct(img):
    H, W, _ = img.shape

    F = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for v in range(T):
                    for u in range(T):
                        for y in range(T):
                            for x in range(T):
                                F[v+yi, u+xi, c] += img[y+yi, x+xi, c] * DCT_w(x,y,u,v)

    return F


# IDCT
def idct(F):
    H, W, _ = F.shape

    out = np.zeros((H, W, channel), dtype=np.float32)

    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):
                for y in range(T):
                    for x in range(T):
                        for v in range(K):
                            for u in range(K):
                                out[y+yi, x+xi, c] += F[v+yi, u+xi, c] * DCT_w(x,y,u,v)

    out = np.clip(out, 0, 255)
    out = np.round(out).astype(np.uint8)

    return out

# Quantization
def quantization(F):
    H, W, _ = F.shape

    Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
                (12, 12, 14, 19, 26, 58, 60, 55),
                (14, 13, 16, 24, 40, 57, 69, 56),
                (14, 17, 22, 29, 51, 87, 80, 62),
                (18, 22, 37, 56, 68, 109, 103, 77),
                (24, 35, 55, 64, 81, 104, 113, 92),
                (49, 64, 78, 87, 103, 121, 120, 101),
                (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

    for ys in range(0, H, T):
        for xs in range(0, W, T):
            for c in range(channel):
                F[ys: ys + T, xs: xs + T, c] =  np.round(F[ys: ys + T, xs: xs + T, c] / Q) * Q

    return F

# MSE
def MSE(img1, img2):
    H, W, _ = img1.shape
    mse = np.sum((img1 - img2) ** 2) / (H * W * channel)
    return mse

# PSNR
def PSNR(mse, vmax=255):
    return 10 * np.log10(vmax * vmax / mse)

# bitrate
def BITRATE():
    return 1. * T * K * K / T / T


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# DCT
F = dct(img)

# quantization
F = quantization(F)

# IDCT
out = idct(F)

# MSE
mse = MSE(img, out)

# PSNR
psnr = PSNR(mse)

# bitrate
bitrate = BITRATE()

print("MSE:", mse)
print("PSNR:", psnr)
print("bitrate:", bitrate)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)

# YCbCr表色系とは、画像を明るさを表すY、輝度と青レベルの差Cb、輝度と赤レベルの差Crに分解する表現方法である。
# RGBからYCbCrへの変換は次式。
# Y = 0.299 * R + 0.5870 * G + 0.114 * B
# Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
# Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

# YCbCrからRGBへの変換は次式。
# R = Y + (Cr - 128) * 1.402
# G = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
# B = Y + (Cb - 128) * 1.7718

# Cannyエッジ検出
# http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_canny/py_canny.html

# Canny法は、
# 1. ガウシアンフィルタを掛ける
# 2. x, y方向のSobelフィルタを掛け、それらからエッジ強度とエッジ勾配を求める
# 3. エッジ勾配の値から、Non-maximum suppression によりエッジの細線化を行う
# 4. ヒステリシスによる閾値処理を行う
# 以上により、画像からエッジ部分を抜き出す手法である。

# 処理手順は、
# 1. 画像をグレースケール化する
# 2. ガウシアンフィルタ(5x5, s=1.4)をかける
# 3. x方向、y方向のsobelフィルタを掛け、画像の勾配画像fx, fyを求め、勾配強度と勾配角度を次式で求める。
# 4. 勾配角度を次式に沿って、量子化する。
# 5. 勾配角度から、Non-maximum suppressionを行い、エッジ線を細くする（細線化）
# 6. 閾値により勾配強度の二値化を行うがCanny法では二つの閾値(HT: high thoresholdとLT: low threshold)を用いる。

# 5
# Non-maximum suppression(NMS)とは非最大値以外を除去する作業の総称である。（他のタスクでもこの名前はよく出る）
# ここでは、注目している箇所の勾配角度の法線方向の隣接ピクセルの３つの勾配強度を比較して、最大値ならそのまま値をいじらずに、最大値でなければ強度を0にする、
# つまり、勾配強度edge(x,y)に注目している際に、勾配角度angle(x,y)によって次式のようにedge(x,y)を変更する。

# 6
# 二つの閾値(HT: high thoresholdとLT: low threshold)を用いる。
# ちなみに閾値の値は結果を見ながら判断するしかない。


# Hough変換を用いた直線検出を行う。

# Hough変換とは、座標を直交座標から極座標に変換することにより数式に沿って直線や円など一定の形状を検出する手法である。 ある直線状の点では極座標に変換すると一定のr, tにおいて交わる。 その点が検出すべき直線を表すパラメータであり、このパラメータを逆変換すると直線の方程式を求めることができる。

# 方法としては、
# 1. エッジ画像からエッジのピクセルにおいてHough変換を行う。
# 2. Hough変換後の値のヒストグラムをとり、極大点を選ぶ。
# 3. 極大点のr, tの値をHough逆変換して検出した直線のパラメータを得る。

# ここでは、1のHough変換を行いヒストグラムを作成する。

# アルゴリズムは、

# 画像の対角線の長さrmaxを求める
# エッジ箇所(x,y)において、t = 0-179で一度ずつtを変えながら、次式によりHough変換を行う
# r = x * cos(t) + y * sin(t)
# 180 x rmaxのサイズの表を用意し、1で得たtable(t, r) に1を足す
# これはすなわち投票(ボーディング)であり、一定の箇所に投票が集中する。


# モルフォロジー処理
# http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

# モルフォロジー処理とは二値化画像の白(255)マス部分を4近傍(上下左右1マス)に膨張、または1マスだけ収縮させる処理をいう。
# この膨張と収縮を何度も繰り返すことで1マスだけに存在する白マスを消したり(Q.49. オープニング処理)、
# 本来つながってほしい白マスを結合させたりできる(Q.50. クロージング処理)。
# モルフォロジー処理の膨張(Dilation)アルゴリズムは、 注目画素I(x, y)=0で、I(x, y-1), I(x-1, y), I(x+1, y), I(x, y+1)のどれか一つが255なら、I(x, y) = 255 とする。

# オープニング処理
# モルフォロジー処理の収縮をN回行った後に膨張をN回行う処理である。
# 一つだけ余分に存在する画素などを削除できる。

# クロージング処理
# モルフォロジー処理の膨張をN回行った後に収縮をN回行う処理である。
# 途中で途切れた画素を結合することができる。

# モルフォロジー勾配
# モルフォロジー膨張の画像と収縮の画像の差分をとることで、物体の境界線を抽出する手法である

# トップハット変換
# 元画像からオープニング処理を行った画像を差し引いた画像であり、細い線状のものやノイズなどを抽出できると言われる。

# ブラックハット変換
# クロージング画像から元画像を差し引いた画像であり、これもトップ変換同様に細い線状やノイズを抽出できると言われる。


# テンプレートマッチング
# テンプレート画像と全体画像の一部分で類似度が高い位置を探す手法であり、物体検出などで使われる。
# 今では物体検出はCNNで行われるが、テンプレートマッチングは最も基本処理となる。

# アルゴリズムとしては、画像I (H x W)、テンプレート画像T (h x w)とすると、
# １　画像Iにおいて、for ( j = 0, H-h) for ( i = 0, W-w)と1ピクセルずつずらしながら画像Aの一部分I(i:i+w, j:j+h)とテンプレート画像の類似度Sを計算する。
# ２　Sが最大もしくは最小の位置がマッチング位置となる。

# Sの選び方は主にSSD, SAD, NCC, ZNCCなどがあり、それぞれ最大値をとるか最小値をとるか異なる。
# SSD(Sum of Squared Difference)
# 画素値の差分の二乗値の和を類似度にする手法であり、Sが最小の位置がマッチング位置となる。
S = Sum_{x=0:w, y=0:h} (I(i+x, j+y) - T(x, y) )^2

# SAD(Sum of Absolute Difference)
# 画素値の差分の絶対値の和を類似度にする手法であり、Sが最小の位置がマッチング位置となる。
S = Sum_{x=0:w, y=0:h} |I(i+x, j+y) - T(x, y)|

# NCC(Normalized Cross Correlation)
# 正規化相互相関を類似度にする手法であり、Sが最大の位置がマッチング位置となる。
# このSは、-1<=S<=1をとる。 NCCは照明変化に強いと言われる
# 類似度が１に近いほど、似ている
     Sum_{x=0:w, y=0:h} I(i+x, j+y) * T(x, y)
S = -----------------------------------------------------------------------------
    Sqrt(Sum_{x=0:w, y=0:h} I(i+x, j+y)^2) * Sqrt(Sum_{x=0:w, y=0:h} T(x, y)^2)

# ZNCC(Zero means Normalized Cross Correlation)
# 零平均正規化相互相関を類似度にする手法であり、Sが最大の位置がマッチング位置となる。
# 画像Iの平均値をmi、画像Tの平均値をmtとすると、Sは次式で計算される。（ただし、平均値はRGB成分ごとに減算する）
# このSは、-1<=S<=1をとる。 ZNCCは平均値を引くことでNCCよりも照明変化に強いと言われる。
       Sum_{x=0:w, y=0:h} (I(i+x, j+y)-mi) * (T(x, y)-mt)
S = --------------------------------------------------------------------------------------
    Sqrt(Sum_{x=0:w, y=0:h} (I(i+x, j+y)-mi)^2) * Sqrt(Sum_{x=0:w, y=0:h} (T(x, y)-mt)^2)


# ちなみにテンプレートマッチングのように画像を左上から右に順に見ていくことを走査(ラスタスキャン)やスライディングウィンドウと呼ぶ。このワードは画像処理でよく出る頻出である。


# ラベリング
https://imagingsolution.net/imaging/labelling/

# アルファブレンド
# アルファブレンドとは透明度（アルファ値）を設定することにより画像の透明度を設定する方法である。 
# OpenCVでは透明度のパラメータはないが、PILなどのライブラリでは存在する。 ここではその透明度を手動で設定する。

# 二つの画像を重ね合わせたい時などに、この手法は有効である。

# img1とimg2を1:1の割合で重ね合わせたい時は、次式となる。 alphaの値を変えることで重ねる時の重みを変えることができる。

alpha = 0.5
out = img1 * alpha + img2 * (1 - alpha)


# 細線化処理
# 細線化とは画素の幅を1にする処理であり、ここでは次のアルゴリズムに沿って処理を行え。

# １　左上からラスタスキャンする。
# ２　x0(x,y)=0ならば、処理なし。x0(x,y)=1ならば次の3条件を満たす時にx0=0に変える。
# 　　(1) 注目画素の4近傍に0が一つ以上存在する
# 　　(2) x0の4-連結数が1である
# 　　(3) x0の8近傍に1が3つ以上存在する
# ３　一回のラスタスキャンで2の変更数が0になるまで、ラスタスキャンを繰り返す。
# 細線化にはヒルディッチのアルゴリズム(Q.64)や、Zhang-Suenのアルゴリズム(Q.65)、田村のアルゴリズムなどが存在する。

# ヒルディッチの細線化
# アルゴリズムは、次の通り。

# １　左上からラスタスキャンする。
# ２　x0(x,y)=0ならば、処理なし。x0(x,y)=1ならば次の5条件を満たす時にx0=-1に変える。
# 　２−１　注目画素の4近傍に0が一つ以上存在する
# 　２−２　x0の8-連結数が1である
# 　２−３　x1〜x8の絶対値の合計が2以上
# 　２−４　x0の8近傍に1が1つ以上存在する
# 　２−５　xn(n=1〜8)全てに対して以下のどちらかが成り立つ
# 　	・xnが-1以外
# 	　・xnを0とした時、x0の8-連結数が1である
# ３　各画素の-1を0に変える
# ４　一回のラスタスキャンで3の変更数が0になるまで、ラスタスキャンを繰り返す

# HOG(Histogram of Oriented Gradients)
# 画像の特徴量表現の一種で

# 画像認識(画像が何を写した画像か)や検出（画像の中で物体がどこにあるか）では、
# (1)画像から特徴量を得て(特徴抽出)、
# (2)特徴量を基に認識や検出を行う(認識・検出)

# ディープラーニングでは特徴抽出から認識までを機械学習により自動で行うため、HOGなどは見られなくなっているが、ディープラーニングが流行る前まではHOGは特徴量表現としてよく使われたらしい。

# HOGは以下のアルゴリズムで得られる。

# 1.画像をグレースケール化し、x、ｙ方向の輝度勾配を求める
# 	x方向: gx = I(x+1, y) - I(x-1, y)
# 	y方向: gy = I(x, y+1) - I(x, y-1)
# 2.gx, gyから勾配強度と勾配角度を求める。
# 	勾配強度: mag = sqrt(gt ** 2 + gy ** 2)
# 	勾配角度: ang = arctan(gy / gx)
# 3.勾配角度を [0, 180]で9分割した値に量子化する。つまり、[0,20]には0、[20, 40]には1というインデックスを求める。
# 4.画像をN x Nの領域に分割し(この領域をセルという)、セル内で3で求めたインデックスのヒストグラムを作成する。ただし、当表示は1でなく勾配角度を求める。
# 5.C x Cのセルを１つとして(これをブロックという)、ブロック内のセルのヒストグラムを次式で正規化する。これを1セルずつずらしながら行うので、一つのセルが何回も正規化される。
# 	h(t) = h(t) / sqrt(Sum h(t) + epsilon)
# 	通常は　epsilon=1

# 以上でHOG特徴量が求められる。


# カラートラッキング
# 特定の色の箇所を抽出する手法である。
# ただし、RGBの状態で色成分を指定するのは256^3のパターンがあり、とても大変
# 手動ではかなり難しいので、HSV変換を用いる。

# HSV変換とは RGBをH(色相)、S(彩度)、V(明度)に変換する手法である。
# Saturation(彩度) 彩度が小さいほど白、彩度が大きいほど色が濃くなる。 0<=S<=1
# Value (明度) 明度が小さいほど黒くなり、明度が大きいほど色がきれいになる。 0<=V<=1
# Hue(色相) 色を0<=H<=360の角度で表し、具体的には次のように表される。
# 	赤 黄色  緑  水色  青  紫   赤
# 	0  60  120  180 240 300 360
# つまり、青色のカラートラッキングを行うにはHSV変換を行い、180<=H<=260となる位置が255となるような二値画像を出力すればよい。

import cv2
import numpy as np
import matplotlib.pyplot as plt

# BGR -> HSV
def BGR2HSV(_img):
	img = _img.copy() / 255.

	hsv = np.zeros_like(img, dtype=np.float32)

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()
	min_arg = np.argmin(img, axis=2)

	# H
	hsv[..., 0][np.where(max_v == min_v)]= 0
	## if min == B
	ind = np.where(min_arg == 0)
	hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
	## if min == R
	ind = np.where(min_arg == 2)
	hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
	## if min == G
	ind = np.where(min_arg == 1)
	hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300
		
	# S
	hsv[..., 1] = max_v.copy() - min_v.copy()

	# V
	hsv[..., 2] = max_v.copy()
	
	return hsv

# make mask
def get_mask(hsv):
	mask = np.zeros_like(hsv[..., 0])
	#mask[np.where((hsv > 180) & (hsv[0] < 260))] = 255
	mask[np.logical_and((hsv[..., 0] > 180), (hsv[..., 0] < 260))] = 255
	return mask


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# RGB > HSV
hsv = BGR2HSV(img)


# color tracking
mask = get_mask(hsv)

out = mask.astype(np.uint8)

# Save result
cv2.imwrite("out.png", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()


# make mask
def get_mask(hsv):
	mask = np.zeros_like(hsv[..., 0])
	#mask[np.where((hsv > 180) & (hsv[0] < 260))] = 255
	mask[np.logical_and((hsv[..., 0] > 180), (hsv[..., 0] < 260))] = 1
	return mask

# masking
def masking(img, mask):
	mask = 1 - mask
	out = img.copy()
	# mask [h, w] -> [h, w, channel]
	mask = np.tile(mask, [3, 1, 1]).transpose([1, 2, 0])
	out *= mask

	return out


# Erosion
def Erode(img, Erode_time=1):
	H, W = img.shape
	out = img.copy()

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each erode
	for i in range(Erode_time):
		tmp = np.pad(out, (1, 1), 'edge')
		# erode
		for y in range(1, H + 1):
			for x in range(1, W + 1):
				if np.sum(MF * tmp[y - 1 : y + 2 , x - 1 : x + 2]) < 1 * 4:
					out[y - 1, x - 1] = 0

	return out

# マスキング(カラートラッキング＋モルフォロジー)

# Dilation
def Dilate(img, Dil_time=1):
	H, W = img.shape

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each dilate time
	out = img.copy()
	for i in range(Dil_time):
		tmp = np.pad(out, (1, 1), 'edge')
		for y in range(1, H + 1):
			for x in range(1, W + 1):
				if np.sum(MF * tmp[y - 1 : y + 2, x - 1 : x + 2]) >= 1:
					out[y - 1, x - 1] = 1

	return out


# Opening morphology
def Morphology_Opening(img, time=1):
    out = Erode(img, Erode_time=time)
    out = Dilate(out, Dil_time=time)
    return out

# Closing morphology
def Morphology_Closing(img, time=1):
	out = Dilate(img, Dil_time=time)
	out = Erode(out, Erode_time=time)
	return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# RGB > HSV
hsv = BGR2HSV(img / 255.)

# color tracking
mask = get_mask(hsv)

# closing
mask = Morphology_Closing(mask, time=5)

# opening
mask = Morphology_Opening(mask, time=5)

# masking
out = masking(img, mask)

out = out.astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ガボールフィルタ
ガウス分布と周波数変換を合わせたフィルタ
画像中にどの向きの線が含まれているかを抽出できるフィルタ
画像の特定方向のみのエッジを抽出する時に使われる。

G(y, x) = exp(-(x'^2 + g^2 y'^2) / 2 s^2) * cos(2 pi x' / l + p)
x' = cosA * x + sinA * y
y' = -sinA * x + cosA * y

y, x はフィルタの位置　フィルタサイズがKとすると、 y, x は [-K//2, k//2]　の値を取る。
g ... gamma ガボールフィルタの楕円率
s ... sigma ガウス分布の標準偏差
l ... lambda 周波数の波長
p ... 位相オフセット
A ... フィルタの回転　抽出したい角度を指定する。

img = cv2.imread("画像のpath") # pathを間違えるとNoneが返ってきます
gabor = cv2.getGaborKernel((30, 30), 4.0, numpy.radians(0), 10, 0.5, 0)
dst = cv2.filter2D(img, -1, gabor)
pylab.imshow(dst) and pylab.show()

help(cv2.getGaborKernel)

# Gabor
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
	# get half size
	d = K_size // 2

	# prepare kernel
	gabor = np.zeros((K_size, K_size), dtype=np.float32)

	# each value
	for y in range(K_size):
		for x in range(K_size):
			# distance from center
			px = x - d
			py = y - d

			# degree -> radian
			theta = angle / 180. * np.pi

			# get kernel x
			_x = np.cos(theta) * px + np.sin(theta) * py

			# get kernel y
			_y = -np.sin(theta) * px + np.cos(theta) * py

			# fill kernel
			gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)

	# kernel normalization
	gabor /= np.sum(np.abs(gabor))

	return gabor


# get gabor kernel
gabor = Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0)

# Visualize
# normalize to [0, 255]
out = gabor - np.min(gabor)
out /= np.max(out)
out *= 255

out = out.astype(np.uint8)
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)

# ガボールフィルタによる特徴量抽出

# Grayscale
def BGR2GRAY(img):
	# Grayscale
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray

# Gabor
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
	# get half size
	d = K_size // 2

	# prepare kernel
	gabor = np.zeros((K_size, K_size), dtype=np.float32)

	# each value
	for y in range(K_size):
		for x in range(K_size):
			# distance from center
			px = x - d
			py = y - d

			# degree -> radian
			theta = angle / 180. * np.pi

			# get kernel x
			_x = np.cos(theta) * px + np.sin(theta) * py

			# get kernel y
			_y = -np.sin(theta) * px + np.cos(theta) * py

			# fill kernel
			gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)

	# kernel normalization
	gabor /= np.sum(np.abs(gabor))

	return gabor


def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    gray = np.pad(gray, (K_size//2, K_size//2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)
        
    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y : y + K_size, x : x + K_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


def Gabor_process(img):
    # get shape
    H, W, _ = img.shape

    # gray scale
    gray = BGR2GRAY(img).astype(np.float32)

    # define angle
    As = [0, 45, 90, 135]

    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, angle=A)

        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# gabor process
out = Gabor_process(img)


cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)

# Hessianのコーナー検出
コーナー検出とはエッジにおける角の点を検出することである。
コーナーは曲率が大きくなる点であり、次式のガウス曲率において、

ガウス曲率 K = det(H) / (1 + Ix^2 + Iy^2)^2

det(H) = Ixx Iyy - IxIy^2
H ... ヘシアン行列。画像の二次微分(グレースケール画像などに対して、Sobelフィルタを掛けて求められる)。画像上の一点に対して、次式で定義される。
Ix ... x方向のsobelフィルタを掛けたもの。 
Iy ... y方向のsobelフィルタを掛けたもの。
H = [ Ix^2  IxIy]
      IxIy  Iy^2

ヘシアンのコーナー検出では、det(H)が極大点をコーナーとみなす。
極大点は注目画素と8近傍を比較して、注目画素の値が最大であれば極大点として扱う。

# Hessian corner detection
def Hessian_corner(img):

	## Grayscale
	def BGR2GRAY(img):
		gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
		gray = gray.astype(np.uint8)
		return gray

	## Sobel
	def Sobel_filtering(gray):
		# get shape
		H, W = gray.shape

		# sobel kernel
		sobely = np.array(((1, 2, 1),
						(0, 0, 0),
						(-1, -2, -1)), dtype=np.float32)

		sobelx = np.array(((1, 0, -1),
						(2, 0, -2),
						(1, 0, -1)), dtype=np.float32)

		# padding
		tmp = np.pad(gray, (1, 1), 'edge')

		# prepare
		Ix = np.zeros_like(gray, dtype=np.float32)
		Iy = np.zeros_like(gray, dtype=np.float32)

		# get differential
		for y in range(H):
			for x in range(W):
				Ix[y, x] = np.mean(tmp[y : y  + 3, x : x + 3] * sobelx)
				Iy[y, x] = np.mean(tmp[y : y + 3, x : x + 3] * sobely)
			
		Ix2 = Ix ** 2
		Iy2 = Iy ** 2
		Ixy = Ix * Iy

		return Ix2, Iy2, Ixy

		

	## Hessian
	def corner_detect(gray, Ix2, Iy2, Ixy):
		# get shape
		H, W = gray.shape

		# prepare for show detection
		out = np.array((gray, gray, gray))
		out = np.transpose(out, (1,2,0))

		# get Hessian value
		Hes = np.zeros((H, W))

		for y in range(H):
			for x in range(W):
				Hes[y,x] = Ix2[y,x] * Iy2[y,x] - Ixy[y,x] ** 2

		## Detect Corner and show
		for y in range(H):
			for x in range(W):
				if Hes[y,x] == np.max(Hes[max(y-1, 0) : min(y+2, H), max(x-1, 0) : min(x+2, W)]) and Hes[y, x] > np.max(Hes) * 0.1:
					out[y, x] = [0, 0, 255]

		out = out.astype(np.uint8)

		return out

	
	# 1. grayscale
	gray = BGR2GRAY(img)

	# 2. get difference image
	Ix2, Iy2, Ixy = Sobel_filtering(gray)

	# 3. corner detection
	out = corner_detect(gray, Ix2, Iy2, Ixy)

	return out

# Harrisのコーナー検出
固有値を計算せずにコーナーかどうか判定する
Harrisのコーナー検出のアルゴリズムは、

1. 画像をグレースケール化。
2. Sobelフィルタにより、ヘシアン行列を求める。
		H = [ Ix^2  IxIy]
      IxIy  Iy^2
3. Ix^2, Iy^2, IxIyにそれぞれガウシアンフィルターをかける。
4. 各ピクセル毎に、R = det(H) - k (trace(H))^2 を計算する。 (kは実験的に0.04 - 0.16 が良いとされる)
5. R >= max(R) * th を満たすピクセルがコーナーとなる。 (thは0.1となることが多い)
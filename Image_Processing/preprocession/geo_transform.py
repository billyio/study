# 幾何変換

import cv2
import numpy as np


# scaling

img = cv2.imread('sample.jpg')

res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

# 引数interpolation
default => cv2.INTER_LINEAR
縮小 => cv2.INTER_AREA
拡大 => cv2.INTER_CUBIC (処理が遅い)

# cv2.imreadの第二引数
cv2.IMREAD_GRAYSCALEを指定
カラーの画像ファイルをgrayscaleで読み込む
cv2.IMREAD_GRAYSCALE == 0 => 0 を指定してもOK
エッジを検出したりするなど、色情報が必要ないときに便利


# translation

img = cv2.imread('sample.jpg',0)
rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# rotation

img = cv2.imread('sample.jpg',0)
rows,cols = img.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)　# 1:回転中心, 2:回転角度（左90°）
dst = cv2.warpAffine(img,M,(cols,rows))


# affine

img = cv2.imread('drawing.png')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)　# [50,50] => [10, 100]のように各移動する、アフィン変換行列を求める
dst = cv2.warpAffine(img,M,(cols,rows))


# projective transformation

img = cv2.imread('sudokusmall.png')
rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)　# 1,2: 頂点座標、これらの透視変換行列を求める
dst = cv2.warpPerspective(img,M,(300,300))　# 透視変換（ホモグラフィー変換）

アフィン変換 => 平行四辺形に変形できますが、辺同士の平行の関係は維持される
ホモグラフィー変換 => 自由に変形することができる
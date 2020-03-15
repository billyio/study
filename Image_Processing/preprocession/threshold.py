# 閾値処理
threshold　閾値

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dave.jpg',0)
img = cv2.medianBlur(img,5) # 平滑化（メディアンフィルタ）

cv2.thresholdの出力 => 1:retval 2:image
cv2.adaptiveThresholdの出力 => image only

# 単純なバイナリ、白黒
# 出力は、1:retval 2:image
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# 近傍領域の中央値をしきい値
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

# 近傍領域の重み付け平均値をしきい値、重みはガウス分布に従う
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


# 大津の二値化
bimodal image (ヒストグラムが双峰性を持つような画像)のとき、有効

ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Gaussian filtering => remove noise 
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Harris Corner

1. グレースケール
2. へシアン行列を求める
3. IxIx, IxIy, IyIyにガウシアンフィルター
4. R = det(H) - k (trace(H))^2 # k=0.04〜0.16が多い
5. R >= max(R) * th を満たすピクセルがコーナー # th=0.1が多い

import cv2
import numpy as np

filename = 'chessboard.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None) # 

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
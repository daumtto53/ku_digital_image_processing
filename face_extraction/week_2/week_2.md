[TOC]

# 환경설정

## Jupyter notebook

New -> python 3

## OpenCV

`pip install opencv-python`
`python`
`import cv2`

 `d:\cms\konkuk\2022_1\destination_drive\digital_image_processing\Face_Extraction\week_2`

# 코드 분석

```python
import cv2
import numpy as np
from math import acos, pi, sqrt
```

```python
src = cv2.imread("경로", FLAG)
height, width = src.shape[0], src.shape[1]


I = np.zeros((height, width))
S = np.zeros((height, width))
H = np.zeros((height, width))

for i in range(height):
    for j in range(width):
        B, G, R = src[i][j][0] / 255., src[i][j][1] / 255., src[i][j][2] / 255.

        I[i][j] = (B + G + R) / 3.
        if B + G +R != 0 :
            S[i][j] = 1 - 3 * np.min([R, G, B]/[R + G + B])
        H[i][j]

dest = np.zeros((height, width, 3), dtype=np.uint8)

for i in range(height) :
    for j in range(width) : 
            if H[i][j] >= 0.25 and H[i][j] <= 0.6 :
                dest[i][j] = src[i][j]

cv2.imshow('dst', dest)
cv2.imshow('src', src)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''
below code ? 
'''
I = I*255
S = S*255
H = H*255/(2*pi)

I = np.asarray(I, dtype=np.uint8)
S = np.asarray(S, dtype=np.uint8)
H = np.asarray(H, dtype=np.uint8)

cv2.imshow('src', src)
cv2.imshow('H', H)
cv2.imshow('S', S)
cv2.imshow('I', I)
cv2.waitkey(0)
cv2.destroyAllWindows()

# Python 문법
```

---

## import VS from

모듈을 import 하느냐, 모듈 안의 funciton, variable을 import 하는가의 차이.

* import [module] as [alias] :     **moudle 에서 alias 하여 import 하겠다.**

* from [module] import [func / var] : **module의 func/var를 import 하겠다.**

## for i in range() :

## 나눗셈 시 실수형으로 변환

`var = 정수 / 3.0` , 또는 `var = float(정수)`

   

## numpy

---

### shape

> [numpy.zeros &#8212; NumPy v1.22 Manual](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html?highlight=zeros#numpy.zeros)
> 
> `numpy.zeros(*shape*, *dtype=float*, *order='C'*, ***, *like=None*)`
> 
> **shape** : int or tuple of ints
> 
> **dtype** : data-type, optional

### np.zeros

> 0으로 가득찬 array 생성.

## OpenCV

### imread

### **Image Dimension**

OpenCV는 기본적으로 이미지를 3차원 배열, 즉 행 X 열 X 채널로 표현

행과 열은 이미지의 크기, 즉 높이와 폭만큼의 길이를 갖고 채널은 컬러인 경우 파랑, 초록, 빨강 3개의 길이를 갖는다.

### image_shape

3D tuple

 ![](https://mblogthumb-phinf.pstatic.net/MjAyMDAxMTRfNDUg/MDAxNTc4OTk1MDM0MTU1.h3C0KMd0NmV4LLwxQW3GdetczPwoBAJY8eYeQ6xWlPwg.9rCdr515CcNwqi6ujGHUl9rR5NfS1lEihr2pCPfZHwwg.PNG.bosongmoon/image.png?type=w800)

### cv2.iamshow()

show image in specified window. `waitKey()` 랑 같이 사용함.

**`waitKey()`** : 버튼입력전까지 대기상태.

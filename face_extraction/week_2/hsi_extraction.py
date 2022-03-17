import cv2
import numpy as np
from math import acos, pi, sqrt

src = cv2.imread("경로", FLAG)
height, width = src.shape[0], src.shape[1]

I = np.zeros((height, width))
S = np.zeros((height, width))
H = np.zeros((height, width))

def rgb_to_hsi(r, g, b) :


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
                dst[i][j] = src[i][j]

cv2.imshow('dst', dst)
cv2.imshow('src', src)
cv2.waitKey(0)
cv2.destroyAllWindows()


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

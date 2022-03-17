import cv2
import numpy as np
from math import acos, pi, sqrt


def rgb_to_hue(r, g, b):
    angle = 0
    if b != g != r:
        angle = 0.5 * ((r - g) + (r - b)) / sqrt(((r - g) ** 2) + (r - b) * (g - b))
    if b <= g:
        return acos(angle)
    else:
        return 2 * pi - acos(angle)


def rgb_to_intensity(r, g, b):
    val = (r + g + b) / 3.
    if val == 0:
        return 0
    else:
        return val


def rgb_to_saturity(r, g, b):
    return 1 - 3 * np.min([r, g, b]) / (r + g + b)


def channel_saturity(src):
    S = np.asarray(src * 255, dtype=np.uint8)
    return S


def channel_intensity(src):
    I = np.asarray(src * 255, dtype=np.uint8)
    return I


def channel_hue(src):
    H = src * 255 / (2 * pi)
    return np.asarray(H, dtype=np.uint8)


def extract_skin_color(file_path):
    src = cv2.imread(file_path, cv2.IMREAD_COLOR)

    height, width = src.shape[0], src.shape[1]

    I = np.zeros((height, width))
    S = np.zeros((height, width))
    H = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            b, g, r = src[i][j][0] / 255., src[i][j][1] / 255., src[i][j][2] / 255.

            I[i][j] = (r + g + b) / 3.
            if r + g + b != 0:
                S[i][j] = 1 - 3 * np.min([r, g, b]) / (r + g + b)
            H[i][j] = rgb_to_hue(r, g, b)

    dst = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if H[i][j] >= 0.25 and H[i][j] <= 0.6:
                dst[i][j] = src[i][j]
    cv2.imshow('dst', dst)
    cv2.imshow('src', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    I = I * 255
    S = S * 255
    H = H * 255 / (2 * pi)

    I = np.asarray(I, dtype=np.uint8)
    S = np.asarray(S, dtype=np.uint8)
    H = np.asarray(H, dtype=np.uint8)

    cv2.imshow('src', src)
    cv2.imshow('H', H)
    cv2.imshow('S', S)
    cv2.imshow('I', I)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# src = cv2.imread("skin_color_example\\black_male_1.jpg", cv2.IMREAD_COLOR)
#
# height, width = src.shape[0], src.shape[1]
#
# I = np.zeros((height, width))
# S = np.zeros((height, width))
# H = np.zeros((height, width))
#
# for i in range(height):
#     for j in range(width):
#         b, g, r = src[i][j][0] / 255., src[i][j][1] / 255., src[i][j][2] / 255.
#
#         I[i][j] = (r + g + b) / 3.
#         if r + g + b != 0:
#             S[i][j] = 1 - 3 * np.min([r, g, b]) / (r + g + b)
#         H[i][j] = rgb_to_hue(r, g, b)
#
# dst = np.zeros((height, width, 3), dtype=np.uint8)
#
# for i in range(height):
#     for j in range(width):
#         if H[i][j] >= 0.25 and H[i][j] <= 0.6:
#             dst[i][j] = src[i][j]
# cv2.imshow('dst', dst)
# cv2.imshow('src', src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# I = I * 255
# S = S * 255
# H = H * 255 / (2 * pi)
#
# I = np.asarray(I, dtype=np.uint8)
# S = np.asarray(S, dtype=np.uint8)
# H = np.asarray(H, dtype=np.uint8)
#
# cv2.imshow('src', src)
# cv2.imshow('H', H)
# cv2.imshow('S', S)
# cv2.imshow('I', I)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


extract_skin_color("skin_color_example\\black_male_1.jpg")
extract_skin_color("skin_color_example\\black_male_2.jpg")
extract_skin_color("skin_color_example\\black_male_3.jpg")
extract_skin_color("skin_color_example\\black_male_4.jpg")
extract_skin_color("skin_color_example\\black_male_5.jpg")
extract_skin_color("skin_color_example\\white_male_1.jpg")
extract_skin_color("skin_color_example\\white_male_2.jpg")
extract_skin_color("skin_color_example\\white_male_3.jpg")
extract_skin_color("skin_color_example\\white_male_4.jpg")
extract_skin_color("skin_color_example\\white_male_5.png")
extract_skin_color("skin_color_example\\yellow_male_1.jpg")
# extract_skin_color("skin_color_example\\yellow_male_2.jpg")   //no image
extract_skin_color("skin_color_example\\yellow_male_3.jpg")
extract_skin_color("skin_color_example\\face_paint_1.jpg")


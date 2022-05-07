import copy
import cv2
import numpy as np
import sys
import util, filtering
import rgb_hsi_conversion
from math import acos, cos, pi, sqrt, radians, degrees, exp

file_name = ["1.jpg", "2.jpg", "3.jpg", "4.jpg","aerial.jpg", "bridge.bmp", "cameraman.bmp", "clown.bmp",
    "crowd.bmp","fig_a_gaussian.jpg","fig_a_org.jpg","fig_a_salt_n_pepper.jpg","fig_b_gaussian.jpg","fig_b_org.jpg",
    "fig_b_salt_n_pepper.jpg","tank.bmp", "tungsten.jpg"]

file_path = "..\\edge_detection\\images\\"
result_path_prewitt_33 = "..\\edge_detection\\images\\prewitt_33\\"
result_path_prewitt_55 = "..\\edge_detection\\images\\prewitt_55\\"
result_path_sobel_33 = "..\\edge_detection\\images\\sobel_33\\"
result_path_sobel_55 = "..\\edge_detection\\images\\sobel_55\\"
result_path_log= "..\\edge_detection\\images\\log\\"

threshold_level = 100

edge_filter_dict = {'prewitt': 1, 'sobel': 2, 'log': 3, "prewitt_threshold": 4, "sobel_threshold": 5}

prewitt_33_x = \
    [ [-1, 0, 1], \
      [-1, 0, 1], \
      [-1, 0, 1] ]

prewitt_33_y = \
    [ [-1, -1, -1], \
      [0, 0, 0], \
      [1, 1, 1] ]

prewitt_55_x = \
    [ [-2, -1, 0, 1, 2], \
      [-2, -1, 0, 1, 2],
      [-2, -1, 0, 1, 2],
      [-2, -1, 0, 1, 2],
      [-2, -1, 0, 1, 2] ]

prewitt_55_y = \
    [ [2, 2, 2, 2, 2], \
      [1, 1, 1, 1, 1],
      [0, 0, 0, 0, 0],
      [-1, -1, -1, -1, -1],
      [-2, -2, -2, -2, -2] ]

sobel_33_x= \
    [ [-1, 0, 1], \
      [-2, 0, 2], \
      [-1, 0, 1] ]

sobel_33_y = \
    [ [-1, -2, -1], \
      [0, 0, 0], \
      [1, 2, 1] ]

sobel_55_x = \
    [ [-5, -4, 0, 4, 5], \
      [-8, -10, 0, 10, 8],
      [-10, -20, 0, 20, 10],
      [-8, -10, 0, 10, 8],
      [-5, -4, 0, 4, 5] ]


sobel_55_y = \
    [ [5, 8, 10, 8, 5], \
      [4, 10, 20, 10, 4],
      [0, 0, 0, 0, 0],
      [-4, -10, -20, -10, -4],
      [-5, -8, -10, -8, -5] ]

log_33 = \
    [[0, 1, 0], \
     [1, -4, 1], \
     [0, 1, 0]]

log_55 = \
    [[0, 0, 1, 0, 0], \
     [0, 1, 2, 1, 0],
     [1, 2, -16, 2, 1],
     [0, 1, 2, 1, 0],
     [0, 0, 1, 0, 0]]

edge_kernel = [[prewitt_33_x, prewitt_33_y], [prewitt_55_x, prewitt_55_y], \
    [sobel_33_x, sobel_33_y], [sobel_55_x, sobel_55_y], \
    log_33, log_55]

def threshold_result(dst, src1, src2, height, width) :
    for i in range(height) :
        for j in range(width) :
            if dst[i][j] < threshold_level :
                dst[i][j] = 0
            else :
                dst[i][j] = 255
            if src1[i][j] < threshold_level :
                src1[i][j] = 0
            else :
                src1[i][j] = 255
            if src2[i][j] < threshold_level :
                src2[i][j] = 0
            else :
                src2[i][j] = 255

def calculate_gradient(dst, src1, src2, height, width) :
    for i in range(height) :
        for j in range(width) :
            dst[i][j] = sqrt(int(src1[i][j]**2) + int(src2[i][j]**2))
            if dst[i][j] >= 255:
                dst[i][j] = 255
            if dst[i][j] <= 0 :
                dst[i][j] = 0

def edge_filter(file, kernel_size, filter_name):

    src_x = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    src_y = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if (edge_filter_dict[filter_name] == 1 and kernel_size == 3) or edge_filter_dict[filter_name] == 4:
        kernel_x, kernel_y = prewitt_33_x, prewitt_33_y
    if edge_filter_dict[filter_name] == 1 and kernel_size == 5:
        kernel_x, kernel_y = prewitt_55_x, prewitt_55_y
    elif edge_filter_dict[filter_name] == 2 and kernel_size == 3 or edge_filter_dict[filter_name] == 5:
        kernel_x, kernel_y = sobel_33_x, sobel_33_y
    elif edge_filter_dict[filter_name] == 2 and kernel_size == 5:
        kernel_x, kernel_y = sobel_55_x, sobel_55_y
    elif edge_filter_dict[filter_name] == 3:
        kernel_x, kernel_y = log_33, log_55

    height, width = src_x.shape[0], src_y.shape[1]
    new_image_x = np.zeros((height, width), dtype=np.uint8)
    new_image_y = np.zeros((height, width), dtype=np.uint8)
    new_image_xy = np.zeros((height, width), dtype=np.uint8)

    filtering.filtering(height, width, kernel_x, src_x, new_image_x)
    filtering.filtering(height, width, kernel_y ,src_y, new_image_y)
    if edge_filter_dict[filter_name] != 3:
        calculate_gradient(new_image_xy, new_image_x, new_image_y, height, width)
    if edge_filter_dict[filter_name] == 4 or edge_filter_dict[filter_name] == 5:
        threshold_result(new_image_xy, new_image_x, new_image_y, height, width)

    np.set_printoptions(threshold=sys.maxsize)

    merged_image = util.concat_four_images(src_x, new_image_x, new_image_y, new_image_xy)

    cv2.imwrite(file_path + util.get_filename(file) + filter_name + str(kernel_size) + ".jpg", merged_image)


def show_edge_filter(file_path, kernel_size, filter_name) :
    for name in file_name :
        file = file_path + name
        edge_filter(file, kernel_size, filter_name)


def edge_filter_canny(file, threshold):
    src = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]
    new_image_1 = np.zeros((height, width), dtype=np.uint8)
    new_image_2 = np.zeros((height, width), dtype=np.uint8)
    new_image_3 = np.zeros((height, width), dtype=np.uint8)

    if 170 < threshold:
        new_image_1 = cv2.Canny(src, 170, threshold)
    if 170 < threshold:
        new_image_1 = cv2.Canny(src, 140, threshold)
    if 170 < threshold:
        new_image_1 = cv2.Canny(src, 110, threshold)
    merged_image = util.concat_four_images(src, new_image_1, new_image_2, new_image_3)
    cv2.imwrite(file_path + util.get_filename(file) + str(threshold) + '_' + str(170) + ".jpg", merged_image)

    new_image_1 = cv2.Canny(src, 80, threshold)
    new_image_2 = cv2.Canny(src, 50, threshold)
    new_image_3 = cv2.Canny(src, 30, threshold)
    merged_image = util.concat_four_images(src, new_image_1, new_image_2, new_image_3)
    cv2.imwrite(file_path + util.get_filename(file) + str(threshold) + '_' + str(80) + ".jpg", merged_image)


def show_edge_filter_canny(file_path, threshold):
    for name in file_name :
        file = file_path + name
        edge_filter_canny(file, threshold)


# show_edge_filter(file_path, 3, "prewitt")
# show_edge_filter(file_path, 5, "prewitt")
# show_edge_filter(file_path, 3, "sobel")
# show_edge_filter(file_path, 5, "sobel")
# show_edge_filter(file_path, 3, "log")

# show_edge_filter(file_path, 3, "prewitt_threshold")

show_edge_filter_canny(file_path, 200)
show_edge_filter_canny(file_path, 150)
show_edge_filter_canny(file_path, 100)

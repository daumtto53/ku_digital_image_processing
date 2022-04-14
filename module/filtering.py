import cv2
import numpy as np
import util
import rgb_hsi_conversion
from math import acos, cos, pi, sqrt, radians, degrees, exp


def apply_kernel(u, v, kernel, kernel_height, kernel_width, old_image, new_image):
    offset_h, offset_w = int(kernel_height / 2), int(kernel_width / 2)
    new_value = 0
    for i in range(-1 * offset_h, offset_h + 1) :
        for j in range(-1 * offset_w, offset_w + 1) :
            new_value = new_value + (old_image[u + i][v + j] * kernel[offset_h + i][offset_w + j])
    new_image[u][v] = int(new_value) # float to int


def find_possible_border(height, width, kernel_height, kernel_width):
    border_h, border_w = height - 1 - kernel_height, width - 1 - kernel_width

    return border_h, border_w


def find_kernel_size(kernel):
    return len(kernel), len(kernel[0])


def define_mean_kernel(size):
    mean_value = 1 / (size * size)
    mean_kernel = np.full((size, size), mean_value, dtype=float)
    return mean_kernel


def gaussian_calculator(u, v, sigma):
    denominator = 2 * pi * (sigma * sigma)
    numerator = np.exp(-1 * (u*u + v*v) / (sigma*sigma))
    return numerator / denominator


def define_gaussian_kernel(size, sigma):
    gaussian_kernel = np.zeros((size,size), dtype=float)
    offset = int(size / 2)

    for i in range(size):
        for j in range(size):
            gaussian_kernel[i][j] = gaussian_calculator(i - offset, j - offset, sigma)
            print(i - offset, j- offset)
    return gaussian_kernel


def filtering(height, width, kernel, old_image, new_image) :
    kernel_height, kernel_width = find_kernel_size(kernel)
    offset_h = int(kernel_height / 2)
    offset_w = int(kernel_width / 2)
    for i in range(offset_h, height - offset_h):
        for j in range(offset_w, width - offset_w):
            apply_kernel(i, j, kernel, kernel_height, kernel_width, old_image, new_image)

def mean_filtering(file_path, kernel_size):
    src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    mean_kernel = define_mean_kernel(kernel_size)
    height, width = src.shape[0],src.shape[1]
    new_image = np.zeros((height,width), dtype=np.uint8)
    filtering(height, width, mean_kernel, src, new_image)
    util.compare_image("mean", new_image, src)

def gaussian_fltering(file_path, kernel_size, sigma):
    src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    gaussian_kernel = define_gaussian_kernel(kernel_size, sigma)
    print(gaussian_kernel)
    height, width = src.shape[0],src.shape[1]
    new_image = np.zeros((height,width), dtype=np.uint8)
    filtering(height, width, gaussian_kernel, src, new_image)
    util.compare_image("gaussian", new_image, src)


def show_mean_filtering_result():
    file_path = "..\\highboost_filtering\\images\\grayscale_noisy\\fig_a_salt_n_pepper.jpg"
    mean_filtering(file_path, 3)
    file_path = "..\\highboost_filtering\\images\\grayscale_noisy\\fig_a_salt_n_pepper.jpg"
    gaussian_fltering(file_path, 1)


def show_gaussian_filtering_result():
    file_path = "..\\highboost_filtering\\images\\grayscale_noisy\\fig_a_gaussian.jpg"
    mean_filtering(file_path, 3)
    file_path = "..\\highboost_filtering\\images\\grayscale_noisy\\fig_a_gaussian.jpg"
    gaussian_fltering(file_path, 3, 1)


show_gaussian_filtering_result()

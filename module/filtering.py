import cv2
import numpy as np
import sys
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
    numerator = np.exp(-1 * (u*u + v*v) / (2 * sigma*sigma))
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


def gaussian_filtering(file_path, kernel_size, sigma):
    src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    gaussian_kernel = define_gaussian_kernel(kernel_size, sigma)
    height, width = src.shape[0],src.shape[1]
    new_image = np.zeros((height,width), dtype=np.uint8)
    filtering(height, width, gaussian_kernel, src, new_image)
    util.compare_image("gaussian", new_image, src)


def apply_median_kernel(u, v, kernel_height, kernel_width, old_image, new_image):
        offset_h, offset_w = int(kernel_height / 2), int(kernel_width / 2)
        to_sort = []
        for i in range(-1 * offset_h, offset_h + 1):
            for j in range(-1 * offset_w, offset_w + 1):
                to_sort.append(old_image[u + i][v + j])
        to_sort.sort()
        median_index = int(kernel_width * kernel_height / 2)
        new_image[u][v] = to_sort[median_index]


def median_filtering(file_path, kernel_height, kernel_width):
    src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    height, width = src.shape[0], src.shape[1]
    new_image = np.zeros((height, width), dtype=np.uint8)
    offset_h = int(kernel_height / 2)
    offset_w = int(kernel_width / 2)
    for i in range(offset_h, height - offset_h):
        for j in range(offset_w, width - offset_w):
            apply_median_kernel(i, j, kernel_height, kernel_width, src, new_image)
    util.compare_image("median", new_image, src)


def colorscale_mean_filtering_RGB(file_path, kernel_size):
    src = cv2.imread(file_path, cv2.IMREAD_COLOR)

    mean_kernel = define_mean_kernel(kernel_size)
    height, width = src.shape[0], src.shape[1]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)
    new_b = np.zeros((height, width), dtype=np.uint8)
    new_g = np.zeros((height, width), dtype=np.uint8)
    new_r = np.zeros((height, width), dtype=np.uint8)

    b_channel = np.zeros((height, width), dtype=np.uint8)
    g_channel = np.zeros((height, width), dtype=np.uint8)
    r_channel = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            b_channel[i][j] = src[i][j][0]
            g_channel[i][j] = src[i][j][1]
            r_channel[i][j] = src[i][j][2]

    filtering(height, width, mean_kernel, b_channel, new_b)
    filtering(height, width, mean_kernel, g_channel, new_g)
    filtering(height, width, mean_kernel, r_channel, new_r)

    for i in range(height):
        for j in range(width):
            new_image[i][j][0] = new_b[i][j]
            new_image[i][j][1] = new_g[i][j]
            new_image[i][j][2] = new_r[i][j]

    np.set_printoptions(threshold=sys.maxsize)
    print(b_channel[1])
    print()
    print(new_b[1])
    util.compare_image("color_mean", new_image, src)


def colorscale_gaussian_filtering_RGB(file_path, kernel_size, sigma):
    src = cv2.imread(file_path, cv2.IMREAD_COLOR)

    gaussian_kernel = define_gaussian_kernel(kernel_size, sigma)
    height, width = src.shape[0], src.shape[1]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)
    new_b = np.zeros((height, width), dtype=np.uint8)
    new_g = np.zeros((height, width), dtype=np.uint8)
    new_r = np.zeros((height, width), dtype=np.uint8)

    b_channel = np.zeros((height, width), dtype=np.uint8)
    g_channel = np.zeros((height, width), dtype=np.uint8)
    r_channel = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            b_channel[i][j] = src[i][j][0]
            g_channel[i][j] = src[i][j][1]
            r_channel[i][j] = src[i][j][2]

    filtering(height, width, gaussian_kernel, b_channel, new_b)
    filtering(height, width, gaussian_kernel, g_channel, new_g)
    filtering(height, width, gaussian_kernel, r_channel, new_r)

    for i in range(height):
        for j in range(width):
            new_image[i][j][0] = new_b[i][j]
            new_image[i][j][1] = new_g[i][j]
            new_image[i][j][2] = new_r[i][j]

    util.compare_image("color_gaussian", new_image, src)


def colorscale_mean_filtering_HSI(file_path, kernel_size) :
    src = cv2.imread(file_path, cv2.IMREAD_COLOR)

    mean_kernel = define_mean_kernel(kernel_size)
    height, width = src.shape[0], src.shape[1]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)
    new_hue = np.zeros((height, width), dtype=float)
    I = np.zeros((height, width))
    S = np.zeros((height, width))
    H = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            b = src[i][j][0] / 255.
            g = src[i][j][1] / 255.
            r = src[i][j][2] / 255.
            H[i][j] = rgb_hsi_conversion.rgb_to_hue(b, g, r)
            S[i][j] = rgb_hsi_conversion.rgb_to_saturity(b, g, r)
            I[i][j] = rgb_hsi_conversion.rgb_to_intensity(b, g, r)

    filtering(height, width, mean_kernel, H, new_hue)

    for i in range(height) :
        for j in range(width) :
            bgr_tuple = rgb_hsi_conversion.HSI_to_bgr(new_hue[i][j], S[i][j], I[i][j])

            new_image[i][j][0] = np.clip(round(bgr_tuple[0] * 255.), 0, 255)
            new_image[i][j][1] = np.clip(round(bgr_tuple[1] * 255.), 0, 255)
            new_image[i][j][2] = np.clip(round(bgr_tuple[2] * 255.), 0, 255)

    util.compare_image("color_mean", new_image, src)



def colorscale_gaussian_filtering_HSI(file_path, kernel_size, sigma):
    src = cv2.imread(file_path, cv2.IMREAD_COLOR)

    gaussian_kernel = define_gaussian_kernel(kernel_size, sigma)
    height, width = src.shape[0], src.shape[1]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)
    new_hue = np.zeros((height, width), dtype=float)
    I = np.zeros((height, width))
    S = np.zeros((height, width))
    H = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            b = src[i][j][0] / 255.
            g = src[i][j][1] / 255.
            r = src[i][j][2] / 255.
            H[i][j] = rgb_hsi_conversion.rgb_to_hue(b, g, r)
            S[i][j] = rgb_hsi_conversion.rgb_to_saturity(b, g, r)
            I[i][j] = rgb_hsi_conversion.rgb_to_intensity(b, g, r)

    filtering(height, width, gaussian_kernel, H, new_hue)

    for i in range(height):
        for j in range(width):
            bgr_tuple = rgb_hsi_conversion.HSI_to_bgr(new_hue[i][j], S[i][j], I[i][j])

            new_image[i][j][0] = np.clip(round(bgr_tuple[0] * 255.), 0, 255)
            new_image[i][j][1] = np.clip(round(bgr_tuple[1] * 255.), 0, 255)
            new_image[i][j][2] = np.clip(round(bgr_tuple[2] * 255.), 0, 255)

    util.compare_image("color_gaussian", new_image, src)


def show_mean_filtering_result():
    file_path = "..\\highboost_filtering\\images\\grayscale_noisy\\fig_a_salt_n_pepper.jpg"
    mean_filtering(file_path, 3)
    file_path = "..\\highboost_filtering\\images\\grayscale_noisy\\fig_a_salt_n_pepper.jpg"
    gaussian_filtering(file_path, 3, 1)


def show_gaussian_filtering_result():
    file_path = "..\\highboost_filtering\\images\\grayscale_noisy\\fig_a_gaussian.jpg"
    mean_filtering(file_path, 3)
    file_path = "..\\highboost_filtering\\images\\grayscale_noisy\\fig_a_gaussian.jpg"
    gaussian_filtering(file_path, 3, 1)

def show_median_filtering_result():
    file_path = "..\\highboost_filtering\\images\\grayscale_noisy\\fig_a_salt_n_pepper.jpg"
    median_filtering(file_path, 3, 3)
    file_path = "..\\highboost_filtering\\images\\grayscale_noisy\\fig_a_gaussian.jpg"
    median_filtering(file_path, 3, 3)


def show_colorscale_mean_filtering_HSI_result(size, sigma):
    file_path = "..\\highboost_filtering\\images\\color_noisy\\Salt&pepper noise.png"
    colorscale_mean_filtering_HSI(file_path, size)
    file_path = "..\\highboost_filtering\\images\\color_noisy\\Salt&pepper noise.png"
    colorscale_gaussian_filtering_HSI(file_path, size, sigma)


def show_colorscale_gaussian_filtering_HSI_result(size, sigma):
    file_path = "..\\highboost_filtering\\images\\color_noisy\\Lena_noise.png"
    colorscale_mean_filtering_HSI(file_path, size)
    file_path = "..\\highboost_filtering\\images\\color_noisy\\Lena_noise.png"
    colorscale_gaussian_filtering_HSI(file_path, size, sigma)


def show_colorscale_mean_filtering_RGB_result(size, sigma):
    file_path = "..\\highboost_filtering\\images\\color_noisy\\Salt&pepper noise.png"
    colorscale_mean_filtering_RGB(file_path, size)
    file_path = "..\\highboost_filtering\\images\\color_noisy\\Salt&pepper noise.png"
    colorscale_gaussian_filtering_RGB(file_path, size, sigma)



def show_colorscale_gaussian_filtering_RGB_result(size, sigma):
    file_path = "..\\highboost_filtering\\images\\grayscale_noisy\\fig_a_gaussian.jpg"
    colorscale_mean_filtering_RGB(file_path, size)
    file_path = "..\\highboost_filtering\\images\\grayscale_noisy\\fig_a_gaussian.jpg"
    colorscale_gaussian_filtering_RGB(file_path, size, sigma)





# show_mean_filtering_result()
# show_gaussian_filtering_result()
# show_median_filtering_result()
#
# size=3
# sigma=1
# show_colorscale_mean_filtering_RGB_result(size, sigma)

# size=5
# sigma=1
# show_colorscale_gaussian_filtering_RGB_result(size, sigma)




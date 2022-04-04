import cv2
import numpy as np
import util
import rgb_hsi_conversion
import matplotlib.pyplot as plt
from math import pi

PATH = "..\\image_enhancing\\test_images\\test_images\\"

bmp = ".bmp"
color_image_tuple = ("airplane", "baboon", "barbara", "BoatsColor", "boy", \
                     "goldhill", "lenna_color", "pepper", "sails")

grayscale_image_tuple = ("boats", "bridge", "cameraman", "clown", "crowd", \
                         "man", "tank", "truck", "zelda")

power_law_value = ((0.5, 1.2), (0.44, 2))
POWER_LAW_ENHANCE_BLACK_1 = 0.5
POWER_LAW_ENHANCE_WHITE_1 = 1.2


def point_process_colorscale_negative_rgb(file_path):
    src = cv2.imread(file_path, cv2.IMREAD_COLOR)

    height, width = src.shape[0], src.shape[1]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            new_image[i][j][0] = 255 - src[i][j][0]
            new_image[i][j][1] = 255 - src[i][j][1]
            new_image[i][j][2] = 255 - src[i][j][2]
    return new_image, src


def show_colorscale_negative_rgb(window_name, file_path):
    res = point_process_colorscale_negative_rgb(file_path)
    util.compare_image(window_name, res[0], res[1])
    # cv2.imwrite('result_image\\'+ 'colorscale_negative_rgb_' + window_name + bmp, res[0])


def show_colorscale_negative_rgb_result():
    for i in range(len(color_image_tuple)):
        show_colorscale_negative_rgb(color_image_tuple[i], PATH + color_image_tuple[i] + bmp)


def point_process_colorscale_negative_intensity(file_path):
    src = cv2.imread(file_path, cv2.IMREAD_COLOR)

    height, width = src.shape[0], src.shape[1]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)
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
            I[i][j] = 1. - I[i][j]

            bgr_tuple = rgb_hsi_conversion.HSI_to_bgr(H[i][j], S[i][j], I[i][j])

            new_image[i][j][0] = np.clip(round(bgr_tuple[0] * 255.), 0, 255)
            new_image[i][j][1] = np.clip(round(bgr_tuple[1] * 255.), 0, 255)
            new_image[i][j][2] = np.clip(round(bgr_tuple[2] * 255.), 0, 255)

    # cv2.imshow('abs diff*50', np.minimum(cv2.absdiff(src, new_image), 5) * 50)
    # cv2.imshow('new_image', new_image)
    return new_image, src


def show_colorscale_negative_intensity(window_name, file_path):
    res = point_process_colorscale_negative_intensity(file_path)
    util.compare_image(window_name, res[0], res[1])
    cv2.imwrite('result_image\\' + 'colorscale_negative_intensity_' + window_name + bmp, res[0])


def show_colorscale_negative_intensity_result():
    for i in range(len(color_image_tuple)):
        show_colorscale_negative_intensity(color_image_tuple[i], PATH + color_image_tuple[i] + bmp)


def point_process_grayscale_negative(file_path):
    src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    height, width = src.shape[0], src.shape[1]
    new_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            new_image[i][j] = 255 - src[i][j]
    return new_image, src


def show_grayscale_negative(window_name, file_path):
    res = point_process_grayscale_negative(file_path)
    util.compare_image(window_name, res[0], res[1])
    # cv2.imwrite('result_image\\' + 'grayscale_negative_' + window_name + bmp, res[0])


def show_grayscale_negative_result():
    show_grayscale_negative("boats", "..\\image_enhancing\\test_images\\test_images\\boats.bmp")
    show_grayscale_negative("bridge", "..\\image_enhancing\\test_images\\test_images\\bridge.bmp")
    show_grayscale_negative("cameraman", "..\\image_enhancing\\test_images\\test_images\\cameraman.bmp")
    show_grayscale_negative("clown", "..\\image_enhancing\\test_images\\test_images\\clown.bmp")
    show_grayscale_negative("crowd", "..\\image_enhancing\\test_images\\test_images\\crowd.bmp")
    show_grayscale_negative("man", "..\\image_enhancing\\test_images\\test_images\\man.bmp")
    show_grayscale_negative("tank", "..\\image_enhancing\\test_images\\test_images\\tank.bmp")
    show_grayscale_negative("truck", "..\\image_enhancing\\test_images\\test_images\\truck.bmp")
    show_grayscale_negative("zelda", "..\\image_enhancing\\test_images\\test_images\\zelda.bmp")


def calculate_grayscale_power_law(coefficient, input_px, gamma_value):
    normalize_value = 255.0
    s = coefficient * (input_px / normalize_value) ** float(gamma_value)
    return s


# need revision.
def point_process_colorscale_power_law_intensity(file_path, gamma_value):
    src = cv2.imread(file_path, cv2.IMREAD_COLOR)

    height, width = src.shape[0], src.shape[1]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)
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

            I[i][j] = calculate_grayscale_power_law(255, I[i][j], gamma_value)

            bgr_tuple = rgb_hsi_conversion.HSI_to_bgr(H[i][j], S[i][j], I[i][j])

            new_image[i][j][0] = np.clip(round(bgr_tuple[0] * 255.), 0, 255)
            new_image[i][j][1] = np.clip(round(bgr_tuple[1] * 255.), 0, 255)
            new_image[i][j][2] = np.clip(round(bgr_tuple[2] * 255.), 0, 255)

    return new_image, src


def show_colorscale_power_law_intensity(window_name, file_path, gamma_value):
    res = point_process_colorscale_power_law_intensity(file_path, gamma_value)
    util.compare_image(window_name, res[0], res[1])


def show_colorscale_power_law_intensity_result():
    show_colorscale_power_law_intensity("BoatsColor.bmp", "..\\image_enhancing\\test_images\\test_images\\BoatsColor.bmp", power_law_value[0][0])
    show_colorscale_power_law_intensity("airplane", "..\\image_enhancing\\test_images\\test_images\\airplane.bmp",
                             power_law_value[0][1])
    show_colorscale_power_law_intensity("baboon", "..\\image_enhancing\\test_images\\test_images\\baboon.bmp",
                             power_law_value[0][1])
    show_colorscale_power_law_intensity("barara", "..\\image_enhancing\\test_images\\test_images\\barbara.bmp", power_law_value[0][0])
    show_colorscale_power_law_intensity("boy", "..\\image_enhancing\\test_images\\test_images\\boy.bmp", power_law_value[0][0])
    show_colorscale_power_law_intensity("goldhill", "..\\image_enhancing\\test_images\\test_images\\goldhill.bmp", power_law_value[0][0])
    show_colorscale_power_law_intensity("lenna_color", "..\\image_enhancing\\test_images\\test_images\\lenna_color.bmp", power_law_value[0][1])
    show_colorscale_power_law_intensity("pepper", "..\\image_enhancing\\test_images\\test_images\\pepper.bmp", power_law_value[0][1])
    show_colorscale_power_law_intensity("sails", "..\\image_enhancing\\test_images\\test_images\\sails.bmp", power_law_value[0][0])

def point_process_colorscale_power_law_rgb(file_path, gamma_value):
    src = cv2.imread(file_path, cv2.IMREAD_COLOR)

    height, width = src.shape[0], src.shape[1]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            new_image[i][j][0] = calculate_grayscale_power_law(255, src[i][j][0], gamma_value)
            new_image[i][j][1] = calculate_grayscale_power_law(255, src[i][j][1], gamma_value)
            new_image[i][j][2] = calculate_grayscale_power_law(255, src[i][j][2], gamma_value)
    return new_image, src


def show_colorscale_power_law_rgb(window_name, file_path, gamma_value):
    res = point_process_colorscale_power_law_rgb(file_path, gamma_value)
    util.compare_image(window_name, res[0], res[1])


def show_colorscale_power_law_rgb_result():
    for i in range(len(color_image_tuple)):
        show_colorscale_power_law_rgb(color_image_tuple[i], PATH + color_image_tuple[i] + bmp, 0.4)


def point_process_grayscale_power_law(file_path, gamma_value):
    src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    height, weight = src.shape[0], src.shape[1]
    new_image = np.zeros((height, weight), dtype=np.uint8)

    for i in range(height):
        for j in range(weight):
            new_image[i][j] = calculate_grayscale_power_law(255, src[i][j], gamma_value)

    return new_image, src


def show_grayscale_power_law(window_name, file_path, gamma_value):
    res = point_process_grayscale_power_law(file_path, gamma_value)
    util.compare_image(window_name, res[0], res[1])
    cv2.imwrite('result_image\\' + str(gamma_value) + '_grayscale_power_law_' + window_name + bmp, res[0])


def show_grayscale_power_law_result():
    show_grayscale_power_law("boats", "..\\image_enhancing\\test_images\\test_images\\boats.bmp", power_law_value[1][0])
    show_grayscale_power_law("bridge", "..\\image_enhancing\\test_images\\test_images\\bridge.bmp",
                             power_law_value[1][1])
    show_grayscale_power_law("cameraman", "..\\image_enhancing\\test_images\\test_images\\cameraman.bmp",
                             power_law_value[1][1])
    show_grayscale_power_law("clown", "..\\image_enhancing\\test_images\\test_images\\clown.bmp", power_law_value[1][0])
    show_grayscale_power_law("crowd", "..\\image_enhancing\\test_images\\test_images\\crowd.bmp", power_law_value[1][0])
    show_grayscale_power_law("man", "..\\image_enhancing\\test_images\\test_images\\man.bmp", power_law_value[1][0])
    show_grayscale_power_law("tank", "..\\image_enhancing\\test_images\\test_images\\tank.bmp", power_law_value[1][1])
    show_grayscale_power_law("truck", "..\\image_enhancing\\test_images\\test_images\\truck.bmp", power_law_value[1][1])
    show_grayscale_power_law("zelda", "..\\image_enhancing\\test_images\\test_images\\zelda.bmp", power_law_value[1][0])




def point_process_grayscale_historgram_equalization(file_path):
    src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return histeq(src)


def show_grayscale_histogram_eqaulization(window_name, file_path):
    new_image, org_hist, new_hist, hist_func = point_process_grayscale_historgram_equalization(file_path)
    src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # show_histogram()
    # plt.plot(org_hist)
    # plt.show()
    # plt.plot(new_hist)
    # plt.show()

    # plot histograms and transfer function
    fig = plt.figure()
    fig.add_subplot(221)
    plt.plot(org_hist)
    plt.title('Original histogram')  # original histogram

    fig.add_subplot(222)
    plt.plot(new_hist)
    plt.title('New histogram')  # hist of eqlauized image

    plt.show()


    util.compare_image(window_name, new_image, src)

def show_grayscale_histogram_equalization_result():
    for i in range(len(grayscale_image_tuple)):
        show_grayscale_histogram_eqaulization(grayscale_image_tuple[i], PATH + grayscale_image_tuple[i] + bmp)






def show_colorscale_histogram_equalization_rgb_result():
    for i in range(len(color_image_tuple)):
        show_colorscale_histogram_equalization_rgb(color_image_tuple[i], PATH + color_image_tuple[i] + bmp)


def show_colorscale_histogram_equalization_rgb(window_name, file_path):
    return;



def point_process_colorscale_histogram_equalization_rgb(file_path):

    src = cv2.imread(file_path, cv2.IMREAD_COLOR)

    height, width = src.shape[0], src.shape[1]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)

    # rgb 각각의 영역에 대해 Histogram Eqaulization 필요.

    histeq_result_bgr = []
    histeq_result_bgr.append(histeq())
    return new_image, src




def imhist(im):
    # calculates normalized histogram of an image
    m, n = im.shape
    h = [0.0] * 256
    for i in range(m):
        for j in range(n):
            h[im[i, j]] += 1
    return np.array(h) / (m * n)


def cumsum(h):
    # finds cumulative sum of a numpy array, list
    return [sum(h[:i + 1]) for i in range(len(h))]


def histeq(im):
    # calculate Histogram
    h = imhist(im)
    cdf = np.array(cumsum(h))  # cumulative distribution function
    sk = np.uint8(255 * cdf)  # finding transfer function values
    s1, s2 = im.shape
    Y = np.zeros_like(im)
    for i in range(0, s1):
        for j in range(0, s2):
            Y[i, j] = sk[im[i, j]]
    H = imhist(Y)
    # return val : Y;new_image, h;org_hist, H:new_hist, hist_func
    return Y, h, H, sk

# def point_process_histogram_

# show_grayscale_negative_result()
# show_colorscale_negative_rgb_result()
# show_colorscale_negative_intensity_result()

# show_grayscale_power_law_result()
# show_colorscale_power_law_rgb_result()
# show_colorscale_power_law_intensity_result()

# show_grayscale_histogram_equalization_result()
import cv2
import numpy as np
import util

PATH = "..\\image_enhancing\\test_images\\test_images\\"

POWER_LAW_ENHANCE_BLACK = 0.5
POWER_LAW_ENHANCE_WHITE = 1.2


def point_proces_colorscale_negative(file_path):

def point_process_grayscale_negative(file_path):

    src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    height, width = src.shape[0], src.shape[1]
    new_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            new_image[i][j] = 255 - src[i][j]
    return src, new_image


def show_grayscale_negative(window_name, file_path):
    res = point_process_grayscale_negative(file_path)
    util.compare_image(window_name, res[0], res[1])


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


def point_process_grayscale_power_law(file_path, gamma_value):

    src = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    height, weight = src.shape[0], src.shape[1]
    new_image = np.zeros((height, weight), dtype=np.uint8)

    for i in range(height):
        for j in range(weight):
            new_image[i][j] = calculate_grayscale_power_law(255, src[i][j], gamma_value)

    return src, new_image


def show_grayscale_power_law(window_name, file_path, gamma_value):
    res = point_process_grayscale_power_law(file_path, gamma_value)
    util.compare_image(window_name, res[0], res[1])


def show_grayscale_power_law_result():
    show_grayscale_power_law("boats", "..\\image_enhancing\\test_images\\test_images\\boats.bmp", POWER_LAW_ENHANCE_BLACK)
    show_grayscale_power_law("bridge", "..\\image_enhancing\\test_images\\test_images\\bridge.bmp", POWER_LAW_ENHANCE_WHITE)
    show_grayscale_power_law("cameraman", "..\\image_enhancing\\test_images\\test_images\\cameraman.bmp", POWER_LAW_ENHANCE_WHITE)
    show_grayscale_power_law("clown", "..\\image_enhancing\\test_images\\test_images\\clown.bmp", POWER_LAW_ENHANCE_BLACK)
    show_grayscale_power_law("crowd", "..\\image_enhancing\\test_images\\test_images\\crowd.bmp", POWER_LAW_ENHANCE_BLACK)
    show_grayscale_power_law("man", "..\\image_enhancing\\test_images\\test_images\\man.bmp", POWER_LAW_ENHANCE_BLACK)
    show_grayscale_power_law("tank", "..\\image_enhancing\\test_images\\test_images\\tank.bmp", POWER_LAW_ENHANCE_WHITE)
    show_grayscale_power_law("truck", "..\\image_enhancing\\test_images\\test_images\\truck.bmp", POWER_LAW_ENHANCE_WHITE)
    show_grayscale_power_law("zelda", "..\\image_enhancing\\test_images\\test_images\\zelda.bmp", POWER_LAW_ENHANCE_BLACK)


def point_process_histogram_


#show_grayscale_negative_result()
show_grayscale_power_law_result()
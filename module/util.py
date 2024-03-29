import cv2
import matplotlib.pyplot as plt
import numpy as np

from os.path import exists

def check_file_exists(file_path):
    if exists(file_path):
        print("file exists")
    else:
        print("file doesn't exists")

def get_filename(file_path):
    src = file_path.split('\\')
    print(src[-1])
    return src[-1].split('.')[0] + '_'

def show_image(window_name, array_of_image):
    return
    # cv2.imshow(window_name, array_of_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def compare_image(window_name, src, dst):
    cv2.imshow(window_name, dst)
    cv2.imshow(window_name+" after", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def concat_four_images(top_left, top_right, bottom_left, bottom_right):
    top_horizontal = np.concatenate((top_left, top_right), axis=1)
    bottom_horizontal = np.concatenate((bottom_left, bottom_right), axis=1)
    top_down = np.concatenate((top_horizontal, bottom_horizontal), axis=0)
    return top_down

def concat_two_images(top_left, top_right):
    top_horizontal = np.concatenate((top_left, top_right), axis=1)
    return top_horizontal

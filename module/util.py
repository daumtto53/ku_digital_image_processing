import cv2
import numpy as np

from os.path import exists

def check_file_exists(file_path):
    if exists(file_path):
        print("file exists")
    else:
        print("file doesn't exists")


def show_image(window_name, array_of_image):
    cv2.imshow(window_name, array_of_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compare_image(window_name, src, dst):
    cv2.imshow(window_name, dst)
    cv2.imshow(window_name+" after", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
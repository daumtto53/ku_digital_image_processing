import cv2
import numpy as np
from math import sqrt, cos, acos, degrees, radians, pi

def HSI_to_bgr(h, s, i):
    b = 0.
    g = 0.
    r = 0.
    h = degrees(h)
    if 0 <= h <= 120 :
        b = i * (1 - s)
        r = i * (1 + (s * cos(radians(h)) / cos(radians(60) - radians(h))))
        g = i * 3 - (r + b)
    elif 120 < h <= 240:
        h -= 120
        r = i * (1 - s)
        g = i * (1 + (s * cos(radians(h)) / cos(radians(60) - radians(h))))
        b = 3 * i - (r + g)
    elif 240 < h <= 360:
        h -= 240
        g = i * (1 - s)
        b = i * (1 + (s * cos(radians(h)) / cos(radians(60) - radians(h))))
        r = i * 3 - (g + b)
    return [b, g, r]


def rgb_to_hue(b, g, r):
    if (b == g == r):
        return 0

    angle = 0.5 * ((r - g) + (r - b)) / sqrt(((r - g) ** 2) + (r - b) * (g - b))
    if b <= g:
        return acos(angle)
    else:
        return 2 * pi - acos(angle)


def rgb_to_intensity(b, g, r):
    val = (b + g + r) / 3.
    if val == 0:
        return 0
    else:
        return val


def rgb_to_saturity(b, g, r):
    if r + g + b != 0:
        return 1. - 3. * np.min([r, g, b]) / (r + g + b)
    else:
        return 0




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
            H[i][j] = rgb_to_hue(b, g, r)
            S[i][j] = rgb_to_saturity(b, g, r)
            I[i][j] = rgb_to_intensity(b, g, r)

            bgr_tuple = HSI_to_bgr(H[i][j], S[i][j], I[i][j])

            new_image[i][j][0] = np.clip(round(bgr_tuple[0] * 255.), 0, 255)
            new_image[i][j][1] = np.clip(round(bgr_tuple[1] * 255.), 0, 255)
            new_image[i][j][2] = np.clip(round(bgr_tuple[2] * 255.), 0, 255)

    return new_image, src

PATH = "..\\image_enhancing\\test_images\\test_images\\"
new_image, src = point_process_colorscale_negative_intensity(PATH + 'baboon' + '.bmp')  # The mandrill image I used is from MATLAB.

cv2.imwrite('new_image.png', new_image)  # Save new_image for testing

cv2.imshow('new_image', new_image)  # Show new_image for testing
cv2.imshow('abs diff*50', np.minimum(cv2.absdiff(src, new_image), 5)*50)  # Show absolute difference of (src - new_image) multiply by 50 for showing small differences.
cv2.waitKey()
cv2.destroyAllWindows()
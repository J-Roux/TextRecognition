import glob

from cv2.cv2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2


def check_lower_border(img, lower_line, a4_y_size_mm, a4_y_size_px, border, threshold):
    if abs((a4_y_size_mm - float(border)) - lower_line[2]) > float(threshold):
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    cv2.line(img, (lower_line[0], int(lower_line[2] / a4_y_size_mm * a4_y_size_px)),
             (lower_line[1],
              int(lower_line[3] / a4_y_size_mm * a4_y_size_px)), color, 7)
    cv2.putText(img,
                str('%.2f' % (a4_y_size_mm - lower_line[2])),
                (int(300), int(lower_line[2] / a4_y_size_mm * a4_y_size_px) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)



def check_upper_border(img, upper_line, a4_y_size_mm, a4_y_size_px, border, threshold):
    if abs(float(border) - upper_line[2]) > float(threshold):
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    cv2.line(img, (upper_line[0], int(upper_line[2] / a4_y_size_mm * a4_y_size_px)),
             (upper_line[1],
              int(upper_line[3] / a4_y_size_mm * a4_y_size_px)), color, 7)
    cv2.putText(img,
                str('%.2f' % upper_line[2]),
                (int(300), int(upper_line[2] / a4_y_size_mm * a4_y_size_px) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)


def check_left_border(img, left_line, a4_x_size_mm, a4_x_size_px, border, threshold):
    if abs(float(border) - left_line[0]) > float(threshold):
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    cv2.line(img, (int(left_line[0] / a4_x_size_mm * a4_x_size_px), left_line[2]),
             (int(left_line[1] / a4_x_size_mm * a4_x_size_px),
              left_line[3]), color, 7)
    cv2.putText(img,
                str('%.2f' % left_line[0]),
                (int(100), 300 ),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

def check_right_border(img, right_line, a4_x_size_mm, a4_x_size_px, border, threshold):
    if abs((a4_x_size_mm - float(border)) - right_line[0]) > float(threshold):
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    cv2.line(img, (int(right_line[0] / a4_x_size_mm * a4_x_size_px), right_line[2]),
             (int(right_line[1] / a4_x_size_mm * a4_x_size_px),
              right_line[3]), color, 7)
    cv2.putText(img,
                str('%.2f' % (a4_x_size_mm - right_line[0])),
                (int(right_line[0] / a4_x_size_mm * a4_x_size_px) +20, 275 ),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)



a4_y_size_mm = 297
a4_x_size_mm = 210

plt.interactive(False)


def check_border(file_name):
    global a4_y_size_px
    img = imread(file_name)
    a4_y_size_px = img.shape[0]
    a4_x_size_px = img.shape[1]
    gray_image = cvtColor(img, COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    ret, gray_image = cv2.threshold(image_final, 180, 255,
                                    cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    gray_image = cv2.dilate(gray_image, kernel, iterations=15)  # dilate , more the iteration more the dilation
    gray_image = Canny(gray_image, 20, 150, apertureSize=7)
    lines = HoughLines(gray_image, 10, np.pi / 180, 54)
    horisontal_lines = []
    vertical_lines = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 3000 * (-b))
        y1 = int(y0 + 3000 * (a))
        y2 = int(y0 - 3000 * (a))
        x2 = int(x0 - 3000 * (-b))
        if abs(y1 - y2) < 2 or x2 == x1:
            if x2 != x1:
                horisontal_lines.append((x1, x2, y1, y2))
            else:
                vertical_lines.append((x1, x2, y1, y2))
            #cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    horisontal_lines.sort(key=lambda x: x[2])
    vertical_lines.sort(key=lambda x: x[0])
    horisontal_lines = map(lambda x: (x[0],
                                      x[1],
                                      float(x[2]) / a4_y_size_px * a4_y_size_mm,
                                      float(x[3]) / a4_y_size_px * a4_y_size_mm),
                           horisontal_lines)
    vertical_lines = map(lambda x: (float(x[0]) / a4_y_size_px * a4_y_size_mm,
                                    float(x[1]) / a4_y_size_px * a4_y_size_mm,
                                    x[2],
                                    x[3]),
                         vertical_lines)
    check_upper_border(img, horisontal_lines[0], a4_y_size_mm, a4_y_size_px, 20.0, 0.5)
    check_lower_border(img, horisontal_lines[-1], a4_y_size_mm, a4_y_size_px, 15.0, 0.5)
    check_left_border(img, vertical_lines[0], a4_x_size_mm, a4_x_size_px, 30, 0.5)
    check_right_border(img, vertical_lines[-1], a4_x_size_mm, a4_x_size_px, 15, 0.5)
    plt.imshow(img, cmap='gray')
    plt.show()


for i in glob.glob('*.png'):
    check_border(i)




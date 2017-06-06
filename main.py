from cv2.cv2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

a4_y_size_mm = 297
a4_x_size_mm = 210

plt.interactive(False)
img = imread('page-33.png')
a4_y_size_px = img.shape[0]
a4_x_size_px = img.shape[1]
print img.shape
gray_image = cvtColor(img, COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)
image_final = cv2.bitwise_and(gray_image, gray_image, mask=mask)
ret, gray_image = cv2.threshold(image_final, 170, 255, cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
gray_image = cv2.dilate(gray_image, kernel, iterations=9)  # dilate , more the iteration more the dilation



gray_image = Canny(gray_image,20,150,apertureSize = 7)



lines = HoughLines(gray_image,10,np.pi/180,54)

result = []
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    y2 = int(y0 - 1000*(a))
    x2 = int(x0 - 1000 * (-b))
    if abs(y1 - y2) < 2 or x2 == x1:
        if x2 != x1:
            result.append((x1, x2, y1, y2))
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

print map(lambda x: float(x[2]) / a4_y_size_px * a4_y_size_mm, result)

#plt.imshow(img, cmap='gray')
#plt.show()




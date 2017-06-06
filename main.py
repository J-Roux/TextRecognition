from cv2.cv2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

a4_y_size_mm = 297

plt.interactive(False)
img = imread('page-33.png')
a4_y_size_px = img.shape[0]
gray_image = cvtColor(img, COLOR_BGR2GRAY)

gray_image = ndimage.minimum_filter(gray_image, size=7)
gray_image = Canny(gray_image,50,150,apertureSize = 5)

#plt.imshow(gray_image, cmap='gray')
#plt.show()

lines = HoughLines(gray_image,1,np.pi/180,200)
for line in lines[:1]:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    y2 = int(y0 - 1000*(a))
    x2 = int(x0 - 1000 * (-b))
    print float(y1) / a4_y_size_px * a4_y_size_mm
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)




plt.imshow(img, cmap='gray')
plt.show()




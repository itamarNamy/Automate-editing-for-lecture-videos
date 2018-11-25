import cv2
import methods as mt
from matplotlib import pyplot as plt
import  numpy as np
import imutils


img = cv2.imread('pics/lecture_pic6.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
border = 5
out = mt.add_borders(img, border, border, border, border)
count = 0

while True:
    out, flag = mt.detect_board(out)
    mt.edge_detector(out)
    if flag and count == 1:
        plt.imshow(out), plt.show()
        break
    if count == 2:
        break
    plt.imshow(out), plt.show()
    out = mt.crop_borders(out, border+1)
    count += 1









import numpy as np
import matplotlib.pyplot as plt
import cv2
import config
from glob import glob
#out = "mask_boundary_thr.png"
images = glob("Images/image/*.png")
masks = glob("Masks/mask1/*.png")
image = config.IMAGE_DATASET
mask =config.MASK_DATASET
count = 0

for img,inp in zip(glob(image),glob(mask)):
    path=img.split("/")
    p=path[-1]
    count +=1
    im = cv2.imread(inp)
    image = cv2.imread(img)
    image = cv2.convertScaleAbs(image)
    im = cv2.convertScaleAbs(im)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #tmp = np.zeros_like(image)
    cv2.drawContours(igray, contours, -1, (255,255,255), 1)
    #cv2.imwrite(f"C:/Users/Deepa N C/PycharmProjects/CarotidArtery1/out/{p}.jpg",igray)
    cv2.imshow("out",igray)
    cv2.waitKey(0)
"""
out = "Images\\out.png"
count +=1
im = cv2.imread("Masks\\mask1\\14-08-29 001.jpg")
image = cv2.imread("Images\\image\\14-08-29 001.png")
image = cv2.convertScaleAbs(image)
im = cv2.convertScaleAbs(im)
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#tmp = np.zeros_like(image)
cv2.drawContours(igray, contours, -1, (255,255,255), 1)
cv2.imwrite('contours.png',igray)
"""
import config
import cv2
from imutils import paths
import os
import numpy as np

imagePaths = sorted(list(paths.list_images(config.IMAGE_BEFORE_PREPROCESS)))
for image in imagePaths:
    imagename=image.split(os.path.sep)[-1]
    imagename = imagename[:-4]
    image=cv2.imread(image)
    #image=cv2.normalize(image, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm = (np.maximum(image, 0) / image.max()) * 255
    #norm = (image - np.min(image)) / (np.max(image) - np.min(image))
    cv2.imwrite(f"Images/image2/{imagename}.png",norm)


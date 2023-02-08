# USAGE
# python predict.py
# import the necessary packages
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from glob import glob
from skimage import color
import pandas as pd

def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print(im_sum)

    return 2. * intersection.sum() / im_sum


def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = np.float64(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float64(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float64(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float64(np.sum((prediction == 0) & (groundtruth == 0)))

    return FP, FN, TP, TN

def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0

def prepare_plot(filename,origImage, out):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	#ax[1].imshow(origMask)
	ax[1].imshow(out)
	# set the titles of the subplots
	#ax[0].set_title("Image")
	ax[0].set_title("Original Mask")
	ax[1].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	#figure.show()
	filename=filename[:-4]
	figure.savefig(f"results-180/{filename}.png")

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (128, 128))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		#groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
		#	filename)
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		#gtMask = cv2.imread(groundTruthPath, 0)
		#gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
		#	config.INPUT_IMAGE_HEIGHT))

# make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)
		# prepare a plot for visualization

		prepare_plot(filename,orig,predMask)
		return orig,predMask,filename

# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
# iterate over the randomly selected test image paths
"""
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, path)
from glob import glob
for path in glob("C:/Users/Deepa N C/PycharmProjects/SegRefCarotidArtery/test/*.png"):
	make_predictions(unet,path)
"""
"""
#for training data
image = config.IMAGE_DATASET
mask =config.MASK_DATASET
SCORE=[]
count=0
diceavg=0

for image,mask in zip(glob(image),glob(mask)):
	# make predictions and visualize the results
	count+=1
	mask=cv2.imread(mask)
	image,predmask,filename=make_predictions(unet, image)
	mask=color.rgb2gray(mask)
	mask = cv2.resize(mask, (128, 128))
	predmask=cv2.resize(predmask,(128,128))

	dice_coefficient=dice_coeff(predmask,mask)
	acc=accuracy_score(predmask,mask)
	diceavg+=dice_coefficient

	SCORE.append([filename, dice_coefficient, acc])

	df = pd.DataFrame(SCORE, columns=["Image", "Dice Coefficient", "Accuracy"])
	df.to_csv("score/scoretrain.csv")
print("Train Dice Average = ",diceavg/count)

#for testing
SCORE=[]
count=0
diceavg=0
image="test/image/*.png"
mask="test/mask/*.jpg"
for image,mask in zip(glob(image),glob(mask)):
	# make predictions and visualize the results
	count+=1
	mask=cv2.imread(mask)
	image,predmask,filename=make_predictions(unet, image)
	mask=color.rgb2gray(mask)
	mask = cv2.resize(mask, (128, 128))
	predmask=cv2.resize(predmask,(128,128))

	dice_coefficient=dice_coeff(predmask,mask)
	acc=accuracy_score(predmask,mask)
	diceavg+=dice_coefficient

	SCORE.append([filename, dice_coefficient, acc])

	df = pd.DataFrame(SCORE, columns=["Image", "Dice Coefficient", "Accuracy"])
	df.to_csv("score/scoretest.csv")
print("Test Dice Average = ",diceavg/count)
"""
#for validation
SCORE=[]
count=0
diceavg=0
image="val/image/*.png"
mask="val/mask/*.jpg"
for image,mask in zip(glob(image),glob(mask)):
	# make predictions and visualize the results
	count+=1
	mask=cv2.imread(mask)
	image,predmask,filename=make_predictions(unet, image)
	mask=color.rgb2gray(mask)
	mask = cv2.resize(mask, (128, 128))
	predmask=cv2.resize(predmask,(128,128))

	dice_coefficient=dice_coeff(predmask,mask)
	acc=accuracy_score(predmask,mask)
	diceavg+=dice_coefficient

	SCORE.append([filename, dice_coefficient, acc])

	df = pd.DataFrame(SCORE, columns=["Image", "Dice Coefficient", "Accuracy"])
	df.to_csv("score/scoreval.csv")
print("Validation Dice Average = ",diceavg/count)



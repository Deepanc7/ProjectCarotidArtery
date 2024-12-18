import torch
import os
# base path of the dataset

IMAGE_BEFORE_PREPROCESS = "Images/image1"
IMAGE_DATASET_PATH = "Images/image2"
MASK_DATASET_PATH = "Masks/mask2"
IMAGE_DATASET = "Images/image2/*.png"
MASK_DATASET = "Masks/mask2/*.png"
# define the test split
TEST_SPLIT = 0.1764
VAL_SPLIT = 0.15
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 4
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 300
BATCH_SIZE = 32
# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
MASK_PATHS = os.path.sep.join([BASE_OUTPUT, "mask_paths.txt"])
VAL_IMAGES = os.path.sep.join([BASE_OUTPUT, "val_images.txt"])
VAL_MASKS = os.path.sep.join([BASE_OUTPUT, "val_masks.txt"])
TRAIN_IMAGES = os.path.sep.join([BASE_OUTPUT, "train_images.txt"])
TRAIN_MASKS = os.path.sep.join([BASE_OUTPUT, "train_masks.txt"])
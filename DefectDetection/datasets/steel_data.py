import numpy as np
import pandas as pd
import os

DATA_DIR = "/Users/rocioliu/Kaggle/severstal-steel-defect-detection"
TRAIN_CSV = "train.csv"
TRAIN_IMG_SUBDIR = "train_images"
TEST_IMG_SUBDIR = "test_images"

TRAIN_IMG_DIR = os.path.join(DATA_DIR, TRAIN_IMG_SUBDIR)
TEST_IMG_DIR = os.path.join(DATA_DIR, TEST_IMG_SUBDIR)


def load_csv(data_dir=DATA_DIR, train_csv=TRAIN_CSV):
    train_df = pd.read_csv(os.path.join(data_dir, train_csv))
    return train_df


# train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

train_df = load_csv(data_dir=DATA_DIR, train_csv=TRAIN_CSV)


def load_img(
        data_dir=DATA_DIR,
        train_csv=TRAIN_CSV,
        train_img_subdir=TRAIN_IMG_SUBDIR,
        test_img_subdir=TEST_IMG_SUBDIR):

    train_df = load_csv(data_dir, train_csv)
    train_img_dir = os.path.join(data_dir, train_img_subdir)
    test_img_dir = os.path.join(data_dir, test_img_subdir)

    train_img_names = sorted(
        [i for i in os.listdir(train_img_dir) if os.path.isfile(os.path.join(train_img_dir, i))])
    test_img_names = sorted(
        [i for i in os.listdir(test_img_dir) if os.path.isfile(os.path.join(test_img_dir, i))])

    defect_names = [x for x in train_img_names if x in train_df.ImageId.tolist()]
    non_defect_names = [x for x in train_img_names if x not in train_df.ImageId.tolist()]

    defect_img = [os.path.join(train_img_dir, fname) for fname in defect_names]
    non_defect_img = [os.path.join(train_img_dir, fname) for fname in non_defect_names]

    return defect_names, defect_img

# #train_images_dir = os.path.join(data_dir, train_img_subdir)
# #test_images_dir = os.path.join(data_dir, test_img_subdir)
#
# #train_images_names = sorted([i for i in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, i))])
# #test_images_names = sorted([i for i in os.listdir(test_images_dir) if os.path.isfile(os.path.join(test_images_dir, i))])
#
#
# # The names in train_images_names but not in train_df are the images that are not defect
# # Extract the non defect images
# defect_names = [x for x in train_images_names if x in train_df.ImageId.tolist()]
# non_defect_names = [x for x in train_images_names if x not in train_df.ImageId.tolist()]
#
# # Split the train images into defect and non-defect
# defect_images = [os.path.join(train_images_dir, fname) for fname in defect_names]
# non_defect_images = [os.path.join(train_images_dir, fname) for fname in non_defect_names]

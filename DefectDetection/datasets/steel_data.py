import numpy as np
import pandas as pd
import os

data_dir = "/Users/rocioliu/Kaggle/severstal-steel-defect-detection"
train_csv = "train.csv"
train_img_subdir = "train_images"
test_img_subdir = "test_images"


def load_csv(data_dir, train_csv):
    train_df = pd.read_csv(os.path.join(data_dir, train_csv))
    return train_df


# train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

train_df = load_csv(data_dir, train_csv)


def load_img(
        data_dir="/Users/rocioliu/Kaggle/severstal-steel-defect-detection",
        train_csv="train.csv",
        train_img_subdir="train_images",
        test_img_subdir="test_images", ):

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

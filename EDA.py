import os

from DefectDetection.datasets.steel_data import load_csv,load_img
from DefectDetection.utils import masks

from absl import flags

# FLAGS = flags.FLAGS
# flags.DEFINE_string("data_dir",
#                     "/Users/rocioliu/Kaggle/severstal-steel-defect-detection",
#                     "")

DATA_DIR = "/Users/rocioliu/Kaggle/severstal-steel-defect-detection"
TRAIN_CSV = "train.csv"
TRAIN_IMG_SUBDIR = "train_images"
TEST_IMG_SUBDIR = "test_images"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, TRAIN_IMG_SUBDIR)
TEST_IMG_DIR = os.path.join(DATA_DIR, TEST_IMG_SUBDIR)


#TODO
def _show_eda_text(train_df):
    print("The number of training data: ",len(train_df))
    print("Total number of training images: ", len(os.listdir(TRAIN_IMG_DIR)))
    print("Total number of test images: ", len(os.listdir(TEST_IMG_DIR)))


#TODO
def _plot_eda(defect_names, train_df, train_img_dir=TRAIN_IMG_DIR):
    for fname in defect_names:
        masks.show_mask_image(fname, train_df, train_img_dir)



def main():
  train_df = load_csv(data_dir=DATA_DIR, train_csv=TRAIN_CSV)
  defect_names, defect_img = load_img(
      data_dir=DATA_DIR, train_csv=TRAIN_CSV,
      train_img_subdir=TRAIN_IMG_SUBDIR, test_img_subdir=TEST_IMG_SUBDIR)
  _plot_eda(defect_names[:10], train_df, train_img_dir=TRAIN_IMG_DIR)
  _show_eda_text(train_df)

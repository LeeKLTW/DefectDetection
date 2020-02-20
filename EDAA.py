from DefectDetection.datasets.steel_data import load_csv,load_img

from DefectDetection.utils import masks


for fname in defect_names[5:10]:
  show_mask_image(fname)
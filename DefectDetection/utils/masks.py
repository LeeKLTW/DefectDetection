import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def make_mask(img_name, train_df):
    # img_name = train_df.ImageId[idx]
    mask_df = train_df[train_df.ImageId.isin([img_name])]
    defect_classes = mask_df.ClassId.values
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)

    for i in range(4):
        if (i + 1) in defect_classes:
            rle = mask_df[mask_df.ClassId == (i + 1)].EncodedPixels.iloc[0].split(" ")
            rle_pos = map(int ,rle[0::2])
            rle_len = map(int ,rle[1::2])

            rle2mask = np.zeros(256 *1600, dtype=np.uint8)
            for pos, leng in zip(rle_pos, rle_len):
                rle2mask[pos-1:pos+leng-1] = 1
            mask[:, :, i] = rle2mask.reshape(256, 1600, order='F')
        else:
            mask[:, :, i] = np.zeros((256, 1600))

    return defect_classes, mask


#TODO: store it instead of showing it
def show_mask_image(img_name, train_df, train_img_dir):
    #defect_classes = train_df[train_df.ImageId.isin([img_name])].ClassId
    defect_classes, mask = make_mask(img_name, train_df)
    img = cv2.imread(os.path.join(train_img_dir, img_name))

    palette = [(255, 89, 143), (0, 179, 44), (255, 215, 0), (21, 178, 211)]
    #fig, ax = plt.subplots(figsize=(15, 5))
    for i in range(4):
        contours, _ = cv2.findContours(mask[:,:,i], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for j in range(len(contours)):
            cv2.polylines(img, contours[j], True, palette[i], 2) #cv2.polylines(影像, 頂點座標, 封閉型, 顏色, 線條寬度)
    #ax.set_title(img_name)
    cv2.imwrite(f'./assets/{img_name}',img)
    # ax.imshow(img)

import numpy as np
import os
import cv2

import tensorflow as tf

from DefectDetection.utils.masks import make_mask


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for tf'

    def __init__(self, df, list_IDs, batch_size=16, img_h=256, img_w=1600,
                 n_classes=4, train_path=train_path, shuffle=True):
        'Initialization'
        self.df = df
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.n_classes = n_classes
        self.train_path = train_path
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_img = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_img)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_img):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.img_h, self.img_w, 3), dtype=np.float32)
        y = np.empty((self.batch_size, self.img_h, self.img_w, self.n_classes), dtype=np.uint8)
        img_names = []

        # Generate data
        for i, img_name in enumerate(list_IDs_img):
            # img_name = self.df.ImageId.iloc[ID]
            img_path = os.path.join(self.train_path, img_name)
            img = cv2.imread(img_path)
            img = img.astype(np.float32) / 255.
            X[i, :, :, :] = img

            # rles = self.df.EncodedPixels.iloc[ID]
            _, masks = make_mask(img_name)
            y[i, :, :, :] = masks

            img_names.append(img_name)

        return X, y
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras import backend as K


def UNet(input_shape):
  x_input = Input(shape=input_shape)

  c1 = Conv2D(8, (3,3), activation='relu', padding='same')(x_input)
  c1 = Conv2D(8, (3,3), activation='relu', padding='same')(c1)
  p1 = MaxPooling2D((2,2))(c1)

  c2 = Conv2D(16, (3,3), activation='relu', padding='same')(p1)
  c2 = Conv2D(16, (3,3), activation='relu', padding='same')(c2)
  p2 = MaxPooling2D((2,2))(c2)

  c3 = Conv2D(32, (3,3), activation='relu', padding='same')(p2)
  c3 = Conv2D(32, (3,3), activation='relu', padding='same')(c3)
  p3 = MaxPooling2D((2,2))(c3)

  c4 = Conv2D(64, (3,3), activation='relu', padding='same')(p3)
  c4 = Conv2D(64, (3,3), activation='relu', padding='same')(c4)
  p4 = MaxPooling2D((2,2))(c4)

  c5 = Conv2D(128, (3,3), activation='relu', padding='same')(p4)
  c5 = Conv2D(128, (3,3), activation='relu', padding='same')(c5)

  u42 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c5)
  u42 = concatenate([u42, c4])
  c42 = Conv2D(64, (3,3), activation='relu', padding='same')(u42)
  c42 = Conv2D(64, (3,3), activation='relu', padding='same')(c42)

  u32 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c42)
  u32 = concatenate([u32, c3])
  c32 = Conv2D(32, (3,3), activation='relu', padding='same')(u32)
  c32 = Conv2D(32, (3,3), activation='relu', padding='same')(c32)

  u22 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c32)
  u22 = concatenate([u22, c2])
  c22 = Conv2D(16, (3,3), activation='relu', padding='same')(u22)
  c22 = Conv2D(16, (3,3), activation='relu', padding='same')(c22)

  u12 = Conv2DTranspose(8, (2,2), strides=(2,2), padding='same')(c22)
  u12 = concatenate([u12, c1])
  c12 = Conv2D(8, (3,3), activation='relu', padding='same')(u12)
  c12 = Conv2D(8, (3,3), activation='relu', padding='same')(c12)

  y_output = Conv2D(4, (1,1), activation='sigmoid')(c12)

  model = Model(inputs=[x_input], outputs=[y_output])

  return model
from DefectDetection.applications.UNet import UNet
from DefectDetection.utils.data_generator import DataGenerator
from DefectDetection.metrics.dice_coef import dice_coef
from DefectDetection.datasets.steel_data import DATA_DIR,TRAIN_CSV,TRAIN_IMG_SUBDIR,TEST_IMG_SUBDIR,load_img,load_csv

from IPython.display import display
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


from absl import flags

# Parameters
flags.DEFINE_integer("img_h", 256, "img_h")
flags.DEFINE_integer("img_w", 1600, "img_w")
flags.DEFINE_integer("batch_size", 16, "batch_size")
flags.DEFINE_integer("n_classes", 4, "n_classes")
flags.DEFINE_boolean("shuffle", True, "shuffle")

#
flags.DEFINE_string("model_path", "steelDefect_model2.h5", "path to save the model")
flags.DEFINE_boolean("do_predict", True, "To predict the model or not.")


FLAGS = flags.FLAGS

#FIXME
def _plot_summary(History):
    # Plot the Loss and the metirc: dice_coef
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(History.history['loss'])
    axes[0].plot(History.history['val_loss'])
    axes[0].set_title("Model Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend(['Training', 'Validation'], loc="upper right")

    axes[1].plot(History.history['dice_coef'])
    axes[1].plot(History.history['val_dice_coef'])
    axes[1].set_title("Model Dice_Coef")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Dice_Coef")
    axes[1].legend(['Training', 'Validation'], loc="upper left");



# #FIXME
# def show_predictions(dataset=None, num=1):
#   if dataset:
#     for image, mask in dataset.take(num):
#       pred_mask = model.predict(image)
#       display([image[0], mask[0], create_mask(pred_mask)])
#   else:
#     display([sample_image, sample_mask,
#              create_mask(model.predict(sample_image[tf.newaxis, ...]))])


def main():
    UNet_model = UNet(input_shape=(256, 1600, 3))
    UNet_model.summary()
    adam = Adam(lr=0.05, epsilon=0.1)
    UNet_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[dice_coef])

    #TODO
    train_df = load_csv(data_dir=DATA_DIR, train_csv=TRAIN_CSV)
    defect_names, defect_img = load_img(
        data_dir=DATA_DIR, train_csv=TRAIN_CSV,
        train_img_subdir=TRAIN_IMG_SUBDIR, test_img_subdir=TEST_IMG_SUBDIR)

    train_idx, valid_idx = train_test_split(defect_names, test_size=0.2, random_state=123)

    params = {'df': train_df,
              'img_h': 256,
              'img_w': 1600,
              'batch_size': 16,
              'n_classes': 4,
              'train_path': '/content/input/train_images',
              'shuffle': True}

    train_data_gen = DataGenerator(list_IDs=train_idx, **params)
    valid_data_gen = DataGenerator(list_IDs=valid_idx, **params)

    History = UNet_model.fit(train_data_gen,
                             validation_data=valid_data_gen,
                             epochs=2,
                             verbose=1)

    _plot_summary(History)

    UNet_model.save(FLAGS.model_path)
    print("Save model at {}".format(FLAGS.model_path))

    if FLAGS.do_predict:
        preds = UNet_model.predict(valid_data_gen, verbose=1)

main()


# Load the model
UNet_model = load_model("steelDefect_model.h5",
                        custom_objects={"dice_coef":dice_coef})
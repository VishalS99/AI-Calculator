import os
import numpy as np
from train import *
import tensorflow as tf
import tensorflow.keras as ks


def character_predict(image, model_path="model/model-bst.h5"):

    model = ks.models.load_model(model_path)
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    # image = cv2.imread("./data/test/nine/481.jpg")
    image = image.astype('float') / 255.0

    image = cv2.resize(
        image, (128, 128))

    image = ks.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)
    ic = pred.argmax(axis=1)[0]

    image_class = [".", "/", "8", "=", "5", "4", "-",
                   "9", "1", "+", "7", "6", "3", "*", "2", "0"]

    return image_class[ic]

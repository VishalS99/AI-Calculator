import os
import numpy as np
from train import *
import tensorflow as tf
import tensorflow.keras as ks


def character_predict(image, model_path="model/model-bst.h5"):

    model = ks.models.load_model(model_path)

    # image = cv2.imread(image)
    image = image.astype('float') / 255.0

    image = ks.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)
    print(pred)
    # TODO: Retrain model in Colab and uncomment below
    ic = pred.argmax(axis=1)[0]
    # ic = np.argsort(-pred, axis=1)[0][1]
    image_class = [".", "/", "8", "=", "5", "4", "-",
                   "9", "1", "+", "7", "6", "3", "*", "2", "0"]

    return image_class[ic]


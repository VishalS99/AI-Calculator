# Base imports for the model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.layers import Flatten, MaxPooling2D, Conv2D, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Model:
    def __init__(self, training_path="./data/train", validation_path="./data/test"):
        self.train = training_path
        self.valid = validation_path

    '''
    Creating image generator for both training and validation data
        - Ensures data augmentation for both
    '''

    def load_data(self):

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        test_datagen = ImageDataGenerator(
            rescale=1./255
        )

        training_gen = train_datagen.flow_from_directory(
            self.train,
            target_size=(128, 128),
            batch_size=30,
            class_mode='categorical',
            shuffle=True)

        validation_gen = test_datagen.flow_from_directory(
            self.valid,
            target_size=(128, 128),
            batch_size=30,
            class_mode='categorical')

        return (training_gen, validation_gen)

    '''
    The CNN model has:
        - layer1: 64 filers 2D convolution layer with Relu Activation
        - layer2: 2D Maxpool layer of size (2, 2)
        
        - layer3: 64 filers 2D convolution layer with Relu Activation
        - layer4: 2D Maxpool layer of size (2, 2)

        - layer5: 64 filers 2D convolution layer with Relu Activation
        - layer6: 2D Maxpool layer of size (2, 2)

        - layer7: Flatten layer
        - layer8: 128 neurons Dense layer with Relu activation
        - layer9: 16 neuron final Dense layer with Softmax activation, having regularization rate of 1E-3

    The model uses:
        - loss: Categorical CrossEntropy loss
        - optimizer: Adam with lr = 1E-3 and decay of 1E-5
        - metric: Accuracy
    '''

    def get_cnn(self):

        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3),
                         kernel_initializer='he_uniform', activation='relu'))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform'))
        model.add(MaxPooling2D(2, 2))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='softmax',
                        kernel_regularizer=ks.regularizers.l1(1e-3)))

        return model

    '''
    Train model: the model ensures
        - checkpoint saves
        - model structure saves
        - weight files save
    '''

    def train_model(self):

        model = self.get_cnn()
        model.summary()
        model.compile(
            loss="categorical_crossentropy",
            optimizer=ks.optimizers.Adam(lr=1e-3, decay=1e-5),
            metrics=['accuracy']
        )

        model_path = "./model"
        model_json_path = "./model/model-desc.json"
        checkpoint_path = "./model/model-bst.h5"

        checkpoint_dir = os.path.dirname(checkpoint_path)

        cp_callback = ks.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            # save_best_only=True,
            verbose=0
        )

        model_json = model.to_json()

        history = model.fit(
            self.load_data()[0],
            epochs=20,
            validation_data=self.load_data()[1],
            callbacks=[cp_callback]
        )
        with open(model_json_path, "w") as json_file:
            json_file.write(model_json)

        return history

    '''
    Plots a summary of the model defined
    '''

    def plot_model_stats(self):
        self.get_cnn().summary()

    '''
    Plot of 
        - Training and validation accuracy
        - Training and validation loss
    '''

    def plot_training_metrics(self):

        history = self.train_model()
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()


if __name__ == '__main__':

    '''
    Fixing directory paths for training and testing data
    To retrain:
        - Create the directory structure mentioned below.
        - Add training and testing data from the link provided in README file
        - Move 15 images from individual training folders to respective testing folders
    '''
    data_dir = './data'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    my_model = Model()
    my_model.train_model()

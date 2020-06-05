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

# Individual Training paths
train_1 = os.path.join(train_dir, 'one')
train_2 = os.path.join(train_dir, 'two')
train_3 = os.path.join(train_dir, 'three')
train_4 = os.path.join(train_dir, 'four')
train_5 = os.path.join(train_dir, 'five')
train_6 = os.path.join(train_dir, 'six')
train_7 = os.path.join(train_dir, 'seven')
train_8 = os.path.join(train_dir, 'eight')
train_9 = os.path.join(train_dir, 'nine')
train_0 = os.path.join(train_dir, 'zero')
train_sign = os.path.join(train_dir, 'equal')
train_dec = os.path.join(train_dir, 'decimal')
train_add = os.path.join(train_dir, 'plus')
train_minus = os.path.join(train_dir, 'minus')
train_mul = os.path.join(train_dir, 'times')
train_div = os.path.join(train_dir, 'div')

# Individual Validation paths
test_1 = os.path.join(test_dir, 'one')
test_2 = os.path.join(test_dir, 'two')
test_3 = os.path.join(test_dir, 'three')
test_4 = os.path.join(test_dir, 'four')
test_5 = os.path.join(test_dir, 'five')
test_6 = os.path.join(test_dir, 'six')
test_7 = os.path.join(test_dir, 'seven')
test_8 = os.path.join(test_dir, 'eight')
test_9 = os.path.join(test_dir, 'nine')
test_0 = os.path.join(test_dir, 'zero')
test_sign = os.path.join(test_dir, 'equal')
test_dec = os.path.join(test_dir, 'decimal')
test_add = os.path.join(test_dir, 'plus')
test_minus = os.path.join(test_dir, 'minus')
test_mul = os.path.join(test_dir, 'times')
test_div = os.path.join(test_dir, 'div')


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
    - optimizer: SGD with lr = 1E-2 and momentum of 0.9
    - metric: Accuracy

'''
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
model.add(Dense(16, activation='softmax',
                kernel_regularizer=ks.regularizers.l1(1e-3)))

model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=ks.optimizers.SGD(lr=0.01, momentum=0.9),
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    # horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

training_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=50,
    class_mode='categorical',
    shuffle=True)

validation_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=50,
    class_mode='categorical')

checkpoint_path = './model/checkpoints/'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ks.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           verbose=0)

model_json = model.to_json()
with open("./model/model2.json", "w") as json_file:
    json_file.write(model_json)
    3
history = model.fit(
    training_gen,
    epochs=20,
    validation_data=validation_gen,
    callbacks=[cp_callback]
)

model.save_weights('./model/model.h5')

'''
Plot of 
    - Training and validation accuracy
    - Training and validation loss
'''
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

# -*- coding: utf-8 -*-
'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import numpy as np
import pandas as pd
from PIL import Image
import pickle
#import matplotlib.pyplot as plt
import keras
from glob import glob
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.regularizers import l2 
from keras.callbacks import History,TensorBoard,EarlyStopping,Callback,ModelCheckpoint
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os

def my_loss(x, x_decoded_mean):  
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)  
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)  
    return xent_loss + kl_loss  
 

batch_size = 32
num_classes = 5
epochs = 200
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_trained_model.h5'

LABEL_FILE = 'data/trainLabels.csv'

def get_image_files(datadir, left_only=False):
    fs = glob('{}/*'.format(datadir))
    if left_only:
        fs = [f for f in fs if 'left' in f]
    return np.array(sorted(fs))

def get_names(files):
    return [os.path.basename(x).split('.')[0] for x in files]

def get_labels(names, labels=None, per_patient=False):
    
    if labels is None:
        labels = pd.read_csv(LABEL_FILE, 
                             index_col=0).loc[names].values.flatten()

    if per_patient:
        left = np.array(['left' in n for n in names])
        return np.vstack([labels[left], labels[~left]]).T
    else:
        return labels

def get_images(files): #这和data里写的不是差不多么？ 下面还删了？smg？？？
    images = []
    for i in range(len(files)):
        img = image.load_img(files[i])
        tmp = image.img_to_array(img)
        #np.subtract(tmp, MEAN[np.newaxis, np.newaxis, :], out=tmp)
        #np.divide(tmp, STD[np.newaxis, np.newaxis, :], out=tmp)
        #tmp = data.augment_color(tmp, sigma=0.25)
        images.append(tmp)
    return images

def print_y_distribution(original_y_train):
    distribution = [0,0,0,0,0,0,0]
    for i in range(len(original_y_train)):
        distribution[original_y_train[i]]+=1
    print(distribution) #[10324, 978, 2117, 350, 284, 0, 0, 0, 0, 0]
# The data, shuffled and split between train and test sets:
original_train_files = get_image_files('data/right_data_256_train_mini')
original_train_names = get_names(original_train_files)
original_y_train = get_labels(original_train_names).astype(np.float32)
print_y_distribution(original_y_train)

qpp=0
for i in range(len(original_y_train)):
    if original_y_train[i]==0:
        if qpp<4000:
            qpp+=1
        else:
            break
print(qpp)

idx = 0
train_files = original_train_files.copy()
for i in range(len(original_train_files)):
    if ( original_y_train[i]==0 and i>qpp):
        continue
    else:
        train_files[idx] = original_train_files[i]
        idx += 1
print(idx)
train_files = train_files[0:idx]
train_names = get_names(original_train_files)
y_train = get_labels(train_names).astype(np.float32)
print_y_distribution(y_train)

test_files = get_image_files('data/right_data_256_val_mini')
test_names = get_names(test_files)
y_test = get_labels(test_names).astype(np.float32)
#net.my_test(test_files, test_labels) #my_test根本就不存在是smg？ make_submis
x_train = np.array(get_images(train_files))
x_test = np.array(get_images(test_files))

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weight_dict = dict(enumerate(class_weight))
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              class_weight = class_weight_dict)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# -*- coding: utf-8 -*-
from __future__ import division, print_function
import time
import os
import numpy as np
import pandas as pd

import data

import keras
import keras.layers
import keras.backend as K
from keras.optimizers import SGD
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Conv2D, MaxPooling2D, Dropout, Flatten, Input, GlobalAveragePooling2D, BatchNormalization, LeakyReLU, AveragePooling2D
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2 
from keras.callbacks import History,TensorBoard,EarlyStopping,Callback,ModelCheckpoint
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config = config))

# channel standard deviations
STD = np.array([70.53946096, 51.71475228, 43.03428563], dtype=np.float32)

# channel means
MEAN = np.array([108.64628601, 75.86886597, 54.34005737], dtype=np.float32)

#generator
train_gen = image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)
test_gen = image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

#计算敏感度和特异度的俩函数

def sensi(y_true, y_pred):

    y_pred = tf.convert_to_tensor(y_pred, np.float32)#newly added 
    y_true = tf.convert_to_tensor(y_true, np.float32)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    return tp / (tp + fn + K.epsilon())

def speci(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    y_true = tf.convert_to_tensor(y_true, np.float32)

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    return tn / (tn + fp + K.epsilon()) 

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

def get_labels(labels):
    targets = to_categorical(labels, 5)
    return targets

def get_testset(files,labels):
    images = get_images(files)
    targets = get_labels(labels)
    return np.array(images), np.array(targets)

def get_trainset(files, labels):
    images = get_images(files)
    #print(np.array(images).shape)
    targets = get_labels(labels)
    return np.array(images), np.array(targets)

def create_net():
    net = NetModel()
    return net

class NetModel():
    def __init__(self):
        self.debug = 1
        self.comment_build()

    def identity_block(input_tensor, kernel_size, filters, stage, block):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, kernel_size, kernel_size,
                        padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = merge([x, input_tensor], mode='sum')
        x = Activation('relu')(x)
        return x


    def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, 1, 1, subsample=strides,
                        name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, kernel_size, kernel_size, padding='same',
                        name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(nb_filter3, 1, 1, subsample=strides,
                                name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = merge([x, shortcut], mode='sum')
        x = Activation('relu')(x)
        return x

    def build_model(self):
        img_input = Input(shape=(512, 512, 3))

        x = ZeroPadding2D((3, 3))(img_input)
        x = Conv2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)

        x = Flatten()(x)
        # x = Dense(1024, activation='relu', name='fc1000')(x)
        x = Dense(5, activation='softmax', name='fc5')(x)

        self.model = Model(inputs, x)

    def comment_build(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3,3),  input_shape=(512, 512, 3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),  padding='same',name='block1_conv1'))
        self.model.add(LeakyReLU(alpha=0.01))
        #self.model.add(Conv2D(32, (3,3), input_shape=(4, 256, 256, 3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation=LeakyReLU(alpha=0.01), padding='same', name='block1_conv1'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool1'))
        self.model.add(Conv2D(32, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same', name='block1_conv2'))
        self.model.add(LeakyReLU(alpha=0.01))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool2'))

        # Block 2
        self.model.add(Conv2D(64, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same', name='block2_conv1'))
        self.model.add(LeakyReLU(alpha=0.01))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block2_pool1'))
        self.model.add(Conv2D(64, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same', name='block2_conv2'))
        self.model.add(LeakyReLU(alpha=0.01))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block2_pool2'))

        # Block 3
        self.model.add(Conv2D(128, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same', name='block3_conv1'))
        activation=LeakyReLU(alpha=0.01), 
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block3_pool1'))
        self.model.add(Conv2D(128, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same', name='block3_conv2'))
        activation=LeakyReLU(alpha=0.01), 
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block3_pool2'))

        # Block 4
        self.model.add(Conv2D(256, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),  padding='same', name='block4_conv1'))
        activation=LeakyReLU(alpha=0.01), 
        self.model.add(Conv2D(256, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same', name='block4_conv2'))
        activation=LeakyReLU(alpha=0.01), 
        self.model.add(Conv2D(256, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same', name='block4_conv3'))
        activation=LeakyReLU(alpha=0.01), 
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block4_pool2'))

        # Block 5
        self.model.add(Conv2D(512, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same', name='block5_conv1'))
        activation=LeakyReLU(alpha=0.01), 
        self.model.add(Conv2D(512, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same', name='block5_conv2'))
        activation=LeakyReLU(alpha=0.01), 
        self.model.add(Conv2D(512, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same', name='block5_conv3'))
        activation=LeakyReLU(alpha=0.01), 
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((3, 3), strides=(2, 2), name='block5_pool2'))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten(name='flatten'))
        self.model.add(Dense(1024))
        activation=LeakyReLU(alpha=0.01), 
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024))
        activation=LeakyReLU(alpha=0.01), 
        self.model.add(Dense(5, activation='softmax'))

    def abort_train_test_split(self, X, y, test=False):
        sss = StratifiedShuffleSplit(n_splits=150, test_size=0.2, random_state=23) #这个看不懂是smg？
        spl = sss.split(X, y)
        tr, te = next(iter(spl))
        X_train, y_train = X[tr], y[tr]
        X_valid, y_valid = X[te], y[te]

        return X_train, X_valid, y_train, y_valid

    def fine_tune(self, X, y):
        #X = np.array(X)
        #y = np.array(y)
        #X_train, X_valid, y_train, y_valid = self.train_test_split(X, y, 0.1)
        X, y = get_trainset(X, y)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=11)
        #X_valid, y_valid = get_trainset(X_val, y_val)
        print("Fine_tune Start")
        batch_size = 3 #本来是4
        train_gen.fit(X)
        test_gen.fit(X)
        if self.debug>0:
            print(X_valid.shape)
            print(y_valid.shape)
            print(len(y_valid))
        train_generator = train_gen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
        validate_generator = test_gen.flow(X_valid, y_valid, batch_size=batch_size, shuffle=True)
        lr = 0.1
        decay = 0.0001
        tb = TensorBoard(log_dir='logs', histogram_freq=5) #这个淘宝打不出来也很尴尬啊！用来打印train&test种的metrics的。会保存到logs文件夹里， 1个epoch打印5次activation&weight histograms
        hs = History()
        es = EarlyStopping(monitor='loss', min_delta=0.0008, patience=2, mode='min',verbose = 0)
        #检测loss 如果连续2轮的min（decrease） 都<min_delta就early stopping， 这个不让加进git_generator很蛋疼啊...
        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=False)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy',
                metrics=['categorical_accuracy', sensi, speci]) 
                #optimizer最常见的就是rmsprop & adagrad  loss最常见的就是⬆️和mse
                #metrics后面的是可以自定义的函数（例如这里的sensi， speci，一般输入都依次是y_true, y_pred）。 
        #callbacks=[tb, hs, es]

              
        epoch = 2 #本来是150
        '''
        hs = self.model.fit_generator(train_generator,
                                samples_per_epoch=len(y_train),
                                nb_epoch = epoch,
                                validation_data=validate_generator,
                                nb_val_samples=len(y_valid),
                                callbacks = [tb, es]
                                )
        self.fit(data, labels, epochs=10, batch_size = 32)
        '''
        hs = self.model.fit_generator(train_generator,
                                steps_per_epoch = len(y_train)/batch_size,
                                epochs = epoch,
                                verbose = 2,
                                validation_data = validate_generator,
                                validation_steps = len(y_valid),
                                initial_epoch = 0                
                                )
                     
        print("Fine_tune Finish")
        self.save_params(epoch, lr, decay)
        df = pd.DataFrame(hs.history)
        #time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        df.to_csv('%s_hist.csv' % time.strftime("%Y%m%d%H%M%S", time.localtime()))


    def save_params(self, epoch, lr, decay):
        self.model.save_weights('weights/incepv3_ep_%d_lr_%f_decay_%f_%s_weights.h5' % (epoch, lr, decay, time.strftime("%Y%m%d%H%M%S", time.localtime())))

    def my_test(self,X,y):
        if self.debug>0:
            print("my_test starts, y is")
            print(y)
        X_test, y_true = get_testset(X,y)
        if self.debug>0:
            print(y_true)
        y_pred = self.model.predict(X_test)
        print("sensitivity={}, specity={}""".format(sensi(y_true, y_pred),speci(y_true, y_pred)))
        
    def make_submission(self, files, names, weights, subm):
        self.model.load_weights(weights)
        X = get_testset(files)
        print('------load done------')
        y_prob = self.model.predict(X, batch_size=2)
        print('------predict done------')
        y_hat = np.argmax(y_prob, axis=1)
        y_dic = {'level':y_hat}
        sub = pd.DataFrame(y_dic)
        sub.insert(0, 'image', names)
        sub.to_csv(subm, index=False)
        print('-------done-------')


        











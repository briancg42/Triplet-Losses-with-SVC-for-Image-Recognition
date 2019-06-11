from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave

from customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D

import theano.tensor as T

def mean_subtract(img):
    img = T.set_subtensor(img[:,0,:,:],img[:,0,:,:] - 123.68)
    img = T.set_subtensor(img[:,1,:,:],img[:,1,:,:] - 116.779)
    img = T.set_subtensor(img[:,2,:,:],img[:,2,:,:] - 103.939)

    return img / 255.0

def AlexNet_base(weights_path=None, input_tensor=None):
    if input_tensor is not None:
        inputs = input_tensor
    else:
        inputs = Input(shape=(3,227,227))
    
    mean_subtraction = Lambda(mean_subtract, name='mean_subtraction')(inputs)

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1')(mean_subtraction)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)


    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    # dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_1)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000,name='dense_3')(dense_3)
    prediction = Activation("softmax",name="softmax")(dense_3)


    model = Model(inputs=inputs, outputs=prediction)

    if weights_path:
        model.load_weights(weights_path)

    x = model.layers[29].output

    new_model = Model(inputs=inputs, outputs = x)

    return new_model

# model = AlexNet_base(weights_path='./weights/alexnet_weights.h5')
# print model.summary()

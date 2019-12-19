"""The Final model for my Research Paper"""
#Changed stddev=0.001 to stddev=0.005 everywhere.
import keras
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, LeakyReLU, Input, add, Activation, UpSampling3D, Conv3DTranspose, Reshape, Dense, Dropout
from keras import optimizers
from keras.models import Model, Sequential
from keras.regularizers import l1
import keras.backend as K
import tensorflow as tf
import os
import numpy as np

def model_inst(input_channels, output_channels):
    img_input = Input(shape = (1, 88, 88, 88))
    #img_input = tf.convert_to_tensor(train_set[0][0])

    
    #Basic3D(1, 16, 3). output_shape = (16, 88, 88, 88)
    x = Conv3D(filters = 16, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = 'he_normal', bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(img_input)
    x = BatchNormalization(axis = 1)(x)
    x = Activation('relu')(x)
    x = Conv3D(filters = 16, 
            kernel_size = 3,
            kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None),
            bias_initializer = 'zeros',
            data_format = 'channels_first',
            kernel_regularizer = l1(0.001),
            padding = 'same')(x)
    x = BatchNormalization(axis = 1)(x)
    x = Activation('relu')(x)

    #Pool3D(2). output_shape = (16, 44, 44, 44)
    x = MaxPooling3D(pool_size=2, strides=2, data_format = 'channels_first')(x)

    #Res3D(16, 32). output_shape = (32, 44, 44, 44)
    shortcut =  Conv3D(filters =  32, kernel_size=1, strides=1, padding='valid', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)
    shortcut = BatchNormalization(axis = 1)(shortcut)

    x = Conv3D(filters = 32, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)
    x = BatchNormalization(axis = 1)(x)
    x = Activation('relu')(x)
    x = Conv3D(filters = 32, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)
    x = BatchNormalization(axis = 1)(x)

    x = add([x, shortcut])
    x = Activation('relu')(x)

    #Res3D(32, 32). output_shape = (32, 44, 44, 44)
    # x = Conv3D(filters = 32, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    # x = BatchNormalization(axis = 1)(x)
    # x = Activation('relu')(x)
    # x = Conv3D(filters = 32, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    # x = BatchNormalization(axis = 1)(x)
    # x = Activation('relu')(x)

    # #Res3D(32, 32). output_shape = (32, 44, 44, 44)
    # x = Conv3D(filters = 32, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    # x = BatchNormalization(axis = 1)(x)
    # x = Activation('relu')(x)
    # x = Conv3D(filters = 32, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    # x = BatchNormalization(axis = 1)(x)
    # x = Activation('relu')(x)

    #Pool3D(2).output_shape = (32, 22, 22, 22)
    x = MaxPooling3D(pool_size=2, strides=2, data_format = 'channels_first')(x)

    #Res3D(32, 64). output_shape = (64, 22, 22, 22)
    shortcut =  Conv3D(filters =  64, kernel_size=1, strides=1, padding='valid', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    shortcut = BatchNormalization(axis = 1)(shortcut)

    x = Conv3D(filters = 64, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)
    x = BatchNormalization(axis = 1)(x)
    x = Activation('relu')(x)
    x = Conv3D(filters = 64, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)
    x = BatchNormalization(axis = 1)(x)

    x = add([x, shortcut])
    x = Activation('relu')(x)

    #Res3D(64, 64).output_shape = (64, 22, 22, 22)
    # x = Conv3D(filters = 64, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)
    # x = BatchNormalization(axis = 1)(x)
    # x = Activation('relu')(x)
    # x = Conv3D(filters = 64, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)
    # x = BatchNormalization(axis = 1)(x)
    # x = Activation('relu')(x)

    # #Res3D(64, 64).output_shape = (64, 22, 22, 22)
    # x = Conv3D(filters = 64, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    # x = BatchNormalization(axis = 1)(x)
    # x = Activation('relu')(x)
    # x = Conv3D(filters = 64, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    # x = BatchNormalization(axis = 1)(x)
    # x = Activation('relu')(x)

    #Pool3D(2).output_shape = (64, 11, 11, 11)
    x = MaxPooling3D(pool_size=2, strides=2, data_format = 'channels_first')(x) #Change

    #Res3D(64, 128) - Encoder. output_shape = (128, 22, 22, 22) #Change
    #shortcut =  Conv3D(filters =  128, kernel_size=1, strides=1, padding='valid', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    #shortcut = BatchNormalization(axis = 1)(shortcut)

    #x = Conv3D(filters = 128, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    #x = BatchNormalization(axis = 1)(x)
    #x = Activation('relu')(x)
    #x = Conv3D(filters = 128, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    #x = BatchNormalization(axis = 1)(x)

    #x = add([x, shortcut])
    #x = Activation('relu')(x)

    #Mid1 - Res3D(128, 128). output_shape = (128, 11, 11, 11) #Change
    #x = Conv3D(filters = 128, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    #x = BatchNormalization(axis = 1)(x)
    #x = Activation('relu')(x)
    #x = Conv3D(filters = 128, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    #x = BatchNormalization(axis = 1)(x)
    #x = Activation('relu')(x)

    # #Mid2 - Res3D(128, 128). output_shape = (128, 11, 11, 11)
    # x = Conv3D(filters = 128, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    # x = BatchNormalization(axis = 1)(x)
    # x = Activation('relu')(x)
    # x = Conv3D(filters = 128, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    # x = BatchNormalization(axis = 1)(x)
    # x = Activation('relu')(x)

    #Upsample3D(128, 64) - Decoder. output_shape = (32, 44, 44, 44) #Change
    #x = Conv3DTranspose(filters = 32, kernel_size=2, strides=2, padding='valid', data_format = 'channels_first', output_padding=0, kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    #x = BatchNormalization(axis = 1)(x)
    #x = Activation('relu')(x)

    #Res3D(64, 64) - Decoder. output_shape = (64, 44, 44, 44) #Change
    #x = Conv3D(filters = 64, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x) #Change
    #x = BatchNormalization(axis = 1)(x)
    #x = Activation('relu')(x)
    #x = Conv3D(filters = 64, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    #x = BatchNormalization(axis = 1)(x)
    #x = Activation('relu')(x)

    #Res3D(64, 64) - Decoder. output_shape = (64, 11, 11, 11)
    x = Conv3D(filters = 64, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)
    x = BatchNormalization(axis = 1)(x)
    x = Activation('relu')(x)
    x = Conv3D(filters = 64, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)
    x = BatchNormalization(axis = 1)(x)
    x = Activation('relu')(x)

    #Upsample3D(64, 32) - Decoder.output_shape = (32, 22, 22, 22)
    x = Conv3DTranspose(filters = 32, kernel_size=2, strides=2, padding='valid', data_format = 'channels_first', output_padding=0, kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)
    x = BatchNormalization(axis = 1)(x)
    x = Activation('relu')(x)

    # #Res3D(32, 32). output_shape = (32, 22, 22, 22)
    x = Conv3D(filters = 32, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)
    x = BatchNormalization(axis = 1)(x)
    x = Activation('relu')(x)
    x = Conv3D(filters = 32, kernel_size=3, strides=1, padding='same', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)
    x = BatchNormalization(axis = 1)(x)
    x = Activation('relu')(x)

    #Basic3D(32, 32, 1) output_shape = (32, 22, 22, 22)
    #x = Conv3D(filters = 32, kernel_size=1, strides=1, padding='valid', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    #x = BatchNormalization(axis = 1)(x)
    #x = Activation('relu')(x)
    #x = Conv3D(filters = 32, kernel_size=1, strides=1, padding='valid', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros')(x)
    #x = BatchNormalization(axis = 1)(x)
    #x = Activation('relu')(x)

    #Output Layer. output_shape = (21, 22, 22, 22) #Changed
    x = Conv3D(filters = output_channels, kernel_size=1, strides=1, padding='valid', data_format = 'channels_first', kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None), bias_initializer = 'zeros', kernel_regularizer = l1(0.001))(x)

    x = Reshape((output_channels, -1))(x)
    
    x = Dense(44, activation = 'relu')(x) #Changed
    x = Dropout(0.5)(x)
    x = Dense(11, activation = 'relu')(x) #Changed
    x = Dropout(0.5)(x)
    x = Dense(3)(x)
    model = Model(img_input, x, name = 'my_network')

    return model
    #return x










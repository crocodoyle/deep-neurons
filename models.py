from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, SpatialDropout2D, Reshape, UpSampling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


def mlp(nb_classes):
    model = Sequential()

    model.add(Dense(1024, input_dim=(1*128*128), init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, init='uniform'))
    model.add(Activation('softmax'))

    return model

def cnn():
    nb_classes = 3

    model = Sequential()

    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(128, 128, 1)))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.3))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.4))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, init='uniform'))
    model.add(Activation('softmax'))

    return model


def discriminator():
    model = Sequential()

    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(128, 128, 1)))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.3))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.4))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, init='uniform'))
    model.add(Activation('softmax'))

    return model


def generator():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=128*128))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))


def gan(generator, discriminator):

    model = Sequential()
    model.add(generator)
    generator.trainable = False
    model.add(discriminator)
    return model
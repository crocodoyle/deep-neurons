from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Deconvolution2D, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, SpatialDropout2D, Reshape, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
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
    model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode='valid', input_shape=(128, 128, 1,)))
    lay = LeakyReLU(alpha=0.2)
    model.add(lay)
    # model.add(SpatialDropout2D(0.5))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='valid'))
    lay2 = LeakyReLU(alpha=0.2)
    model.add(lay2)

    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='valid'))
    lay3 = LeakyReLU(alpha=0.2)
    model.add(lay3)

    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='valid'))
    lay4 = LeakyReLU(alpha=0.2)
    model.add(lay4)

    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='valid'))
    lay5 = LeakyReLU(alpha=0.2)
    model.add(lay5)

    model.add(Convolution2D(4, 1, 1, border_mode='valid'))

    lay3 = LeakyReLU(alpha=0.2)
    model.add(lay3)
    model.add(Convolution2D(1, 1, 1, border_mode='valid'))
    model.add(Flatten())
    return model

def generator():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=16*16))
    lay1 = LeakyReLU(alpha=0.2)
    model.add(lay1)
    model.add(Dense(1*16*16))
    # model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Reshape((16, 16, 1), input_shape=(1*16*16,)))

    model.add(Deconvolution2D(128, 5, 5, output_shape=(None, 32, 32, 128), subsample=(2, 2), border_mode='same'))
    lay2 = LeakyReLU(alpha=0.2)
    model.add(lay2)
    # model.add(SpatialDropout2D(0.5))
    model.add(BatchNormalization())

    model.add(Deconvolution2D(256, 5, 5, output_shape=(None, 128, 128, 256), subsample=(4, 4), border_mode='same'))
    lay3 = LeakyReLU(alpha=0.2)
    model.add(lay3)
    # model.add(SpatialDropout2D(0.5))
    model.add(BatchNormalization())

    # model.add(Deconvolution2D(16, 5, 5, output_shape=(None, 128, 128, 16), subsample=(2, 2), border_mode='same'))
    # lay4 = LeakyReLU(alpha=0.2)
    # model.add(lay4)
    # model.add(SpatialDropout2D(0.5))
    # model.add(BatchNormalization())

    # model.add(Convolution2D(16, 5, 5, border_mode='same'))
    # model.add(Activation('tanh'))
    # model.add(SpatialDropout2D(0.5))
    # model.add(Convolution2D(1, 5, 5, border_mode='same'))
    # model.add(Activation('tanh'))
    # model.add(SpatialDropout2D(0.5))

    model.add(Convolution2D(1, 1, 1, border_mode='same'))
    model.add(Activation('tanh'))

    return model

def gan(generator, discriminator):
    print('GENERATOR:')
    print(generator.summary())
    print('DISCRIMINATOR:')
    print(discriminator.summary())

    model = Sequential()
    model.add(generator)


    generator.trainable = False
    model.add(discriminator)
    return model
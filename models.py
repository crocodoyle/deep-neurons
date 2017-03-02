from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, SpatialDropout2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


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

    sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=["accuracy"])

    return model

def gan():
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

    sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=["accuracy"])

    return model
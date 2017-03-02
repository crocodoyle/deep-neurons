from keras.optimizers import SGD
from keras.models import Sequential

import make_dataset

from sklearn.model_selection import StratifiedShuffleSplit

import models
import numpy as np

import h5py

def load_data():
    f = h5py.File('E:/deep-neurons/deep-neurons.hdf5', 'r')

    labels = f['labels']

    sss_validation = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

    train_indices, validation_indices, test_indices = None, None, None

    for train_index, validation_index in sss_validation.split(np.zeros(len(labels)), labels):
        train_indices      = train_index
        validation_indices = validation_index

    # for validation_index, test_index in sss_test.split(np.zeros(len(labels[validation_indices])), labels[validation_indices]):
    #     validation_indices = validation_index
    #     test_indices       = test_index

    print("training images:", len(train_index))
    print("validation images:", len(validation_index))
    # print("test_index:", len(test_index))


    return train_indices, validation_indices

if __name__ == "__main__":
    f = h5py.File('E:/deep-neurons/deep-neurons.hdf5', 'r')

    train_index, val_index = load_data()

    images = f['images']
    labels = f['labels']

    model = models.mlp(3)

    sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=["accuracy"])

    x_training = np.asarray(images)[train_index]

    # print(np.shape(x_training), len(images))


    x_train = np.reshape(x_training, (np.shape(x_training)[0],128*128))

    y_train = np.asarray(labels)[train_index]

    model.fit(x_train, y_train, nb_epoch=200)



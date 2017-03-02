import keras
import make_dataset

import models

import h5py


if __name___ == "__main__":


    f = h5py.File('E:/deep-neurons/raw/deep-neruons.hdf5', 'rb')

    images = f['images']
    labels = f['labels']

    model = models.gan()

    sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=["accuracy"])

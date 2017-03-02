import h5py

import numpy as np
import matplotlib.pyplot as plt

import openslide

import os, sys, csv


if __name__ == "__main__":

    input_path = 'E:/deep-neurons/raw/'
    label_file = 'E:/deep-neurons/raw/labels.csv'

    f = h5py.File('E:/deep-neurons/deep-neurons.hdf5', 'w')

    images = f.create_dataset('images', shape=(128,128), dtype='float32')
    labels = f.create_dataset('labels', shape=(3,1),     dtype='bool')

    numNeurons = 0

    filenames = []

    for root, dirs, files in os.walk(input_path, topdown=False):
        for name in files:
            print(name)
            filename = os.path.join(root, name)
            filenames.append(filename)
            numNeurons += 1

    label_list = []
    with open(label_file, 'r') as f:
        reader = csv.reader(f)
        label_tuples = list(reader)

    for i, filename in enumerate(filenames):
        images[i] = plt.imread(filename)

        for label_tuple in label_tuples:
            if label_tuple[0] in filename:
                if label_tuple == 0:
                    labels[i] = [1, 0, 0]
                if label_tuple == 1:
                    labels[i] = [0, 1, 0]
                if label_tuple == 2:
                    labels[i] = [0, 0, 1]

import h5py

import numpy as np
import matplotlib.pyplot as plt

import openslide

import os, sys, csv


if __name__ == "__main__":

    input_path = 'E:/deep-neurons/'
    label_file = 'E:/deep-neurons/labels.txt'

    f = h5py.File('E:/deep-neurons/deep-neurons.hdf5', 'w')

    numNeurons = 0

    filenames = []

    for root, dirs, files in os.walk(input_path, topdown=False):
        for name in files:
            if '.tif' in name:
                print(name)
                filename = os.path.join(root, name)
                filenames.append(filename)
                numNeurons += 1

    print(numNeurons, 'neurons')
    images = f.create_dataset('images', shape=(numNeurons, 128,128), dtype='float32')
    labels = f.create_dataset('labels', shape=(numNeurons, 3,1),     dtype='bool')

    label_list = []
    with open(label_file, 'r') as f:
        reader = csv.reader(f)
        label_tuples = list(reader)

    for i, filename in enumerate(filenames):
        images[i,...] = plt.imread(filename)

        for label_tuple in label_tuples:
            if (label_tuple[0]+ '.tif') in filename:
                if 'pyr' in label_tuple[1]:
                    labels[i, ...] = [1, 0, 0]
                if 'not' in label_tuple[1]:
                    labels[i,...] = [0, 1, 0]
                if 'unk' in label_tuple[1]:
                    labels[i,...] = [0, 0, 1]
    f.close()

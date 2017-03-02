import h5py

import numpy as np
import matplotlib.pyplot as plt

import openslide

from PIL import Image

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
    images = f.create_dataset('images', shape=(numNeurons, 128, 128), dtype='float32')
    labels = f.create_dataset('labels', shape=(numNeurons, 3),     dtype='bool')

    label_list = []
    csvReader = csv.reader(open(label_file, newline='\n'), delimiter=' ')

    for line in csvReader:
        label_list.append((line[0], line[1]))

    for i, (filename, label_tuple) in enumerate(zip(filenames, label_list)):
        images[i,...] = Image.open(filename).convert('L')

        print(label_tuple)

        if 'pyr' in label_tuple[1]:
            labels[i, ...] = [True, False, False]
        if 'not' in label_tuple[1]:
            labels[i,...] = [False, True, False]
        if 'unk' in label_tuple[1]:
            labels[i,...] = [False, False, True]

    f.close()

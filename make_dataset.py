import h5py

import numpy as np
import matplotlib.pyplot as plt
import argparse

from PIL import Image

import os, sys, csv

def build_parser():
    parser = argparse.ArgumentParser(description="Model for convolution-graph network (CGN)")
    parser.add_argument('--input-path', default="/Users/martinweiss/code/deep-neurons/labeled/", type=str)
    return parser

def parse_args(argv):
    print(argv)
    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv
    return opt

def normalise_zero_one(image):
   """Image normalisation. Normalises image to fit [0, 1] range."""

   image = image.astype(np.float32)
   ret = (image - np.min(image))
   ret /= (np.max(image) + 0.000001)
   return ret

def get_label(filename):
    return filename.split("/")[-1].split("_")[-1].split(".")[0]

def main(argv=None):
    opt = parse_args(argv)

    f = h5py.File(os.getcwd() + '/deep-neurons.hdf5', 'w')

    numNeurons = 0

    filenames = []
    for root, dirs, files in os.walk(opt.input_path):
        for name in files:
            if '.tif' in name:
                print(name)
                filename = os.path.join(root, name)
                filenames.append(filename)
                numNeurons += 1

    print(numNeurons, 'neurons')
    images = f.create_dataset('images', shape=(numNeurons, 128, 128), dtype='float32')
    labels = f.create_dataset('labels', shape=(numNeurons, 3), dtype='bool')

    for i, filename in enumerate(filenames):
        images[i,...] = normalise_zero_one(np.array(Image.open(filename).convert('L')))
        label = get_label(filename)
        if 'pyr' in label:
            labels[i, ...] = [True, False, False]
        if 'not' in label:
            labels[i,...] = [False, True, False]
        if 'unk' in label:
            labels[i,...] = [False, False, True]

    f.close()

if __name__ == "__main__":
    main()

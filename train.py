from keras.optimizers import SGD, Adam
from keras.models import Sequential

import make_dataset

from sklearn.model_selection import StratifiedShuffleSplit

import os
import models
import numpy as np
import matplotlib.pyplot as plt

import h5py


input_file = 'deep-neurons.hdf5'

def load_data():

    f = h5py.File(os.getcwd() + '/deep-neurons.hdf5', 'r')

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


def train_gan():

    BATCH_SIZE = 10
    GENERATOR_PARAMS = 100

    train_index, test_index = load_data()

    f = h5py.File(input_file, 'r')

    images = f['images']
    labels = f['labels']

    x_train = np.asarray(images)[train_index]
    y_train = np.asarray(labels)[train_index]

    x_train = (x_train - 127.5) / 127.5


    x_test = np.asarray(images)[test_index]

    x_test = (x_test - 127.5) / 127.5
    y_test = np.asarray(labels)[test_index]

    print(np.max(x_train))

    discriminator = models.discriminator()
    generator = models.generator()
    gan = models.gan(generator, discriminator)

    d_opt = Adam(lr=0.000001, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
    g_opt = Adam(lr=0.000002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gan_opt = Adam(lr=0.00002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)

    generator.compile(loss='binary_crossentropy', optimizer=g_opt)
    gan.compile(loss='binary_crossentropy', optimizer=gan_opt)

    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    noise = np.zeros((BATCH_SIZE, GENERATOR_PARAMS)) # parameters that control generator degrees of freedom


    for epoch in range(200):
        print("Training epoch: " + str(epoch))
        for index in range(int(x_train.shape[0] / BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, GENERATOR_PARAMS)

            image_batch = np.reshape(x_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE], (BATCH_SIZE, 128, 128, 1))
            generated_images = generator.predict(noise, verbose=0)

            # print(np.shape(image_batch))
            # print(np.shape(generated_images))

            #each batch only real/fake images
            if epoch % 2 == 0:
                x = image_batch
                y = [0.7 + np.random.normal(0, .5 / float(epoch + 1))] * BATCH_SIZE
            else:
                x = generated_images
                y = [0.3 + np.random.normal(0, .5 / float(epoch + 1))] * BATCH_SIZE

            #occasionally flip labels?

            # print("x shape:", np.shape(x))
            # print("y shape:", np.shape(y))

            d_loss = discriminator.train_on_batch(x, y)
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, GENERATOR_PARAMS)
            discriminator.trainable = False

            y_gan = [0.7 + np.random.normal(0, 5/(epoch+1))] * BATCH_SIZE
            g_loss = gan.train_on_batch(noise, y_gan)

        print("Generator loss:", str(g_loss))
        print("Discriminator loss:", str(d_loss))

    discriminator.save_weights('discriminator', True)
    generator.save_weights('generator', True)

def train_discriminator(model):
    f = h5py.File(input_file, 'r')

    train_index, val_index = load_data()

    images = f['images']
    labels = f['labels']

    sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=["accuracy"])

    x_training = np.asarray(images)[train_index]

    # print(np.shape(x_training), len(images))


    x_train = np.reshape(x_training, (np.shape(x_training)[0],128*128))

    y_train = np.asarray(labels)[train_index]

    model.fit(x_train, y_train, nb_epoch=200)

def generate():
    BATCH_SIZE = 2
    GENERATOR_PARAMS = 100

    opt = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)

    discriminator = models.discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)
    discriminator.load_weights('discriminator')

    generator = models.generator()
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    generator.load_weights('generator')

    noise = np.zeros((BATCH_SIZE * 20, GENERATOR_PARAMS))

    for i in range(BATCH_SIZE * 20):
        noise[i, :] = np.random.uniform(-1, 1, GENERATOR_PARAMS)

    generated_images = np.zeros((20*BATCH_SIZE, 128, 128, 1))

    for i in range(BATCH_SIZE * 20):
        generated_images[i, ...] = generator.predict(np.reshape(noise[i, :], (1, GENERATOR_PARAMS)), verbose=1)
    # d_pret = discriminator.predict(generated_images, verbose=1)
    #
    # index = np.arange(0, BATCH_SIZE * 20)
    # index.resize((BATCH_SIZE * 20, 1))
    # pre_with_index = list(np.append(d_pret, index, axis=1))
    # pre_with_index.sort(key=lambda x: x[0], reverse=True)
    # nice_images = np.zeros((BATCH_SIZE, 1) + (generated_images.shape[2:]), dtype=np.float32)
    for i in range(len(generated_images)):
        plt.imshow(generated_images[i, :, :, 0])
        plt.axis('off')
        plt.savefig('E:/deep-neurons/generated-' + str(i) + '.png')
        plt.clf()

if __name__ == "__main__":
    train_gan()
    generate()

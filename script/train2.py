import sys
import os
import random
import numpy as np

# SETTAGGIO SCHEDA GRAFICA
if len(sys.argv) == 2:
    gpu_id = sys.argv[1]
    # cnmem = sys.argv[2]
else:
    gpu_id = "cpu"
    cnmem = "0.7"
# print("Argument: gpu={}, mem={}".format(gpu_id, cnmem))
print("Argument: gpu={}".format(gpu_id))
# os.environ["THEANO_FLAGS"] = "device=" + gpu_id + ", lib.cnmem=" + cnmem
os.environ["THEANO_FLAGS"] = "device=" + gpu_id

from models import foamNet2
from batch_generators import load_names, load_images, load_names_val
from custom_loss import euclidean_distance
from keras.optimizers import Adam

import warnings
warnings.filterwarnings("ignore")

random.seed(1769)

from matplotlib import pyplot as plt
import keras as k
from keras.callbacks import EarlyStopping, ModelCheckpoint

class LossHistory(k.callbacks.Callback):

    def __init__(self):
        plt.ion()
        fig = plt.figure()
        self.plot_loss = fig.add_subplot(211)
        self.plot_val_loss = fig.add_subplot(212)

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.val_loss = 0
        self.loss = 0

    def on_epoch_end(self, epoch, logs={}):

        self.val_loss = logs.get('val_loss')
        self.loss = logs.get('loss')
        self.losses.append(self.loss)
        self.val_losses.append(self.val_loss)

        self.plot_loss.plot(self.val_losses, 'r')
        self.plot_val_loss.plot(self.losses, 'r')

        plt.draw()
        plt.pause(0.0001)


if __name__ == '__main__':

    # dimensioni immagine (originali 770x1024)
    rows = 256
    cols = 192

    # parametri modelli deep
    patience = 100
    batch_size = 16
    n_epoch = 100

    # parametri training
    data_augmentation = True
    b_crop = False

    model = foamNet2(rows, cols)

    opt = Adam(lr=1e-6)
    model.compile(loss=euclidean_distance, optimizer=opt)
    model.summary()

    # carico nomi di train
    train_data_names = load_names()

    # carico nomi sequenza validazione (attenzione! nome hardcodato nel codice, da sistemare)
    val_data_names = load_names_val()

    # carico immagini validazione
    val_data_X, val_data_Y = load_images(val_data_names, crop=b_crop, rows=rows, cols=cols)

    # Data augmentation
    if data_augmentation:
        # flip
        tmp = load_names(augm=1)
        train_data_names = train_data_names + tmp

    # DEBUG
    # X, Y = load_images(train_data_names[:2], crop=True, rows, cols)

    random.shuffle(train_data_names)



    def generator():
        random.shuffle(train_data_names)

        while True:
            for it in range(0, len(train_data_names), batch_size):
                X, Y = load_images(train_data_names[it:it + batch_size], crop=b_crop, rows=rows, cols=cols)
                yield X, Y

    his = LossHistory()

    model.fit_generator(generator(),
                        nb_epoch=n_epoch,
                        validation_data=[val_data_X, val_data_Y],
                        samples_per_epoch=len(train_data_names),
                        callbacks=[his, EarlyStopping(patience=patience),
                        ModelCheckpoint("weights2/weights.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=False)]
                            )
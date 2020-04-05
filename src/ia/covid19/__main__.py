#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: abr-02 of 2020
"""

# For model definition/training
import gc
import logging

import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# For saving Keras model scheme graphically and visualizing
from keras.utils import plot_model
import matplotlib.pyplot as plt

# For general purposes
import datetime
import numpy as np
import os
import random
import tensorflow as tf
import time
from sklearn import preprocessing

# Custom imports
from covid19 import utils
from covid19 import keras_utils
import efficientnet.keras as efn

# Set seed for reproducing
seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)


# noinspection DuplicatedCode
def protocol(_network_class,
             _load_data_function,
             _batch_size=400,
             _epochs=200,
             _number_of_classes=3,
             _shuffle_in_training=True,
             _plot_loss_epochs=5,
             _lr=0.001,
             _train_portion=0.7,
             _model_epochs_checkpoint=200,
             _save_model=False,
             _save_intermediate_models=False,):

    _input_image_shape = _network_class.get_input_size()
    # The data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = _load_data_function((_input_image_shape[0], _input_image_shape[1]))

    _create_base_network = _network_class.create_net
    current_dt = datetime.datetime.now()
    output_path = '/home/pedro/Documentos/sources/python/pessoal/playground/covid19/results/{}{:02d}{:02d}-{:02d}{:02d}{:02d}/'.format(current_dt.year,
                                                                                                                                       current_dt.month,
                                                                                                                                       current_dt.day,
                                                                                                                                       current_dt.hour,
                                                                                                                                       current_dt.minute,
                                                                                                                                       current_dt.second)
    os.mkdir(output_path)
    utils.copy_python_files(os.path.dirname(os.path.abspath(__file__)), output_path)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    # preprocessing.normalize([x_train])
    # # Create the Scaler object
    # scaler = preprocessing.StandardScaler()
    # # Fit your data on the scaler object
    # scaled_df = scaler.fit_transform(df)
    # scaled_df = pd.DataFrame(scaled_df, columns=names)

    x_train, y_train, x_val, y_val = utils.separate_data(x_train, y_train, _input_image_shape, (0, 1, 2), _train_portion)

    y_train = keras_utils.to_categorical(y_train.reshape(-1), (x for x in range(_number_of_classes)))
    y_val = keras_utils.to_categorical(y_val.reshape(-1), (x for x in range(_number_of_classes)))
    y_test = keras_utils.to_categorical(y_test.reshape(-1), (x for x in range(_number_of_classes)))

    ####################################################################################################################
    # -- Starting training
    ####################################################################################################################
    model = _create_base_network(_number_of_classes=_number_of_classes, _output_path=output_path)

    model.summary()
    plot_model(model, to_file='{}/model.png'.format(output_path), show_shapes=True, show_layer_names=True)

    # train session
    opt = Adam(lr=_lr)  # choose optimiser. RMS is good too!
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if _save_intermediate_models:
        filepath = "%s/model_semiH_trip_MNIST_v13_ep{epoch:02d}_BS%d.hdf5" % (output_path, _batch_size)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, period=_model_epochs_checkpoint)
        callbacks_list = [checkpoint,
                          keras_utils.PlotLosses(epochs=_plot_loss_epochs, output_folder=output_path),
                          keras_utils.SaveIntermediateData(file_path=output_path)]
    else:
        callbacks_list = [keras_utils.PlotLosses(epochs=_plot_loss_epochs, output_folder=output_path),
                          keras_utils.SaveIntermediateData(file_path=output_path)]

    start = time.time()
    history_model = model.fit(
        x=x_train,
        y=y_train,
        batch_size=_batch_size,
        shuffle=_shuffle_in_training,
        epochs=_epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks_list
    )
    stop = time.time()

    if _save_model:
        model.save('{}/model_final.hdf5'.format(output_path), overwrite=True)

    print('Time spent to train: {}s'.format(stop - start))

    plt.figure(figsize=(8, 8))
    plt.plot(history_model.history['loss'], label='training loss')
    plt.plot(history_model.history['val_loss'], label='validation loss')
    plt.legend()
    plt.suptitle('Loss history after {} epochs'.format(_epochs))
    plt.savefig(output_path + 'loss{}.png'.format(_epochs))
    plt.show()

    ####################################################################################################################
    # -- Done training
    ####################################################################################################################

    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=64)
    print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))


if __name__ == "__main__":
    from covid19 import models  # Model

    protocol(
        _network_class=models.EfficientNetB0(
            use_weights=True,
            use_regularization=False,
            regularizer=tf.keras.regularizers.l2(0.001)),
        _load_data_function=utils.load_covid_dataset,
        _number_of_classes=3,
        _batch_size=10,
        _epochs=20,
        _shuffle_in_training=True,
        _plot_loss_epochs=2,
        _lr=0.001,
        _train_portion=0.8,
        _model_epochs_checkpoint=200,
        _save_intermediate_models=False,
        _save_model=True)

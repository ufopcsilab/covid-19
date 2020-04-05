#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: set-28 of 2019
"""

import keras
from keras.models import Model
from keras.layers import Input, Dense, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, add, Flatten
from keras import backend

# For model plot
from keras.utils import plot_model

from covid19.models.base_model import BaseModel


class ResNet50(BaseModel):

    def __init__(self, use_weights=True, custom_weights_path=None, use_regularization=False, regularizer=None):
        self.use_weights = use_weights
        self._custom_weights = custom_weights_path
        self._use_regularization = use_regularization
        self._regularizer = regularizer
        self._input_shape = (224, 224, 3)
        self.model_name = '/media/data/Profissional/Doc/KerasModels/resnet50.h5'

    def create_net(self,  _number_of_classes=3, _output_path=None):

        if self.use_weights:
            base_model = keras.models.load_model(self.model_name)

            model = Model(inputs=base_model.input, outputs=base_model.get_layer('pool1').output)
            # input_image = Input(shape=_image_input_shape)
            # model = Model(inputs=input_image, outputs=base_model.get_layer('pool1').output)

            x = model.output
            x = BatchNormalization()(x)

            # Output Layer
            classification_layer = Dense(_number_of_classes)(x)

            print(base_model.summary())

            _base_network = Model(inputs=base_model.input, outputs=classification_layer)
        else:
            _base_network = self.create_net_from_scratch(self._input_shape, _number_of_classes)

        if self._use_regularization:
            _base_network = self.add_regularization(_base_network, regularizer=self._regularizer)

        if _output_path is not None:
            plot_model(_base_network, to_file=_output_path + '/model_base_network.png', show_shapes=True, show_layer_names=True)

        return _base_network

    def get_input_size(self):
        return 224, 224, 3

    def get_net_name(self):
        return 'ResNet50'

    def create_net_from_scratch(self, _image_input_shape, _embedding_size):

        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        # Determine proper input shape
        input_shape = Input(shape=_image_input_shape)

        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_shape)
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = Flatten()(x)

        embedding = Dense(_embedding_size)(x)

        model = Model(input_shape, embedding, name='resnet50')

        return model

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1),
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same',
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1),
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def conv_block(self,
                   input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2)):
        """A block that has a conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.
        # Returns
            Output tensor for the block.
        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1),
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x

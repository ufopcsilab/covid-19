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
from keras.layers import Input, Dense, ZeroPadding2D, Conv2D
from keras.layers import BatchNormalization, Activation, Add, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout
from keras import backend

# For model plot
from keras.utils import plot_model

from covid19.models.base_model import BaseModel


class ResNet18(BaseModel):

    def __init__(self, use_weights=True, custom_weights_path=None, use_regularization=False, regularizer=None):
        self.use_weights = use_weights
        self.model_name = '/media/data/Profissional/Doc/KerasModels/resnet18.h5'
        self._custom_weights = custom_weights_path
        self._use_regularization = use_regularization
        self._regularizer = regularizer
        self._input_shape = (224, 224, 3)

    def create_net(self,  _number_of_classes=3, _output_path=None):

        if self.use_weights:
            base_model = keras.models.load_model(self.model_name)

            model = Model(inputs=base_model.input, outputs=base_model.get_layer('relu1').output)
            # model = Model(inputs=input_image, outputs=base_model.get_layer('pool1').output)

            x = model.output
            # x = BatchNormalization()(x)
            x = Flatten()(x)
            x = Dense(units=4096, activation='relu')(x)  # 1st Fully Connected Layer
            x = Dropout(0.4)(x)  # Add Dropout to prevent overfitting
            x = Dense(4096, activation='relu')(x)  # 2nd Fully Connected Layer
            # x = Dropout(0.4)(x)  # Add Dropout

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
        return 'ResNet18'

    def create_net_from_scratch(self, _image_input_shape, _num_classes):

        # Determine proper input shape
        img_input = Input(shape=_image_input_shape)

        # choose residual block type
        ResidualBlock = self.residual_conv_block
        Attention = None

        # get parameters for model layers
        no_scale_bn_params = self.get_bn_params(scale=False)
        bn_params = self.get_bn_params()
        conv_params = self.get_conv_params()
        init_filters = 64

        # resnet bottom
        x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
        x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
        x = BatchNormalization(name='bn0', **bn_params)(x)
        x = Activation('relu', name='relu0')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

        # resnet body
        for stage, rep in enumerate((2, 2, 2, 2)):
            for block in range(rep):

                filters = init_filters * (2 ** stage)

                # first block of first stage without strides because we have maxpooling before
                if block == 0 and stage == 0:
                    x = ResidualBlock(filters, stage, block, strides=(1, 1),
                                      cut='post', attention=Attention)(x)

                elif block == 0:
                    x = ResidualBlock(filters, stage, block, strides=(2, 2),
                                      cut='post', attention=Attention)(x)

                else:
                    x = ResidualBlock(filters, stage, block, strides=(1, 1),
                                      cut='pre', attention=Attention)(x)

        x = BatchNormalization(name='bn1', **bn_params)(x)
        x = Activation('relu', name='relu1')(x)

        x = GlobalAveragePooling2D(name='pool1')(x)
        x = Flatten()(x)

        classification = Dense(_num_classes, activation='softmax')(x)

        model = Model(img_input, classification, name='resnet18')

        return model

    # -------------------------------------------------------------------------
    #   Helpers functions
    # -------------------------------------------------------------------------

    def handle_block_names(self, stage, block):
        name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
        conv_name = name_base + 'conv'
        bn_name = name_base + 'bn'
        relu_name = name_base + 'relu'
        sc_name = name_base + 'sc'
        return conv_name, bn_name, relu_name, sc_name

    def get_conv_params(self, **params):
        default_conv_params = {
            'kernel_initializer': 'he_uniform',
            'use_bias': False,
            'padding': 'valid',
        }
        default_conv_params.update(params)
        return default_conv_params

    def get_bn_params(self, **params):
        axis = 3 if backend.image_data_format() == 'channels_last' else 1
        default_bn_params = {
            'axis': axis,
            'momentum': 0.99,
            'epsilon': 2e-5,
            'center': True,
            'scale': True,
        }
        default_bn_params.update(params)
        return default_bn_params

    # -------------------------------------------------------------------------
    #   Residual blocks
    # -------------------------------------------------------------------------

    def residual_conv_block(self, filters, stage, block, strides=(1, 1), attention=None, cut='pre'):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            cut: one of 'pre', 'post'. used to decide where skip connection is taken
        # Returns
            Output tensor for the block.
        """

        def layer(input_tensor):

            # get params and names of layers
            conv_params = self.get_conv_params()
            bn_params = self.get_bn_params()
            conv_name, bn_name, relu_name, sc_name = self.handle_block_names(stage, block)

            x = BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
            x = Activation('relu', name=relu_name + '1')(x)

            # defining shortcut connection
            if cut == 'pre':
                shortcut = input_tensor
            elif cut == 'post':
                shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
            else:
                raise ValueError('Cut type not in ["pre", "post"]')

            # continue with convolution layers
            x = ZeroPadding2D(padding=(1, 1))(x)
            x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

            x = BatchNormalization(name=bn_name + '2', **bn_params)(x)
            x = Activation('relu', name=relu_name + '2')(x)
            x = ZeroPadding2D(padding=(1, 1))(x)
            x = Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

            # use attention block if defined
            if attention is not None:
                x = attention(x)

            # add residual connection
            x = Add()([x, shortcut])
            return x

        return layer

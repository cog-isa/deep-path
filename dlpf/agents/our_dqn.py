import keras, logging, itertools
import keras.backend as K
from keras.layers import Dense, Flatten, Dropout, Reshape, \
    Convolution2D, MaxPooling2D, AveragePooling2D, \
    BatchNormalization, Activation

from .base import BaseKerasAgent


logger = logging.getLogger()


class OneLayerAgent(BaseKerasAgent):
    def _build_inner_model(self, input_layer):
        h = Flatten()(input_layer)
        return h


class TwoLayerAgent(BaseKerasAgent):
    def __init__(self,
                 hidden_size = 10,
                 hidden_activation = 'relu',
                 dropout = 0.8,
                 *args, **kwargs):
        self.hidden_size = hidden_size
        self.hidden_activation = hidden_activation
        self.dropout = dropout
        super(TwoLayerAgent, self).__init__(*args, **kwargs)

    def _build_inner_model(self, input_layer):
        h = Flatten()(input_layer)
        h = Dense(self.hidden_size,
                  activation = self.hidden_activation)(h)
        h = Dropout(self.dropout)(h)
        return h


class ConvAndDenseAgent(BaseKerasAgent):
    def __init__(self,
                 hidden_sizes = [10],
                 hidden_activations = ['relu'],
                 hidden_dropouts = [0.2],
                 hidden_batchnorm = [False],
                 conv_cores = [10],
                 conv_core_sizes = [4],
                 conv_strides = [2],
                 conv_activations = ['relu'],
                 conv_dropouts = [0.2],
                 conv_pooling = ['max'],
                 conv_batchnorm = [False],
                 *args, **kwargs):
        self.hidden_sizes = hidden_sizes
        self.hidden_activations = hidden_activations
        self.hidden_dropouts = hidden_dropouts
        self.hidden_batchnorm = hidden_batchnorm
        self.conv_cores = conv_cores
        self.conv_core_sizes = conv_core_sizes
        self.conv_strides = conv_strides
        self.conv_activations = conv_activations
        self.conv_dropouts = conv_dropouts
        self.conv_pooling = conv_pooling
        self.conv_batchnorm = conv_batchnorm
        super(ConvAndDenseAgent, self).__init__(*args, **kwargs)

    def _build_inner_model(self, input_layer):
        h = input_layer

        is_theano = K.image_dim_ordering() == 'th'
        if len(self.input_shape) == 2:
            if is_theano: # theano is (layers, rows, cols)
                reshape_input_to = (1,) + self.input_shape
            else:
                reshape_input_to = self.input_shape + (1,) # tensorflow is (rows, cols, layers)
            h = Reshape(reshape_input_to)(h)

        bn_axis = 0 if is_theano else 2
        for cores, core_size, stride, act, dropout, pool, bn in itertools.izip_longest(self.conv_cores,
                                                                                       self.conv_core_sizes,
                                                                                       self.conv_strides,
                                                                                       self.conv_activations,
                                                                                       self.conv_dropouts,
                                                                                       self.conv_pooling,
                                                                                       self.conv_batchnorm):
            h = Convolution2D(cores,
                              core_size,
                              core_size,
                              subsample = (stride, stride))(h)
            if not dropout is None and 0 < dropout < 1:
                h = Dropout(dropout)(h)
            if bn:
                h = BatchNormalization(axis = bn_axis)(h)
            if pool in ('max', 'min'):
                h = (MaxPooling2D if pool == 'max' else AveragePooling2D)()(h)
            h = Activation(act)(h)
        h = Flatten()(h)

        for size, act, dropout, bn in itertools.izip_longest(self.hidden_sizes,
                                                             self.hidden_activations,
                                                             self.hidden_dropouts,
                                                             self.hidden_batchnorm):
            h = Dense(size)(h)
            if not dropout is None and 0 < dropout < 1:
                h = Dropout(dropout)(h)
            if bn:
                h = BatchNormalization()(h)
            h = Activation(act)(h)

        return h

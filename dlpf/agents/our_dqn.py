import keras
from keras.layers import Dense, Flatten, Dropout, Reshape, Convolution2D

from .base import BaseKerasAgent


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


class OneConvPlusTwoLayerAgent(BaseKerasAgent):
    def __init__(self,
                 hidden_size = 10,
                 hidden_activation = 'relu',
                 convolution_cores = 10,
                 convolution_core_size = 4,
                 convolution_activation = 'relu',
                 dropout1 = 0.8,
                 dropout2 = 0.8,
                 *args, **kwargs):
        self.convolution_cores = convolution_cores
        self.convolution_core_size = convolution_core_size
        self.convolution_activation = convolution_activation
        self.hidden_size = hidden_size
        self.hidden_activation = hidden_activation
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        super(OneConvPlusTwoLayerAgent, self).__init__(*args, **kwargs)

    def _build_inner_model(self, input_layer):
        h = Reshape((1, input_layer.output_shape))(input_layer)
        h = Convolution2D(self.convolution_cores,
                          self.convolution_core_size, self.convolution_core_size,
                          input_shape=(1, input_layer.output_shape),
                          activation=self.convolution_activation)(h)
        h = Flatten()(h)
        h = Dropout(self.dropout1)(h)
        h = Dense(self.hidden_size,
                  activation = self.hidden_activation)(h)
        h = Dropout(self.dropout2)(h)
        return h
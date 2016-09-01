import keras
from keras.layers import Dense

from .base import BaseKerasAgent


class OneLayerAgent(object):
    def _build_inner_model(self, input_layer):
        h = Flatten()(input_layer)
        return h


class TwoLayerAgent(object):
    def __init__(self,
                 hidden_size = 10,
                 hidden_activation = 'relu',
                 dropout = 0.8,
                 *args, **kwargs):
        super(BaseKerasAgent, self).__init__(*args, **kwargs)
        self.hidden_size = 10
        self.hidden_activation = hidden_activation
        self.dropout = dropout

    def _build_inner_model(self, input_layer):
        h = Flatten()(input_layer)
        h = Dense(self.hidden_size,
                  activation = self.hidden_activation)(h)
        h = Dropout(self.dropout)(h)
        return h

import logging

from scipy.ndimage.interpolation import zoom

from .architectures import OneLayer, TwoLayer, ConvAndDense, DeepPreproc, \
    Inception
from .base import BaseStandaloneKerasAgent

logger = logging.getLogger()


###############################################################################
############################# Standalone agents ###############################
###############################################################################
class OneLayerAgent(OneLayer, BaseStandaloneKerasAgent):
    pass


class TwoLayerAgent(TwoLayer, BaseStandaloneKerasAgent):
    pass


class ConvAndDenseAgent(ConvAndDense, BaseStandaloneKerasAgent):
    pass


class DeepPreprocAgent(DeepPreproc, BaseStandaloneKerasAgent):
    def _predict_action_probabilities(self, observation):
        return self.model.predict(zoom(observation.reshape((1,) + observation.shape), self.scale_factor))


class InceptionAgent(Inception, BaseStandaloneKerasAgent):
    pass

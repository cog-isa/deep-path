import logging

from scipy.ndimage.interpolation import zoom

from .architectures import OneLayer, TwoLayer, ConvAndDense, DeepPreproc, \
    Inception
from .base import BaseStandaloneKerasAgent
from .ranking import BasePointwiseRankingAgent, BasePairwiseRankingAgent, \
    SimpleMaxValueRankingAgent

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


###############################################################################
########################## Pointwise ranking agents ###########################
###############################################################################
class TwoLayerPointwiseAgent(TwoLayer, BasePointwiseRankingAgent):
    pass


class ConvAndDensePointwiseAgent(ConvAndDense, BasePointwiseRankingAgent):
    pass


class DeepPreprocPointwiseAgent(DeepPreproc, BasePointwiseRankingAgent):
    pass


class InceptionPointwiseAgent(Inception, BasePointwiseRankingAgent):
    pass


###############################################################################
########################### Pairwise ranking agents ###########################
###############################################################################
class TwoLayerPairwiseAgent(TwoLayer, BasePairwiseRankingAgent):
    pass


class ConvAndDensePairwiseAgent(ConvAndDense, BasePairwiseRankingAgent):
    pass


class DeepPreprocPairwiseAgent(DeepPreproc, BasePairwiseRankingAgent):
    pass


class InceptionPairwiseAgent(Inception, BasePairwiseRankingAgent):
    pass


###############################################################################
######################## Greedy agents without learning #######################
###############################################################################
class SimpleMaxValueRankingNoLearningAgent(OneLayer, SimpleMaxValueRankingAgent):
    pass

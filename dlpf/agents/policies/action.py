import logging
import numpy
import random

logger = logging.getLogger()


class BaseActionPolicy(object):
    def reset(self):
        pass

    def new_episode(self):
        pass

    def choose_action(self, actions):
        raise NotImplemented()


class PolynomiallyAnnealedPolicyMixin(object):
    def __init__(self, eps=0.1, final_eps=0.01, episodes_number=5000, degree=2, *args, **kwargs):
        self.init_eps = eps
        self.final_eps = final_eps
        self.episodes_number = episodes_number
        self.degree = degree
        self.amp = (self.init_eps - self.final_eps) / (self.episodes_number ** self.degree)
        self.episode_i = 0

        super(PolynomiallyAnnealedPolicyMixin, self).__init__(eps=eps, *args, **kwargs)

        logger.debug('PolynomiallyAnnealedPolicyMixin: eps=%s' % [(i, self._calc_eps(i)) for i in
                                                                  xrange(0, self.episodes_number,
                                                                         self.episodes_number / 10)])

    def reset(self):
        self.eps = self.init_eps

    def new_episode(self):
        self.episode_i += 1
        self.eps = self._calc_eps(self.episode_i)
        logger.debug('PolynomiallyAnnealedPolicyMixin: current eps=%f' % self.eps)

    def _calc_eps(self, episode_i):
        if episode_i <= self.episodes_number:
            return self.amp * (abs(episode_i - self.episodes_number) ** self.degree) + self.final_eps
        else:
            return self.final_eps


class EpsilonGreedyPolicy(BaseActionPolicy):
    def __init__(self, eps=0.05, rand=None):
        self.eps = eps
        self.rand = rand or random.Random()

    def choose_action(self, action_probabilities):
        if self.rand.random() < self.eps:
            max_variant = action_probabilities.shape[-1] - 1
            logger.debug('EpsilonGreedyPolicy: make random decision from %d variants' % (max_variant + 1))
            return self.rand.randint(0, max_variant)
        else:
            logger.debug('EpsilonGreedyPolicy: make maximum probability decision')
            return action_probabilities.argmax()


class AnnealedEpsilonGreedyPolicy(PolynomiallyAnnealedPolicyMixin, EpsilonGreedyPolicy):
    pass


def softmax_with_temperature(p, t):
    p = p / t
    e_x = numpy.exp(p - numpy.max(p))
    return e_x / e_x.sum()


class SoftmaxSamplePolicy(BaseActionPolicy):
    def __init__(self, eps=0.7):
        self.eps = eps

    def choose_action(self, action_probabilities):
        action_probabilities = numpy.asarray(action_probabilities).reshape(-1)
        smoothed_probs = softmax_with_temperature(action_probabilities,
                                                  self.eps)
        logger.debug('SoftmaxSamplePolicy: sampling from %s' % ', '.join('%.3f' % p for p in smoothed_probs))
        return numpy.random.choice(range(action_probabilities.shape[-1]),
                                   p=smoothed_probs)


class AnnealedSoftmaxSamplePolicy(PolynomiallyAnnealedPolicyMixin, SoftmaxSamplePolicy):
    pass


class IdentityPolicy(BaseActionPolicy):
    def choose_action(self, actions):
        return actions


_ACTION_POLICIES = {
    'epsilon_greedy': EpsilonGreedyPolicy,
    'annealed_epsilon_greedy': AnnealedEpsilonGreedyPolicy,
    'softmax_sample': SoftmaxSamplePolicy,
    'annealed_softmax_sample': AnnealedSoftmaxSamplePolicy,
    'identity': IdentityPolicy
}
DEFAULT_ACTION_POLICY = 'epsilon_greedy'


def get_available_action_policies():
    return list(_ACTION_POLICIES.keys())


def get_action_policy(ctor=DEFAULT_ACTION_POLICY, *args, **kwargs):
    assert ctor in _ACTION_POLICIES, 'Unknown action policy %s' % ctor
    return _ACTION_POLICIES[ctor](*args, **kwargs)

import random


class BaseActionPolicy(object):
    def reset(self):
        pass
    
    def new_episode(self):
        pass
    
    def choose_action(self, actions):
        raise NotImplemented()


class EpsilonGreedyPolicy(BaseActionPolicy):
    def __init__(self, eps = 0.05, rand = None):
        self.eps = eps
        self.rand = rand or random.Random()

    def choose_action(self, action_probabilities):
        if self.rand.random() < self.eps:
            return self.rand.randint(0, len(action_probabilities))
        else:
            return action_probabilities.argmax()


class AnnealedEpsilonGreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self, eps = 0.1, decrease_coef = 0.99, rand = None):
        super(AnnealedEpsilonPolicy, self).__init__(eps, rand = rand)
        self.init_eps = eps
        self.decrease_coef = decrease_coef

    def reset(self):
        self.eps = self.init_eps

    def new_episode(self):
        self.eps *= self.decrease_coef


_ACTION_POLICIES = {
    'epsilon_greedy' : EpsilonGreedyPolicy,
    'annealed_epsilon_greedy' : AnnealedEpsilonGreedyPolicy,
}
DEFAULT_ACTION_POLICY = 'epsilon_greedy'

def get_available_action_policies():
    return list(_ACTION_POLICIES.keys())

def get_action_policy(ctor = DEFAULT_ACTION_POLICY, *args, **kwargs):
    assert ctor in _ACTION_POLICIES, 'Unknown action policy %s' % ctor
    return _ACTION_POLICIES[ctor](*args, **kwargs)

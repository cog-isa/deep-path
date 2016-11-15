import collections, itertools, numpy
from scipy.spatial.distance import euclidean

from .base import BaseKerasAgent, split_train_val_replay_gens, MemoryRecord


def sort_episode_steps(episode):
    # where did we go from each state?
    forward_links = collections.defaultdict(set)
    for step in episode.viewvalues():
        if not step.observation.prev_id is None:
            forward_links[step.observation.prev_id].add(step.observation._id)

    # nodes without continuation
    queue = collections.deque(state_id
                              for state_id, next_states
                              in forward_links.viewitems()
                              if len(next_states) == 0)
    chains = { state_id : (state_id, 0)
              for state_id in queue }
    winning_chains = { state_id
                      for state_id in queue
                      if episode[state_id].done }
    while len(queue) > 0:
        state_id = queue.popleft()
        chain_id, number_from_end = chains[state_id]
        prev_state_id = episode[state_id].observation._id
        if not prev_state_id is None:
            queue.append(prev_state_id)
            chains[prev_state_id] = (chain_id, number_from_end + 1)

    def min_distance_to_winning_state(s):
        if winning_chains:
            return min(euclidean(s, w) for w in winning_chains)
        return 0

    def compare_states(s1, s2):
        c1, i1 = chains[s1]
        c2, i2 = chains[s2]
        if c1 == c2:
            return i2 - i1 # closer to the end of chain (i is less) - bigger the state
        if c1 in winning_chains:
            return 1 # c1 should go right
        if c2 in winning_chains:
            return -1 # c2 should go right
        d1 = min_distance_to_winning_state(s1)
        d2 = min_distance_to_winning_state(s2)
        return d2 - d1

    result = chains.keys()
    result.sort(cmp = compare_states)
    return result


class EpisodeWithPreparedInfo(object):
    def __init__(self):
        self.all_states = {}
        self.prepared_info = None


class BaseRankingAgent(BaseKerasAgent):
    def __init__(self):
        super(BaseRankingAgent, self).__init__()
        self.number_of_actions = 1

    def _init_memory_for_new_episode():
        return EpisodeWithPreparedInfo()

    def _update_memory(self, episode_memory, observation = None, action_probabilities = None, action = None, reward = None, done = None):
        for node in observation:
            episode_memory.all_states[node._id] = MemoryRecord(node, None, None, done)

    def _gen_train_val_data_from_memory(self):
        for episode in self.memory:
            if episode.prepared_info is None:
                episode.prepared_info = self._prepare_episode_info(episode.all_states)

        return split_train_val_replay_gens([episode.prepared_info for episode in self.memory],
                                           self.batch_size,
                                           self.number_of_actions,
                                           val_part = self.validation_part,
                                           output_type = self.train_data_output_type,
                                           rand = self.split_rand)


def linear_weight(i, n):
    return float(i) / n

_WEIGHTING_FUNCTIONS = {
    'linear' : linear_weight
}
DEFAULT_WEIGHTING = 'linear'


class BasePointwiseRankingAgent(BaseRankingAgent):
    def __init__(self,
                 weighting_function = DEFAULT_WEIGHTING,
                 *args, **kwargs):
        super(PointwiseRankingAgent, self).__init__(*args, **kwargs)
        self.weighting_function = _WEIGHTING_FUNCTIONS.get(weighting_function, None)
        if self.weighting_function is None:
            self.weighting_function = _WEIGHTING_FUNCTIONS[DEFAULT_WEIGHTING]

    def _prepare_episode_info(self, episode_states):
        sorted_state_ids = sort_episode_steps(episode)
        n = len(sorted_state_ids)
        return [MemoryRecord(state_info.observation.viewport, 0, self.weighting_function(i, n), state_info.done)
                for i, state_id in enumerate(sorted_state_ids)
                for state_info in [episode_states[state_id]] ]


class BasePairwiseRankingAgent(BaseRankingAgent):
    def __init__(self, *args, **kwargs):
        super(BasePairwiseRankingAgent, self).__init__(*args, **kwargs)
        self.input_shape = (2,) + self.input_shape

    def _prepare_episode_info(self, episode_states):
        sorted_state_ids = sort_episode_steps(episode)
        return [ MemoryRecord(numpy.stack([s1_info.observation.viewport,
                                           s2_info.observation.viewport]),
                              0,
                              self.weighting_function(i, n) - self.weighting_function(j, n),
                              state_info.done)
                for i, s1_id in enumerate(sorted_state_ids)
                for j, s2_id in enumerate(sorted_state_ids)
                if i != j
                for s1_info in [ episode_states[s1_id] ]
                for s2_info in [ episode_states[s2_id] ] ]

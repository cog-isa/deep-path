import collections, itertools, numpy, json
from scipy.spatial.distance import euclidean

from .base import BaseKerasAgent, split_train_val_replay_gens, MemoryRecord


def sort_episode_steps(episode):
    # where did we go from each state?
    forward_links = { step.observation.cur_id : set() for step in episode.viewvalues() }
    for step in episode.viewvalues():
        if not step.observation.prev_id is None:
            forward_links[step.observation.prev_id].add(step.observation.cur_id)

    # nodes without continuation
    queue = collections.deque(state_id
                              for state_id, next_states
                              in forward_links.viewitems()
                              if len(next_states) == 0)
    chains = collections.defaultdict(dict)
    for state_id in queue:
        chains[state_id][state_id] = 0
    winning_chains = { state_id
                      for state_id in queue
                      if episode[state_id].done }

    while len(queue) > 0:
        state_id = queue.popleft()
        cur_chains_info = chains[state_id]
        prev_state_id = episode[state_id].observation.prev_id
        if not prev_state_id is None:
            queue.append(prev_state_id)
            prev_state_chains_info = chains[prev_state_id]
            for chain_id, number_from_end in cur_chains_info.viewitems():
                prev_state_chains_info[chain_id] = number_from_end + 1

    def min_distance_to_winning_state(s):
        if winning_chains:
            return min(euclidean(s, w) for w in winning_chains)
        return 0

    def compare_states(s1, s2):
        chains1 = chains[s1]
        chains2 = chains[s2]
        common_chain_ids = { k for k in chains1.viewkeys() if k in chains2 }

        if len(common_chain_ids) > 0:
            i1 = min(chains1[k] for k in common_chain_ids)
            i2 = min(chains2[k] for k in common_chain_ids)
            return i2 - i1 # closer to the end of chain (i is less) - bigger the state

        if len(winning_chains.intersection(chains1.viewkeys())) > 0:
            return 1 # c1 should go right

        if len(winning_chains.intersection(chains2.viewkeys())) > 0:
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

    def __len__(self):
        # needed to calculate number of train samples in memory
        return len(self.prepared_info)


def linear_weight(i, n):
    return float(i) / n

_WEIGHTING_FUNCTIONS = {
    'linear' : linear_weight
}
DEFAULT_WEIGHTING = 'linear'


class BaseRankingAgent(BaseKerasAgent):
    def __init__(self,
                 weighting_function = DEFAULT_WEIGHTING,
                 *args, **kwargs):
        super(BaseRankingAgent, self).__init__(*args, **kwargs)
        self.number_of_actions = 1
        self.weighting_function = _WEIGHTING_FUNCTIONS.get(weighting_function, None)
        if self.weighting_function is None:
            self.weighting_function = _WEIGHTING_FUNCTIONS[DEFAULT_WEIGHTING]

    def _init_memory_for_new_episode(self):
        return EpisodeWithPreparedInfo()

    def _update_memory(self, episode_memory, observation = None, action_probabilities = None, action = None, reward = None, done = None):
        for node in observation:
            if not node.cur_id in episode_memory.all_states:
                episode_memory.all_states[node.cur_id] = MemoryRecord(node, None, None, done)

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


class BasePointwiseRankingAgent(BaseRankingAgent):
    def _predict_action_probabilities(self, observation):
        all_observations = numpy.stack([node.viewport for node in observation])
        ratings = self.model.predict(all_observations)
        return { node.cur_id : rating for node, rating in itertools.izip(observation, ratings[:, 0]) }

    def _prepare_episode_info(self, episode_states):
        sorted_state_ids = sort_episode_steps(episode_states)
        n = len(sorted_state_ids)
        return [MemoryRecord(state_info.observation.viewport, 0, self.weighting_function(i, n), state_info.done)
                for i, state_id in enumerate(sorted_state_ids)
                for state_info in [episode_states[state_id]] ]


class BasePairwiseRankingAgent(BaseRankingAgent):
    def __init__(self, *args, **kwargs):
        super(BasePairwiseRankingAgent, self).__init__(*args, **kwargs)
        self.input_shape = (2,) + self.input_shape

    def _predict_action_probabilities(self, observation):
        unique_pairs = [(n1, n2)
                        for i, n1 in enumerate(observation)
                        for n2 in observation[i+1:] ]
        all_samples = numpy.stack([(n1.viewport, n2.viewport) for n1, n2 in unique_pairs])
        comparisons_raw = self.model.predict(all_samples)
        comparisons = { (n1.cur_id, n2.cur_id) : cmp_value
                       for (n1, n2), cmp_value
                       in itertools.izip(unique_pairs, comparisons_raw) }
        result = [node.cur_id for node in observation]

        def precalculated_comparator(s1_id, s2_id):
            direct_cmp = comparisons.get((s1_id, s2_id), None)
            if not direct_cmp is None:
                return direct_cmp
            return -comparisons[(s2_id, s1_id)]

        result.sort(cmp = precalculated_comparator)
        n = len(result)
        return { state_id : self.weighting_function(i, n)
                for i, state_id
                in enumerate(result) }

    def _prepare_episode_info(self, episode_states):
        sorted_state_ids = sort_episode_steps(episode_states)
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


class SimpleMaxValueRankingAgent(BaseRankingAgent):
    def _predict_action_probabilities(self, observation):
        return { node.cur_id : node.viewport.max()
                for node in observation }

    def _prepare_episode_info(self, episode_states):
        return []

    def train_on_memory(self):
        pass

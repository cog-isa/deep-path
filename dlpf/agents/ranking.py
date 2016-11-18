import collections, itertools, numpy, json, logging, pandas
from scipy.spatial.distance import euclidean

from .base import BaseKerasAgent, split_train_val_replay_gens, MemoryRecord
from ..keras_utils import get_backend

logger = logging.getLogger(__name__)


_PERCENTILES = range(20, 101, 20)

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
                      if episode[state_id].observation.goal }
    unique_chain_ids = set(chains.viewkeys())
    winning_chains_number = len(winning_chains)

    while len(queue) > 0:
        state_id = queue.popleft()
        cur_chains_info = chains[state_id]
        prev_state_id = episode[state_id].observation.prev_id
        if not prev_state_id is None:
            queue.append(prev_state_id)
            prev_state_chains_info = chains[prev_state_id]
            for chain_id, number_from_end in cur_chains_info.viewitems():
                prev_state_chains_info[chain_id] = number_from_end + 1

    # Calculate number of chains that were terminated.
    # These are chains that are:
    # * not winning
    # * have at least two nodes after last bifurcation (one really visited and one just considered)
    # This is simply the list of unique nodes that are previous for chain ends
    terminated_chains = set() # identifiers of last visited nodes (last bifurcation points)
    for state_id in unique_chain_ids:
        prev_id = episode[state_id].observation.prev_id
        if not prev_id is None:
            terminated_chains.add(prev_id)

    # also, calculate quantiles of ratio chain_len / winning_chain_len
    initial_states = { node.observation.cur_id
                      for node in episode.viewvalues()
                      if node.observation.prev_id is None }
    winning_chain_size = float(max(chains[init_state_id][win_chain_id]
                                   for init_state_id in initial_states
                                   for win_chain_id in winning_chains))
    chain_length_ratios = [chain_size / winning_chain_size
                           for init_state_id in initial_states
                           for _, chain_size in chains[init_state_id].viewitems()]
    chain_length_percentiles = numpy.percentile(chain_length_ratios,
                                                _PERCENTILES)

    stats = {'winning_chains_number' : winning_chains_number,
             'terminated_chains_ratio' : float(len(terminated_chains)) / len(unique_chain_ids) }
    stats.update(('chain_length_percentile_%.2f' % perc, perc_value)
                 for perc, perc_value in itertools.izip(_PERCENTILES, chain_length_percentiles))

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
        if d1 == d2:
            return 0
        elif d2 > d1:
            return 1
        else:
            return -1

    result = chains.keys()
    result.sort(cmp = compare_states)
    return result, stats


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
        self._chains_stat = []

    def _init_memory_for_new_episode(self):
        return EpisodeWithPreparedInfo()

    def _update_memory(self, episode_memory, observation = None, action_probabilities = None, action = None, reward = None, done = None):
        #logger.info('Done %r' % done)
        #logger.info('obs %s' % [(n.cur_id, n.goal) for n in observation])
        for node in observation:
            episode_memory.all_states[node.cur_id] = MemoryRecord(node, None, None, done)

    def _gen_train_val_data_from_memory(self):
        self._chains_stat = []

        for episode in self.memory:
            if episode.prepared_info is None:
                episode.prepared_info, episode_stats = self._prepare_episode_info(episode.all_states)
                self._chains_stat.append(episode_stats)

        return split_train_val_replay_gens([episode.prepared_info for episode in self.memory],
                                           self.batch_size,
                                           self.number_of_actions,
                                           val_part = self.validation_part,
                                           output_type = self.train_data_output_type,
                                           rand = self.split_rand)

    def get_episode_stat(self):
        result = super(BaseRankingAgent, self).get_episode_stat()
        result.update(pandas.DataFrame(data = self._chains_stat).mean(axis = 0))
        return result


class BasePointwiseRankingAgent(BaseRankingAgent):
    def _predict_action_probabilities(self, observation):
        all_observations = numpy.stack([node.viewport for node in observation])
        ratings = self.model.predict(all_observations)
        return { node.cur_id : rating for node, rating in itertools.izip(observation, ratings[:, 0]) }

    def _prepare_episode_info(self, episode_states):
        sorted_state_ids, stats = sort_episode_steps(episode_states)
        n = len(sorted_state_ids)
        result = [MemoryRecord(state_info.observation.viewport, 0, self.weighting_function(i, n), state_info.done)
                  for i, state_id in enumerate(sorted_state_ids)
                  for state_info in [episode_states[state_id]] ]
        return result, stats


class BasePairwiseRankingAgent(BaseRankingAgent):
    def __init__(self, *args, **kwargs):
        super(BasePairwiseRankingAgent, self).__init__(*args, **kwargs)
        self._comparison_cache = {}

    def new_episode(self, *args, **kwargs):
        super(BasePairwiseRankingAgent, self).new_episode(*args, **kwargs)
        self._comparison_cache = {}

    def _build_model(self):
        if get_backend() == 'tf':
            self.input_shape = self.input_shape + (2,)
        else:
            self.input_shape = (2,) + self.input_shape
        super(BasePairwiseRankingAgent, self)._build_model()

    def _predict_action_probabilities(self, observation):
        result = [node.cur_id for node in observation]

        if len(observation) > 1:
            new_unique_pairs = [(n1, n2)
                                for i, n1 in enumerate(observation)
                                for n2 in observation[i+1:]
                                if not (n1.cur_id, n2.cur_id) in self._comparison_cache
                                 and not (n2.cur_id, n1.cur_id) in self._comparison_cache]
            if len(new_unique_pairs) > 0:
                new_samples = numpy.stack([(n1.viewport, n2.viewport) for n1, n2 in new_unique_pairs])
                new_comparisons_raw = self.model.predict(new_samples)
                self._comparison_cache.update(((n1.cur_id, n2.cur_id), cmp_value)
                                              for (n1, n2), cmp_value
                                              in itertools.izip(new_unique_pairs,
                                                                new_comparisons_raw[:, 0]))

            result.sort(cmp = self._compare_with_cache)

        n = len(result)
        return { state_id : self.weighting_function(i, n)
                for i, state_id
                in enumerate(result) }

    def _compare_with_cache(self, s1_id, s2_id):
        result = self._comparison_cache.get((s1_id, s2_id), None)
        if result is None:
            result = -self._comparison_cache[(s2_id, s1_id)]
        if result > 0:
            return 1
        elif result < 0:
            return -1
        else:
            return 0

    def _prepare_episode_info(self, episode_states):
        sorted_state_ids, stats = sort_episode_steps(episode_states)
        n = len(episode_states)

        if get_backend() == 'tf':
            reshape = lambda m: numpy.moveaxis(m, 0, -1) # tensorflow is (rows, cols, layers)
        else:
            reshape = lambda m: m

        result = [ MemoryRecord(reshape(numpy.stack([s1_info.observation.viewport,
                                                     s2_info.observation.viewport])),
                                0,
                                self.weighting_function(i, n) - self.weighting_function(j, n),
                                None)
                  for i, s1_id in enumerate(sorted_state_ids)
                  for j, s2_id in enumerate(sorted_state_ids)
                  for s1_info in [ episode_states[s1_id] ]
                  for s2_info in [ episode_states[s2_id] ] ]
        return result, stats


class SimpleMaxValueRankingAgent(BaseRankingAgent):
    def _predict_action_probabilities(self, observation):
        return { node.cur_id : -euclidean(node.cur_id, self.goal) for node in observation }

    def _prepare_episode_info(self, episode_states):
        sorted_state_ids, stats = sort_episode_steps(episode_states)
        return [], stats

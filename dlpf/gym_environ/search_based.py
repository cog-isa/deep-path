import collections, gym, logging

from .base import BasePathFindingEnv, InfoValues
from .flat import WithDistanceMapMixin, FlatObservationMixin
from .search_algo import get_search_algo, DEFAULT_SEARCH_ALGO


logger = logging.getLogger(__name__)


VisitedNodeInfo = collections.namedtuple('VisitedNodeInfo',
                                         'cur_id prev_id viewport goal'.split(' '))


class MapSpace(gym.Space):
    def __init__(self, shape):
        self.shape = shape

    def sample(self):
        return {}

    def contains(self, x):
        return True


class BaseSearchBasedPathFindingEnv(BasePathFindingEnv):
    action_space = gym.spaces.Discrete(1)

    def __init__(self):
        super(BaseSearchBasedPathFindingEnv, self).__init__()
        self._searcher = None
        self._visited_nodes = None
        self._considered_nodes = None
        self.max_positions_to_consider = None

    def _configure(self,
                   search_algo = DEFAULT_SEARCH_ALGO,
                   max_positions_to_consider = 1000,
                   *args, **kwargs):
        super(BaseSearchBasedPathFindingEnv, self)._configure(*args, **kwargs)
        self._searcher = get_search_algo(search_algo)
        self.max_positions_to_consider = max_positions_to_consider

    def _init_state(self):
        self._searcher.reset(self.cur_task.local_map,
                             self.path_policy.get_start_position(),
                             self.path_policy.get_global_goal())
        self._visited_nodes = set()
        init_pos = self.path_policy.get_start_position()
        self._considered_nodes = { init_pos : VisitedNodeInfo(init_pos,
                                                              None,
                                                              self._get_base_state(init_pos),
                                                              init_pos == self.path_policy.get_global_goal()) }
        return super(BaseSearchBasedPathFindingEnv, self)._init_state()

    def _get_state(self):
        return [n for n in self._considered_nodes.viewvalues()
                if not n.cur_id in self._visited_nodes]

    def _step(self, action):
        self._searcher.update_ratings(action)
        search_res = self._searcher.step()

        done = self._searcher.goal_achieved()
        info = InfoValues.DONE if done else InfoValues.OK

        if search_res.must_continue:
            goal = self.path_policy.get_global_goal()
            self._visited_nodes.add(search_res.best_next)
            self._considered_nodes.update((new_pos,
                                           VisitedNodeInfo(new_pos,
                                                           search_res.best_next,
                                                           self._get_base_state(new_pos),
                                                           new_pos == goal))
                                          for new_pos, _
                                          in search_res.new_variants_with_ratings)
        else:
            if not done:
                info = InfoValues.NOTHING

        return self._get_state(), 0.0, done, info

    def _get_base_state(self, position):
        raise NotImplemented()


class FlatSearchBasedPathFindingEnv(WithDistanceMapMixin, FlatObservationMixin, BaseSearchBasedPathFindingEnv):
    def _get_new_agent_positions(self):
        return self._visited_nodes

    def _get_observation_space(self, map_shape):
        return MapSpace(self._get_flat_observation_shape(map_shape))

import collections, gym, logging, numpy

from .base import BasePathFindingEnv, InfoValues
from .flat import WithDistanceMapMixin, FlatObservationMixin
from .search_algo import get_search_algo, DEFAULT_SEARCH_ALGO
from ..plot_utils import scatter_plot
from ..keras_utils import get_tensor_reshaper, add_depth


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
        self._considered_nodes = None
        self.max_positions_to_consider = None

    def _configure(self,
                   search_algo = DEFAULT_SEARCH_ALGO,
                   max_positions_to_consider = 1000,
                   stack_previous_viewports = 0,
                   *args, **kwargs):
        self._searcher = get_search_algo(search_algo)
        self.max_positions_to_consider = max_positions_to_consider

        assert stack_previous_viewports >= 0
        self.stack_previous_viewports = stack_previous_viewports
        return super(BaseSearchBasedPathFindingEnv, self)._configure(*args, **kwargs)

    def _init_state(self):
        self._searcher.reset(self.cur_task.local_map,
                             self.path_policy.get_start_position(),
                             self.path_policy.get_global_goal())
        init_pos = self.path_policy.get_start_position()
        self._considered_nodes = {}
        self._considered_nodes = { init_pos : VisitedNodeInfo(init_pos,
                                                              None,
                                                              self._get_viewport_with_history(init_pos),
                                                              init_pos == self.path_policy.get_global_goal()) }
        return super(BaseSearchBasedPathFindingEnv, self)._init_state()

    def _get_state(self):
        visited = self._searcher.visited_nodes
        return [n for n in self._considered_nodes.viewvalues()
                if not n.cur_id in visited]

    def _step(self, action):
        self._searcher.update_ratings(action)
        search_res = self._searcher.step()

        done = self._searcher.goal_achieved()
        info = InfoValues.DONE if done else InfoValues.OK

        if search_res.must_continue:
            goal = self.path_policy.get_global_goal()
            self._considered_nodes.update((new_pos,
                                           VisitedNodeInfo(new_pos,
                                                           search_res.best_next,
                                                           self._get_viewport_with_history(new_pos, prev = search_res.best_next),
                                                           new_pos == goal))
                                          for new_pos, _
                                          in search_res.new_variants_with_ratings)
        else:
            if not done:
                info = InfoValues.NOTHING

        return self._get_state(), 0.0, done, info

    def _get_viewport_with_history(self, position, prev = None):
        result = [self._get_base_state(position)]
        empty = numpy.zeros_like(result[0])

        if prev is None and position in self._considered_nodes:
            prev = self._considered_nodes[position].prev_id

        for _ in xrange(self.stack_previous_viewports):
            if prev is None:
                result.append(empty)
            else:
                result.append(self._get_base_state(prev))
                prev = self._considered_nodes[prev].prev_id
        return get_tensor_reshaper()(numpy.stack(result))

    def _get_base_state(self, position):
        raise NotImplementedError()


class FlatSearchBasedPathFindingEnv(WithDistanceMapMixin, FlatObservationMixin, BaseSearchBasedPathFindingEnv):
    def _get_new_agent_positions(self):
        return self._searcher.visited_nodes

    def _get_observation_space(self, map_shape):
        return MapSpace(add_depth(self._get_flat_observation_shape(map_shape),
                                  depth = self.stack_previous_viewports + 1))

    def _visualize_episode(self, out_file):
        scatter_plot(({'label' : 'obstacle',
                       'data' : self.obstacle_points_for_vis,
                       'color' : 'black',
                       'marker' : 's'},
                      {'label' : 'considered',
                       'data' : [(x, y) for y, x in self._considered_nodes.viewkeys()],
                       'color' : 'gold',
                       'marker' : 'x'},
                      {'label' : 'visited',
                       'data' : [(x, y) for y, x in self._searcher.visited_nodes],
                       'color' : 'green',
                       'marker' : '.'},
                      {'label' : 'goal',
                       'data' : [reversed(self.path_policy.get_global_goal())],
                       'color' : 'red',
                       'marker' : '*'},
                      {'label' : 'start',
                       'data' : [reversed(self.path_policy.get_start_position())],
                       'color' : 'red',
                       'marker' : '^'}),
                     x_lim = (0, self.cur_task.local_map.shape[1]),
                     y_lim = (0, self.cur_task.local_map.shape[0]),
                     offset = (0.5, 0.5),
                     out_file = out_file)

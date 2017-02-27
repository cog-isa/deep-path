import gym.spaces
import logging
import numpy
from scipy.spatial.distance import euclidean

from .base import BasePathFindingByPixelEnv
from .utils_compiled import build_distance_map, get_flat_state
from ..keras_utils import get_tensor_reshaper, add_depth
from ..plot_utils import scatter_plot

logger = logging.getLogger(__name__)


class WithDistanceMapMixin(object):
    def __init__(self, *args, **kwargs):
        super(WithDistanceMapMixin, self).__init__(*args, **kwargs)
        self.distance_map = None
        self._unique_positions = None
        self._sum_reward = None

    def get_episode_stat(self):
        stat = super(WithDistanceMapMixin, self).get_episode_stat()
        if not self.distance_map is None:
            optimal_path_length = self.distance_map[self.path_policy.get_start_position()]
            if optimal_path_length > 0:
                real_path_length = float(len(self._unique_agent_positions))
                stat['path_len_rate'] = real_path_length / optimal_path_length
                #logger.info('path_len_rate %f, %d, %r' % (stat['path_len_rate'],
                #                                          optimal_path_length,
                #                                          self._unique_agent_positions))
            else:
                stat['path_len_rate'] = numpy.inf

            minimal_sum_reward = optimal_path_length + self._get_done_reward()
            stat['sum_reward_rate'] = self._sum_reward / minimal_sum_reward
        else:
            stat['path_len_rate'] = numpy.inf
            stat['sum_reward_rate'] = numpy.inf
        return stat

    def _init_state(self):
        self.distance_map = build_distance_map(numpy.array(self.cur_task.local_map, dtype = numpy.int),
                                               numpy.array(self.path_policy.get_global_goal(), dtype = numpy.int))
        self._unique_agent_positions = { self.path_policy.get_start_position() }
        self._sum_reward = 0.0
        return super(WithDistanceMapMixin, self)._init_state()

    def _get_usual_reward(self, old_position, new_position):
        old_height = self.distance_map[tuple(old_position)]
        new_height = self.distance_map[tuple(new_position)]
        true_gain = old_height - new_height

        local_target = self.path_policy.get_local_goal()
        old_dist = euclidean(old_position, local_target)
        new_dist = euclidean(new_position, local_target)
        greedy_gain = old_dist - new_dist

        start_height = self.distance_map[tuple(self.path_policy.get_start_position())]
        abs_gain = numpy.exp(-new_height / start_height)

        total_gain = sum(((1 - self.greedy_distance_reward_weight - self.absolute_distance_reward_weight) * true_gain,
                          self.greedy_distance_reward_weight * greedy_gain,
                          self.absolute_distance_reward_weight * abs_gain))
        logger.debug('true_gain %f, greedy gain %f, abs_gain %f, total %f' % (true_gain,
                                                                              greedy_gain,
                                                                              abs_gain,
                                                                              total_gain))
        return total_gain

    def _configure(self,
                   greedy_distance_reward_weight = 0.1,
                   absolute_distance_reward_weight = 0.1,
                   *args, **kwargs):
        self.greedy_distance_reward_weight = greedy_distance_reward_weight
        self.absolute_distance_reward_weight = absolute_distance_reward_weight
        super(WithDistanceMapMixin, self)._configure(*args, **kwargs)

    def _step(self, action):
        observation, reward, done, info = super(WithDistanceMapMixin, self)._step(action)
        self._sum_reward += abs(reward)
        self._unique_agent_positions.update(tuple(p) for p in self._get_new_agent_positions())
        return observation, reward, done, info
    
    def _get_new_agent_positions(self):
        raise NotImplemented()


class FlatObservationMixin(object):
    def _init_state(self):
        result = super(FlatObservationMixin, self)._init_state()
        m = self.cur_task.local_map
        self.obstacle_points_for_vis = [(x, y)
                                        for y in xrange(m.shape[0])
                                        for x in xrange(m.shape[1])
                                        if m[y, x] > 0]
        return result

    def _get_base_state(self, cur_position_discrete):
        return get_flat_state(self.cur_task.local_map,
                              tuple(cur_position_discrete),
                              self.vision_range,
                              self._get_done_reward(),
                              self.target_on_border_reward,
                              self.path_policy.get_start_position(),
                              self.path_policy.get_global_goal(),
                              self.absolute_distance_observation_weight)

    def _get_flat_observation_shape(self, map_shape):
        return (2 * self.vision_range + 1, 2 * self.vision_range + 1)

    def _configure(self,
                   vision_range = 10,
                   target_on_border_reward = 5,
                   absolute_distance_observation_weight = 0.1,
                   *args, **kwargs):
        self.vision_range = vision_range
        self.target_on_border_reward = target_on_border_reward
        self.absolute_distance_observation_weight = absolute_distance_observation_weight
        super(FlatObservationMixin, self)._configure(*args, **kwargs)


class PathFindingByPixelWithDistanceMapEnv(WithDistanceMapMixin, FlatObservationMixin, BasePathFindingByPixelEnv):
    def _init_state(self):
        self.cur_episode_state_id_seq = [tuple(self.path_policy.get_start_position())]
        return super(PathFindingByPixelWithDistanceMapEnv, self)._init_state()

    def _get_state(self):
        cur_pos = tuple(self.cur_position_discrete)
        if cur_pos != self.cur_episode_state_id_seq[-1]:
            self.cur_episode_state_id_seq.append(cur_pos)
        result = [self._get_base_state(pos)
                  for pos in self.cur_episode_state_id_seq[:-2-self.stack_previous_viewports:-1]]
        if len(result) < self.stack_previous_viewports + 1:
            empty = numpy.zeros_like(result[0])
            for _ in xrange(self.stack_previous_viewports + 1 - len(result)):
                result.append(empty)
        return get_tensor_reshaper()(numpy.stack(result))

    def _get_observation_space(self, map_shape):
        return gym.spaces.Box(low = 0,
                              high = 1,
                              shape = add_depth(self._get_flat_observation_shape(map_shape),
                                                depth = self.stack_previous_viewports + 1))

    def _get_new_agent_positions(self):
        return (self.cur_position_discrete, )

    def _visualize_episode(self, out_file):
        scatter_plot(({'label' : 'obstacle',
                       'data' : self.obstacle_points_for_vis,
                       'color' : 'black',
                       'marker' : 's'},
                      {'label' : 'path',
                       'data' : [(x, y) for y, x in self._unique_agent_positions],
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

    def _configure(self,
                   stack_previous_viewports = 0,
                   *args, **kwargs):
        assert stack_previous_viewports >= 0
        self.stack_previous_viewports = stack_previous_viewports
        return super(PathFindingByPixelWithDistanceMapEnv, self)._configure(*args, **kwargs)
        

import itertools, numpy, logging
import gym.spaces
from scipy.spatial.distance import euclidean
from .base import BasePathFindingByPixelEnv
# from .utils import line_intersection
from .utils_compiled import build_distance_map, get_flat_state


logger = logging.getLogger(__name__)


class WithDistanceMapMixin(object):
    def _init_state(self):
        self.distance_map = build_distance_map(numpy.array(self.cur_task.local_map, dtype = numpy.int),
                                               numpy.array(self.path_policy.get_global_goal(), dtype = numpy.int))
        return super(WithDistanceMapMixin, self)._init_state()

    def _current_optimal_score(self):
        dm = getattr(self, 'distance_map', None)
        if dm is None:
            return 0
        return self.distance_map[self.path_policy.get_start_position()] + self._get_done_reward()

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


class FlatObservationMixin(object):
    def _get_base_state(self, cur_position_discrete):
        return get_flat_state(self.cur_task.local_map,
                              tuple(cur_position_discrete),
                              self.vision_range,
                              self._get_done_reward(),
                              self.target_on_border_reward,
                              self.path_policy.get_start_position(),
                              self.path_policy.get_global_goal(),
                              self.absolute_distance_observation_weight)
    
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
    def _get_state(self):
        return self._get_base_state(self.cur_position_discrete)

    def _get_observation_space(self, map_shape):
        return gym.spaces.Box(low = 0,
                              high = 1,
                              shape = (2 * self.vision_range + 1, 2 * self.vision_range + 1))


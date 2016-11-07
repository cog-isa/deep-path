import itertools, numpy, logging
import gym.spaces
from scipy.spatial.distance import euclidean
from .base import BasePathFindingByPixelEnv
# from .utils import line_intersection
from .utils_compiled import build_distance_map, get_flat_state


logger = logging.getLogger(__name__)


class PathFindingByPixelWithDistanceMapEnv(BasePathFindingByPixelEnv):
    def _get_usual_reward(self, new_position):
        old_height = self.distance_map[tuple(self.cur_position_discrete)]
        new_height = self.distance_map[tuple(new_position)]
        true_gain = old_height - new_height

        local_target = self.path_policy.get_local_goal()
        old_dist = euclidean(self.cur_position_discrete, local_target)
        new_dist = euclidean(new_position, local_target)
        greedy_gain = old_dist - new_dist

        start_height = self.distance_map[tuple(self.path_policy.get_start_position())]
        abs_gain = numpy.exp(-new_height / start_height)

        total_gain = sum(((1 - self.greedy_distance_weight - self.absolute_distance_weight) * true_gain,
                          self.greedy_distance_weight * greedy_gain,
                          self.absolute_distance_weight * abs_gain))
        logger.debug('true_gain %f, greedy gain %f, abs_gain %f, total %f' % (true_gain,
                                                                              greedy_gain,
                                                                              abs_gain,
                                                                              total_gain))
        return total_gain

    def _init_state(self):
        self.distance_map = build_distance_map(numpy.array(self.cur_task.local_map, dtype = numpy.int),
                                               numpy.array(self.path_policy.get_global_goal(), dtype = numpy.int))
        return self._get_state()

    def _get_state(self):
        return get_flat_state(self.cur_task.local_map,
                              tuple(self.cur_position_discrete),
                              self.vision_range,
                              self._get_done_reward(),
                              self.target_on_border_reward,
                              self.path_policy.get_start_position(),
                              self.path_policy.get_global_goal(),
                              self.absolute_distance_weight)

    '''
    def _get_state(self):
        result = numpy.ones((2 * self.vision_range + 1,
                             2 * self.vision_range + 1))
        result *= -1 # everything is obstacle by default

        local_map = self.cur_task.local_map

#         logger.debug('Map:\n%s' % local_map)
        
        y_viewport_left_top = self.cur_position_discrete[0] - self.vision_range
        x_viewport_left_top = self.cur_position_discrete[1] - self.vision_range

        y_from = max(0, y_viewport_left_top)
        y_to = min(self.cur_position_discrete[0] + self.vision_range + 1, local_map.shape[0])
        x_from = max(0, x_viewport_left_top)
        x_to = min(self.cur_position_discrete[1] + self.vision_range + 1, local_map.shape[1])
#         logger.debug('Pos %s, viewport lt %s, cropped %s' % (self.cur_position_discrete,
#                                                              (y_viewport_left_top, x_viewport_left_top),
#                                                              ((y_from, x_from), (y_to, x_to))))
        
        for point in itertools.product(range(y_from, y_to), range(x_from, x_to)):
            if local_map[point] > 0:
                continue
            viewport_offset = (point[0] - y_viewport_left_top, point[1] - x_viewport_left_top)
#             logger.debug((point, viewport_offset))
            result[viewport_offset] = 0

        goal = self.path_policy.get_global_goal()
        if x_from <= goal[1] < x_to and y_from <= goal[0] < y_to:
            result[goal[0] - y_viewport_left_top, goal[1] - x_viewport_left_top] = self._get_done_reward()
            #cur_y, cur_x = self.cur_position_discrete
            #goal_y, goal_x = self.cur_task.finish
            #real_dist = ((cur_y-goal_y)**2+(cur_x-goal_x)**2)**0.5
            #print 'dst=', real_dist
        else: # find intersection of line <cur_pos, goal> with borders of view range and mark it
            cur_y, cur_x = self.cur_position_discrete
#             logger.debug('Target out of view range %s, %s' % (self.cur_position_discrete,
#                                                               goal))
            # NW, NE, SE, SW
            corners = [(y_viewport_left_top,                   x_viewport_left_top),
                       (y_viewport_left_top,                   x_viewport_left_top + result.shape[1] - 1),
                       (y_viewport_left_top + result.shape[0] - 1, x_viewport_left_top + result.shape[1] - 1),
                       (y_viewport_left_top + result.shape[0] - 1, x_viewport_left_top)]

            # top, right, bottom, left
            borders = [(corners[0], corners[1]),
                       (corners[1], corners[2]),
                       (corners[2], corners[3]),
                       (corners[3], corners[0])]

            line_to_goal = (self.cur_position_discrete, goal)

            best_dist = numpy.inf
            inter_point = None
            for border_i, border in enumerate(borders):
#                 logger.debug('border %s' % repr(border))
                cur_inter_point = line_intersection(line_to_goal, border)
#                 logger.debug('inter %s' % repr(cur_inter_point))
                if cur_inter_point is None:
                    continue
                cur_dist = euclidean(cur_inter_point, goal)
#                 logger.debug('inter dist %s' % repr(cur_dist))
                if 0 <= cur_inter_point[0] - y_viewport_left_top < result.shape[0] \
                    and 0 <= cur_inter_point[1] - x_viewport_left_top < result.shape[1] \
                    and cur_dist < best_dist:
                    best_dist = cur_dist
                    inter_point = cur_inter_point

            abs_dist_normed = numpy.exp(-best_dist / euclidean(self.path_policy.get_start_position(), goal))
            abs_distance_reward = self.absolute_distance_weight * (self._get_done_reward() - self.target_on_border_reward) * abs_dist_normed

            result[inter_point[0] - y_viewport_left_top, inter_point[1] - x_viewport_left_top] = self.target_on_border_reward + abs_distance_reward
#         logger.debug('Viewport:\n%s' % result)
        return result
    '''

    def _get_observation_space(self, map_shape):
        return gym.spaces.Box(low = 0,
                              high = 1,
                              shape = (2 * self.vision_range + 1, 2 * self.vision_range + 1))
    
    def _configure(self,
                   vision_range = 10,
                   target_on_border_reward = 5,
                   greedy_distance_weight = 0.1,
                   absolute_distance_weight = 0.1,
                   *args, **kwargs):
        self.vision_range = vision_range
        self.target_on_border_reward = target_on_border_reward
        self.greedy_distance_weight = greedy_distance_weight
        self.absolute_distance_weight = absolute_distance_weight
        super(PathFindingByPixelWithDistanceMapEnv, self)._configure(*args, **kwargs)

    def _current_optimal_score(self):
        dm = getattr(self, 'distance_map', None)
        if dm is None:
            return 0
        return self.distance_map[self.path_policy.get_start_position()] + self._get_done_reward()

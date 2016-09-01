import itertools, numpy
from scipy.spatial.distance import euclidean
from .base import BasePathFindingByPixelEnv
from .utils import line_intersection


class PathFindingByPixelWithDistanceMapEnv(BasePathFindingByPixelEnv):
    def _get_usual_reward(self, new_position):
        old_height = self.distance_map[self.cur_position_discrete]
        new_height = self.distance_map[tuple(new_position)]
        return old_height - new_height

    def _init_state(self):
        return self._get_state()

    def _get_state(self):
        result = numpy.ones((2 * self.vision_range + 1,
                             2 * self.vision_range + 1))
        result *= -1 # everything is invisible by default

        local_map = self.cur_task.local_map

        x_from = max(0, self.cur_position_discrete[1] - self.vision_range)
        x_to = min(self.cur_position_discrete[1] + self.vision_range + 1, local_map.shape[1])
        y_from = max(0, self.cur_position_discrete[0] - self.vision_range)
        y_to = min(self.cur_position_discrete[0] + self.vision_range + 1, local_map.shape[0])

        for point in itertools.product(range(y_from, y_to), range(x_from, x_to)):
            if local_map[points] > 0:
                continue
            result[point] = 0

        goal = self.path_policy.get_global_goal()
        if x_from <= goal[1] < x_to and y_from <= goal[0] < y_to:
            result[goal] = self._get_done_reward()
        else: # find intersection of line <cur_pos, goal> with borders of view range and mark it
            cur_y, cur_x = self.cur_position_discrete
            
            # NW, NE, SE, SW
            corners = [(cur_y - self.vision_range, cur_x - self.vision_range),
                       (cur_y - self.vision_range, cur_x +self.vision_range),
                       (cur_y + self.vision_range, cur_x +self.vision_range),
                       (cur_y + self.vision_range, cur_x -self.vision_range)]

            # top, right, bottom, left
            borders = [(corners[0], corners[1]),
                       (corners[1], corners[2]),
                       (corners[2], corners[3]),
                       (corners[3], corners[0])]

            line_to_goal = (self.cur_position_discrete, self.cur_task.finish)

            best_dist = numpy.inf
            inter_point = None
            for border_i, border in enumerate(borders):
                cur_inter_point = line_intersection(line_to_goal, border)
                cur_dist = euclidean(self.cur_position_discrete, cur_inter_point)
                if 0 <= cur_inter_point[0] - from_y < result.shape[0] \
                    and 0 <= cur_inter_point[1] - from_x < result.shape[1] \
                    and cur_dist < best_dist:
                    best_dist = cur_dist
                    iter_point = cur_inter_point

            result[inter_point[0] - y_from][inter_point[1] - x_from] = self.target_on_border_reward
        return seen

    def _get_observation_space(self, map_shape):
        return gym.spaces.Box(low = 0,
                              high = 1,
                              shape = map_shape)

    def _configure(self,
                   vision_range = 10,
                   target_on_border_reward = 5,
                   *args, **kwargs):
        super(PathFindingByPixelWithDistanceMapEnv, self)._configure(*args, **kwargs)
        self.vision_range = vision_range
        self.target_on_border_reward = target_on_border_reward

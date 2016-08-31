from dlpf.io import *
from scipy.spatial.distance import euclidean
from random import randint
import gym, gym.spaces, gym.utils
from math import hypot


logger = logging.getLogger(__name__)


BY_PIXEL_ACTIONS = {
    0 : 'N',
    1 : 'NE',
    2 : 'E',
    3 : 'SE',
    4 : 'S',
    5 : 'SW',
    6 : 'W',
    7 : 'NW'
}

BY_PIXEL_ACTION_DIFFS = {
    0 : numpy.array([-1,  0], dtype = 'int8'),
    1 : numpy.array([-1,  1], dtype = 'int8'),
    2 : numpy.array([ 0,  1], dtype = 'int8'),
    3 : numpy.array([ 1,  1], dtype = 'int8'),
    4 : numpy.array([ 1,  0], dtype = 'int8'),
    5 : numpy.array([ 1, -1], dtype = 'int8'),
    6 : numpy.array([ 0, -1], dtype = 'int8'),
    7 : numpy.array([-1, -1], dtype = 'int8')
}

DONE_REWARD = 10
OBSTACLE_PUNISHMENT = 1

class StateLayers:
    OBSTACLE = 0
    GOAL = 1
    WALKED = 2

    LAYERS_NUM = WALKED + 1


def map_state_elem_to_ascii(elem):
    if elem[StateLayers.GOAL] > 0:
        return '*'
    if elem[StateLayers.WALKED] > 0:
        return '.'
    if elem[StateLayers.OBSTACLE] > 0:
        return '#'
    return ' '


def map_state_elem_to_rgb(elem):
    if elem[StateLayers.GOAL] > 0:
        return (255, 0, 0)
    if elem[StateLayers.WALKED] > 0:
        return (0, 255, 0)
    if elem[StateLayers.OBSTACLE] > 0:
        return (0, 0, 0)
    return (255, 255, 255)


def render_ansi(state, **kwargs):
    return '\n'.join(''.join(map_state_elem_to_ascii(state[:, row_i, column_i])
                             for column_i in xrange(state.shape[2]))
                     for row_i in xrange(state.shape[1])) + '\n'


def render_rgb(state, scale = 2):
    base_array = numpy.array([[map_state_elem_to_rgb(state[:,
                                                           min(row_i, state.shape[2] - 1),
                                                           min(column_i, state.shape[1] - 1)])
                               for column_i in xrange(state.shape[2] + state.shape[2] % 2)]
                              for row_i in xrange(state.shape[1] + state.shape[1] % 2)],
                             dtype = 'uint8')
    return numpy.repeat(numpy.repeat(base_array,
                                     scale,
                                     axis = 1),
                        scale,
                        axis = 0)


STATE_RENDERERS = {
    'ansi': render_ansi,
    'rgb_array': render_rgb
}


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


heatmap = {}


def render_state(state, mode = 'human', **kwargs):
    assert mode in STATE_RENDERERS
    return STATE_RENDERERS[mode](state, **kwargs)


class PathFindingByPixelEnv(gym.Env):
    metadata = { 'render.modes': STATE_RENDERERS.keys() }
    action_space = gym.spaces.Discrete(len(BY_PIXEL_ACTIONS))

    def __init__(self):
        self.task_set = None
        self.cur_task = None
        self.observation_space = None
        self.state = None
        self.cur_position_discrete = None
        self.step_number = None
        self.goal_error = None
        self.cur_target_i = None
        self.obstacle_punishment = None
        self.local_goal_reward = None
        self.monitor_scale = None
        self.stop_game_after_invalid_action = None

    def _step(self, action):
        new_position = self.cur_position_discrete + BY_PIXEL_ACTION_DIFFS[action]
        logger.debug('Actor decided to go %s from %s to %s' % (BY_PIXEL_ACTIONS[action],
                                                               tuple(self.cur_position_discrete),
                                                               tuple(new_position)))
        
        done = all(new_position == self.cur_task.finish)
        if done:
            logger.debug('Finished!')
            reward = DONE_REWARD
            print 'FINISHED'
        else:
            goes_out_of_field = any(new_position < 0) or any(new_position + 1 > self.cur_task.local_map.shape)
            invalid_step = goes_out_of_field or self.state[(StateLayers.OBSTACLE,) + tuple(new_position)] > 0
            if invalid_step:
                reward = -self.obstacle_punishment
                logger.debug('Obstacle or out!')
                if self.stop_game_after_invalid_action:
                    done = True
                print 'WALL'
            else:
                old_target_dist = euclidean(self.cur_position_discrete, self.cur_task.path[self.cur_target_i])
                cur_target_dist = euclidean(new_position, self.cur_task.path[self.cur_target_i])
                if cur_target_dist < self.goal_error:
                    reward = self.local_goal_reward
                    if self.cur_target_i < len(self.cur_task.path) - 1:
                        self.cur_target_i += 1
                    else:
                        done = True
                else:
                    reward = (old_target_dist - cur_target_dist)*0
                    reward = 0.01

                logger.debug('Cur target dist is %s' % cur_target_dist)

                self.cur_position_discrete += BY_PIXEL_ACTION_DIFFS[action]
                self.state[(StateLayers.WALKED,) + tuple(self.cur_position_discrete)] = 1
        logger.debug('Reward is %f' % reward)
        return self.state, reward, done, None

    def _reset(self):
        logger.info('Reset environment %s' % self.__class__.__name__)
        task_id = self.np_random.choice(self.task_set.keys())
        self.cur_task = self.task_set[task_id]
        #self.cur_position_discrete = numpy.array(self.cur_task.start, dtype = 'int8')
        self.cur_position_discrete = numpy.array((randint(0,9), randint(0,9)), dtype = 'int32')

        self.state = numpy.zeros(self.observation_space.shape,
                                 dtype = 'uint8')
        self.state[StateLayers.OBSTACLE] = self.cur_task.local_map
        #self.state[(StateLayers.WALKED,) + self.cur_position_discrete] = 1
        self.state[(StateLayers.GOAL,) + self.cur_task.finish] = 1

        self.step_number = 0
        self.cur_target_i = 0
        logger.info('Environment %s has been reset' % self.__class__.__name__)
        return self.state

    def _render(self, mode = 'human', close = False):
        return render_state(self.state, mode = mode, scale = self.monitor_scale)

    def _configure(self,
                   tasks_dir = 'data/samples/imported',
                   map_shape = (501, 501),
                   goal_error = 1,
                   obstacle_punishment = OBSTACLE_PUNISHMENT,
                   local_goal_reward = 5,
                   monitor_scale = 2,
                   stop_game_after_invalid_action = False):
        self.observation_space = gym.spaces.Box(low = 0,
                                                high = 1,
                                                shape = (StateLayers.LAYERS_NUM,) + map_shape)
        self.task_set = TaskSet(tasks_dir)
        self.goal_error = goal_error
        self.obstacle_punishment = obstacle_punishment
        self.local_goal_reward = local_goal_reward
        self.monitor_scale = monitor_scale
        self.stop_game_after_invalid_action = stop_game_after_invalid_action

    def _seed(self, seed = None):
        self.np_random, seed1 = gym.utils.seeding.np_random(seed)
        return [seed1]


class PathFindingByHeightEnv(gym.Env):
    metadata = { 'render.modes': STATE_RENDERERS.keys() }
    action_space = gym.spaces.Discrete(len(BY_PIXEL_ACTIONS))

    def __init__(self):
        self.cur_visible_map = None
        self.task_set = None
        self.cur_task = None
        self.observation_space = None
        self.state = None
        self.cur_position_discrete = None
        self.step_number = None
        self.goal_error = None
        self.cur_target_i = None
        self.obstacle_punishment = None
        self.local_goal_reward = None
        self.monitor_scale = None
        self.stop_game_after_invalid_action = None
        self.heights = None

    def set_heights(self, state):
        labyrinth = state[StateLayers.OBSTACLE]
        heights = numpy.ndarray(shape=labyrinth.shape, dtype='int32')
        heights.fill(0)
        processed = set()
        to_process_next = {self.cur_task.finish}
        current_height = 0
        for i in range(labyrinth.shape[0]):
            for j in range(labyrinth.shape[1]):
                if labyrinth[i][j]:
                    heights[i][j] = -1#labyrinth[i][j]
                    processed.add((i,j))
                else:
                    heigth = 0
        while len(to_process_next):
            processing_now = to_process_next.copy()
            to_process_next = set()
            for pair in processing_now:
                y, x = pair
                heights[y][x] = current_height
                processed.add((y, x))
                for d in BY_PIXEL_ACTION_DIFFS.values():
                    if 0 <= y+d[0] < labyrinth.shape[0] \
                            and 0 <= x+d[1] < labyrinth.shape[1] \
                            and (y+d[0],x+d[1]) not in processed \
                            and (y+d[0],x+d[1]) not in processing_now:
                        to_process_next.add((y+d[0], x+d[1]))
            current_height += 1
        return heights

    def cut_seen(self, state):
        pos = self.cur_position_discrete
        seen = numpy.ndarray(shape=(1, 2*self.vision_range+1, 2*self.vision_range+1))
        walls, target, path = state
        center_y, center_x = pos
        height, width = len(walls), len(walls[0])
        for dy in range(-self.vision_range, self.vision_range+1):
            for dx in range(-self.vision_range, self.vision_range+1):
                x, y = center_x+dx, center_y+dy
                if x < 0 or x >= width or y < 0 or y >= height:
                    seen[0][dx][dy] = -1
                elif walls[y][x]:
                    seen[0][dx][dy] = -1
                else:
                    seen[0][dx][dy] = target[y][x]*10
        return seen

    def _step(self, action):
        shout = False
        new_position = self.cur_position_discrete + BY_PIXEL_ACTION_DIFFS[action]
        logger.debug('Actor decided to go %s from %s to %s' % (BY_PIXEL_ACTIONS[action],
                                                               tuple(self.cur_position_discrete),
                                                               tuple(new_position)))
        wall = False
        done = all(new_position == self.cur_task.finish)
        if done:
            logger.debug('Finished!')
            reward = DONE_REWARD
            if shout:
                print 'FINISHED'
        else:
            goes_out_of_field = any(new_position < 0) or any(new_position + 1 > self.cur_task.local_map.shape)
            invalid_step = goes_out_of_field or self.state[(StateLayers.OBSTACLE,) + tuple(new_position)] > 0
            if invalid_step:
                reward = -self.obstacle_punishment
                logger.debug('Obstacle or out!')
                if self.stop_game_after_invalid_action:
                    done = True
                if shout:
                    print 'WALL'
                wall = True
            else:
                old_target_dist = euclidean(self.cur_position_discrete, self.cur_task.path[self.cur_target_i])
                cur_target_dist = euclidean(new_position, self.cur_task.path[self.cur_target_i])
                old_height = self.heights[self.cur_position_discrete[0],
                                          self.cur_position_discrete[1]]
                new_height = self.heights[new_position[0], new_position[1]]
                if cur_target_dist < self.goal_error:
                    reward = self.local_goal_reward
                    if self.cur_target_i < len(self.cur_task.path) - 1:
                        self.cur_target_i += 1
                    else:
                        done = True
                else:
                    reward = (old_height - new_height)

                logger.debug('Cur target dist is %s' % cur_target_dist)

                self.cur_position_discrete += BY_PIXEL_ACTION_DIFFS[action]
                self.state[(StateLayers.WALKED,) + tuple(self.cur_position_discrete)] = 1
        logger.debug('Reward is %f' % reward)
        self.cur_visible_map = self.cut_seen(self.state)
        return self.cur_visible_map, reward, done, wall

    def _reset(self):
        logger.info('Reset environment %s' % self.__class__.__name__)
        task_id = self.np_random.choice(self.task_set.keys())
        self.cur_task = self.task_set[task_id]
        self.cur_position_discrete = numpy.array(self.cur_task.start, dtype = 'int32')
        #self.cur_position_discrete = numpy.array((randint(0,9), randint(0,9)), dtype = 'int8')

        self.state = numpy.zeros(self.observation_space.shape,
                                 dtype = 'uint32')
        self.state[StateLayers.OBSTACLE] = self.cur_task.local_map
        #self.state[(StateLayers.WALKED,) + self.cur_position_discrete] = 1
        self.state[(StateLayers.GOAL,) + self.cur_task.finish] = 1

        self.step_number = 0
        self.cur_target_i = 0
        self.cur_visible_map = self.cut_seen(self.state)
        self.heights = self.set_heights(self.state)
        logger.info('Environment %s has been reset' % self.__class__.__name__)
        return self.cur_visible_map

    def _render(self, mode = 'human', close = False):
        pass
        #return render_state(self.state, mode = mode, scale = self.monitor_scale)

    def _configure(self,
                   tasks_dir = 'data/samples/imported',
                   map_shape = (501, 501),
                   goal_error = 1,
                   obstacle_punishment = OBSTACLE_PUNISHMENT,
                   local_goal_reward = 5,
                   monitor_scale = 2,
                   stop_game_after_invalid_action = False,
                   vision_range = 10):
        self.observation_space = gym.spaces.Box(low = 0,
                                                high = 1,
                                                shape = (StateLayers.LAYERS_NUM,) + map_shape)
        self.task_set = TaskSet(tasks_dir)
        self.goal_error = goal_error
        self.obstacle_punishment = obstacle_punishment
        self.local_goal_reward = local_goal_reward
        self.monitor_scale = monitor_scale
        self.stop_game_after_invalid_action = stop_game_after_invalid_action
        self.vision_range = vision_range

    def _seed(self, seed = None):
        self.np_random, seed1 = gym.utils.seeding.np_random(seed)
        return [seed1]


class PathFindingByHeightWithControlEnv(gym.Env):
    metadata = { 'render.modes': STATE_RENDERERS.keys() }
    action_space = gym.spaces.Discrete(len(BY_PIXEL_ACTIONS))

    def __init__(self):
        self.cur_visible_map = None
        self.train_task_set = None
        self.test_task_set = None
        self.validate_task_set = None
        self.cur_task = None
        self.observation_space = None
        self.state = None
        self.cur_position_discrete = None
        self.step_number = None
        self.goal_error = None
        self.cur_target_i = None
        self.obstacle_punishment = None
        self.local_goal_reward = None
        self.monitor_scale = None
        self.stop_game_after_invalid_action = None
        self.heights = None
        self.vision_range = None
        self.mode = None

    def set_heights(self, state):
        labyrinth = state[StateLayers.OBSTACLE]
        heights = numpy.ndarray(shape=labyrinth.shape, dtype='int32')
        heights.fill(0)
        processed = set()
        to_process_next = {self.cur_task.finish}
        current_height = 0
        for i in range(labyrinth.shape[0]):
            for j in range(labyrinth.shape[1]):
                if labyrinth[i][j]:
                    heights[i][j] = -1#labyrinth[i][j]
                    processed.add((i,j))
                else:
                    heigth = 0
        while len(to_process_next):
            processing_now = to_process_next.copy()
            to_process_next = set()
            for pair in processing_now:
                y, x = pair
                heights[y][x] = current_height
                processed.add((y, x))
                for d in BY_PIXEL_ACTION_DIFFS.values():
                    if 0 <= y+d[0] < labyrinth.shape[0] \
                            and 0 <= x+d[1] < labyrinth.shape[1] \
                            and (y+d[0],x+d[1]) not in processed \
                            and (y+d[0],x+d[1]) not in processing_now:
                        to_process_next.add((y+d[0], x+d[1]))
            current_height += 1
        return heights

    def cut_seen(self, state):
        pos = self.cur_position_discrete
        seen = numpy.ndarray(shape=(1, 2*self.vision_range+1, 2*self.vision_range+1))
        walls, target, path = state
        center_y, center_x = pos
        height, width = len(walls), len(walls[0])
        for dy in range(-self.vision_range, self.vision_range+1):
            for dx in range(-self.vision_range, self.vision_range+1):
                x, y = center_x+dx, center_y+dy
                if x < 0 or x >= width or y < 0 or y >= height:
                    seen[0][dy+self.vision_range][dx+self.vision_range] = -1
                elif walls[y][x]:
                    seen[0][dy+self.vision_range][dx+self.vision_range] = -1
                else:
                    seen[0][dy+self.vision_range][dx+self.vision_range] = target[y][x]*10

        # When agent can`t see the target, it senses the direction
        if max([max(i) for i in seen[0]]) == 0:
            corners = [(center_y-self.vision_range, center_x-self.vision_range),
                       (center_y-self.vision_range, center_x+self.vision_range),
                       (center_y+self.vision_range, center_x+self.vision_range),
                       (center_y+self.vision_range, center_x-self.vision_range)]

            borders = [(corners[0], corners[1]),
                       (corners[1], corners[2]),
                       (corners[2], corners[3]),
                       (corners[3], corners[0])]

            target_line = [(center_y, center_x), self.cur_task.finish]

            intersections = []
            for i in range(4):
                ci = line_intersection(target_line, borders[i])
                if not ci or abs(ci[0]-center_y) > self.vision_range or abs(ci[1]-center_x) > self.vision_range:
                    intersections.append(None)
                else:
                    intersections.append(ci)

            min_dist = 1000
            nearest = intersections[0]
            for i in range(4):
                if intersections[i]:
                    dst = hypot(intersections[i][0]-self.cur_task.finish[0],
                                intersections[i][1]-self.cur_task.finish[1])
                    if dst < min_dist:
                        min_dist = dst
                        nearest = intersections[i]
            seen[0][nearest[0]-center_y+self.vision_range][nearest[1]-center_x+self.vision_range] = 5
        return seen

    def _step(self, action):
        shout = False
        new_position = self.cur_position_discrete + BY_PIXEL_ACTION_DIFFS[action]
        logger.debug('Actor decided to go %s from %s to %s' % (BY_PIXEL_ACTIONS[action],
                                                               tuple(self.cur_position_discrete),
                                                               tuple(new_position)))
        wall = False
        done = all(new_position == self.cur_task.finish)
        if done:
            logger.debug('Finished!')
            reward = DONE_REWARD
            if shout:
                print 'FINISHED'
        else:
            goes_out_of_field = any(new_position < 0) or any(new_position + 1 > self.cur_task.local_map.shape)
            invalid_step = goes_out_of_field or self.state[(StateLayers.OBSTACLE,) + tuple(new_position)] > 0
            if invalid_step:
                reward = -self.obstacle_punishment
                logger.debug('Obstacle or out!')
                if self.stop_game_after_invalid_action:
                    done = True
                if shout:
                    print 'WALL'
                wall = True
            else:
                old_target_dist = euclidean(self.cur_position_discrete, self.cur_task.path[self.cur_target_i])
                cur_target_dist = euclidean(new_position, self.cur_task.path[self.cur_target_i])
                old_height = self.heights[self.cur_position_discrete[0],
                                          self.cur_position_discrete[1]]
                new_height = self.heights[new_position[0], new_position[1]]
                if cur_target_dist < self.goal_error:
                    reward = self.local_goal_reward
                    if self.cur_target_i < len(self.cur_task.path) - 1:
                        self.cur_target_i += 1
                    else:
                        done = True
                else:
                    reward = (old_height - new_height)

                logger.debug('Cur target dist is %s' % cur_target_dist)

                self.cur_position_discrete += BY_PIXEL_ACTION_DIFFS[action]
                self.state[(StateLayers.WALKED,) + tuple(self.cur_position_discrete)] = 1
        logger.debug('Reward is %f' % reward)
        self.cur_visible_map = self.cut_seen(self.state)
        return self.cur_visible_map, reward, done, wall

    def generate_new_task(self, mode):
        if mode == 'train':
            task_set = self.train_task_set
        elif mode == 'test':
            task_set = self.test_task_set
        else:
            task_set = self.validate_task_set
        task_name = self.np_random.choice(task_set.keys())
        task_start = (randint(0, 500), randint(0, 500))
        while task_set[(task_name, task_start, (0, 0))].local_map[task_start[0], task_start[1]]:
            task_start = (randint(0, 500), randint(0, 500))
        task_finish = (randint(0, 500), randint(0, 500))
        while task_set[(task_name, task_start, task_finish)].local_map[task_start[0], task_start[1]] \
                or task_start == task_finish:
            task_finish = (randint(0, 500), randint(0, 500))
        return task_set[(task_name, task_start, task_finish)]

    def _reset(self):
        logger.info('Reset environment %s' % self.__class__.__name__)
        generated = False
        while not generated:
            try:
                self.cur_task = self.generate_new_task(self.mode)
                generated = True
            except IOError:
                pass

        self.cur_position_discrete = numpy.array(self.cur_task.start, dtype = 'int32')
        self.state = numpy.zeros(self.observation_space.shape,
                                 dtype = 'uint32')
        self.state[StateLayers.OBSTACLE] = self.cur_task.local_map
        self.state[(StateLayers.GOAL,) + self.cur_task.finish] = 1

        self.step_number = 0
        self.cur_target_i = 0
        self.cur_visible_map = self.cut_seen(self.state)
        self.heights = self.set_heights(self.state)
        logger.info('Environment %s has been reset' % self.__class__.__name__)
        return self.cur_visible_map

    def _render(self, mode = 'human', close = False):
        pass
        #return render_state(self.state, mode = mode, scale = self.monitor_scale)

    def _configure(self,
                   tasks_dir = 'data/samples/imported',
                   map_shape = (501, 501),
                   goal_error = 1,
                   obstacle_punishment = OBSTACLE_PUNISHMENT,
                   local_goal_reward = 5,
                   monitor_scale = 2,
                   stop_game_after_invalid_action = False,
                   vision_range = 10):
        self.observation_space = gym.spaces.Box(low = 0,
                                                high = 1,
                                                shape = (StateLayers.LAYERS_NUM,) + map_shape)
        self.train_task_set = UltimateTaskSet(tasks_dir+'/train')
        self.test_task_set = UltimateTaskSet(tasks_dir+'/test')
        self.validate_task_set = UltimateTaskSet(tasks_dir+'/validate')
        self.goal_error = goal_error
        self.obstacle_punishment = obstacle_punishment
        self.local_goal_reward = local_goal_reward
        self.monitor_scale = monitor_scale
        self.stop_game_after_invalid_action = stop_game_after_invalid_action
        self.vision_range = vision_range

    def _seed(self, seed = None):
        self.np_random, seed1 = gym.utils.seeding.np_random(seed)
        return [seed1]

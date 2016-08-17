from dlpf.io import *
from scipy.spatial.distance import euclidean
import gym, gym.spaces, gym.utils


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
OBSTACLE_PUNISHMENT = 10

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
    'ansi' : render_ansi,
    'rgb_array' : render_rgb
}

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
                    reward = 0.1

                logger.debug('Cur target dist is %s' % cur_target_dist)

                self.cur_position_discrete += BY_PIXEL_ACTION_DIFFS[action]
                self.state[(StateLayers.WALKED,) + tuple(self.cur_position_discrete)] = 1
        logger.debug('Reward is %f' % reward)
        return self.state, reward, done, None

    def _reset(self):
        logger.info('Reset environment %s' % self.__class__.__name__)
        task_id = self.np_random.choice(self.task_set.keys())
        self.cur_task = self.task_set[task_id]
        self.cur_position_discrete = numpy.array(self.cur_task.start, dtype = 'int8')

        self.state = numpy.zeros(self.observation_space.shape,
                                 dtype = 'uint8')
        self.state[StateLayers.OBSTACLE] = self.cur_task.local_map
        self.state[(StateLayers.WALKED,) + self.cur_task.start] = 1
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

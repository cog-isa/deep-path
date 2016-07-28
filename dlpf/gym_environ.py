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
    7 : 'WN'
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

SQRT_OF_2 = 2.0 ** 0.5
BY_PIXEL_STEP_SIZES = {
    0 : 1.0,
    1 : SQRT_OF_2,
    2 : 1.0,
    3 : SQRT_OF_2,
    4 : 1.0,
    5 : SQRT_OF_2,
    6 : SQRT_OF_2,
    7 : 1.0
}

DONE_REWARD = 10000

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
        self.cur_segment_i = None
        self.steps_done_along_cur_segment = None
        self.min_segment_leftover = None
        self.min_obstacle_punishment = None
        self.reward_type = None
        self.scale = None

    def _step(self, action):
        new_position = self.cur_position_discrete + BY_PIXEL_ACTION_DIFFS[action]
        logger.debug('Actor decided to go %s from %s to %s' % (BY_PIXEL_ACTIONS[action],
                                                               tuple(self.cur_position_discrete),
                                                               tuple(new_position)))
        
        done = all(new_position == self.cur_task.finish)
        if done:
            logger.debug('Finished!')
            reward = DONE_REWARD
        else:
            step_size = BY_PIXEL_STEP_SIZES[action]
            while True:
                cur_seg_start = numpy.array(self.cur_task.path[self.cur_segment_i], dtype = 'int8')
                cur_seg_end = numpy.array(self.cur_task.path[self.cur_segment_i + 1], dtype = 'int8')
                cur_seg_dir = cur_seg_end - cur_seg_start
                cur_gold_pos = cur_seg_dir * self.steps_done_along_cur_segment
                next_gold_pos = cur_gold_pos + cur_seg_dir * step_size
                if euclidean(next_gold_pos, cur_seg_end) > self.min_segment_leftover:
                    break
                else:
                    self.cur_segment_i += 1
                    self.steps_done_along_cur_segment = 0

            
            cur_dist = euclidean(self.cur_position_discrete, cur_gold_pos)
            next_dist = euclidean(new_position, next_gold_pos)
            
            if self.reward_type == 'abs':
                reward = -next_dist
            elif self.reward_type == 'diff':
                reward = cur_dist - next_dist # if next distance is smaller, we encourage the agent and punish otherwise

            if self.state[(StateLayers.OBSTACLE,) + tuple(new_position)] > 0: # if agent tries to go through wall, we punish it strongly
                # if agent intended to break the wall, pay a fine and win nevertheless, we increase the punishment
                reward = - (self.min_obstacle_punishment + abs(reward))
            logger.debug('Reward is %f' % reward)
            self.cur_position_discrete += BY_PIXEL_ACTION_DIFFS[action]
            self.state[(StateLayers.WALKED,) + tuple(self.cur_position_discrete)] = 1
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
        self.cur_segment_i = 0
        self.steps_done_along_cur_segment = 0
        logger.info('Environment %s has been reset' % self.__class__.__name__)
        return self.state

    def _render(self, mode = 'human', close = False):
        return render_state(self.state, mode = mode, scale = self.scale)

    def _configure(self,
                   tasks_dir = 'data/samples/imported',
                   map_shape = (501, 501),
                   min_segment_leftover = 0.1,
                   min_obstacle_punishment = 10000.0,
                   reward_type = 'abs',
                   scale = 2):
        assert reward_type in ('abs', 'diff')
        self.observation_space = gym.spaces.Box(low = 0,
                                                high = 1,
                                                shape = (StateLayers.LAYERS_NUM,) + map_shape)
        self.task_set = TaskSet(tasks_dir)
        self.min_segment_leftover = min_segment_leftover
        self.min_obstacle_punishment = min_obstacle_punishment
        self.reward_type = reward_type
        self.scale = scale

    def _seed(self, seed = None):
        self.np_random, seed1 = gym.utils.seeding.np_random(seed)
        return [seed1]

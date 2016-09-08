import gym
from .base import BasePathFindingByPixelEnv


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


def render_state(state, mode = 'human', **kwargs):
    assert mode in STATE_RENDERERS
    return STATE_RENDERERS[mode](state, **kwargs)


class MultilayerPathFindingByPixelEnv(BasePathFindingByPixelEnv):
    metadata = { 'render.modes': STATE_RENDERERS.keys() }

    def _get_usual_reward(self, new_position):
        local_target = self.path_policy.get_local_goal()
        #old_target_dist = euclidean(self.cur_position_discrete, local_target)
        cur_target_dist = euclidean(new_position, local_target)
        #return (old_target_dist - cur_target_dist)
        return cur_target_dist

    def _init_state(self):
        self.state = numpy.zeros(self.observation_space.shape,
                                 dtype = 'uint8')
        self.state[StateLayers.OBSTACLE] = self.cur_task.local_map
        self.state[(StateLayers.WALKED,) + self.cur_position_discrete] = 1
        self.state[(StateLayers.GOAL,) + self.cur_task.finish] = 1

    def _get_state(self):
        self.state[(StateLayers.WALKED,) + tuple(self.cur_position_discrete)] = 1
        return self.state

    def _get_observation_space(self, map_shape):
        return gym.spaces.Box(low = 0,
                              high = 1,
                              shape = (StateLayers.LAYERS_NUM,) + map_shape)

    def _render(self, mode = 'human', close = False):
        return render_state(self.state,
                            mode = mode,
                            scale = self.monitor_scale)

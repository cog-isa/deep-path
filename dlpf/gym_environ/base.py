import logging, numpy
from scipy.spatial.distance import euclidean
import gym, gym.spaces, gym.utils

from dlpf.io import TaskSet
from dlpf.base_utils import load_object_by_name, load_yaml, copy_and_update
from .policies import get_path_policy, get_task_policy, \
    DEFAULT_PATH_POLICY, DEFAULT_TASK_POLICY
from .utils import BY_PIXEL_ACTIONS, BY_PIXEL_ACTION_DIFFS

logger = logging.getLogger(__name__)

DEFAULT_DONE_REWARD = 10
DEFAULT_GOAL_REWARD = 5
DEFAULT_OBSTACLE_PUNISHMENT = 1


class InfoValues:
    OK = 'OK'
    NOTHING = 'NOTHING'
    DONE = 'DONE'

    OBSTACLE = 'OBS'
    OUT_OF_FIELD = 'OOF'
    
    GOOD = frozenset({ OBSTACLE, OUT_OF_FIELD })
    BAD = frozenset({ OK, DONE })


class BasePathFindingEnv(gym.Env):
    action_space = gym.spaces.Discrete(len(BY_PIXEL_ACTIONS))

    def __init__(self):
        self.task_set = None
        self.cur_task = None
        self.task_policy = None
        self.path_policy = None
        self.observation_space = None
        self.obstacle_punishment = None
        self.local_goal_reward = None
        self.done_reward = None

    def current_optimal_score(self):
        return self._current_optimal_score()

    def get_episode_stat(self):
        return {}

    ####################################################
    ######## Default environment implementation ########
    ####################################################
    def _reset(self):
        logger.info('Reset environment %s' % self.__class__.__name__)

        self.cur_task = self.task_policy.choose_next_task()
        self.path_policy.reset(self.cur_task)

        logger.info('Environment %s has been reset' % self.__class__.__name__)
        return self._init_state()

    def _configure(self,
                   tasks_dir = 'data/samples/imported/tasks',
                   maps_dir = 'data/samples/imported/maps',
                   map_shape = (501, 501),
                   path_policy = DEFAULT_PATH_POLICY,
                   task_policy = DEFAULT_TASK_POLICY,
                   obstacle_punishment = DEFAULT_OBSTACLE_PUNISHMENT,
                   local_goal_reward = DEFAULT_GOAL_REWARD,
                   done_reward = DEFAULT_DONE_REWARD):
        self.observation_space = self._get_observation_space(map_shape)
        self.task_set = TaskSet(tasks_dir, maps_dir)
        self.task_policy = get_task_policy(task_policy)
        self.task_policy.reset(self.task_set)
        self.path_policy = get_path_policy(path_policy)

        self.obstacle_punishment = abs(obstacle_punishment)
        self.local_goal_reward = local_goal_reward
        self.done_reward = done_reward
        self.stop_game_after_invalid_action = stop_game_after_invalid_action

    def _seed(self, seed = None):
        self.np_random, seed1 = gym.utils.seeding.np_random(seed)
        return [seed1]

    def _check_out_of_field(self, position):
        return any(position < 0) or any(position + 1 > self.cur_task.local_map.shape)

    def _goes_to_obstacle(self, position):
        return self.cur_task.local_map[tuple(position)] > 0

    ####################################################
    ########### Methods optional to implement ##########
    ####################################################
    def _get_obstacle_punishment(self):
        return -self.obstacle_punishment

    def _get_local_goal_reward(self):
        return self.local_goal_reward

    def _get_done_reward(self):
        return self.done_reward

    def _init_state(self):
        return self._get_state()

    def _render(self, mode = 'human', close = False):
        pass

    ####################################################
    ########## Methods mandatory to implement ##########
    ####################################################
    def _step(self, action):
        raise NotImplemented()

    def _get_usual_reward(self, old_position, new_position):
        raise NotImplemented()

    def _get_state(self):
        raise NotImplemented()

    def _get_observation_space(self, map_shape):
        raise NotImplemented()
    
    def _current_optimal_score(self):
        raise NotImplemented()


class BasePathFindingByPixelEnv(BasePathFindingEnv):
    action_space = gym.spaces.Discrete(len(BY_PIXEL_ACTIONS))

    def __init__(self):
        super(BasePathFindingByPixelEnv, self).__init__()
        self.cur_position_discrete = None
        self.goal_error = None

    ####################################################
    ######## Default environment implementation ########
    ####################################################
    def _step(self, action):
        new_position = self.cur_position_discrete + BY_PIXEL_ACTION_DIFFS[action]
        logger.debug('Actor decided to go %s from %s to %s' % (BY_PIXEL_ACTIONS[action],
                                                               tuple(self.cur_position_discrete),
                                                               tuple(new_position)))
        
        info = InfoValues.OK
        done = numpy.allclose(new_position, self.path_policy.get_global_goal())
        if done:
            logger.debug('Finished %s %s!' % (new_position, self.path_policy.get_global_goal()))
            reward = self._get_done_reward()
            info = InfoValues.DONE
        else:
            goes_out_of_field = self._check_out_of_field(new_position)
            invalid_step = goes_out_of_field or self._goes_to_obstacle(new_position)
            if invalid_step:
                info = InfoValues.OUT_OF_FIELD if goes_out_of_field else InfoValues.OBSTACLE
                reward = self._get_obstacle_punishment()
                logger.debug('Obstacle or out!')
                if self.stop_game_after_invalid_action:
                    done = True
            else:
                local_target = self.path_policy.get_local_goal()
                cur_target_dist = euclidean(new_position, local_target)
                if cur_target_dist < self.goal_error:
                    reward = self._get_local_goal_reward()
                    done = self.path_policy.move_to_next_goal()
                else:
                    reward = self._get_usual_reward(self.cur_position_discrete, new_position)

                logger.debug('Cur target dist is %s' % cur_target_dist)

                self.cur_position_discrete = self._update_cur_position(action)

        logger.debug('Reward is %f' % reward)
        return self._get_state(), reward, done, info

    def _reset(self):
        result = super(BasePathFindingByPixelEnv, self)._reset()
        self.cur_position_discrete = self.path_policy.get_start_position()
        return self._init_state()

    def _configure(self,
                   goal_error = 1,
                   *args, **kwargs):
        super(BasePathFindingByPixelEnv, self)._configure(*args, **kwargs)
        self.goal_error = goal_error

    ####################################################
    ########### Methods optional to implement ##########
    ####################################################
    def _update_cur_position(self, action):
        return self.cur_position_discrete + BY_PIXEL_ACTION_DIFFS[action]


def load_environment_from_yaml(fname, **kwargs_override):
    info = load_yaml(fname)
    result = gym.make(info['ctor'])
    result.configure(*info.get('args', []),
                     **copy_and_update(info.get('kwargs', {}),
                                       **kwargs_override))
    return result

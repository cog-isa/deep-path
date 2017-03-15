import gym.envs.registration

from dlpf.gym_environ.base import InfoValues, load_environment_from_yaml
from dlpf.gym_environ.policies import *

gym.envs.registration.register('MultilayerPathFindingByPixelEnv-v1',
                               entry_point='dlpf.gym_environ.multilayer_state:MultilayerPathFindingByPixelEnv')

gym.envs.registration.register('PathFindingByPixelWithDistanceMapEnv-v1',
                               entry_point='dlpf.gym_environ.flat:PathFindingByPixelWithDistanceMapEnv')

gym.envs.registration.register('FlatSearchBasedPathFindingEnv-v1',
                               entry_point='dlpf.gym_environ.search_based:FlatSearchBasedPathFindingEnv')

import gym.envs.registration

#gym.envs.registration.register('PathFindingByPixel-v1',
#                               entry_point = 'dlpf.gym_environ:PathFindingByPixelEnv')

gym.envs.registration.register('PathFindingByPixel-v2',
                               entry_point = 'dlpf.gym_environ:PathFindingByHeightEnv')

gym.envs.registration.register('PathFindingByPixel-v3',
                               entry_point = 'dlpf.gym_environ:PathFindingByHeightWithControlEnv')
import gym.envs.registration

gym.envs.registration.register('MultilayerPathFindingByPixelEnv-v1',
                               entry_point = 'dlpf.gym_environ.multilayer_state:MultilayerPathFindingByPixelEnv')

gym.envs.registration.register('PathFindingByPixelWithDistanceMapEnv-v1',
                               entry_point = 'dlpf.gym_environ.flat:PathFindingByPixelWithDistanceMapEnv')

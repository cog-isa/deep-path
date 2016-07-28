import gym.envs.registration

gym.envs.registration.register('PathFindingByPixel-v1',
                               entry_point = 'dlpf.gym_environ:PathFindingByPixelEnv')

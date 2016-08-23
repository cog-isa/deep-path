#!/usr/bin/env python

import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import gym
from dlpf.agents import DqnAgent, RandomAgent, FlatAgent
from dlpf.io import *

logger = init_log(out_file = 'import.log', stderr = False)
import_tasks_from_xml_to_compact('data/sample/raw/', 'data/sample/imported/')

logger = init_log(out_file = 'testbed.log', stderr = False)


env = gym.make('PathFindingByPixel-v2')
env.configure(tasks_dir = os.path.abspath('data/sample/imported/'), monitor_scale = 10, map_shape = (10, 10))
env.monitor.start('data/sample/results/basic_dqn', force=True, seed=0)


def objective(space):
    stepslog = []
    agent = FlatAgent(state_size = env.observation_space.shape,
                     number_of_actions = env.action_space.n,
                     save_name = env.__class__.__name__)
    agent.build_model(number_of_neurons=space['neurons'],
                      desc_name=space['desc'],
                      loss_fn=space['lf'],
                      dropout1=space['dropout1'],
                      activation=space['activation'],
                      lr=space['lr'])

    episode_count = 5000
    max_steps = 100

    for _ in xrange(episode_count):
        observation = env.reset()
        agent.new_episode()
        walls = 0
        for __ in range(max_steps):
            action, values = agent.act(observation, epsilon=0.05+0.95*0.999**(_))
            observation, reward, done, info = env.step(action)
            if info:
                walls += 1
            agent.observe(reward, action)
            if done:
                break
        steps = __
        stepslog.append(steps+walls)
        if _ % 100 == 99:
            # print 'iteration:', _ + 1
            agent.plot_layers(to_save='iteration'+str(_+1))
        if _ % 10 == 9:
            agent.train_with_full_experience(output=0, batch=space['batch'])
    print 'result: ', sum(stepslog[:10])
    return {'loss': sum(stepslog[:10]), 'status': STATUS_OK}





trials = Trials()
best = fmin(objective,
            space={'neurons': hp.choice('neurons', [4, 8, 12, 16, 32]),
                   'batch': hp.choice('batch', [4, 16, 32, 64, 126, 256, 512, 1024]),
                   'desc': hp.choice('desc', ['adadelta', 'rmsprop', 'adagrad', 'nadam']),
                   'lf': hp.choice('lf', ['mean_squared_error', 'categorical_crossentropy']),
                   'lr': hp.loguniform('lr', numpy.log(0.001), numpy.log(0.1)),
                   'dropout1': hp.uniform('dropout1', 0, 1),
                   'dropout2': hp.uniform('dropout2', 0, 1),
                   'activation': hp.choice('activation', ['relu', 'linear', 'tanh', 'sigmoid'])},
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)

print best
for i in trials.trials:
    print i#['result']


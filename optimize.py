import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import gym
from dlpf.agents import DqnAgent, RandomAgent, FlatAgent, FlatAgentWithLossLogging
from dlpf.io import *
from data_shuffle import *

logger = init_log(out_file = 'import.log', stderr = False)
to_import = True
if to_import:
    import_tasks_from_xml_to_compact('data/sample/raw/', 'data/sample/imported/')
    shuffle_imported_paths(to_split=True, val=False)
    shuffle_imported_maps(to_split=True, val=False)

logger = init_log(out_file = 'testbed.log', stderr = False)


env = gym.make('PathFindingByPixel-v3')
env.configure(tasks_dir = os.path.abspath('data/sample/imported/'), monitor_scale = 10, map_shape = (501, 501))
env.monitor.start('data/sample/results/basic_dqn', force=True, seed=0)


def to_continue(agent, critical_enlargement):
    history = agent.history.get_flat_array()
    if len(history) < 1000:
        return True
    elif (sum(history[-100:])-sum(history[-200:-100]))/100 < critical_enlargement:
        return True
    else:
        return False


def objective(space):
    reslog = []
    agent = FlatAgentWithLossLogging(state_size = env.observation_space.shape,
                     number_of_actions = env.action_space.n,
                     save_name = env.__class__.__name__)
    agent.build_model(number_of_neurons=space['neurons'],
                      desc_name=space['desc'],
                      loss_fn=space['lf'],
                      dropout1=space['dropout1'],
                      activation=space['activation'])

    episode_count = space['episodes']
    max_steps = 500

    for _ in xrange(episode_count):
        env.mode = 'train'
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
        #if _ % 100 == 99:
        #    print 'iteration:', _ + 1
        #    agent.plot_layers(to_save='iteration'+str(_+1))
        if _ % 10 == 9:
            agent.train_with_full_experience()

    for _ in xrange(episode_count):
        env.mode = 'test'
        observation = env.reset()
        agent.new_episode()
        walls = 0
        for __ in range(max_steps):
            action, values = agent.act(observation, epsilon=0)
            observation, reward, done, info = env.step(action)
            if info:
                walls += 1
            agent.observe(reward, action)
            if done:
                break
        steps = __
    if env.heights[env.cur_task.start[0]][env.cur_task.start[1]] > 0:
        reslog.append(steps+walls/env.heights[env.cur_task.start[0]][env.cur_task.start[1]])
    else:
        reslog.append(3)
        print 'something weird happened'
    print 'result: ', sum(reslog)/len(reslog)
    return {'loss': sum(reslog)/len(reslog), 'status': STATUS_OK}





trials = Trials()
best = fmin(objective,
            space={'neurons': hp.choice('neurons', [16, 32, 64, 128]),
                   'batch': hp.choice('batch', [5, 10, 20, 50, 100, 500, 1000]),
                   'desc': hp.choice('desc', ['adadelta', 'rmsprop', 'adagrad', 'nadam']),
                   'lf': hp.choice('lf', ['mean_squared_error', 'categorical_crossentropy', 'squared_hinge']),
                   'dropout1': hp.choice('dropout1', [0, 0.25, 0.5, 0.75]),
                   'dropout2': hp.choice('dropout2', [0, 0.25, 0.5, 0.75]),
                   'activation': hp.choice('activation', ['relu', 'linear', 'tanh', 'sigmoid']),
                   'episodes': hp.choice('episodes', [10000, 25000, 50000, 100000])},
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)

print best
for i in trials.trials:
    print i#['result']


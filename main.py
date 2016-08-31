from dlpf.base_utils import *
import gym
from dlpf.agents import DqnAgent, RandomAgent, FlatAgent, FlatAgentWithLossLogging
import keras
from dlpf.io import *
from data_shuffle import *

logger = init_log(out_file = 'import.log', stderr = False)
to_import = False
if to_import:
    import_tasks_from_xml_to_compact('data/sample/raw/', 'data/sample/imported/')
    shuffle_imported_paths(to_split=True, val=False)
    shuffle_imported_maps(to_split=True, val=False)

logger = init_log(out_file = 'testbed.log', stderr = False)


env = gym.make('PathFindingByPixel-v3')
env.configure(tasks_dir = os.path.abspath('data/sample/imported/'), monitor_scale = 10)#, map_shape = (10, 10))
env.monitor.start('data/sample/results/basic_dqn', force=True, seed=0)
agent = FlatAgentWithLossLogging(state_size = env.observation_space.shape,
                 number_of_actions = env.action_space.n,
                 save_name = env.__class__.__name__)
agent.build_model()

episode_count = 5000
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
    print 'GAME #', _+1
    if done:
        print 'DONE: ', steps, 'moves.',
    else:
        print 'FAIL: ',
    print 'Found', walls, 'walls'
    #if _ % 100 == 99:
    #    print 'iteration:', _ + 1
    #    agent.plot_layers(to_save='iteration'+str(_+1))
    if _ % 10 == 9:
        agent.train_with_full_experience()

print 'NN finished learning. Starting test'

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
    if done:
        print 'DONE: ', steps, 'moves.',
    else:
        print 'FAIL: ',
    print 'Found', walls, 'walls'
    print 'SCORE:', steps+walls, '(PERFECT:', env.heights[env.cur_task.start[0]][env.cur_task.start[1]], ')'

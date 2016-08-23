from dlpf.base_utils import *
import gym
from dlpf.agents import DqnAgent, RandomAgent, FlatAgent
from dlpf.io import *

logger = init_log(out_file = 'import.log', stderr = False)
import_tasks_from_xml_to_compact('data/sample/raw/', 'data/sample/imported/')

logger = init_log(out_file = 'testbed.log', stderr = False)


env = gym.make('PathFindingByPixel-v2')
env.configure(tasks_dir = os.path.abspath('data/sample/imported/'), monitor_scale = 10, map_shape = (10, 10))
env.monitor.start('data/sample/results/basic_dqn', force=True, seed=0)
agent = FlatAgent(state_size = env.observation_space.shape,
                 number_of_actions = env.action_space.n,
                 save_name = env.__class__.__name__)
agent.build_model()

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
    if done:
        print 'DONE: ', steps, 'moves.',
    else:
        print 'FAIL: ',
    print 'Found', walls, 'walls'
    if _ % 100 == 99:
        print 'iteration:', _ + 1
        agent.plot_layers(to_save='iteration'+str(_+1))
    if _ % 10 == 9:
        agent.train_with_full_experience()

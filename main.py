from dlpf.base_utils import *
import gym
from dlpf.agents import DqnAgent, RandomAgent
from dlpf.io import *

logger = init_log(out_file = 'import.log', stderr = False)
import_tasks_from_xml_to_compact('data/sample/raw/', 'data/sample/imported/')

logger = init_log(out_file = 'testbed.log', stderr = False)


env = gym.make('PathFindingByPixel-v1')
env.configure(tasks_dir = os.path.abspath('data/sample/imported/'), monitor_scale = 10, map_shape = (10, 10))
env.monitor.start('data/sample/results/basic_dqn', force=True, seed=0)
agent = DqnAgent(state_size = env.observation_space.shape,
                 number_of_actions = env.action_space.n,
                 save_name = env.__class__.__name__)

episode_count = 10000
max_steps = 50

for _ in xrange(episode_count):
    observation = env.reset()
    agent.new_episode()
    for __ in range(max_steps):
        action, values = agent.act(observation)
        observation, reward, done, info = env.step(action)
        agent.observe(reward)
        if done:
            break
    if _ % 100 == 99:
        print _ + 1

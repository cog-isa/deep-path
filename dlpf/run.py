import logging
import os

from .gym_environ.base import InfoValues
from .stats import StatHolder
from .utils.base_utils import rename_and_update

logger = logging.getLogger(__name__)


def apply_agent(environment,
                agent,
                episodes_number=5000,
                max_steps=100,
                initial_reward=0,
                allow_train=False,
                train_each_episodes=10,
                visualization_dir=None,
                visualize_each=10):
    logger.info('Applying agent %s to env %s in %d episodes, %d max steps, %s training' % (repr(agent),
                                                                                           repr(environment),
                                                                                           episodes_number,
                                                                                           max_steps,
                                                                                           'with' if allow_train else 'without'))
    stat = StatHolder()
    prev_result = InfoValues.NOTHING

    need_visualize = visualize_each and visualization_dir

    for episode_i in xrange(1, episodes_number + 1):
        logger.info('Start episode %d' % episode_i)

        new_episode_info = dict(prev_result=prev_result)
        rename_and_update(new_episode_info, 'env %s', **environment.get_episode_stat())
        rename_and_update(new_episode_info, 'agent %s', **agent.get_episode_stat())
        stat.new_episode(**new_episode_info)

        observation = environment.reset()
        agent.new_episode(environment.get_global_goal())

        reward, done = (initial_reward, False) if allow_train else (None, None)

        for step_i in range(max_steps):
            action = agent.act(observation)
            next_observation, reward, done, info = environment.step(action)
            stat.add_step(reward=reward, info=info)

            if allow_train:
                agent.update_memory(observation, action, reward, next_observation, done)
            observation = next_observation
            if done:
                break

        if need_visualize and episode_i % visualize_each == 0:
            environment.visualize_episode(os.path.join(visualization_dir, '%05d.png' % episode_i))

        prev_result = InfoValues.DONE if done else InfoValues.NOTHING

        if allow_train and episode_i % train_each_episodes == 0:
            agent.train_on_memory()

    logger.info('Agent %s completed %d episodes with env %s' % (repr(agent),
                                                                episodes_number,
                                                                repr(environment)))
    return stat

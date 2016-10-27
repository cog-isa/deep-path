import logging
from .stats import StatHolder
from .gym_environ.base import InfoValues

logger = logging.getLogger(__name__)


def apply_agent(environment,
                agent,
                episodes_number = 3000,
                max_steps = 100,
                initial_reward = 0,
                allow_train = False,
                train_each_episodes = 10):
    logger.info('Applying agent %s to env %s in %d episodes, %d max steps, %s training' % (repr(agent),
                                                                                           repr(environment),
                                                                                           episodes_number,
                                                                                           max_steps,
                                                                                           'with' if allow_train else 'without'))
    stat = StatHolder()
    prev_result = InfoValues.NOTHING

    for episode_i in xrange(episodes_number):
        logger.info('Start episode %d' % episode_i)

        observation = environment.reset()
        agent.new_episode()
        stat.new_episode(optimal_score = environment.current_optimal_score(),
                         prev_result = prev_result)

        reward, done = (initial_reward, False) if allow_train else (None, None)

        for step_i in range(max_steps):
            action = agent.act(observation, reward = reward, done = done)
            observation, reward, done, info = environment.step(action)
            #print '-'*50 #environment.cur_task.local_map
            #print observation
            #print reward, observation[4][4], done
            stat.add_step(reward = reward, info = info)

            if not allow_train:
                reward, done = (None, None)

            if done:
                agent.act(observation, reward = reward, done = done)
                break

        prev_result = InfoValues.DONE if done else InfoValues.NOTHING

        if allow_train and (episode_i + 1) % train_each_episodes == 0:
            agent.train_on_memory()

    logger.info('Agent %s completed %d episodes with env %s' % (repr(agent),
                                                                episodes_number,
                                                                repr(environment)))
    return stat

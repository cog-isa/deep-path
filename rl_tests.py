import gym
import numpy as np
import copy

# Deep-Q learning Agent
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.memory = []
        self.gamma = 0.9  # decay rate
        self.epsilon = 1  # exploration
        self.epsilon_decay = .995
        self.epsilon_min = 0.1
        self.learning_rate = 0.0001
        self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=4, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        batches = min(batch_size, len(self.memory))
        batches = np.random.choice(len(self.memory), batches)
        for i in batches:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                target = reward + self.gamma * \
                                  np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":

    # initialize gym environment and the agent
    env = gym.make('CartPole-v0')
    agent = DQNAgent(env)

    episodes = 5000

    # Iterate the game
    for e in range(episodes):

        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, 4])

        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible
        # the more time_t the more score
        for time_t in range(5000):
            # turn this on if you want to render
            # env.render()

            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            # (reward defaults to 1)
            # reward the agent 1 for every frame it lived
            # and punish -100 for dying
            reward = -100 if done else reward

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = copy.deepcopy(next_state)

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, time_t))
                break
        # train the agent with the experience of the episode
        agent.replay(32)

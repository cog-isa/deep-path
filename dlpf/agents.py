import random
import numpy
from scipy.spatial.distance import euclidean
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge
from keras.optimizers import RMSprop, Adam
from keras import backend as K




class DqnAgent(object):
    def __init__(self, state_size=None, number_of_actions=1,
                 epsilon=0.1, mbsz=32, discount=0.9, memory=50,
                 save_name='basic', save_freq=10):
        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.mbsz = mbsz
        self.discount = discount
        self.memory = memory
        self.save_name = save_name
        self.states = []
        self.actions = []
        self.rewards = []
        self.experience = []
        self.i = 1
        self.save_freq = save_freq
        self.vision_range = 3

    def cut_seen(self, pos):
        seen = numpy.ndarray(shape=(1, 2, 2*self.vision_range+1, 2*self.vision_range+1))
        walls, target, path = self.states[-1][-1]
        center_y, center_x = pos
        height, width = len(walls), len(walls[0])
        for dy in range(-self.vision_range, self.vision_range+1):
            for dx in range(-self.vision_range, self.vision_range+1):
                x, y = center_x+dx, center_y+dy
                if x < 0 or x >= width or y < 0 or y >= height:
                    seen[0][0][dx][dy] = 1
                    seen[0][1][dx][dy] = 0
                else:
                    seen[0][0][dx][dy] = walls[y][x]
                    seen[0][1][dx][dy] = target[y][x]*numpy.log1p(euclidean(pos, (y, x)))
        #if max([max(i) for i in target]) == 0:
        return seen


    def build_model(self):
        #S = Input(shape=self.state_size)
        S = Input(shape=(2, 2*self.vision_range+1, 2*self.vision_range+1))
        #h = Convolution2D(16, 8, 8, subsample=(4, 4),
        #    border_mode='same', activation='relu')(S)
        #h = Convolution2D(32, 4, 4, subsample=(2, 2),
        #    border_mode='same', activation='relu')(h)
        h = Flatten()(S)
        h = Dense(50, activation='relu')(h)
        V = Dense(self.number_of_actions)(h)
        self.model = Model(S, V)
        try:
            self.model.load_weights('{}.h5'.format(self.save_name))
            print "loading from {}.h5".format(self.save_name)
        except:
            print "Training a new model"
        self.model.compile(RMSprop(), loss='categorical_crossentropy')

    def new_episode(self):
        self.states.append([])
        self.actions.append([])
        self.rewards.append([])
        self.states = self.states[-self.memory:]
        self.actions = self.actions[-self.memory:]
        self.rewards = self.rewards[-self.memory:]
        self.i += 1
        if self.i % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.save_name), True)

    def act(self, state, pos):
        self.states[-1].append(state)
        #values = self.model.predict([state[None, :]])
        #print self.cut_seen(pos)
        values = self.model.predict(self.cut_seen(pos))
        if numpy.random.random() < self.epsilon:
            action = numpy.random.randint(self.number_of_actions)
        else:
            action = values.argmax()
        self.actions[-1].append(action)
        return action, values

    def observe(self, reward, action, pos):
        input_layer = self.cut_seen(pos)
        self.rewards[-1].append(reward)
        #input_layer = numpy.ndarray(shape=(1, 3, 10, 10))
        #input_layer[0] = [[[self.states[-1][-1][x][y][z]
        #                    for z in range(len(self.states[-1][-1][x][y]))]
        #                   for y in range(len(self.states[-1][-1][x]))]
        #                  for x in range(len(self.states[-1][-1]))]
        filled_reward = numpy.ndarray(shape=(1, self.number_of_actions))
        filled_reward[0][action] = reward
        self.model.fit(input_layer, filled_reward, nb_epoch=1, verbose=0)



class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

import random
import numpy
from scipy.spatial.distance import euclidean
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge
from keras.optimizers import RMSprop, Adam
from keras import backend as K





class DqnAgent(object):
    def __init__(self, state_size=None, number_of_actions=1,
                 epsilon=0.1, mbsz=32, discount=0.9, memory=1000,
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
        self.vision_range = 5

    def cut_seen(self, state, pos):
        seen = numpy.ndarray(shape=(1, 2, 2*self.vision_range+1, 2*self.vision_range+1))
        walls, target, path = state
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
        #print self.memory
        self.states.append([])
        self.actions.append([])
        self.rewards.append([])
        self.states = self.states[-self.memory:]
        self.actions = self.actions[-self.memory:]
        self.rewards = self.rewards[-self.memory:]
        self.i += 1
        if self.i % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.save_name), True)

    def generate_batches(self, batch_size, starting_from=0, num_of_layers=2, width_of_layers=11, height_of_layers=11):
        input_batch = numpy.zeros((batch_size, num_of_layers, width_of_layers, height_of_layers),
                                  dtype = 'float32')
        output_batch = numpy.zeros((batch_size, self.number_of_actions),
                                   dtype = 'float32')
        random.seed()
        inputs = []
        outputs = []
        print starting_from, len(self.states), len(self.states[0])
        for i in range(len(self.states)):
            inputs += self.states[i]
            outputs += self.rewards[i]
        print 'length:', len(inputs)
        if len(inputs) == len(outputs):
            unused = set(range(len(outputs)))
            while len(unused) > 0:
                batch_indexes = random.sample(unused, batch_size)
                #for i in batch_indexes:
                #    unused.remove(i)
                j = 0
                for i in batch_indexes:
                    for x in xrange(num_of_layers):
                        for y in xrange(width_of_layers):
                            for z in xrange(height_of_layers):
                                input_batch[j][x][y][z] = inputs[i][0][x][y][z]
                    j += 1
                j = 0
                for i in batch_indexes:
                    for x in xrange(self.number_of_actions):
                        output_batch[j][x] = outputs[i][0][x]
                    j += 1

                yield {'input': input_batch,
                       'output': output_batch}
                #yield [input_batch, output_batch]

    def act(self, state, pos):
        self.states[-1].append(self.cut_seen(state, pos))
        #values = self.model.predict([state[None, :]])
        #print self.cut_seen(pos)
        values = self.model.predict(self.states[-1][-1])
        if numpy.random.random() < self.epsilon:
            action = numpy.random.randint(self.number_of_actions)
        else:
            action = values.argmax()
        self.actions[-1].append(action)
        return action, values

    def observe(self, reward, action, pos):
        input_layer = self.states[-1][-1]
        #input_layer = numpy.ndarray(shape=(1, 3, 10, 10))
        #input_layer[0] = [[[self.states[-1][-1][x][y][z]
        #                    for z in range(len(self.states[-1][-1][x][y]))]
        #                   for y in range(len(self.states[-1][-1][x]))]
        #                  for x in range(len(self.states[-1][-1]))]
        filled_reward = numpy.ndarray(shape=(1, self.number_of_actions))
        filled_reward[0][action] = reward
        self.rewards[-1].append(filled_reward)
        #self.model.fit(input_layer, filled_reward, nb_epoch=1, verbose=0)

    def train_with_full_experience(self, start):
        batches = self.generate_batches(batch_size=100,
                                        starting_from=start)
        self.model.fit_generator(batches,
                                 samples_per_epoch=len(self.states*100),
                                 nb_epoch=10,
                                 verbose=1)


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

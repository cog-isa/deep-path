import random, numpy, collections
from scipy.spatial.distance import euclidean

import keras
from keras.models import Model
from keras.layers import Dense, Input

from dlpf.keras_utils import get_optimizer, choose_samples_per_epoch
from .policies import get_action_policy


def replay_train_data_generator(episodes, batch_size, actions_number, rand = random.Random()):
    '''
    Episodes is a list of list of MemoryRecord or compatible type
    '''
    if not episodes:
        return

    input_batch = numpy.zeros((batch_size, ) + episodes[0][0].observation.shape,
                              dtype = 'float32')
    output_batch = numpy.zeros((batch_size, actions_number),
                               dtype = 'float32')
    batch_i = 0

    while True:
        episode = episodes[rand.randint(0, len(episodes))]
        step = episode[rand.randint(0, len(episodes[episode_i]))]

        input_batch[batch_i] = step.observation
        output_batch[batch_i, step.action] = step.reward

        batch_i += 1
        if batch_i % batch_size == 0:
            yield (input_batch, output_batch)
            output_batch[:] = 0
            batch_i = 0


def split_train_val_replay_gens(episodes, batch_size, actions_number, val_part = 0.1, rand = random.Random()):
    indices = range(len(observations))
    rand.shuffle(indices)
    
    split_i = int(len(indices) * (1 - val_part))
    train_indices = indices[:split_i]
    val_indices = indices[split_i:]

    return (replay_train_data_generator(memory[train_indices], batch_size, actions_number, rand = rand),
            replay_train_data_generator(memory[val_indices], batch_size, actions_number, rand = rand))


MemoryRecord = collections.namedtuple('MemoryRecord',
                                      'observation action reward')


class BaseKerasAgent(object):
    def __init__(self,
                 input_shape = None,
                 number_of_actions = 1,
                 action_policy = get_action_policy(),
                 max_memory_size = 250,
                 loss = 'squared_hinge',
                 optimizer = get_optimizer(),
                 model_metrics = [],
                 model_callbacks = [],
                 epoch_number = 100,
                 passes_over_train_data = 2,
                 validation_part = 0.1,
                 batch_size = 32,
                 keras_verbose = 0,
                 split_rand = random.Random()):
        self.input_shape = input_shape
        self.number_of_actions = number_of_actions
        self.action_policy = action_policy
        self.max_memory_size = max_memory_size
        self.loss = loss
        self.optimizer = optimizer
        self.model_metrics = model_metrics
        self.model_callbacks = model_callbacks
        self.epoch_number = epoch_number
        self.passes_over_train_data = passes_over_train_data
        self.validation_part = validation_part
        self.batch_size = batch_size
        self.keras_verbose = keras_verbose
        self.split_rand = split_rand

        self.memory = []

    ##############################################################
    ###################### Basic agent logic #####################
    ##############################################################
    def build_model(self):
        input_layer = Input(shape = self.input_shape)
        inner_model = self._build_inner_model(input_layer)
        V = Dense(self.number_of_actions)(inner_model)
        self.model = Model(S, V)
        self.model.compile(self.optimizer,
                           loss = self.loss,
                           metrics = self.model_metrics)

    def new_episode(self):
        self.memory.append([])
        self.memory = self.memory[-self.memory_size:]

    def act(self, observation, reward = None, done = 0):
        action_probabilities = self.model.predict(observation)
        action = self.action_policy.choose_action(action_probabilities)

        if not reward is None: # that means that we can learn
            mem_rec = self._prepare_memory_record(observation,
                                                  action_probabilities,
                                                  action,
                                                  reward,
                                                  done)
            self.memory[-1].append(mem_rec)

        return action

    def train_on_memory(self):
        train_gen, val_gen = self._gen_train_val_data_from_memory()

        total_samples = sum(len(ep) for ep in self.memory)
        (train_samples_per_epoch,
         val_samples_per_epoch) = choose_samples_per_epoch(total_samples,
                                                           self.batch_size,
                                                           self.validation_part,
                                                           self.passes_over_train_data,
                                                           self.epoch_number)
        self.model.fit_generator(train_gen,
                                 train_samples_per_epoch,
                                 self.epoch_number,
                                 verbose = self.keras_verbose,
                                 callbacks = self.model_callbacks,
                                 validation_data = val_gen,
                                 nb_val_samples = val_samples_per_epoch)

    ##############################################################
    ################# Methods optional to implement ##############
    ##############################################################
    def plot_layers(self, to_save=''):
        pass

    def _prepare_memory_record(self, observation, action_probabilities, action, reward, done):
        return MemoryRecord(observation, action, reward)

    def _gen_train_val_data_from_memory(self):
        '''If we have another self.memory strucutre, then we will need to override this'''
        return split_train_val_replay_gens(self.memory,
                                           self.batch_size,
                                           self.number_of_actions,
                                           self.validation_part,
                                           rand = self.split_rand)

    ##############################################################
    ################ Methods mandatory to implement ##############
    ##############################################################
    def _build_inner_model(self, input_layer):
        raise NotImplemented()

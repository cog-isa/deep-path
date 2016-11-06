import random, numpy, collections, logging
from scipy.spatial.distance import euclidean

import keras
from keras.models import Model
from keras.layers import Dense, Input, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from dlpf.base_utils import load_object_from_dict
from dlpf.keras_utils import get_optimizer, choose_samples_per_epoch
from .policies import get_action_policy, BaseActionPolicy

from .training_data_gen import replay_train_data_generator

logger = logging.getLogger(__name__)


def split_train_val_replay_gens(episodes, batch_size, actions_number, val_part = 0.1, output_type = 'free_hinge', rand = random.Random()):
    indices = range(len(episodes))
    rand.shuffle(indices)

    split_i = int(len(indices) * (1 - val_part))
    train_indices = indices[:split_i]
    val_indices = indices[split_i:]

    return (replay_train_data_generator([episodes[i] for i in train_indices],
                                        batch_size,
                                        actions_number,
                                        output_type,
                                        rand = rand),
            replay_train_data_generator([episodes[i] for i in val_indices],
                                        batch_size,
                                        actions_number,
                                        output_type,
                                        rand = rand))


MemoryRecord = collections.namedtuple('MemoryRecord',
                                      'observation action reward')


class BaseKerasAgent(object):
    def __init__(self,
                 input_shape = None,
                 number_of_actions = 1,
                 action_policy = get_action_policy(),
                 max_memory_size = 250,
                 output_activation = 'linear',
                 loss = 'mean_squared_error',
                 optimizer = get_optimizer(),
                 model_metrics = [],
                 model_callbacks = [],
                 epoch_number = 100,
                 passes_over_train_data = 2,
                 validation_part = 0.1,
                 batch_size = 32,
                 keras_verbose = 0,
                 train_gen_processes_number = 4,
                 train_gen_queue_size = 100,
                 early_stopping_patience = 20,
                 reduce_lr_on_plateau_factor = 0.2,
                 reduce_lr_on_plateau_patience = 10,
                 train_data_output_type = 'free_hinge',
                 split_rand = random.Random()):
        self.input_shape = input_shape
        self.number_of_actions = number_of_actions
        self.action_policy = get_action_policy(action_policy) \
            if isinstance(action_policy, (str, unicode)) \
            else get_action_policy(**action_policy)
        self.max_memory_size = max_memory_size
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = get_optimizer(optimizer) \
            if isinstance(optimizer, (str, unicode)) \
            else get_optimizer(**optimizer)
        self.model_metrics = model_metrics
        self.model_callbacks = model_callbacks
        self.epoch_number = epoch_number
        self.passes_over_train_data = passes_over_train_data
        self.validation_part = validation_part
        self.batch_size = batch_size
        self.keras_verbose = keras_verbose
        self.train_gen_processes_number = train_gen_processes_number
        self.train_gen_queue_size = train_gen_queue_size
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_on_plateau_factor = reduce_lr_on_plateau_factor
        self.reduce_lr_on_plateau_patience = reduce_lr_on_plateau_patience
        self.train_data_output_type = train_data_output_type
        self.split_rand = split_rand

        self.model_callbacks.append(ReduceLROnPlateau(monitor = 'val_loss',
                                                      factor = self.reduce_lr_on_plateau_factor,
                                                      patience = self.reduce_lr_on_plateau_patience,
                                                      verbose = self.keras_verbose,
                                                      mode = 'min'))
        self.model_callbacks.append(EarlyStopping(monitor = 'val_loss',
                                                  patience = self.early_stopping_patience,
                                                  verbose = self.keras_verbose,
                                                  mode = 'min'))

        self.memory = []
        self.prev_step_info = None
        
        self._build_model()

    ##############################################################
    ###################### Basic agent logic #####################
    ##############################################################
    def _build_model(self):
        input_layer = Input(shape = self.input_shape)
        inner_model = self._build_inner_model(input_layer)
        last_layer = Dense(self.number_of_actions)(inner_model)
        output_layer = Activation(self.output_activation)(last_layer)
        self.model = Model(input_layer, output_layer)
        self.model.compile(self.optimizer,
                           loss = self.loss,
                           metrics = self.model_metrics)
        self.model.summary()

    def new_episode(self):
        self.memory.append([])
        self.memory = self.memory[-self.max_memory_size:]
        self.prev_step_info = None
        self.action_policy.new_episode()

    def act(self, observation, reward = None, done = None):
        action_probabilities = self.model.predict(observation.reshape((1,) + observation.shape))
        action = self.action_policy.choose_action(action_probabilities)

        if not reward is None: # that means that we can learn
            if not self.prev_step_info is None:
                self.prev_step_info['reward'] = reward
                self.memory[-1].append(self._prepare_memory_record(**self.prev_step_info))
            self.prev_step_info = dict(observation = observation,
                                       action_probabilities = action_probabilities,
                                       action = action,
                                       reward = reward,
                                       done = done)

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
                                 nb_val_samples = val_samples_per_epoch,
                                 max_q_size = self.train_gen_queue_size,
                                 nb_worker = self.train_gen_processes_number,
                                 pickle_safe = False)

    def get_episode_stat(self):
        return {}

    ##############################################################
    ################# Methods optional to implement ##############
    ##############################################################
    def plot_layers(self, to_save=''):
        pass

    def _prepare_memory_record(self, observation = None, action_probabilities = None, action = None, reward = None, done = None):
        rec = MemoryRecord(observation, action, reward)
        # logger.debug('Add memory record\n%s' % repr(rec))
        return rec

    def _gen_train_val_data_from_memory(self):
        '''If we have another self.memory strucutre, then we will need to override this'''
        return split_train_val_replay_gens(self.memory,
                                           self.batch_size,
                                           self.number_of_actions,
                                           val_part = self.validation_part,
                                           output_type = self.train_data_output_type,
                                           rand = self.split_rand)

    ##############################################################
    ################ Methods mandatory to implement ##############
    ##############################################################
    def _build_inner_model(self, input_layer):
        raise NotImplemented()

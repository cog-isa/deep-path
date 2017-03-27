import collections
import logging
import random
import numpy as np

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.optimizers import Optimizer

from dlpf.agents.policies.action import DEFAULT_ACTION_POLICY
from dlpf.utils.keras_utils import get_optimizer, choose_samples_per_epoch, DEFAULT_OPTIMIZER
from .policies import get_action_policy
from .training_data_gen import replay_train_data_generator

logger = logging.getLogger(__name__)

MemoryRecord = collections.namedtuple('MemoryRecord',
                                      'observation action reward next_observation done')


class BaseKerasAgent(object):
    def __init__(self,
                 input_shape=None,
                 number_of_actions=1,
                 episodes_number=5000,
                 action_policy=DEFAULT_ACTION_POLICY,
                 max_memory_size=250,
                 q_gamma=0.1,
                 output_activation='linear',
                 loss='mean_squared_error',
                 optimizer=DEFAULT_OPTIMIZER,
                 model_metrics=None,
                 model_callbacks=None,
                 epoch_number=100,
                 passes_over_train_data=2,
                 validation_part=0.1,
                 batch_size=32,
                 keras_verbose=0,
                 train_gen_processes_number=4,
                 train_gen_queue_size=100,
                 early_stopping_patience=20,
                 reduce_lr_on_plateau_factor=0.2,
                 reduce_lr_on_plateau_patience=10,
                 train_data_output_type='free_hinge',
                 split_rand=random.Random()):
        self.input_shape = input_shape
        self.number_of_actions = number_of_actions
        self.q_gamma = q_gamma
        self.action_policy = get_action_policy(action_policy) \
            if isinstance(action_policy, (str, unicode)) \
            else get_action_policy(episodes_number=episodes_number, **action_policy)
        self.max_memory_size = max_memory_size
        self.output_activation = output_activation
        self.loss = loss

        if isinstance(optimizer, (str, unicode)):
            self.optimizer = get_optimizer(optimizer)
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = get_optimizer(**optimizer)

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

        self.goal = None

        if self.model_callbacks is None:
            self.model_callbacks = []
        self.model_callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                                      factor=self.reduce_lr_on_plateau_factor,
                                                      patience=self.reduce_lr_on_plateau_patience,
                                                      verbose=self.keras_verbose,
                                                      mode='min'))
        self.model_callbacks.append(EarlyStopping(monitor='val_loss',
                                                  patience=self.early_stopping_patience,
                                                  verbose=self.keras_verbose,
                                                  mode='min'))

        self.memory = []
        self.prev_step_info = None

        self._build_model()

    def __repr__(self):
        return self.__class__.__name__

    ##############################################################
    ###################### Basic agent logic #####################
    ##############################################################
    def _build_model(self):
        input_layer = Input(shape=self.input_shape)
        inner_model = self._build_inner_model(input_layer)
        last_layer = Dense(self.number_of_actions)(inner_model)
        output_layer = Activation(self.output_activation)(last_layer)
        self.model = Model(input_layer, output_layer)
        self.model.compile(self.optimizer,
                           loss=self.loss,
                           metrics=self.model_metrics)
        logger.info('Making new agent: {}'.format(self.__class__.__name__))
        self.model.summary()

    def new_episode(self, goal):
        self.memory.append([])
        self.memory = self.memory[-self.max_memory_size:]
        self.prev_step_info = None
        self.action_policy.new_episode()
        self.goal = goal

    def act(self, observation):
        action_probabilities = self._predict_action_probabilities(observation)
        action = self.action_policy.choose_action(action_probabilities)

        return action

    def train_on_memory(self):
        total_samples = sum(len(ep) for ep in self.memory)
        if total_samples == 0:
            return

        indices = range(len(self.memory))
        self.split_rand.shuffle(indices)

        split_i = int(len(indices) * (1 - self.validation_part))
        train_indices = indices[:split_i]
        val_indices = indices[split_i:]

        def predict_feature_r(indices):
            q_predictions = []
            targets = []
            for i in indices:
                episode_predictions = []
                episode_targets = []
                for j in range(len(self.memory[i])):
                    obs = self.memory[i][j].observation
                    next_obs = self.memory[i][j].next_observation
                    episode_predictions.append(self.q_gamma * np.max(self._predict_action_probabilities(next_obs)))
                    episode_targets.append(self._predict_action_probabilities(obs))
                q_predictions.append(episode_predictions)
                targets.append(episode_targets)
            return q_predictions, targets

        train_q_predictions, train_targets = predict_feature_r(train_indices)
        val_q_predictions, val_targets = predict_feature_r(val_indices)

        logger.info(
            'Start training: \n\t q_predictions = {}\n\t targets = {}'.format(train_q_predictions, train_targets))

        train_gen = replay_train_data_generator([self.memory[i] for i in train_indices],
                                                train_q_predictions,
                                                train_targets,
                                                self.batch_size,
                                                self.number_of_actions,
                                                self.train_data_output_type,
                                                rand=self.split_rand)
        val_gen = replay_train_data_generator([self.memory[i] for i in val_indices],
                                              val_q_predictions,
                                              val_targets,
                                              self.batch_size,
                                              self.number_of_actions,
                                              self.train_data_output_type,
                                              rand=self.split_rand)

        (train_samples_per_epoch,
         val_samples_per_epoch) = choose_samples_per_epoch(total_samples,
                                                           self.batch_size,
                                                           self.validation_part,
                                                           self.passes_over_train_data,
                                                           self.epoch_number)
        self.model.fit_generator(train_gen,
                                 train_samples_per_epoch,
                                 self.epoch_number,
                                 verbose=self.keras_verbose,
                                 callbacks=self.model_callbacks,
                                 validation_data=val_gen,
                                 nb_val_samples=val_samples_per_epoch,
                                 max_q_size=self.train_gen_queue_size,
                                 nb_worker=self.train_gen_processes_number,
                                 pickle_safe=False)

    def get_episode_stat(self):
        return {}

    ##############################################################
    ################# Methods optional to implement ##############
    ##############################################################
    def update_memory(self, observation, action, reward, next_observation, done):
        self.memory[-1].append(MemoryRecord(observation, action, reward, next_observation, done))

    ##############################################################
    ################ Methods mandatory to implement ##############
    ##############################################################
    def _predict_action_probabilities(self, observation):
        raise NotImplementedError()

    def _build_inner_model(self, input_layer):
        raise NotImplementedError()


class BaseStandaloneKerasAgent(BaseKerasAgent):
    def _predict_action_probabilities(self, observation):
        return self.model.predict(observation.reshape((1,) + observation.shape))

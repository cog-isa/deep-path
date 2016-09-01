from keras.callbacks import Callback
from keras.optimizers import RMSprop, Adagrad, Nadam, Adadelta
from .plot_utils import basic_plot_df
from .base_utils import add_filename_suffix, copy_except, floor_to_number


LOGS_TO_IGNORE = ['batch', 'size']

class LossHistory(Callback):
    def on_train_begin(self, logs = {}, logs_to_ignore = LOGS_TO_IGNORE):
        self.batch_data = []
        self.epoch_data = []
        self.logs_to_ignore = logs_to_ignore

    def on_batch_end(self, batch, logs = {}):
        self.batch_data.append(copy_except(logs, self.logs_to_ignore))

    def on_epoch_end(self, epoch, logs = {}):
        self.epoch_data.append(copy_except(logs, self.logs_to_ignore))

    def plot(self, out_file):
        basic_plot_df(self.batch_data,
                      out_file = add_filename_suffix(out_file, '_batch') if out_file else None)
        basic_plot_df(self.epoch_data,
                      out_file = add_filename_suffix(out_file, '_epoch') if out_file else None)


_OPTIMIZERS = {
    'rmsprop' : RMSprop,
    'adagrad' : Adagrad,
    'nadam' : Nadam,
    'adadelta' : Adadelta
}
DEFAULT_OPTIMIZER = 'rmsprop'

def get_available_optimizers():
    return list(_OPTIMIZERS.keys())

def get_optimizer(name = DEFAULT_OPTIMIZER, *args, **kwargs):
    assert name in _OPTIMIZERS, 'Unknown optimizer %s' % name
    return _OPTIMIZERS[name](*args, **kwargs)

def choose_samples_per_epoch(total_samples_number, batch_size, val_part, passes_over_train_data, epoch_number):
    train_samples_total = total_samples * (1 - self.validation_part)
    train_samples_per_epoch_raw = train_samples_total * passes_over_train_data / epoch_number

    val_samples_total = total_samples * self.validation_part
    val_samples_per_epoch_raw = val_samples_total * passes_over_train_data / epoch_number
    
    return (floor_to_number(train_samples_per_epoch_raw, batch_size),
            floor_to_number(val_samples_per_epoch_raw, batch_size))
import re, os, logging, numpy
from keras.callbacks import Callback
from keras.optimizers import RMSprop, Adagrad, Nadam, Adadelta
from .base_utils import add_filename_suffix, copy_except, floor_to_number
from .stats import StatHolder


logger = logging.getLogger(__name__)


LOGS_TO_IGNORE = ['batch', 'size']


def copy_except_and_map(src, keys_to_ignore, converter):
    return { k : converter(v) for k, v in copy_except(src, keys_to_ignore).viewitems() }


def try_ensure_float(obj):
    try:
        return float(obj)
    except TypeError:
        return obj


class LossHistory(Callback):
    def __init__(self, logs_to_ignore = LOGS_TO_IGNORE):
        self.logs_to_ignore = logs_to_ignore
        self.batch_stats = StatHolder()
        self.epoch_stats = StatHolder()

    def on_train_begin(self, logs = {}):
        self.batch_stats = StatHolder()
        self.batch_stats.new_episode()
        self.epoch_stats = StatHolder()
        self.epoch_stats.new_episode()

    def on_batch_end(self, batch, logs = {}):
        self.batch_stats.add_step(**copy_except_and_map(logs,
                                                        self.logs_to_ignore,
                                                        try_ensure_float))

    def on_epoch_end(self, epoch, logs = {}):
        self.epoch_stats.add_step(**copy_except_and_map(logs,
                                                        self.logs_to_ignore,
                                                        try_ensure_float))


_OPTIMIZERS = {
    'rmsprop' : RMSprop,
    'adagrad' : Adagrad,
    'nadam' : Nadam,
    'adadelta' : Adadelta
}
DEFAULT_OPTIMIZER = 'rmsprop'

def get_available_optimizers():
    return list(_OPTIMIZERS.keys())

def get_optimizer(ctor = DEFAULT_OPTIMIZER, *args, **kwargs):
    assert name in _OPTIMIZERS, 'Unknown optimizer %s' % name
    return _OPTIMIZERS[name](*args, **kwargs)

def choose_samples_per_epoch(total_samples_number, batch_size, val_part, passes_over_train_data, epoch_number):
    train_samples_total = total_samples_number * (1 - val_part)
    train_samples_per_epoch_raw = train_samples_total * passes_over_train_data / epoch_number

    val_samples_total = total_samples_number * val_part
    val_samples_per_epoch_raw = val_samples_total * passes_over_train_data / epoch_number
    
    return (int(floor_to_number(train_samples_per_epoch_raw, batch_size)),
            int(floor_to_number(val_samples_per_epoch_raw, batch_size)))


_THEANO_DEVICES_TO_TRY = ['gpu1', 'gpu0']
_GET_DEVICE=re.compile('device=([^,]+)')
def try_assign_theano_on_free_gpu():
    match = _GET_DEVICE.search(os.environ['THEANO_FLAGS'])
    if match and match.group(1) == 'cpu':
        return

    import theano.sandbox.cuda
    for dev in _THEANO_DEVICES_TO_TRY:
        try:
            theano.sandbox.cuda.use(dev)
            return
        except:
            print traceback.format_exc()
            pass
    raise RuntimeError('no GPUs available')

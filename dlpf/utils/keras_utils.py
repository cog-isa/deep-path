import logging
import os
import re
import traceback

import keras.backend as K
import numpy
from keras.callbacks import Callback
from keras.optimizers import RMSprop, Adagrad, Nadam, Adadelta

from .base_utils import copy_except, floor_to_number
from dlpf.stats import StatHolder

logger = logging.getLogger(__name__)

LOGS_TO_IGNORE = ['batch', 'size']


class LossHistory(Callback):
    def __init__(self, logs_to_ignore=LOGS_TO_IGNORE):
        super(LossHistory, self).__init__()
        self.logs_to_ignore = logs_to_ignore
        self.batch_stats = StatHolder()
        self.epoch_stats = StatHolder()

    def on_train_begin(self, logs=None):
        self.batch_stats.new_episode()
        self.epoch_stats.new_episode()

    def on_batch_end(self, batch, logs=None):
        self.batch_stats.add_step(**copy_except_and_map(logs,
                                                        self.logs_to_ignore,
                                                        try_ensure_float))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_stats.add_step(**copy_except_and_map(logs,
                                                        self.logs_to_ignore,
                                                        try_ensure_float))


def copy_except_and_map(src, keys_to_ignore, converter):
    return {k: converter(v) for k, v in copy_except(src, keys_to_ignore).viewitems()}


def try_ensure_float(obj):
    try:
        return float(obj)
    except TypeError:
        return obj


_OPTIMIZERS = {
    'rmsprop': RMSprop,
    'adagrad': Adagrad,
    'nadam': Nadam,
    'adadelta': Adadelta
}
DEFAULT_OPTIMIZER = 'rmsprop'


def get_available_optimizers():
    return list(_OPTIMIZERS.keys())


def get_optimizer(ctor=DEFAULT_OPTIMIZER, *args, **kwargs):
    assert ctor in _OPTIMIZERS, 'Unknown optimizer %s' % ctor
    return _OPTIMIZERS[ctor](*args, **kwargs)


_THEANO_DEVICES_TO_TRY = ['gpu', 'gpu0', 'gpu1']
_GET_DEVICE = re.compile('device=([^,]+)')


def try_assign_theano_on_free_gpu():
    match = _GET_DEVICE.search(os.environ.get('THEANO_FLAGS', ''))
    if match and match.group(1) == 'cpu':
        return

    import theano.sandbox.cuda
    for dev in _THEANO_DEVICES_TO_TRY:
        try:
            theano.sandbox.cuda.use(dev, force=True)
            return
        except:
            logger.warning(traceback.format_exc())
    raise RuntimeError('no GPUs available')


def set_theano_compiledir(dirname):
    theano_config = os.environ.get('THEANO_FLAGS', '')
    if not 'base_compiledir' in theano_config:
        if theano_config:
            os.environ['THEANO_FLAGS'] += ',base_compiledir=' + dirname
        else:
            os.environ['THEANO_FLAGS'] = 'base_compiledir=' + dirname


def choose_samples_per_epoch(total_samples_number, batch_size, val_part, passes_over_train_data, epoch_number):
    train_samples_total = total_samples_number * (1 - val_part)
    train_samples_per_epoch_raw = train_samples_total * passes_over_train_data / epoch_number

    val_samples_total = total_samples_number * val_part
    val_samples_per_epoch_raw = val_samples_total * passes_over_train_data / epoch_number

    return (int(floor_to_number(train_samples_per_epoch_raw, batch_size)),
            int(floor_to_number(val_samples_per_epoch_raw, batch_size)))


def get_backend():
    return K.image_dim_ordering()


def get_tensor_reshaper():
    if get_backend() == 'tf':
        return lambda m: numpy.moveaxis(m, 0, -1)  # tensorflow is (rows, cols, layers)
    else:
        return lambda m: m


def add_depth(base_shape, depth=1):
    if get_backend() == 'tf':
        return base_shape + (depth,)
    else:
        return (depth,) + base_shape


def get_depth(base_shape):
    return base_shape[-1 if get_backend() == 'tf' else 0]


def update_depth(base_shape, new_depth):
    if get_backend() == 'tf':
        return base_shape[:-1] + (new_depth,)
    else:
        return (new_depth,) + base_shape[1:]


def increase_depth(base_shape, factor):
    return update_depth(base_shape, get_depth(base_shape) * factor)


def concat_tensors(a, b):
    if get_backend() == 'tf':
        return numpy.concatenate((a, b), axis=-1)
    else:
        return numpy.concatenate((a, b), axis=0)

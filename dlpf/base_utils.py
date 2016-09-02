# This is a file to just to include everything you need to not to worry much about it
# from base_utils import *
# logger = init_log(out_file = 'some_log.log', stderr = False)

import logging, sys, os, cPickle, itertools, \
    math, multiprocessing as mp, glob, pandas, \
    collections, functools, traceback, re, numpy, \
    ujson, importlib, yaml, shutil


LOGGING_LEVELS = {
    'debug' : logging.DEBUG,
    'info' : logging.INFO,
    'warning' : logging.WARNING,
    'error' : logging.ERROR,
    'critical' : logging.CRITICAL
}

def init_log(out_file = None, stderr = False, level = logging.DEBUG):
    all_loggers = [logging.getLogger()]

    log_formatter = logging.Formatter('%(asctime)-15s %(levelname)10s %(message)s')

    if stderr:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(log_formatter)
        stderr_handler.setLevel(logging.INFO)

    if out_file:
        if os.path.exists(out_file):
            os.remove(out_file)
        file_handler = logging.FileHandler(out_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.DEBUG)

    for logger in all_loggers:
        logger.handlers = []
        logger.setLevel(level)

        if stderr:
            logger.addHandler(stderr_handler)

        if out_file:
            logger.addHandler(file_handler)

    main_logger = all_loggers[0]
    main_logger.info('Logger initialized')
    return main_logger

def reporting_gen(in_gen, logger, report_every = 100, template = 'Processed %d items'):
    for i, el in enumerate(in_gen):
        yield el
        if i % report_every == 0:
            logger.info(template % i)

def save_obj(obj, fname):
    with open(fname, 'wb') as f:
        cPickle.dump(obj, f, 2)

def load_obj(fname):
    with open(fname, 'rb') as f:
        return cPickle.load(f)

def load_json(fname):
    with open(fname, 'r') as f:
        return ujson.load(f)

def save_json(obj, fname):
    with open(fname, 'w') as f:
        ujson.dump(obj, f, indent = 2)


class MultiIndexableDataset(object):
    def __init__(self, values, parser = ujson.loads):
        self.values = values
        self.parser = parser

    def __getitem__(self, idx):
        parser = self.parser
        if isinstance(idx, slice):
            return (parser(obj) for obj in self.values[idx])
        try:
            return (parser(self.values[i]) for i in idx)
        except TypeError:
            
            return parser(self.values[idx])


class FileReadingDataset(object):
    def __init__(self, parser = load_json):
        self.parser = parser

    def __getitem__(self, idx):
        return self.parser(idx)


def set_theano_compiledir(dirname):
    theano_config = os.environ.get('THEANO_FLAGS', '')
    if not 'base_compiledir' in theano_config:
        if theano_config:
            os.environ['THEANO_FLAGS'] += ',base_compiledir=' + dirname
        else:
            os.environ['THEANO_FLAGS'] = 'base_compiledir=' + dirname


_THEANO_DEVICES_TO_TRY = ['gpu1', 'gpu0']
_GET_DEVICE = re.compile('device=([^,]+)')
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


def no_copy_update(d, **updates):
    d.update(updates)
    return d


def copy_and_update(d, **updates):
    d = dict(d)
    d.update(updates)
    return d


def copy_except(src, fields_to_skip):
    return { k : v for k, v in src.viewitems() if not k in fields_to_skip}


def add_filename_suffix(fname, suffix):
    base_name, ext = os.path.splitext(fname)
    return base_name + suffix + ext


def floor_to_number(what, how):
    return int((float(what) / how) * how)


def load_yaml(fname):
    with open(fname, 'r') as f:
        return yaml.load(f)


def import_name_from_module(name):
    mod_name, cls_name = name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def load_object_by_name(name, *args, **kwargs):
    return import_name_from_module(name)(*args, **kwargs)


def load_object_from_dict(info, **override_kwargs):
    # print info
    return load_object_by_name(info['ctor'],
                               *info.get('args', []),
                               **copy_and_update(info.get('kwargs', {}),
                                                 **override_kwargs))


def load_object_from_yaml(fname, **override_kwargs):
    return load_object_from_dict(load_yaml(fname),
                                 **override_kwargs)


def ensure_dir_exists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def copy_files(from_dir, fnames, to_dir):
    for fname in fnames:
        shutil.copy2(os.path.join(from_dir, fname),
                     os.path.join(to_dir, fname))

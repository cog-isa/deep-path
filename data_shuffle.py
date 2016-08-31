import os
from random import randint


def split_data(path, dest, use_validation=True):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for f in onlyfiles:
        i = randint(0, 9)
        if i == 9:
            os.rename(os.path.join(path, f), os.path.join(dest['test'], f))
        elif i == 8 and use_validation:
            os.rename(os.path.join(path, f), os.path.join(dest['validate'], f))
        else:
            os.rename(os.path.join(path, f), os.path.join(dest['train'], f))


def join_data(path, dest, use_validation=True):
    onlyfiles = [f for f in os.listdir(dest['test']) if os.path.isfile(os.path.join(dest['test'], f))]
    for f in onlyfiles:
        os.rename(os.path.join(dest['test'], f), os.path.join(path, f))
    if use_validation:
        onlyfiles = [f for f in os.listdir(dest['validate']) if os.path.isfile(os.path.join(dest['validate'], f))]
        for f in onlyfiles:
            os.rename(os.path.join(dest['validate'], f), os.path.join(path, f))
    onlyfiles = [f for f in os.listdir(dest['train']) if os.path.isfile(os.path.join(dest['train'], f))]
    for f in onlyfiles:
        os.rename(os.path.join(dest['train'], f), os.path.join(path, f))


def shuffle_raw(to_join=False, to_split=False, val=True):
    path = 'data/sample/raw/'
    dest = {'test': path+'test/',
            'train': path+'train/',
            'validate': path+'validate/'}
    if to_join:
        join_data(path, dest, use_validation=val)
    if to_split:
        split_data(path, dest, use_validation=val)


def shuffle_imported_maps(to_join=False, to_split=False, val=True):
    path = 'data/sample/imported/maps/'
    dest = {'test': 'data/sample/imported/test/maps/',
            'train': 'data/sample/imported/train/maps/',
            'validate': 'data/sample/imported/validate/maps/'}
    if to_join:
        join_data(path, dest, use_validation=val)
    if to_split:
        split_data(path, dest, use_validation=val)


def shuffle_imported_paths(to_join=False, to_split=False, val=True):
    path = 'data/sample/imported/paths/'
    dest = {'test': 'data/sample/imported/test/paths/',
            'train': 'data/sample/imported/train/paths/',
            'validate': 'data/sample/imported/validate/paths/'}
    if to_join:
        join_data(path, dest, use_validation=val)
    if to_split:
        split_data(path, dest, use_validation=val)


shuffle_imported_paths(to_join=True, val=False)
shuffle_imported_maps(to_join=True, val=False)
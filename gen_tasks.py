#!/usr/bin/env python

import argparse, os, random

os.environ['KERAS_BACKEND'] = 'theano'

from dlpf.io import load_map_from_compact, PathFindingTask, save_to_compact
from dlpf.gym_environ.search_algo import EuclideanAStar


def _gen_point(map_shape):
    return (random.randint(0, map_shape[0] - 1), # randint is inclusive
            random.randint(0, map_shape[1] - 1))


if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-n',
                         type = int,
                         default = 10,
                         help = 'How much tasks to generate')
    aparser.add_argument('map_fname',
                         type = str,
                         help = 'Path to .npz-file with map to generate tasks for')
    aparser.add_argument('out_dir',
                         type = str,
                         help = 'Where to put tasks')
    
    args = aparser.parse_args()

    path_builder = EuclideanAStar()

    map_dir = os.path.dirname(args.map_fname)
    local_map = load_map_from_compact(args.map_fname)
    for i in xrange(args.n):
        while True:
            start = _gen_point(local_map.shape)
            finish = _gen_point(local_map.shape)
            if local_map[start] != 0 or local_map[finish] != 0:
                continue
            path_builder.reset(local_map, start, finish)
            path = path_builder.get_best_path()
            if not path is None:
                break
        task = PathFindingTask(str(i), local_map, start, finish, path)
        print task
        save_to_compact(task, map_dir, args.out_dir)

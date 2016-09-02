#!/usr/bin/env python

import argparse, os

from dlpf.benchmark import prepare_evaluation_splits

if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--folds', type = int, default = 3, help = 'How many splits to generate')
    aparser.add_argument('--test-part', type = float, default = 0.3, help = 'How many tasks will be reseved for test (evaluation)')
    aparser.add_argument('src_tasks_dir', type = str, help = 'Where source tasks reside')
    aparser.add_argument('target_folds_dir', type = str, help = 'Root directory to put folds under')

    args = aparser.parse_args()

    prepare_evaluation_splits(args.src_tasks_dir,
                              args.target_folds_dir,
                              folds = args.folds,
                              test_part = args.test_part)

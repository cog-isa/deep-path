#!/usr/bin/env python

import argparse, logging, os

from dlpf.base_utils import init_log, LOGGING_LEVELS, ensure_dir_exists
from dlpf.benchmark import evaluate_agent_with_configs
from dlpf.plot_utils import basic_plot_from_df
from dlpf.fglab_utils import create_scores_file
from dlpf.keras_utils import try_assign_theano_on_free_gpu


logger = logging.getLogger()


STATS_TITLES = 'train test batch epoch'.split(' ')


if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--env', type = str, help = 'Path to environment config to use')
    aparser.add_argument('--agent', type = str, help = 'Path to agent config to use')
    aparser.add_argument('--folds', type = str, help = 'Path to directory with cross-validation data')
    aparser.add_argument('--apply', type = str, help = 'Path to apply_agent config to use')
    aparser.add_argument('--output', type = str, help = 'Where to store results')
    aparser.add_argument('--level', type = str,
                         choices = LOGGING_LEVELS.keys(),
                         default = 'info',
                         help = 'Logging verbosity')

    args = aparser.parse_args()

    logger = init_log(stderr = True, level = LOGGING_LEVELS[args.level])

    try_assign_theano_on_free_gpu()

    all_stats = evaluate_agent_with_configs(args.env,
                                            args.agent,
                                            args.folds,
                                            args.apply)

    ensure_dir_exists(args.output)
    for stat_title, stat in zip(STATS_TITLES, all_stats):
        basic_plot_from_df(stat.episodes,
                           out_file = os.path.join(args.output, '%s_episodes.png' % stat_title))
        basic_plot_from_df(stat.full,
                           out_file = os.path.join(args.output, '%s_full.png' % stat_title))    
    create_scores_file(os.path.join(args.output, 'scores.json'),
                       train_score = all_stats[0].score,
                       test_score = all_stats[1].score)

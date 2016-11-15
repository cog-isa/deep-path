#!/usr/bin/env python

import argparse, logging, os

os.environ['KERAS_BACKEND'] = 'theano'

from dlpf.base_utils import init_log, LOGGING_LEVELS, ensure_dir_exists, \
    copy_yaml_configs_to_json, rename_and_update
from dlpf.benchmark import evaluate_agent_with_configs
from dlpf.plot_utils import basic_plot_from_df, basic_plot_from_df_rolling_mean
from dlpf.fglab_utils import create_scores_file, create_charts_file
from dlpf.keras_utils import try_assign_theano_on_free_gpu
from dlpf.perf_utils import Profiler

logger = logging.getLogger()


STATS_TITLES = 'train test batch epoch'.split(' ')

SERIES_NOT_TO_PLOT = frozenset({ 'optimal_score' })

if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--env', type = str, help = 'Path to environment config to use')
    aparser.add_argument('--agent', type = str, help = 'Path to agent config to use')
    aparser.add_argument('--folds', type = str, help = 'Path to directory with cross-validation data')
    aparser.add_argument('--apply', type = str, help = 'Path to apply_agent config to use')
    aparser.add_argument('--output', type = str, default = '.', help = 'Where to store results')
    aparser.add_argument('--_id', type = str, default = None, help = 'FGLab experiment id')
    aparser.add_argument('--level', type = str,
                         choices = LOGGING_LEVELS.keys(),
                         default = 'info',
                         help = 'Logging verbosity')

    args = aparser.parse_args()

    if args._id:
        args.output = os.path.join(args.output, args._id)
    ensure_dir_exists(args.output)

    logger = init_log(stderr = True,
                      level = LOGGING_LEVELS[args.level],
                      out_file = os.path.join(args.output, 'evaluate.log'))

    try_assign_theano_on_free_gpu()

    with Profiler(logger):
        all_stats = evaluate_agent_with_configs(args.env,
                                                args.agent,
                                                args.folds,
                                                args.apply)

    for stat_title, stat in zip(STATS_TITLES, all_stats):
        basic_plot_from_df(stat.episodes,
                           out_file = os.path.join(args.output, '%s_episodes.png' % stat_title),
                           ignore = SERIES_NOT_TO_PLOT)
        basic_plot_from_df_rolling_mean(stat.episodes,
                                        out_file = os.path.join(args.output, '%s_episodes_smoothed.png' % stat_title),
                                        ignore = SERIES_NOT_TO_PLOT)
        basic_plot_from_df(stat.full,
                           out_file = os.path.join(args.output, '%s_full.png' % stat_title),
                           ignore = SERIES_NOT_TO_PLOT)
        basic_plot_from_df_rolling_mean(stat.full,
                                        out_file = os.path.join(args.output, '%s_full_smoothed.png' % stat_title),
                                        ignore = SERIES_NOT_TO_PLOT)

    all_scores = {}
    rename_and_update(all_scores, 'train %s', all_stats[0].scores)
    rename_and_update(all_scores, 'test %s', all_stats[0].scores)
    create_scores_file(os.path.join(args.output, 'scores.json'),
                       **all_scores)
    create_charts_file(os.path.join(args.output, 'charts.json'),
                       **{ '_'.join((stat_title, attr)) : getattr(stat, attr)
                          for stat_title, stat in zip(STATS_TITLES, all_stats)
                          for attr in ('episodes', 'full') })

    copy_yaml_configs_to_json(os.path.join(args.output, 'configs.json'),
                              env = args.env,
                              agent = args.agent,
                              apply = args.apply)

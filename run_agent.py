#!/usr/bin/env python

import argparse, logging, os, time

#os.environ['KERAS_BACKEND'] = 'theano'
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['THEANO_FLAGS'] = 'device=cpu'

from dlpf.gym_environ import load_environment_from_yaml
from dlpf.base_utils import init_log, LOGGING_LEVELS, ensure_dir_exists, \
    load_object_from_yaml, load_yaml
from dlpf.benchmark import evaluate_agent_with_configs
from dlpf.plot_utils import basic_plot_from_df, basic_plot_from_df_rolling_mean
from dlpf.keras_utils import try_assign_theano_on_free_gpu, LossHistory
from dlpf.stats import aggregate_application_run_stats, aggregate_application_base_stats
from dlpf.run import apply_agent
from dlpf.fglab_utils import create_scores_file
from dlpf.perf_utils import Profiler

logger = logging.getLogger()

STATS_NAMES = 'run epoch batch'.split(' ')

SERIES_NOT_TO_PLOT = frozenset({ 'optimal_score' })

if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--env',
                         type = str,
                         default = 'configs/env/sample.yaml',
                         help = 'Path to environment config to use')
    aparser.add_argument('--agent',
                         type = str,
                         default = 'configs/agent/sample.yaml',
                         help = 'Path to agent config to use')
    aparser.add_argument('--apply',
                         type = str,
                         default = 'configs/apply/sample.yaml',
                         help = 'Path to apply_agent config to use')
    aparser.add_argument('--output',
                         type = str,
                         default = '.',
                         help = 'Where to store results')
    aparser.add_argument('--level',
                         type = str,
                         choices = LOGGING_LEVELS.keys(),
                         default = 'info',
                         help = 'Logging verbosity')
    aparser.add_argument('--_id',
                         type = str,
                         default = None,
                         help = 'FGLab experiment id')

    args = aparser.parse_args()

    if args._id:
        args.output = os.path.join(args.output, args._id)
    ensure_dir_exists(args.output)

    logger = init_log(stderr = True,
                      level = LOGGING_LEVELS[args.level],
                      out_file = os.path.join(args.output, 'run_agent.log'))

    try_assign_theano_on_free_gpu()

    env = load_environment_from_yaml(args.env)
    
    keras_hist = LossHistory()
    agent = load_object_from_yaml(args.agent,
                                  input_shape = env.observation_space.shape,
                                  number_of_actions = env.action_space.n,
                                  model_callbacks = [keras_hist])
    
    apply_kwargs = load_yaml(args.apply)
    if int(apply_kwargs.get('visualize_each', 0)) > 0:
        apply_kwargs['visualization_dir'] = args.output

    with Profiler(logger):
        run_stats = apply_agent(env, agent, **apply_kwargs)

    stats = (aggregate_application_run_stats([run_stats]),
             aggregate_application_base_stats([keras_hist.epoch_stats]),
             aggregate_application_base_stats([keras_hist.batch_stats]))

    for stat_name, stat in zip(STATS_NAMES, stats):
        basic_plot_from_df(stat.episodes,
                           out_file = os.path.join(args.output, '%s_episodes.png' % stat_name),
                           ignore = SERIES_NOT_TO_PLOT)
        basic_plot_from_df_rolling_mean(stat.episodes,
                                        out_file = os.path.join(args.output, '%s_episodes_smoothed.png' % stat_name),
                                        ignore = SERIES_NOT_TO_PLOT)
        basic_plot_from_df(stat.full,
                           out_file = os.path.join(args.output, '%s_full.png' % stat_name),
                           ignore = SERIES_NOT_TO_PLOT)
        basic_plot_from_df_rolling_mean(stat.full,
                                        out_file = os.path.join(args.output, '%s_full_smoothed.png' % stat_name),
                                        ignore = SERIES_NOT_TO_PLOT)

    create_scores_file(os.path.join(args.output, 'scores.json'),
                       **stats[0].scores)
    create_scores_file(os.path.join(args.output, 'scores.js'),
                       **stats[0].scores)
    time.sleep(1)

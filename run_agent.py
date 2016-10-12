#!/usr/bin/env python

import argparse, logging, os

from dlpf.gym_environ import load_environment_from_yaml
from dlpf.base_utils import init_log, LOGGING_LEVELS, ensure_dir_exists, \
    load_object_from_yaml, load_yaml
from dlpf.benchmark import evaluate_agent_with_configs
from dlpf.plot_utils import basic_plot_from_df
from dlpf.keras_utils import try_assign_theano_on_free_gpu, LossHistory
from dlpf.stats import aggregate_application_run_stats, aggregate_application_base_stats
from dlpf.run import apply_agent
from dlpf.fglab_utils import create_scores_file
from dlpf.perf_utils import Profiler

logger = logging.getLogger()

STATS_NAMES = 'run epoch batch'.split(' ')

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

    args = aparser.parse_args()

    logger = init_log(stderr = True, level = LOGGING_LEVELS[args.level])

    try_assign_theano_on_free_gpu()

    env = load_environment_from_yaml(args.env)
    
    keras_hist = LossHistory()
    agent = load_object_from_yaml(args.agent,
                                  input_shape = env.observation_space.shape,
                                  number_of_actions = env.action_space.n,
                                  model_callbacks = [keras_hist])
    with Profiler(logger):
        run_stats = apply_agent(env, agent, **load_yaml(args.apply))

    stats = (aggregate_application_run_stats([run_stats]),
             aggregate_application_base_stats([keras_hist.epoch_stats]),
             aggregate_application_base_stats([keras_hist.batch_stats]))

    ensure_dir_exists(args.output)
    for stat_name, stat in zip(STATS_NAMES, stats):
        basic_plot_from_df(stat.episodes,
                           out_file = os.path.join(args.output, '%s_episodes.png' % stat_name))
        basic_plot_from_df(stat.full,
                           out_file = os.path.join(args.output, '%s_full.png' % stat_name))

    create_scores_file(os.path.join(args.output, 'scores.json'),
                       train_score = stats[0].score)

import os, glob, numpy, shutil, logging
from sklearn.cross_validation import ShuffleSplit

from .base_utils import ensure_dir_exists, copy_files, copy_and_update, \
    load_yaml, load_object_from_yaml
from .gym_environ import load_environment_from_yaml
from .run import apply_agent
from .stats import aggregate_application_base_stats, aggregate_application_run_stats
from .keras_utils import LossHistory


logger = logging.getLogger(__name__)


TRAIN_DIR = 'train'
TEST_DIR = 'test'


def prepare_evaluation_splits(tasks_dir, to_dir, folds = 3, test_part = 0.3):
    all_task_fnames = numpy.array([fname
                                   for fname in os.listdir(tasks_dir)
                                   if os.path.isfile(os.path.join(tasks_dir, fname))])
    for fold_i, (train_idx, test_idx) in enumerate(ShuffleSplit(len(all_task_fnames),
                                                                n_iter = folds,
                                                                test_size = test_part)):
        train_dir = os.path.join(to_dir, str(fold_i), TRAIN_DIR)
        ensure_dir_exists(train_dir)
        copy_files(tasks_dir, all_task_fnames[train_idx], train_dir)

        test_dir = os.path.join(to_dir, str(fold_i), TEST_DIR)
        ensure_dir_exists(test_dir)
        copy_files(tasks_dir, all_task_fnames[test_idx], test_dir)


def evaluate_agent(environment_ctor,
                   agent_ctor,
                   folds_dir,
                   **apply_kwargs):
    
    fold_dirs = filter(os.path.isdir, glob.glob(os.path.join(folds_dir, '*')))
    fold_dirs.sort()
    train_stats = []
    test_stats = []
    if len(fold_dirs) > 0:
        logger.info('Evaluating on %d folds from %s' % (len(fold_dirs),
                                                        folds_dir))
        for fold_i, fold_dir in enumerate(fold_dirs):
            logger.info('Fold %d train' % fold_i)
            train_env = environment_ctor(os.path.join(fold_dir, TRAIN_DIR))
            agent = agent_ctor(train_env.observation_space.shape,
                               train_env.action_space.n)
            cur_train_stat = apply_agent(train_env,
                                         agent,
                                         **copy_and_update(apply_kwargs,
                                                           allow_train = True))
            train_stats.append(cur_train_stat)

            logger.info('Fold %d test' % fold_i)
            test_env = environment_ctor(os.path.join(fold_dir, TEST_DIR))
            cur_test_stat = apply_agent(test_env,
                                        agent,
                                        **copy_and_update(apply_kwargs,
                                                          allow_train = False))
            test_stats.append(cur_test_stat)
    return train_stats, test_stats


def evaluate_agent_with_configs(environment_conf_fname,
                                agent_conf_fname,
                                folds_dir,
                                apply_conf):
    def environment_ctor(tasks_dir):
        return load_environment_from_yaml(environment_conf_fname,
                                          tasks_dir = tasks_dir)

    inner_batch_stats = []
    inner_epoch_stats = []
    keras_hist = LossHistory()
    first_agent_ctor_call = []

    def agent_ctor(input_shape, number_of_actions):
        if not first_agent_ctor_call:
            inner_batch_stats.append(keras_hist.batch_stats)
            inner_epoch_stats.append(keras_hist.epoch_stats)
        else:
            first_agent_ctor_call.append(1)
        return load_object_from_yaml(agent_conf_fname,
                                     input_shape = input_shape,
                                     number_of_actions = number_of_actions,
                                     model_callbacks = [keras_hist])

    apply_kwargs = load_yaml(apply_conf)
    
    train_stats, test_stats = evaluate_agent(environment_ctor,
                                             agent_ctor,
                                             folds_dir,
                                             **apply_kwargs)

    train_stats = aggregate_application_stats(train_stats)
    test_stats = aggregate_application_stats(test_stats)
    
    return (aggregate_application_run_stats(train_stats),
            aggregate_application_run_stats(test_stats),
            aggregate_application_base_stats(inner_batch_stats),
            aggregate_application_base_stats(inner_epoch_stats))

import collections, pandas, itertools

from .base_utils import no_copy_update


EpisodeInfo = collections.namedtuple('EpisodeStat',
                                     'info steps'.split(' '))


class StatHolder(object):
    def __init__(self):
        self.episodes_data = []

    def new_episode(self, **episode_info):
        self.episodes_data.append(EpisodeInfo(episode_info, []))
        
    def add_step(self, **kwargs):
        self.episodes_data[-1].steps.append(kwargs)

    def plot(self, to_file):
        pass

BaseStats = collections.namedtuple('BaseStats',
                                   'episodes full'.split(' '))
RunStats = collections.namedtuple('RunStats',
                                  'score episodes full'.split(' '))
    
def aggregate_application_base_stats(stats_lst):
    full_step_stats = pandas.DataFrame(data = itertools.chain.from_iterable(ep.steps
                                                                            for stat in stats_lst
                                                                            for ep in stat.episodes_data))
    episodes_stats = pandas.DataFrame(data = [ep.info
                                              for stat in stats_lst
                                              for ep in stat.episodes_data])
    return BaseStats(episodes_stats, full_step_stats)


def aggregate_application_run_stats(stats_lst):
    full_step_stats = pandas.DataFrame(data = itertools.chain.from_iterable(ep.steps
                                                                            for stat in stats_lst
                                                                            for ep in stat.episodes_data))
    episodes_stats = pandas.DataFrame(data = [no_copy_update(ep.info,
                                                             score = (sum(d['reward'] for d in ep.steps)
                                                                      / float(ep.info['optimal_score'])) )
                                              for stat in stats_lst
                                              for ep in stat.episodes_data])
    return RunStats(episodes_stats['score'].mean(), episodes_stats, full_step_stats)

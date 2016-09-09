import cProfile, pstats, StringIO


class Profiler(object):
    def __init__(self, logger):
        self.logger = logger

    def __enter__(self):
        self.profile = cProfile.Profile()
        self.profile.enable()
        return self

    def __exit__(self, *args):
        self.profile.disable()
        buf = StringIO.StringIO()
        pstats.Stats(self.profile, stream = buf) \
            .sort_stats('cumulative') \
            .print_stats()
        self.logger.info(buf.getvalue())

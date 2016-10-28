import collections

from scipy.spatial.distance import euclidean

from .utils import BY_PIXEL_ACTION_DIFFS


class BaseSearchAlgo(object):
    def __init__(self):
        pass

    def reset(self, local_map, start, finish):
        self.local_map = local_map
        self.start = start
        self.finish = finish

        self.queue = [(self.start, 0)]
        self.backrefs = { self.start : self.start }

    def walk_to_finish(self):
        while self.step():
            pass

    def step(self):
        if self.goal_achieved() or len(self.queue) == 0:
            return False

        best_next, _ = self.queue.pop()

        new_variants = self._gen_new_variants(best_next)
        self.queue.extend(new_variants)
        self._reorder_queue()

        self.backrefs.update((new_point, best_next) for new_point, _ in new_variants)
        return True

    def goal_achieved(self):
        return self.finish in self.backrefs

    def get_best_path(self):
        self.walk_to_finish()
        if not self.goal_achieved():
            return None

        result = [self.finish]
        while result[-1] != self.start:
            result.append(self.backrefs[result[-1]])
        result.reverse()
        return result

    def _reorder_queue(self):
        self.queue.sort(key = lambda el: el[-1])

    def _gen_new_variants(self, pos):
        '''Proceed from the given state.
        Should return an iterable of pairs (new_state, rating).
        The bigger the rating the better the new_state is.'''
        raise NotImplemented()


class EuclideanAStar(BaseSearchAlgo):
    def _gen_new_variants(self, pos):
        y, x = pos
        all_new_points = ((y + dy, x + dx)
                          for dy, dx
                          in BY_PIXEL_ACTION_DIFFS.viewvalues())
        return [(point, -euclidean(point, self.finish))
                for point in all_new_points
                if (not point in self.backrefs)
                    and (0 <= point[0] < self.local_map.shape[0])
                    and (0 <= point[1] < self.local_map.shape[1])
                    and (self.local_map[point] == 0)]
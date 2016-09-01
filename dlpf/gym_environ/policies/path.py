import random
from .utils import build_distance_map


class BasePathPolicy(object):
    def __init__(self):
        self.reset(None)

    def reset(self, task):
        self.task = task

    def get_start_position(self):
        return self.task.path[0]
    
    def get_global_goal(self):
        return self.task.path[-1]

    def get_local_goal(self):
        raise NotImplemented()

    def move_to_next_goal(self):
        raise NotImplemented()


class FollowGoldPath(BasePathPolicy):
    def reset(self, task):
        super(FollowGoldPath, self).reset(task)
        self.cur_target_i = 1 # skip start position

    def get_local_goal(self):
        return self.task.path[self.cur_target_i]

    def move_to_next_goal(self):
        if self.cur_target_i < len(self.task.path) - 1:
            self.cur_target_i += 1
            return False
        return True


class GoStraightToFinish(BasePathPolicy):
    def get_local_goal(self):
        return self.task.path[-1]

    def move_to_next_goal(self):
        return True


class RandomStartAndFinishMixin(object):
    def __init__(self, rand = None, *args, **kwargs):
        super(RandomStartAndFinishMixin, self).__init__(*args, **kwargs)
        self.rand = rand or random.Random()

    def reset(self, task):
        super(RandomStartAndFinish, self).reset(task)
        local_map = self.task.local_map # shortcut
        while True:
            self.start = self._gen_point()
            self.finish = self._gen_point()
            if local_map[self.start] == 0 \
                and local_map[self.finish] == 0 \
                and check_finish_achievable(local_map, self.start, self.finish):
                break
        
    def get_start_position(self):
        return self.start

    def get_global_goal(self):
        return self.finish

    def _gen_point(self):
        return (self.rand.randint(0, self.task.local_map.shape[0] - 1), # randint is inclusive
                self.rand.randint(0, self.task.local_map.shape[1] - 1))


class RandomStartAndFinishStraight(RandomStartAndFinishMixin, GoStraightToFinish):
    pass


_PATH_POLICIES = {
    'follow_gold' : FollowGoldPath,
    'go_straight' : GoStraightToFinish,
    'random_start_and_finish_straight' : RandomStartAndFinishStraight,
}


def get_available_path_policies():
    return list(_PATH_POLICIES.keys())


DEFAULT_PATH_POLICY = 'follow_gold'

def get_path_policy(name = DEFAULT_PATH_POLICY, *args, **kwargs):
    assert name in _PATH_POLICIES, "Unknown path policy %s" % name
    return _PATH_POLICIES[name](*args, **kwargs)

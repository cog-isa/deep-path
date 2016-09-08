import random


class BaseTaskPolicy(object):
    def reset(self, task_set):
        self.task_set = task_set

    def choose_next_task(self):
        raise NotImplemented()


class SequentialTaskPolicy(BaseTaskPolicy):
    def reset(self, task_set):
        super(SequentialTaskPolicy, self).reset(task_set)
        self.task_ids = list(task_set.keys())
        self.cur_task_i = 0

    def choose_next_task(self):
        result = self.task_set[self.task_ids[self.cur_task_i]]
        self.cur_task_i += 1
        if self.cur_task_i >= len(self.task_ids):
            self.cur_task_i = 0
        return result


class RandomTaskPolicy(BaseTaskPolicy):
    def __init__(self, rand = None):
        self.rand = rand or random.Random()

    def choose_next_task(self):
        task_id = self.rand.choice(self.task_set.keys())
        return self.task_set[task_id]


_TASK_POLICIES = {
    'sequential' : SequentialTaskPolicy,
    'random' : RandomTaskPolicy,
}
DEFAULT_TASK_POLICY = 'random'


def get_available_task_policies():
    return list(_TASK_POLICIES.keys())


def get_task_policy(name = DEFAULT_TASK_POLICY, *args, **kwargs):
    assert name in _TASK_POLICIES, "Unknown task policy %s" % name
    return _TASK_POLICIES[name](*args, **kwargs)

import numpy
cimport numpy

import threading #

class threadsafe_iter: #
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f): #
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


ctypedef numpy.float_t FLOAT_T


def assign_outputs_free_hinge(numpy.ndarray[numpy.float32_t, ndim=1] result, int action, float reward):
    result[action] = reward


def assign_outputs_tanh_hinge(numpy.ndarray[numpy.float32_t, ndim=1] result, int action, float reward):
    result[action] = numpy.tanh(reward)


def softmax(numpy.ndarray[numpy.float32_t, ndim=1] p):
    e = numpy.exp(p - p.max())
    return e / e.sum()


def assign_outputs_softmax(numpy.ndarray[numpy.float32_t, ndim=1] result, int action, float reward):
    cdef numpy.ndarray[numpy.float32_t, ndim=1] p = numpy.zeros((result.shape[0],), dtype = 'float32')
    p[action] = reward
    result[:] = softmax(p)


_OUTPUT_TYPES = {
    'free_hinge' : assign_outputs_free_hinge,
    'tanh_hinge' : assign_outputs_tanh_hinge,
    'softmax' : assign_outputs_softmax
}


@threadsafe_generator #
def replay_train_data_generator(list episodes, int batch_size, int actions_number, str output_type_name, rand):
    assert output_type_name in _OUTPUT_TYPES, 'Unknown output type %s' % output_type_name
    if len(episodes) == 0:
        return

    input_batch = numpy.zeros((batch_size, ) + episodes[0][0].observation.shape,
                              dtype = 'float32')
    output_batch = numpy.zeros((batch_size, actions_number),
                               dtype = 'float32')
    cdef int batch_i = 0

    output_assigner = _OUTPUT_TYPES[output_type_name]
    while True:
        episode = episodes[rand.randint(0, len(episodes) - 1)]
        step = episode[rand.randint(0, len(episode) - 1)]

        input_batch[batch_i] = step.observation
        output_assigner(output_batch[batch_i], step.action, step.reward)

        batch_i += 1
        if batch_i % batch_size == 0:
            yield (input_batch, output_batch)
            output_batch[:] = 0
            batch_i = 0

import numpy
cimport numpy
import scipy.ndimage.interpolation

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


def assign_outputs_free_hinge(int action, float reward, float q_prediction, numpy.ndarray[numpy.float32_t, ndim=1] target):
    target[action] = reward + q_prediction


def assign_outputs_tanh_hinge(int action, float reward, float q_prediction, numpy.ndarray[numpy.float32_t, ndim=1] target):
    target[action] = numpy.tanh(reward + q_prediction)


def softmax(numpy.ndarray[numpy.float32_t, ndim=1] p):
    e = numpy.exp(p - p.max())
    return e / e.sum()


def assign_outputs_softmax(int action, float reward, float q_prediction, numpy.ndarray[numpy.float32_t, ndim=1] target):
    cdef numpy.ndarray[numpy.float32_t, ndim=1] p = numpy.zeros((target.shape[0],), dtype = 'float32')
    p[action] = reward + q_prediction
    target[:] = softmax(p)


_OUTPUT_TYPES = {
    'free_hinge' : assign_outputs_free_hinge,
    'tanh_hinge' : assign_outputs_tanh_hinge,
    'softmax' : assign_outputs_softmax
}


@threadsafe_generator
def replay_train_data_generator(list episodes,
                                list q_predictions,
                                list targets,
                                int batch_size,
                                int actions_number,
                                str output_type_name,
                                rand):
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
        e_i = rand.randint(0, len(episodes) - 1)
        episode = episodes[e_i]
        s_i = rand.randint(0, len(episode) - 1)
        step = episode[s_i]

        input_batch[batch_i] = step.observation
        output_assigner(step.action, step.reward, q_predictions[e_i][s_i], targets[e_i][s_i])
        output_batch[batch_i] = targets[e_i][s_i]

        batch_i += 1
        if batch_i % batch_size == 0:
            yield (input_batch, output_batch)
            output_batch[:] = 0
            batch_i = 0

def scale_tensors(base_gen, tuple scale_factor, int order):
    for input_batch, output_batch in base_gen:
        yield (scipy.ndimage.interpolation.zoom(input_batch, scale_factor, order = order),
               output_batch)

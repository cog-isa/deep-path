import numpy
cimport numpy


ctypedef numpy.float_t FLOAT_T


def replay_train_data_generator(list episodes, int batch_size, int actions_number, rand):
    if len(episodes) == 0:
        return

    input_batch = numpy.zeros((batch_size, ) + episodes[0][0].observation.shape,
                              dtype = 'float32')
    output_batch = numpy.zeros((batch_size, actions_number),
                               dtype = 'float32')
    cdef int batch_i = 0

    while True:
        episode = episodes[rand.randint(0, len(episodes) - 1)]
        step = episode[rand.randint(0, len(episode) - 1)]

        input_batch[batch_i] = step.observation
        output_batch[batch_i, step.action] = step.reward

        batch_i += 1
        if batch_i % batch_size == 0:
            yield (input_batch, output_batch)
            output_batch[:] = 0
            batch_i = 0

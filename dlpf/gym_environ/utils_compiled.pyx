import numpy as np, collections
cimport numpy as np


DIFFS = np.array([
        [-1, 0],
        [-1, 1],
        [0,  1],
        [1,  1],
        [1,  0],
        [1, -1],
        [0, -1],
        [-1, -1]
], dtype = 'int32')

DTYPE = np.int
ctypedef np.int_t DTYPE_t


def build_distance_map(np.ndarray[DTYPE_t, ndim = 2] local_map, np.ndarray[DTYPE_t] finish):
    cdef np.ndarray[DTYPE_t, ndim = 2] result = -np.array(local_map, dtype=DTYPE)
    
    queue = collections.deque()
    queue.append(((finish[0], finish[1]), 0))
    result[finish[0], finish[1]] = 0

    cdef DTYPE_t new_y, new_x, new_dist, max_y = local_map.shape[0], max_x = local_map.shape[1], cur_dist
    cdef tuple cur_point

    while len(queue) > 0:
        cur_point, cur_dist = queue.popleft()
        new_dist = cur_dist + 1

        for new_y, new_x in DIFFS + cur_point:
            if (0 <= new_y < max_y
                and 0 <= new_x < max_x
                and new_y != finish[0]
                and new_x != finish[1]
                and result[new_y, new_x] == 0): # we are not going to obstacle and we have not filled this cell yet
                queue.append(((new_y, new_x), new_dist))
                result[new_y, new_x] = new_dist

    return result


def check_finish_achievable(np.ndarray[DTYPE_t, ndim = 2] local_map, np.ndarray[DTYPE_t] start, np.ndarray[DTYPE_t] finish):
    if start == finish:
        return True
    return build_distance_map(local_map, finish)[start[0], start[1]] > 0

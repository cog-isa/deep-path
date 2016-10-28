import numpy as np, collections
from scipy.spatial.distance import euclidean
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

'''
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
'''

def build_distance_map(np.ndarray[DTYPE_t, ndim = 2] local_map, np.ndarray[DTYPE_t] finish):
    cdef np.ndarray[DTYPE_t, ndim = 2] result = -np.array(local_map, dtype=DTYPE)

    queue = collections.deque()
    queue.append(((finish[0], finish[1]), 0))
    result[finish[0], finish[1]] = 0

    cdef DTYPE_t new_y, new_x, new_dist, max_y = local_map.shape[0], max_x = local_map.shape[1], cur_dist, off_i, cur_x, cur_y

    cdef DTYPE_t all_dx[8]
    cdef DTYPE_t all_dy[8]
    all_dx[:] = [0, 1, 1, 1, 0, -1, -1, -1]
    all_dy[:] = [-1, -1, 0, 1, 1, 1, 0, -1]

    while len(queue) > 0:
        (cur_y, cur_x), cur_dist = queue.popleft()
        new_dist = cur_dist + 1

        for off_i in range(0, 8, 1):
            new_y = cur_y + all_dy[off_i]
            new_x = cur_x + all_dx[off_i]
            if (0 <= new_y
                and new_y < max_y
                and 0 <= new_x
                and new_x < max_x
                and (new_y != finish[0] or new_x != finish[1])
                and result[new_y, new_x] == 0): # we are not going to obstacle and we have not filled this cell yet
                queue.append(((new_y, new_x), new_dist))
                result[new_y, new_x] = new_dist

    return result


def check_finish_achievable(np.ndarray[DTYPE_t, ndim = 2] local_map, np.ndarray[DTYPE_t] start, np.ndarray[DTYPE_t] finish):
    if  np.allclose(start, finish):
        return True
    return build_distance_map(local_map, finish)[start[0], start[1]] > 0


def line_intersection(tuple line1, tuple line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(tuple a, tuple b):
        return a[0] * b[1] - a[1] * b[0]

    cdef float div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    y = det(d, ydiff) / div
    x = det(d, xdiff) / div
    return x, y


def get_flat_state(np.ndarray[np.uint8_t, ndim = 2] local_map,
                   tuple cur_pos,
                   int vision_range,
                   float done_reward,
                   float target_on_border_reward,
                   tuple start,
                   tuple goal,
                   float absolute_distance_weight):
    cdef np.ndarray[np.float_t, ndim = 2] result = np.ones((2 * vision_range + 1,
                                                            2 * vision_range + 1))
    result *= -1 # everything is obstacle by default

#     logger.debug('Map:\n%s' % local_map)

    cdef int cur_pos_y = cur_pos[0]
    cdef int cur_pos_x = cur_pos[1]
    cdef int y_viewport_left_top = cur_pos_y - vision_range
    cdef int x_viewport_left_top = cur_pos_x - vision_range

    cdef int y_from = max(0, y_viewport_left_top)
    cdef int y_to = min(cur_pos_y + vision_range + 1, local_map.shape[0])
    cdef int x_from = max(0, x_viewport_left_top)
    cdef int x_to = min(cur_pos_x + vision_range + 1, local_map.shape[1])
#         logger.debug('Pos %s, viewport lt %s, cropped %s' % (self.cur_position_discrete,
#                                                              (y_viewport_left_top, x_viewport_left_top),
#                                                              ((y_from, x_from), (y_to, x_to))))

    cdef int x, y, border_i
    cdef tuple border
    for y in range(y_from, y_to):
        for x in range(x_from, x_to):
            if local_map[y, x] == 0:
                result[y - y_viewport_left_top, x - x_viewport_left_top] = 0

    cdef int y_goal = goal[0]
    cdef int x_goal = goal[1]
    if y_from <= y_goal < y_to and x_from <= x_goal < x_to:
        result[y_goal - y_viewport_left_top, x_goal - x_viewport_left_top] = done_reward
    else: # find intersection of line <cur_pos, goal> with borders of view range and mark it
#             logger.debug('Target out of view range %s, %s' % (self.cur_position_discrete,
#                                                               goal))
        # NW, NE, SE, SW
        corners = [(y_viewport_left_top,                   x_viewport_left_top),
                   (y_viewport_left_top,                   x_viewport_left_top + result.shape[1] - 1),
                   (y_viewport_left_top + result.shape[0] - 1, x_viewport_left_top + result.shape[1] - 1),
                   (y_viewport_left_top + result.shape[0] - 1, x_viewport_left_top)]

        # top, right, bottom, left
        borders = [(corners[0], corners[1]),
                   (corners[1], corners[2]),
                   (corners[2], corners[3]),
                   (corners[3], corners[0])]

        line_to_goal = ((cur_pos_y, cur_pos_x), goal)

        best_dist = np.inf
        inter_point = None
        for border_i, border in enumerate(borders):
#                 logger.debug('border %s' % repr(border))
            cur_inter_point = line_intersection(line_to_goal, border)
#                 logger.debug('inter %s' % repr(cur_inter_point))
            if cur_inter_point is None:
                continue
            cur_dist = euclidean(cur_inter_point, goal)
#                 logger.debug('inter dist %s' % repr(cur_dist))
            if 0 <= cur_inter_point[0] - y_viewport_left_top < result.shape[0] \
                and 0 <= cur_inter_point[1] - x_viewport_left_top < result.shape[1] \
                and cur_dist < best_dist:
                best_dist = cur_dist
                inter_point = cur_inter_point

        abs_dist_normed = np.exp(-best_dist / euclidean(start, goal))
        abs_distance_reward = absolute_distance_weight * (done_reward - target_on_border_reward) * abs_dist_normed

        result[inter_point[0] - y_viewport_left_top, inter_point[1] - x_viewport_left_top] = target_on_border_reward + abs_distance_reward
#         logger.debug('Viewport:\n%s' % result)
    return result

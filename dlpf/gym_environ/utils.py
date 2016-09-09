import itertools, collections, numpy

BY_PIXEL_ACTIONS = {
    0 : 'N',
    1 : 'NE',
    2 : 'E',
    3 : 'SE',
    4 : 'S',
    5 : 'SW',
    6 : 'W',
    7 : 'NW'
}

BY_PIXEL_ACTION_DIFFS = {
    0 : numpy.array([-1,  0], dtype = 'int8'),
    1 : numpy.array([-1,  1], dtype = 'int8'),
    2 : numpy.array([ 0,  1], dtype = 'int8'),
    3 : numpy.array([ 1,  1], dtype = 'int8'),
    4 : numpy.array([ 1,  0], dtype = 'int8'),
    5 : numpy.array([ 1, -1], dtype = 'int8'),
    6 : numpy.array([ 0, -1], dtype = 'int8'),
    7 : numpy.array([-1, -1], dtype = 'int8')
}


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    y = det(d, ydiff) / div
    x = det(d, xdiff) / div
    return x, y


def build_distance_map(local_map, finish):
    result = numpy.array(-local_map,
                         dtype='int32')
    
    queue = collections.deque()
    queue.append((finish, 0))
    result[finish] = 0

    while queue:
        cur_point, cur_dist = queue.popleft()
        new_dist = cur_dist + 1

        for dy, dx in BY_PIXEL_ACTION_DIFFS.viewvalues():
            new_point = (cur_point[0] + dy, cur_point[1] + dx)

            if (0 <= new_point[0] < local_map.shape[0] and 0 <= new_point[1] < local_map.shape[1] # we are in boundaries
                and new_point != finish
                and result[new_point] == 0): # we are not going to obstacle and we have not filled this cell yet
                queue.append((new_point, new_dist))
                result[new_point] = new_dist

    return result


def check_finish_achievable(local_map, start, finish):
    if start == finish:
        return True
    return build_distance_map(local_map, finish)[start] > 0

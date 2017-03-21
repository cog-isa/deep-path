from dlpf.utils.base_utils import *
from dlpf.utils.plot_utils import scatter_plot
from dlpf.utils.task_utils import load_map_from_compact, PathFindingTask

if __name__ == "__main__":
    task_name = '25'
    task = load_obj(os.path.join('../data/20x20/imported/paths', task_name + '.pickle'))
    local_map = load_map_from_compact(os.path.join('../data/20x20/imported/maps', task.map_id + '.npz'))

    cur_task = PathFindingTask(task_name,
                               local_map,
                               task.start,
                               task.finish,
                               task.path)
    obstacle_points_for_vis = [(x, y)
                               for y in xrange(cur_task.local_map.shape[0])
                               for x in xrange(cur_task.local_map.shape[1])
                               if cur_task.local_map[y, x] > 0]
    scatter_plot(({'label': 'obstacle',
                   'data': obstacle_points_for_vis,
                   'color': 'black',
                   'marker': 's'},),
                 x_lim=(0, cur_task.local_map.shape[1]),
                 y_lim=(0, cur_task.local_map.shape[0]),
                 offset=(0.5, 0.5),
                 out_file='map_example_%s.png' % task_name)

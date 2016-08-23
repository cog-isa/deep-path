from .base_utils import *
import lxml.etree, hashlib


logger = logging.getLogger(__name__)


PathFindingTask = collections.namedtuple('PathFindingTask',
                                         'title local_map start finish path'.split(' '))

CompactPathFindingTask = collections.namedtuple('CompactPathFindingTask',
                                                'map_id start finish path'.split(' '))


COMPACT_MAP_EXT = '.npz'
COMPACT_TASK_EXT = '.pickle'


def load_from_xml(fname, ctor = PathFindingTask):
    try:
        logger.info('Loading task from xml %s' % fname)

        with open(fname, 'r') as f:
            tree = lxml.etree.parse(f)

        title = os.path.splitext(os.path.basename(fname))[0]

        local_map = numpy.array([map(int, row.split(' '))
                                 for row in tree.xpath('/root/map/grid/row/text()')],
                                dtype = 'uint8')
        start_x = int(tree.xpath('/root/map/startx/text()')[0])
        start_y = int(tree.xpath('/root/map/starty/text()')[0])
        finish_x = int(tree.xpath('/root/map/finishx/text()')[0])
        finish_y = int(tree.xpath('/root/map/finishy/text()')[0])

        sections = tree.xpath('/root/log[1]/hplevel/section')
        sections.sort(key = lambda n: int(n.get('number')))
        path = [(int(s.get('finish.y')),
                 int(s.get('finish.x')))
                for s in sections]
        #if sections:
        #    path.append((int(sections[-1].get('finish.y')),
        #                 int(sections[-1].get('finish.x'))))
        #print path
        result = ctor(title,
                      local_map,
                      (start_y, start_x),
                      (finish_y, finish_x),
                      path)
        logger.info('Task %s loaded' % fname)
        return result
    except:
        logger.error('Could not load task from %s:\n%s' % (fname, traceback.format_exc()))
        return None


def save_to_compact(task, maps_dir, paths_dir):
    try:
        logger.info('Saving %s to compact' % task.title)
        map_id = hashlib.md5(task.local_map.tostring()).hexdigest()
        map_fname = os.path.join(maps_dir, map_id)
        if not os.path.exists(map_fname + '.npz'):
            logger.info('Saving map to %s' % map_fname)
            numpy.savez_compressed(map_fname, task.local_map)
            
        compact = CompactPathFindingTask(map_id, task.start, task.finish, task.path)
        task_fname = os.path.join(paths_dir, task.title + COMPACT_TASK_EXT)
        save_obj(compact, task_fname)
        logger.info('Saved task %s' % task_fname)
    except:
        logger.error('Could not save task %s:\n%s' % (task.title, traceback.format_exc()))


def import_tasks_from_xml_to_compact(in_dir, out_dir, maps_subdir = 'maps', paths_subdir = 'paths', report_every = 100):
    logger.info('Importing tasks from %s to %s' % (in_dir, out_dir))

    maps_dir = os.path.join(out_dir, maps_subdir)
    paths_dir = os.path.join(out_dir, paths_subdir)

    for in_fname in reporting_gen(os.listdir(in_dir),
                                  logger,
                                  report_every = report_every,
                                  template = 'Imported %d tasks'):
        task = load_from_xml(os.path.join(in_dir, in_fname))
        if task is None:
            continue
        save_to_compact(task, maps_dir, paths_dir)
    logger.info('Imported tasks from %s to %s' % (in_dir, out_dir))


class TaskSet(object):
    def __init__(self, in_dir, maps_subdir = 'maps', paths_subdir = 'paths'):
        self.in_dir = in_dir
        self.map_dir = os.path.join(self.in_dir, maps_subdir)
        self.paths_dir = os.path.join(self.in_dir, paths_subdir)
        self.task_names = [os.path.splitext(fn)[0] for fn in os.listdir(self.paths_dir)]
        self.maps_cache = {}

    def keys(self):
        return self.task_names

    def __getitem__(self, task_name):
        task = load_obj(os.path.join(self.paths_dir, task_name + COMPACT_TASK_EXT))

        local_map = self.maps_cache.get(task.map_id)
        if local_map is None:
            with numpy.load(os.path.join(self.map_dir, task.map_id + COMPACT_MAP_EXT)) as f:
                local_map = f['arr_0']
                self.maps_cache[task.map_id] = local_map

        return PathFindingTask(task_name,
                               local_map,
                               task.start,
                               task.finish,
                               task.path)

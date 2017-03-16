#!/usr/bin/env python

import argparse
import os

from dlpf.utils.base_utils import init_log, LOGGING_LEVELS
from dlpf.utils.task_utils import import_tasks_from_xml_to_compact

if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument('src_dir', type=str)
    aparser.add_argument('target_dir', type=str)

    args = aparser.parse_args()

    logger = init_log(stderr=True)

    import_tasks_from_xml_to_compact(args.src_dir, args.target_dir)

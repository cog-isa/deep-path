#!/usr/bin/env python

import argparse, os

from dlpf.io import import_tasks_from_xml_to_compact

if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument('src_dir', type = str)
    aparser.add_argument('target_dir', type = str)

    args = aparser.parse_args()

    import_tasks_from_xml_to_compact(args.src_dir, args.target_dir)

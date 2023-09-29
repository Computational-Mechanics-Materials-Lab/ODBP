#!/usr/bin/env python

"""
ODBPlotter odb_to_npz.py

ODBPlotter
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBPlotter
MIT License (c) 2023

This is a Python 2 file which implements an Abaqus Python interface to
convert data from within a .odb file into a hierarchical directory of .npz
compressed numpy arrays. This is used to translate .odb data from the Python 2
environment to a modern Python 3 environment

Originally authored by CMML member CJ Nguyen, based before on extraction
script written by CMML members Will Furr and Matt Dantin
"""

import sys
import argparse
import pickle
from odbAccess import openOdb


def main():
    input_args = "input args"
    parser = argparse.ArgumentParser()
    parser.add_argument(input_args, nargs="*")
    odb_path, result_path = vars(parser.parse_args())[input_args]
    results = get_odb_info(odb_path)
    try:
        result_file = open(result_path, "wb")
        pickle.dump(results, result_file)
    finally:
        result_file.close()


def get_odb_info(odb_path):
    result = dict()
    try:
        odb = openOdb(odb_path, readOnly=True)

    except Exception as e:
        print("Abaqus Error:")
        print(e)
        sys.exit(1)

    try:
        # What are our targets:
        # Frames
        # Steps --> Frames
        # Nodesets
        # Parts --> Nodesets
        # # Nodes --> Nodesets
        steps = odb.steps
        step_lens = dict()
        all_frames = list()
        all_parts = list()
        frame_keys = list()
        frame_keys_per_step = dict()
        frame_range = 0
        for step_key, step_data in steps.items():
            step_lens[step_key] = len(step_data.frames)
            frame_range += step_lens[step_key]
            frame_keys_per_step[step_key] = list(
                step_data.frames[0].fieldOutputs.keys()
            )

        for v in frame_keys_per_step.values():
            for k in v:
                if k not in frame_keys:
                    frame_keys.append(k)

        assembly = odb.rootAssembly
        nodesets = assembly.nodeSets.keys()

        all_parts = assembly.instances.keys()

        all_nodes = list()
        parts_to_nodes = dict()
        for key in all_parts:
            parts_to_nodes[key] = list()
            for node in assembly.instances[key].nodes:
                parts_to_nodes[key].append(node.label)
                all_nodes.append(node.label)

        # Temporal
        result["frame_keys"] = frame_keys
        result["frame_keys_per_step"] = frame_keys_per_step
        result["frame_range"] = frame_range
        result["step_names"] = list(step_lens.keys())
        result["step_lens"] = step_lens

        # Spatial
        result["nodeset_names"] = nodesets
        result["part_names"] = list(all_parts)

        result["node_range"] = (all_nodes[0], all_nodes[-1])
        node_ranges_per_part = dict()
        for s, f in parts_to_nodes.items():
            node_ranges_per_part[s] = (min(f), max(f))
        result["node_ranges_per_part"] = node_ranges_per_part

    finally:
        odb.close()

    return result


if __name__ == "__main__":
    main()

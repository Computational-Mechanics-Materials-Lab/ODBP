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
    results = collect_state(odb_path)
    try:
        result_file = open(result_path, "wb")
        pickle.dump(results, result_file)
    finally:
        result_file.close()

def collect_state(odb_path):
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
        steps_to_lens = dict()
        steps_to_frames = dict()
        all_frames = list()
        all_parts = list()
        frame_keys = list()
        for step_key, step_data in steps.items():
            steps_to_lens[step_key] = len(step_data.frames)

        frame_keys = list(steps[steps.keys()[0]].frames[0].fieldOutputs.keys())

        prev = 0
        for s, l in steps_to_lens.items():
            all_frames.append((prev, l + prev))
            steps_to_frames[s] = (prev, l + prev)
            prev += l + 1

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

        #Temporal
        result["frame_range"] = (all_frames[0][0], all_frames[-1][-1])
        result["frame_keys"] = frame_keys
        result["step_names"] = list(steps_to_frames.keys())
        frame_ranges_per_steps = dict()
        for s, f in steps_to_frames.items():
            frame_ranges_per_steps[s] = f
        result["frame_ranges_per_steps"] = frame_ranges_per_steps 
        
        #Spatial
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
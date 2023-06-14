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

import argparse
from odbAccess import openOdb

def main():
    input_args = "input args"
    parser = argparse.ArgumentParser()
    parser.add_argument(input_args, nargs="*")
    args = vars(parser.parse_args())[input_args]
    if len(args) > 0:
        arg = args[0]
        if arg.lower() in ("true", "t", "verbose", "v"):
            print_state(True)
        else:
            print_state(False)
    else:
        print_state(False)

def print_state(verbose):
    odb_path = r"C:\Users\ch3136\Testing\odbs\v3_05mm_i0_01_T_coord.odb"
    try:
        odb = openOdb(odb_path, readOnly=True)

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

        if verbose:
            all_nodes = list()
            parts_to_nodes = dict()
            for key in all_parts:
                parts_to_nodes[key] = list()
                for node in assembly.instances[key].nodes:
                    parts_to_nodes[key].append(node.label)
                    all_nodes.append(node.label)

        print("Temporal:")
        print("Range of Frames: {} to {}".format(all_frames[0][0], all_frames[-1][-1]))
        print("")
        print("Keys within each frame: {}".format(frame_keys))
        print("")
        print("List of Steps: {}".format(list(steps_to_frames.keys())))
        for s, f in steps_to_frames.items():
            print('Range of Frames for Step "{}": {} to {}'.format(s, *f))
        
        print("")
        print("Spatial:")
        print("Predefined Nodesets: {}".format(nodesets))
        print("")
        print("List of Parts: {}".format(list(all_parts)))

        if verbose:
            print("")
            print("Range of All Nodes: {} to {}".format(all_nodes[0], all_nodes[-1]))
            for s, f in parts_to_nodes.items():
                max_node = max(f)
                min_node = min(f)
                print('Range of Nodes for Part "{}": {} to {}'.format(s, min_node, max_node))

    
    finally:
        odb.close()


if __name__ == "__main__":
    main()
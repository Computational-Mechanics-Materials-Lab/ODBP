#!/usr/bin/env python

"""
ODBPlotter convert.py

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
import os
import pickle
import numpy as np
import argparse
import multiprocessing
import collections
from odbAccess import openOdb


def main():
    """
    Helper function to parse command-line arguments from subprocess.run
    and format these values to correctly convert the .odb to the .npz files.
    This reads the pickle file passed by file path and passes these values
    to the convert_odb_to_npz() method
    """

    # Parse the subprocess input args
    input_args = "input args"
    parser = argparse.ArgumentParser()
    parser.add_argument(input_args, nargs="*")
    odb_path, pickle_path, result_path = vars(parser.parse_args())[input_args]

    # Try our best to manage Python 2 file handling
    pickle_file = open(pickle_path, "rb")
    try:
        input_dict = pickle.load(pickle_file)

    finally:
        pickle_file.close()

    # Now we can remove the file
    os.remove(pickle_path)

    user_nodes = input_dict.get("nodes")
    if isinstance(user_nodes, collections.Mapping):
        temp_nodes = dict()
        for key, val in user_nodes.items():
            temp_nodes[str(key)] = val
        user_nodes = temp_nodes

    unicode_nodesets = input_dict.get("nodesets")
    if unicode_nodesets is not None:
        user_nodesets = list()
        for nodeset in unicode_nodesets:
            user_nodesets.append(str(nodeset))
    else:
        user_nodesets = None

    user_frames = input_dict.get("frames")

    unicode_parts = input_dict.get("parts")
    if unicode_parts is not None:
        user_parts = list()
        for part in unicode_parts:
            user_parts.append(str(part))
    else:
        user_parts = None

    unicode_steps = input_dict.get("steps")
    if unicode_steps is not None:
        user_steps = list()
        for step in unicode_steps:
            user_steps.append(str(step))
    else:
        user_steps = None

    coord_key = str(input_dict.get("coord_key", "COORD"))
    temp_key = str(input_dict.get("temp_key", "NT11"))
    num_cpus = int(input_dict.get("cpus"))

    result_name = convert_odb_to_npz(odb_path, user_nodesets, user_nodes, user_frames, user_parts, user_steps, coord_key, temp_key, num_cpus)
    try:
        result_file = open(result_path, "wb")
        pickle.dump(result_name, result_file)
    finally:
        result_file.close()


def convert_odb_to_npz(odb_path, user_nodesets, user_nodes, user_frames, user_parts, user_steps, coord_key, temp_key, num_cpus):
    """
    Based on the 4 lists given, convert the .odb data to .npz files
    odb_path: str path to the .odb file
    user_parts: list[str] which parts to convert (default to all)
    user_nodesets: list[str] which nodesets to convert (default to all)
    user_nodes: list[int] which nodes to turn into a new nodeset (default to None)
    frames: list[int] which frames to convert (default to all)
    user_steps: list[str] which steps to convert (default to alll)

    return the name of the directory of .npz files created
    """

    # Create the results directory
    parent_dir = os.path.join(os.getcwd(), "tmp_npz")
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    temps_dir = os.path.join(parent_dir, "temps")
    if not os.path.exists(temps_dir):
        os.mkdir(temps_dir)

    time_dir = os.path.join(parent_dir, "step_frame_times")
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)

    try:
        odb = openOdb(odb_path, readOnly=True)

    except Exception as e:
        print("Abaqus Error:")
        print(e)
        sys.exit(1)

    try:
        steps = odb.steps

        base_times = [
            (step_key, step_val.totalTime) for step_key, step_val in steps.items()
            ]

        assembly = odb.rootAssembly

        target_frames = set()
        if user_steps is not None:
            for step in user_steps:
                step_data = steps[step]
                for frame in step_data.frames:
                    target_frames.add(frame.frameId)

        if user_frames is not None:
            for frame in user_frames:
                target_frames.add(frame.frameId)

        target_frames = sorted(list(target_frames))

        if len(target_frames) == 0:
            for step_data in steps.values():
                for i, _ in enumerate(step_data.frames):
                    target_frames.append(i) 

        target_nodesets = set()
        if user_nodes is not None:
            if isinstance(user_nodes, collections.Mapping):
                for key, val in user_nodes.items():
                    assembly.NodeSetFromNodeLabels(name=key, nodeLabels=(val,))
                    target_nodesets.add(key)
            elif isinstance(user_nodes, collections.Iterable):
                if isinstance(user_nodes[0], int):
                    user_nodes = [user_nodes]
                for i, val in enumerate(user_nodes):
                    assembly.NodeSetFromNodeLabels(name=i, nodeLabels=(val,))
                    target_nodesets.add(i)

        if user_parts is not None:
            for part in assembly.instances.keys():
                for nodeset in assembly.instances[part].nodeSets:
                    target_nodesets.append(nodeset)

        if user_nodesets is not None:
            for nodeset in user_nodesets:
                target_nodesets.add(nodeset)

        target_nodesets = sorted(list(target_nodesets))

        if len(target_nodesets) == 0:
            target_nodesets = list(assembly.nodeSets.keys())

    finally:
        odb.close()

    for nodeset in target_nodesets:
        for step_key, base_time in base_times:
            coord_file = os.path.join(parent_dir, "node_coords.npz")
            read_nodeset_coords(odb_path, nodeset, coord_file, step_key, coord_key)
            read_step_data(odb_path, temps_dir, time_dir, step_key, base_time, target_frames, nodeset, temp_key, num_cpus)

    return parent_dir


def read_step_data(odb_path, temps_dir, time_dir, step_key, base_time, target_frames, nodeset, temp_key, num_cpus):
    try:
        odb = openOdb(odb_path, readOnly=True)

    except Exception as e:
        print("Abaqus Error:")
        print(e)
        sys.exit(1)

    try:
        steps = odb.steps

        curr_step_dir = os.path.join(temps_dir, step_key)
        if not os.path.exists(curr_step_dir):
            os.mkdir(curr_step_dir)

        max_frame = max(target_frames)

        max_pad = len(str(max_frame))

        manager = multiprocessing.Manager()
        frame_times = manager.list()
        if len(steps[step_key].frames) > 0:
            idx_list = [i for i in range(len(steps[step_key].frames))]
            idx_list_len = len(idx_list)
            # TODO: what if the length isn't divisible by the number of processors (is it now?)
            final_idx_list = [idx_list[i: i + int(idx_list_len / num_cpus)] for i in range(0, idx_list_len, max(int(idx_list_len / num_cpus), 1))]

            temp_procs = list()
            for idx_list in final_idx_list:
                p = multiprocessing.Process(target=read_single_frame_temp, args=(odb_path, idx_list, max_pad, target_frames, step_key, curr_step_dir, frame_times, base_time, nodeset, temp_key))
                p.start()
                temp_procs.append(p)

            for p in temp_procs:
                p.join()

            np.savez_compressed(
                "{}.npz".format(
                    os.path.join(
                        time_dir,
                        step_key
                        )
                    ),
                np.sort(
                    np.array(
                        frame_times
                        )
                    )
                )

    finally:
        odb.close()


def read_single_frame_temp(odb_path, idx_list, max_pad, target_frames, step_key, curr_step_dir, frame_times, base_time, nodeset, temp_key):

    try:
        odb = openOdb(odb_path, readOnly=True)

    except Exception as e:
        print("Abaqus Error:")
        print(e)
        sys.exit(1)

    try:
        steps = odb.steps
        assembly = odb.rootAssembly

        for idx in idx_list:
            if idx not in target_frames:
                continue
            
            frame = steps[step_key].frames[idx]

            field = frame.fieldOutputs[temp_key].getSubset(region=assembly.nodeSets[nodeset])
            frame_times.append(float(format(round(frame.frameValue + base_time, 5), ".2f")))
            node_temps = list()
            for item in field.values:
                temp = item.data
                node_temps.append(temp)

            num = str(idx + 1).zfill(max_pad)
            np.savez_compressed(
                    os.path.join(
                        curr_step_dir,
                        "frame_{}".format(num)
                        ),
                    np.array(node_temps)
                    )

    finally:
        odb.close()


def read_nodeset_coords(odb_path, nodeset, coord_file, step_key, coord_key):

    # Only extract coordinates from the first given nodeset/step
    if not os.path.exists(coord_file):

        try:
            odb = openOdb(odb_path, readOnly=True)

        except Exception as e:
            print("Abaqus Error:")
            print(e)
            sys.exit(1)

        try:
            assembly = odb.rootAssembly
            steps = odb.steps

            frame = steps[step_key].frames[0]

            try:
                coords = frame.fieldOutputs[coord_key].getSubset(region=assembly.nodeSets[nodeset])

                results_list = list()
                for item in coords.values:
                    node = item.nodeLabel
                    coord = item.data
                    xyz = [node]
                    for axis in coord:
                        xyz.append(axis)
                    results_list.append(xyz)
                np.savez_compressed(coord_file, np.array(results_list))

            except KeyError:
                pass

        finally:
            odb.close()

    
if __name__ == "__main__":
    main()

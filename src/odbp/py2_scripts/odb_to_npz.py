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


import os
import pickle
import numpy as np
import argparse
import multiprocessing
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
        data_to_extract = pickle.load(pickle_file)

    finally:
        pickle_file.close()

    # Now we can remove the file
    os.remove(pickle_path)

    nodesets = data_to_extract["nodesets"]
    frames = data_to_extract["frames"]
    num_cpus = data_to_extract["cpus"]

    result_name = convert_odb_to_npz(odb_path, nodesets, frames, num_cpus)
    result_file = open(result_path, "wb")
    try:
        pickle.dump(result_name, result_file)
    finally:
        result_file.close()



def convert_odb_to_npz(odb_path, nodesets, frames, num_cpus):
    """
    Based on the 4 lists given, convert the .odb data to .npz files
    odb_path: str path to the .odb file
    nodes: list[int] which nodes to convert (default to all)
    parts: list[str] which parts to convert (default to no specific part)
    nodesets: list[str] which nodesets to convert (default to the first
    nodeset)
    frames: list[int] which frames to convert (default to all)

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

    odb = openOdb(odb_path, readOnly=True)

    steps = odb.steps

    base_times = [
        (step_key, step_val.totalTime) for step_key, step_val in steps.items()
        ]
    
    assembly = odb.rootAssembly
    nodeset_keys = assembly.nodeSets.keys()
    if nodesets is None:
        nodesets = nodeset_keys
    else:
        for nodeset in nodesets:
            if nodeset not in nodeset_keys:
                raise ValueError(
                    '"{0}" is not a valid nodeset key.' \
                        'Possible values in this .odb are "{1}"'
                .format(nodeset, nodeset_keys)
                )

    odb.close()

    for nodeset in nodesets:
        for step_key, base_time in base_times:
            coord_file = os.path.join(parent_dir, "node_coords.npz")
            read_nodeset_coords(odb_path, nodeset, coord_file, step_key)
            read_step_data(odb_path, temps_dir, time_dir, step_key, base_time, frames, nodeset, num_cpus)

    return parent_dir


def read_step_data(odb_path, temps_dir, time_dir, step_key, base_time, frames, nodeset, num_cpus):
    odb = openOdb(odb_path, readOnly=True)
    steps = odb.steps

    curr_step_dir = os.path.join(temps_dir, step_key)
    if not os.path.exists(curr_step_dir):
        os.mkdir(curr_step_dir)

    if frames is not None:
        max_frame = max(frames)

    else:
        max_frame = len(steps[step_key].frames)

    max_pad = len(str(max_frame))

    manager = multiprocessing.Manager()
    frame_times = manager.list()
    #frame_times = list()
    if len(steps[step_key].frames) > 0:
        idx_list = [i for i in range(len(steps[step_key].frames))]
        idx_list_len = len(idx_list)
        idx_list_max = idx_list[-1]
        # TODO: what if the length isn't divisible by the number of processors (is it now?)
        final_idx_list = [idx_list[i: i + int(idx_list_len / num_cpus)] for i in range(0, idx_list_len, max(int(idx_list_len / num_cpus), 1))]
        odb.close()

        temp_procs = list()
        for idx_list in final_idx_list:
            p = multiprocessing.Process(target=read_single_frame_temp, args=(odb_path, idx_list, max_pad, frames, step_key, curr_step_dir, frame_times, base_time, nodeset))
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


def read_single_frame_temp(odb_path, idx_list, max_pad, frames, step_key, curr_step_dir, frame_times, base_time, nodeset):

    odb = openOdb(odb_path, readOnly=True)
    steps = odb.steps
    assembly = odb.rootAssembly

    for idx in idx_list:
        if frames is not None and idx not in frames:
            continue
        
        frame = steps[step_key].frames[idx]

        field = frame.fieldOutputs["NT11"].getSubset(region=assembly.nodeSets[nodeset])
        frame_times.append(float(format(round(frame.frameValue + base_time, 5), ".2f")))
        node_temps = list()
        for item in field.values:
            temp = item.data
            node_temps.append(temp)

        num = str(idx + 1).zfill(max_pad)
        np.savez_compressed(
                os.path.join(
                    curr_step_dir,
                    "frame_{".format(num)
                    ),
                np.array(node_temps)
                )

    odb.close()


def read_nodeset_coords(odb_path, nodeset, coord_file, step_key):

    # Only extract coordinates from the first given nodeset/step
    if not os.path.exists(coord_file):

        odb = openOdb(odb_path, readOnly=True)
        assembly = odb.rootAssembly
        steps = odb.steps

        frame = steps[step_key].frames[0]

        try:
            coords = frame.fieldOutputs["COORD"].getSubset(region=assembly.nodeSets[nodeset])

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

        odb.close()

    
if __name__ == "__main__":
    main()

#!/usr/bin/env python

"""
ODBPlotter converter.py

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
    target_outputs = input_dict.get("target_outputs", ["NT11"])
    if isinstance(target_outputs, collections.Iterable):
        target_outputs = [str(key) for key in target_outputs]
    elif isinstance(target_outputs, str):
        target_outputs = [str(target_outputs)]
    num_cpus = int(input_dict.get("cpus"))
    time_step = int(input_dict.get("time_step", 1))

    #data_model = int(input_dict["data_model"])

    result_name = convert_odb_to_npz(
        odb_path,
        user_nodesets,
        user_nodes,
        user_parts,
        user_steps,
        coord_key,
        target_outputs,
        num_cpus,
        time_step,
        #data_model,
    )
    try:
        result_file = open(result_path, "wb")
        pickle.dump(result_name, result_file)
    finally:
        result_file.close()


def convert_odb_to_npz(
    odb_path,
    user_nodesets,
    user_nodes,
    user_parts,
    user_steps,
    coord_key,
    target_outputs,
    num_cpus,
    time_step,
    #data_model,
):
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

    data_dir = os.path.join(parent_dir, "data")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

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

        base_times = list()
        total_idx = 0
        for step_key, step_val in steps.items():
            base_times.append((step_key, step_val.totalTime, total_idx))
            total_idx += len(step_val.frames)

        max_idx = base_times[-1][2] + len(steps[base_times[-1][0]].frames) - 1
        assembly = odb.rootAssembly

        target_frames = dict()
        if user_steps is not None:
            for step in user_steps:
                step_data = steps[step]
                target_frames[step] = set()
                for frame in step_data.frames:
                    target_frames[step].add(frame.frameId)

            for step in target_frames:
                target_frames[step] = sorted(list(target_frames[step]))

        else:
            for step, step_data in steps.items():
                target_frames[step] = list()
                for frame in step_data.frames:
                    target_frames[step].append(frame.frameId)

        for step in target_frames:
            target_frames[step] = target_frames[step][::time_step]

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

        if target_outputs is None or len(target_outputs) == 0:
            target_outputs = list(steps[steps.keys()[0]].frames[0].fieldOutputs.keys())
            try:
                target_outputs.remove(coord_key)
            except ValueError:
                raise ValueError(
                    "Coordinate Key {} was not found in this .odb file.".format(
                        coord_key
                    )
                )

    finally:
        odb.close()

    for nodeset in target_nodesets:
        for step_key, base_time, base_idx in base_times:
            coord_file = os.path.join(parent_dir, "node_coords.npz")
            #if data_model == 0:
            read_nodeset_coords(odb_path, nodeset, coord_file, step_key, coord_key)
            read_step_data(
                odb_path,
                data_dir,
                time_dir,
                step_key,
                base_time,
                base_idx,
                max_idx,
                target_frames[step_key],
                nodeset,
                target_outputs,
                num_cpus,
                #data_model,
                #coord_key,
            )

    return parent_dir


def read_step_data(
    odb_path,
    data_dir,
    time_dir,
    step_key,
    base_time,
    base_idx,
    max_idx,
    target_frames,
    nodeset,
    target_outputs,
    num_cpus,
    #data_model,
    #coord_key,
):
    try:
        odb = openOdb(odb_path, readOnly=True)

    except Exception as e:
        print("Abaqus Error:")
        print(e)
        sys.exit(1)

    try:
        steps = odb.steps

        curr_step_dir = os.path.join(data_dir, step_key)
        if not os.path.exists(curr_step_dir):
            os.mkdir(curr_step_dir)

        max_pad = len(str(max_idx))

        #manager = multiprocessing.Manager()
        #frame_times = manager.list()
        if len(steps[step_key].frames) > 0:
            frame_list_len = len(target_frames)
            # TODO: what if the length isn't divisible by the number of processors (is it now?)
            combined_frame_list = [
                target_frames[i : i + max(int(frame_list_len / num_cpus), 1)]
                for i in range(
                    0, frame_list_len, max(int(frame_list_len / num_cpus), 1)
                )
            ]

            temp_procs = list()
            for frame_list in combined_frame_list:
                ##if data_model == 1:
                #    p = multiprocessing.Process(
                #        target=read_single_frame_data,
                #        args=(
                #            odb_path,
                #            frame_list,
                #            max_pad,
                #            step_key,
                #            curr_step_dir,
                #            frame_times,
                #            base_time,
                #            base_idx,
                #            nodeset,
                #            target_outputs,
                #            coord_key,
                #        ),
                #    )
                #else:
                p = multiprocessing.Process(
                    target=read_single_frame_temp,
                    args=(
                        odb_path,
                        frame_list,
                        max_pad,
                        step_key,
                        curr_step_dir,
                        #frame_times,
                        base_time,
                        base_idx,
                        nodeset,
                        target_outputs,
                    ),
                )
                p.start()
                temp_procs.append(p)

            for p in temp_procs:
                p.join()

            #np.savez_compressed(
            #    "{}.npz".format(os.path.join(time_dir, step_key)),
            #    np.array(frame_times),
            #)

    finally:
        odb.close()


def read_single_frame_temp(
    odb_path,
    frame_list,
    max_pad,
    step_key,
    curr_step_dir,
    #frame_times,
    base_time,
    base_idx,
    nodeset,
    target_outputs,
):
    try:
        odb = openOdb(odb_path, readOnly=True)

    except Exception as e:
        print("Abaqus Error:")
        print(e)
        sys.exit(1)

    try:
        steps = odb.steps
        assembly = odb.rootAssembly

        for idx, frame in enumerate(steps[step_key].frames):
            if frame.frameId not in frame_list:
                continue

            num = str(idx + base_idx).zfill(max_pad)

            for output in target_outputs:
                field = frame.fieldOutputs[output].getSubset(
                    region=assembly.nodeSets[nodeset]
                )
                node_vals = list()
                for item in field.values:
                    val = item.data
                    node_vals.append(val)

                if len(node_vals) > 0:
                    np.savez_compressed(
                        os.path.join(curr_step_dir, "{}_{}".format(output, num)),
                        np.array(node_vals),
                    )

            
            np.savez_compressed(
                os.path.join(curr_step_dir, "Time_{}".format(num)),
                np.asarray([round(frame.frameValue + base_time, 5)])
            )

    finally:
        odb.close()


#def read_single_frame_data(
#    odb_path,
#    frame_list,
#    max_pad,
#    step_key,
#    curr_step_dir,
#    frame_times,
#    base_time,
#    base_idx,
#    nodeset,
#    target_outputs,
#    coord_key,
#):
#    all_data_outputs = target_outputs[:]
#    all_data_outputs.append(coord_key)
#
#    try:
#        odb = openOdb(odb_path, readOnly=True)
#
#    except Exception as e:
#        print("Abaqus Error:")
#        print(e)
#        sys.exit(1)
#
#    try:
#        steps = odb.steps
#        assembly = odb.rootAssembly
#
#        for idx, frame in enumerate(steps[step_key].frames):
#            if frame.frameId not in frame_list:
#                continue
#
#            frame_times.append(
#                float(format(round(frame.frameValue + base_time, 5), ".2f"))
#            )
#
#            for output in all_data_outputs:
#                field = frame.fieldOutputs[output].getSubset(
#                    region=assembly.nodeSets[nodeset]
#                )
#                node_vals = list()
#                if output == coord_key:
#                    for item in field.values:
#                        node = item.nodeLabel
#                        coord = item.data
#                        xyz = [node]
#                        for axis in coord:
#                            xyz.append(axis)
#                        node_vals.append(xyz)
#
#                    if len(node_vals) > 0:
#                        node_vals = np.array(node_vals)
#                        num = str(idx + base_idx).zfill(max_pad)
#                        for i, key in enumerate(["Node Label", "X", "Y", "Z"]):
#                            np.savez_compressed(
#                                os.path.join(curr_step_dir, "{}_{}".format(key, num)),
#                                node_vals[:, i],
#                            )
#                else:
#                    for item in field.values:
#                        val = item.data
#                        node_vals.append(val)
#
#                    if len(node_vals) > 0:
#                        num = str(idx + base_idx).zfill(max_pad)
#                        np.savez_compressed(
#                            os.path.join(curr_step_dir, "{}_{}".format(output, num)),
#                            np.array(node_vals),
#                        )
#
#    finally:
#        odb.close()


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
                coords = frame.fieldOutputs[coord_key].getSubset(
                    region=assembly.nodeSets[nodeset]
                )

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

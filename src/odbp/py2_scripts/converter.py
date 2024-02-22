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

    user_elements = input_dict.get("elements")
    if isinstance(user_elements, collections.Mapping):
        temp_elements = dict()
        for key, val in user_elements.items():
            temp_elements[str(key)] = val
        user_elements = temp_elements

    unicode_elementsets = input_dict.get("elementsets")
    if unicode_elementsets is not None:
        user_elementsets = list()
        for elementset in unicode_elementsets:
            user_elementsets.append(str(elementset))
    else:
        user_elementsets = None

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
    num_cpus = int(input_dict.get("cpus", multiprocessing.cpu_count()))
    time_step = int(input_dict.get("time_step", 1))

    result_name = convert_odb_to_npz(
        odb_path,
        user_nodesets,
        user_nodes,
        user_elementsets,
        user_elements,
        user_parts,
        user_steps,
        coord_key,
        num_cpus,
        time_step,
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
    user_elementsets,
    user_elements,
    user_parts,
    user_steps,
    coord_key,
    num_cpus,
    time_step,
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
                    target_nodesets.add(nodeset)

        if user_nodesets is not None:
            for nodeset in user_nodesets:
                target_nodesets.add(nodeset)

        target_nodesets = sorted(list(target_nodesets))

        if len(target_nodesets) == 0:
            target_nodesets = [
                list(assembly.nodeSets.keys())[0]
            ]  # If none is specified, grab the first

        target_elementsets = set()
        if user_elements is not None:
            if isinstance(user_elements, collections.Mapping):
                for key, val in user_elements.items():
                    assembly.ElementSetFromElementLabels(name=key, elementLabels=(val,))
                    target_elementsets.add(key)
            elif isinstance(user_elements, collections.Iterable):
                if isinstance(user_elements[0], int):
                    user_elements = [user_elements]
                for i, val in enumerate(user_elements):
                    assembly.ElementSetFromElementLabels(name=i, elementLabels=(val,))
                    target_elementsets.add(i)

        if user_parts is not None:
            for part in assembly.instances.keys():
                for elementset in assembly.instances[part].elementSets:
                    target_elementsets.add(elementset)

        if user_elementsets is not None:
            for elementset in user_elementsets:
                target_elementsets.add(elementset)

        target_elementsets = sorted(list(target_elementsets))

        if len(target_elementsets) == 0:
            target_elementsets = [
                list(assembly.elementSets.keys())[0]
            ]  # If none is specified, grab the first

        all_outputs = list(steps[steps.keys()[0]].frames[0].fieldOutputs.keys())
        if coord_key not in all_outputs:
            raise ValueError(
                "Coordinate Key {} was no found in this .odb file".format(coord_key)
            )

        first_step = list(steps.values())[0]
        first_frame = first_step.frames[0]

        if len(target_nodesets) > 0:
            first_nodeset = target_nodesets[0]
            first_nodeset_field = first_frame.fieldOutputs[coord_key].getSubset(
                region=assembly.nodeSets[first_nodeset]
            )
            nodeset_coords = len(first_nodeset_field.values) != 0
        else:
            nodeset_coords = False

        if len(target_elementsets) > 0:
            first_elset = target_elementsets[0]
            first_elset_field = first_frame.fieldOutputs[coord_key].getSubset(
                region=assembly.elementSets[first_elset]
            )
            elset_coords = len(first_elset_field.values) != 0
        else:
            elset_coords = False

        if not nodeset_coords and not elset_coords:
            print(
                "ERROR! {} not mapped to either Nodesets or Elementsets".format(
                    coord_key
                )
            )
            sys.exit(-1)

    finally:
        odb.close()

    if nodeset_coords:
        use_nodeset = True
        for nodeset in target_nodesets:
            for step_key, base_time, base_idx in base_times:
                read_step_data(
                    odb_path,
                    data_dir,
                    step_key,
                    base_time,
                    base_idx,
                    max_idx,
                    target_frames[step_key],
                    nodeset,
                    num_cpus,
                    use_nodeset,
                )

    else:
        use_nodeset = False
        for elset in target_elementsets:
            for step_key, base_time, base_idx in base_times:
                read_step_data(
                    odb_path,
                    data_dir,
                    step_key,
                    base_time,
                    base_idx,
                    max_idx,
                    target_frames[step_key],
                    elset,
                    num_cpus,
                    use_nodeset,
                )

    return parent_dir


def read_step_data(
    odb_path,
    data_dir,
    step_key,
    base_time,
    base_idx,
    max_idx,
    target_frames,
    elset_or_nodeset,
    num_cpus,
    use_nodeset,
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
                p = multiprocessing.Process(
                    target=read_single_frame_data,
                    args=(
                        odb_path,
                        frame_list,
                        max_pad,
                        step_key,
                        curr_step_dir,
                        base_time,
                        base_idx,
                        elset_or_nodeset,
                        use_nodeset,
                    ),
                )
                p.start()
                temp_procs.append(p)

            for p in temp_procs:
                p.join()

    finally:
        odb.close()


def read_single_frame_data(
    odb_path,
    frame_list,
    max_pad,
    step_key,
    curr_step_dir,
    base_time,
    base_idx,
    elset_or_nodeset,
    use_nodeset,
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

            all_outputs = list(steps[steps.keys()[0]].frames[0].fieldOutputs.keys())
            for output in all_outputs:
                if use_nodeset:
                    nodeset = elset_or_nodeset
                    field = frame.fieldOutputs[output].getSubset(
                        region=assembly.nodeSets[nodeset]
                    )
                else:
                    elset = elset_or_nodeset
                    field = frame.fieldOutputs[output].getSubset(
                        region=assembly.elementSets[elset]
                    )
                node_vals = []
                if len(field.componentLabels) > 0:
                    for val in field.values:
                        coord_triple = {}
                        for item, lab in zip(val.data, field.componentLabels):
                            coord_triple[lab] = item

                        node_vals.append(coord_triple)

                else:
                    for item in field.values:
                        val = item.data
                        node_vals.append(val)

                if len(node_vals) > 0:
                    if type(node_vals[0]) is dict:
                        reordered_dict = {}
                        for k in node_vals[0].keys():
                            reordered_dict[k] = []

                        for node_dict in node_vals:
                            for k, v in node_dict.items():
                                reordered_dict[k].append(v)

                        for k, v in reordered_dict.items():
                            np.savez_compressed(
                                os.path.join(
                                    curr_step_dir, "{}_{}_{}".format(output, k, num)
                                ),
                                np.array(v),
                            )
                    else:
                        np.savez_compressed(
                            os.path.join(curr_step_dir, "{}_{}".format(output, num)),
                            np.array(node_vals),
                        )

            np.savez_compressed(
                os.path.join(curr_step_dir, "Time_{}".format(num)),
                np.asarray([round(frame.frameValue + base_time, 5)]),
            )

    finally:
        odb.close()
        raise SystemExit


if __name__ == "__main__":
    main()

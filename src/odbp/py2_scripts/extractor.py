# -*- coding: utf-8 -*-
"""
Written by J. Logan Betts for CMML 
Abaqus Python (2.7) 
Extracts a user defined node set, all_nodes, part_nodes, 
sub_nodes for avg, min, max, time. The user node set 
extracts as one 2D array to be split in the corresponding 
Python 3 script: 

"""
"""
Updated by Clark Hensley for CMML
"""

import sys
import pickle
import argparse
import collections
import numpy as np
from odbAccess import openOdb


def main():
    input_args = "input args"
    parser = argparse.ArgumentParser()
    parser.add_argument(input_args, nargs="*")
    odb_path, pickle_path, save_path = vars(parser.parse_args())[input_args]

    try:
        input_file = open(pickle_path, "rb")
        input_dict = pickle.load(input_file)
    finally:
        input_file.close()

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

    target_outputs = str(input_dict.get("target_outputs", ["NT11"]))

    try:
        odb = openOdb(odb_path, readOnly=True)

    except Exception as e:
        print("Abaqus Error:")
        print(e)
        sys.exit(1)

    try:
        steps = odb.steps
        assembly = odb.rootAssembly

        target_frames = dict()
        if user_steps is not None:
            for step in user_steps:
                step_data = steps[step]
                target_frames[step] - set()
                for frame in step_data.frames:
                    target_frames[step].add(frame.frameId)

            for step in target_frames():
                target_frames[step] = sorted(list(target_frames[step]))

        else:
            for step, step_data in steps.items():
                target_frames[step] = list()
                for frame in step_data.frames:
                    target_frames[step].append(frame.frameId)

        time_step = int(input_dict.get("time_step"), 1)
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

    finally:
        odb.close()

    extract(odb_path, save_path, target_nodesets, target_frames, target_outputs)


def extract(odb_path, save_path, target_nodesets, target_frames, target_outputs):

    try:
        odb = openOdb(odb_path, readOnly=True)

    except Exception as e:
        print("Abaqus Error:")
        print(e)
        sys.exit(1)

    try:
        steps = odb.steps
        assembly = odb.rootAssembly

        final_record = dict()
        for output in target_outputs:
            final_record[output] = list()
            for step in steps.keys():
                step_start_time = steps[step].totalTime
                for frame in steps[step].frames:
                    if frame.frameId not in target_frames[step]:
                        continue
                    frame_time = step_start_time + frame.frameValue
                    selected_output_results = frame.fieldOutputs[output]
                    for nodeset in target_nodesets:

                        region = assembly.nodeSets[nodeset]
                        output_subset = selected_output_results.getSubset(region=region)
                        output_vals = np.copy(output_subset.bulkDataBlocks[0].data).astype("float64")
                        if output in ("NT11",):
                            output_vals[output_vals == 0] = np.nan
                            output_vals[output_vals == 300] = np.nan
                        output_vals = output_vals[~np.isnan(output_vals)]

                        final_record[output].append({
                            "time": frame_time,
                            "max": np.max(output_vals) if len(output_vals) > 0 else np.nan,
                            "mean": np.mean(output_vals) if len(output_vals) > 0 else np.nan,
                            "min": np.min(output_vals) if len(output_vals) > 0 else np.nan,
                        })

    finally:
        odb.close()

    try:
        save_file = open(save_path, "wb")
        pickle.dump(final_record, save_file, protocol=2)

    finally:
        save_file.close()

 
if __name__ == "__main__":
    main()

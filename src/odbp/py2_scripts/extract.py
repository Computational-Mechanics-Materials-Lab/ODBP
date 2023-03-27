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
import pickle
import argparse
import numpy as np
from odbAccess import openOdb

def main(odb_path, save_path, parts=None, nodesets=None, nodes=None):
    odb = openOdb(odb_path, readOnly=True)
    steps = odb.steps
    root_assembly = odb.rootAssembly
    instances = root_assembly.instances

    final_record = dict()
    if parts is not None:
        parts_to_values = dict()
        nodesets_to_values = dict()

        # If nodes is not none, it should be a dictionary
        # of str: list[int]
        if nodes is not None:
            for key, val in nodes:
                root_assembly.NodeSetFromNodeLabels(name=key, nodeLabels=(val,))

        if nodesets is None:
            nodesets = dict()
            for part in parts:
                parts_to_values[part] = instances[part]
                nodesets_to_values[part] = parts_to_values[part].nodeSets
                nodesets[part] = nodesets_to_values[part].keys()

        elif isinstance(nodesets, list):
            new_nodesets_dict = dict()
            for part in parts:
                new_nodesets = list()
                for n in nodesets:
                    parts_to_values[part] = instances[part]
                    nodesets_to_values[part] = parts_to_values[part].nodeSets
                    if n in nodesets_to_values[part].keys():
                        new_nodesets.append(n)
                new_nodesets_dict[part] = new_nodesets
            nodesets = new_nodesets_dict

        for part in parts:
            final_record[part] = dict()
            print("Extracting for Part {}".format(part))
            for step_key in steps.keys():
                print("\tExtracting for Step {}".format(steps[step_key].name))
                final_record[part][step_key] = dict()
                final_record[part][step_key]["time"] = list()
                ## Building in time output 
                step_start_time = steps[step_key].totalTime # time till the 'Step-1'
                for frame in steps[step_key].frames:
                    current_time = step_start_time + frame.frameValue
                    print("\t\tExtracting for Time {}".format(current_time))
                    selected_temp_results = frame.fieldOutputs["NT11"]
                    final_record[part][step_key]["time"].append(current_time)
                    for nodeset in nodesets[part]:
                        final_record[part][step_key][nodeset] = {
                                "max": list(),
                                "avg": list(),
                                "min": list(),
                                }
                        region = nodesets_to_values[part][nodeset]
                        temp_subset = selected_temp_results.getSubset(region = region)
                        temp = np.copy(temp_subset.bulkDataBlocks[0].data).astype("float64")
                        temp[temp == 0] = np.nan
                        temp[temp == 300] = np.nan
                        updated_temp = temp[~np.isnan(temp)]

                        final_record[part][step_key][nodeset]["max"].append(np.max(updated_temp) if len(updated_temp) > 0 else np.nan)
                        final_record[part][step_key][nodeset]["avg"].append(np.mean(updated_temp) if len(updated_temp) > 0 else np.nan)
                        final_record[part][step_key][nodeset]["min"].append(np.min(updated_temp) if len(updated_temp) > 0 else np.nan)

    else:
        for step_key in steps.keys():
            print("Extracting for Step {}".format(steps[step_key].name))
            final_record[step_key] = dict()
            final_record[step_key]["time"] = list()
            step_start_time = steps[step_key].totalTime
            for frame in steps[step_key].frames:
                current_time = step_start_time + frame.frameValue
                print("\t\tExtracting for Time {}".format(current_time))
                final_record[step_key]["time"].append(current_time)
                selected_temp_results = frame.fieldOutputs["NT11"]
                final_record[step_key]["ALL NODES"] = {
                    "max": list(),
                    "avg": list(),
                    "min": list(),
                }
                region = root_assembly.nodeSets["ALL NODES"]
                temp_subset = selected_temp_results.getSubset(region=region)
                temp = np.copy(temp_subset.bulkDataBlocks[0].data).astype("float64")
                temp[temp == 0] = np.nan
                temp[temp == 300] = np.nan
                updated_temp = temp[~np.isnan(temp)]

                final_record[part][step_key]["ALL NODES"]["max"].append(np.max(updated_temp) if len(updated_temp) > 0 else np.nan)
                final_record[part][step_key]["ALL NODES"]["avg"].append(np.mean(updated_temp) if len(updated_temp) > 0 else np.nan)
                final_record[part][step_key]["ALL NODES"]["min"].append(np.min(updated_temp) if len(updated_temp) > 0 else np.nan)

    odb.close()

    save_file = open(save_path, "wb")
    pickle.dump(final_record, save_file, protocol=2)
    save_file.close()

 
if __name__ == "__main__":
    input_args = "input args"
    parser = argparse.ArgumentParser()
    parser.add_argument(input_args, nargs="*")
    odb_path, save_path = vars(parser.parse_args())[input_args]

    input_file = open(save_path, "rb")
    input_dict = pickle.load(input_file)
    input_file.close()
    old_parts = input_dict["parts"]
    old_nodesets = input_dict["nodesets"]
    old_nodes = input_dict["nodes"]

    if old_parts is not None:
        parts = list()
        for p in old_parts:
            parts.append(str(p))
    else:
        parts = old_parts

    if old_nodesets is not None:
        nodesets = list()
        for n in old_nodesets:
            nodesets.append(str(n))
    else:
        nodesets = old_nodesets

    if old_nodes is not None:
        nodes = dict()
        for k, v in old_nodes:
            nodes[str(k)] = int(v)
    else:
        nodes = old_nodes

    main(odb_path, save_path, parts, nodesets, nodes)

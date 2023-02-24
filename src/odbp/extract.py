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
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from odbAccess import openOdb

def main(odb_path, save_path, parts=None, nodesets=None, nodes=None):
    odb = openOdb(odb_path, readOnly=True)

    final_record = dict()
    if parts is not None:

        # If nodes is not none, it should be a dictionary
        # of str: list[int]
        if nodes is not None:
            for part in parts:
                for key, val in nodes:
                    odb.rootAssembly.NodeSetFromNodeLabels(name=key, nodeLabels=(val,))

        if nodesets is None:
            nodesets = dict()
            for part in parts:
                nodesets[part] = odb.rootAssembly.instances[part].nodeSets.keys()

        elif isinstance(nodesets, list):
            new_nodesets_dict = dict()
            for part in parts:
                new_nodesets = list()
                for n in nodesets:
                    if n in odb.rootAssembly.instances[part].nodeSets.keys():
                        new_nodesets.append(n)
                new_nodesets_dict[part] = new_nodesets
            nodesets = new_nodesets_dict

        for part in parts:
            final_record[part] = dict()
            for step_key, step_val in odb.steps.values():
                final_record[part][step_key] = dict()
                final_record[part][step_key]["time"] = list()
                ## Building in time output 
                step_start_time = step_val.totalTime # time till the 'Step-1'
                for frame in odb.steps[step_val.name].frames:
                    selected_temp_results = frame.fieldOutputs["NT11"]
                    final_record[part][step_key]["time"].append(step_start_time + frame.frameValue)
                    for nodeset in nodesets:
                        final_record[part][step_key][nodeset] = {
                                "max": list(),
                                "avg": list(),
                                "min": list(),
                                }
                        region = odb.rootAssembly.instances[part].nodesets[nodeset]
                        temp_subset = selected_temp_results.getSubset(region = region)
                        temp = np.copy(temp_subset.bulkDataBlocks[0].data).astype("float64")
                        temp[temp == 0] = np.nan
                        temp[temp == 300] = np.nan
                        updated_temp = temp[~np.isnan(temp)]

                        final_record[part][step_key][nodeset]["max"].append(np.max(updated_temp) if len(updated_temp) > 0 else np.nan)
                        final_record[part][step_key][nodeset]["avg"].append(np.mean(updated_temp) if len(updated_temp) > 0 else np.nan)
                        final_record[part][step_key][nodeset]["min"].append(np.min(updated_temp) if len(updated_temp) > 0 else np.nan)

    else:
        for step_key, step_val in odb.steps.values():
            final_record[step_key] = dict()
            final_record[step_key]["time"] = list()
            step_start_time = step_val.totalTime
            for frame in odb.steps[step_val.name].frames:
                selected_temp_results = frame.fieldOutputs["NT11"]
                final_record[part][step_key]["time"].append(step_start_time + frame.frameValue)
                final_record[part][step_key]["ALL NODES"] = {
                    "max": list(),
                    "avg": list(),
                    "min": list(),
                }
                region = odb.rootAssembly.nodeSets["ALL NODES"]
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
    pickle.dump(pd.DataFrame.from_dict(final_record), save_file)
    save_file.close()

 
if __name__ == "__main__":
    input_args = "input args"
    parser = argparse.ArgumentParser()
    parser.add_argument(input_args, nargs="*")
    odb_path, save_path, parts, nodesets, nodes = vars(parser.parser_args())[input_args]
    main(odb_path, save_path, parts, nodesets, nodes)

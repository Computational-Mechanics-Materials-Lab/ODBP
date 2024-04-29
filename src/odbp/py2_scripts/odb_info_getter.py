#!/usr/bin/env python

"""
ODBP odb_to_npz.py

ODBP
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBP
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
        step_lens = {}
        all_parts = []
        frame_keys = []
        frame_keys_per_step = {}
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
        elementsets = assembly.elementSets.keys()

        all_parts = assembly.instances.keys()

        target_part = all_parts[0]

        target_nodeset = list(assembly.instances[target_part].elementSets.keys())[0]
        node_range = len(assembly.instances[key].nodeSets[target_nodeset].nodes)
        nodes_per_part = {}
        nodeset_per_part = {}
        nodes_per_nodeset = {}
        for key in all_parts:
            nodes_per_part[key] = []
            nodeset_per_part[key] = []
            for node in assembly.instances[key].nodeSets[target_nodeset].nodes:
                nodes_per_part[key].append(node.label)

            for nodeset in assembly.instances[key].nodeSets.keys():
                nodeset_per_part[key].append(nodeset)
                if nodeset not in nodes_per_nodeset:
                    nodes_per_nodeset[nodeset] = []
                    for node in assembly.instances[key].nodeSets[nodeset].nodes:
                        nodes_per_nodeset[nodeset].append(node.label)

        target_elementset = list(assembly.instances[target_part].elementSets.keys())[0]
        element_range = len(
            assembly.instances[key].elementSets[target_elementset].elements
        )
        elements_per_part = {}
        elementset_per_part = {}
        elements_per_elementset = {}
        for key in all_parts:
            elements_per_part[key] = []
            for element in (
                assembly.instances[key].elementSets[target_elementset].elements
            ):
                elements_per_part[key].append(element.label)

            for elementset in assembly.instances[key].elementsets.keys():
                elementset_per_part[key].append(elementset)
                if elementset not in elements_per_elementset:
                    elements_per_elementset[elementset] = []
                    for element in (
                        assembly.instances[key].elementSets[elementset].elements
                    ):
                        elements_per_elementset[elementset].append(element.label)

        # Temporal
        result["frame_keys"] = frame_keys
        result["frame_keys_per_step"] = frame_keys_per_step
        result["frame_range"] = frame_range
        result["step_names"] = list(step_lens.keys())
        result["step_lens"] = step_lens

        # Spatial
        result["nodeset_names"] = nodesets
        result["elementset_names"] = elementsets
        result["part_names"] = list(all_parts)

        result["node_range"] = node_range
        result["element_range"] = element_range
        result["nodes_per_part"] = nodes_per_part
        result["nodeset_per_part"] = nodeset_per_part
        result["nodes_per_nodeset"] = nodes_per_nodeset
        result["elements_per_part"] = elements_per_part
        result["elementset_per_part"] = elementset_per_part
        result["elements_per_elementset"] = elements_per_elementset

    finally:
        odb.close()

    return result


if __name__ == "__main__":
    main()

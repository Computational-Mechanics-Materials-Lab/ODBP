#!/usr/bin/env python

"""
ODBP converter.py

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
import os
import pickle
import numpy as np
import argparse
import multiprocessing

# import collections
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

    except Exception as e:
        print("Error with pickling:")
        print(e)

    finally:
        pickle_file.close()

    # Now we can remove the file
    os.remove(pickle_path)

    # TODO
    # user_nodes = input_dict.get("nodes")
    # if isinstance(user_nodes, collections.Mapping):
    #    temp_nodes = dict()
    #    for key, val in user_nodes.items():
    #        temp_nodes[str(key)] = val
    #    user_nodes = temp_nodes

    # unicode_nodesets = input_dict.get("nodesets")
    # if unicode_nodesets is not None:
    #    user_nodesets = list()
    #    for nodeset in unicode_nodesets:
    #        user_nodesets.append(str(nodeset))
    # else:
    #    user_nodesets = None

    # user_elements = input_dict.get("elements")
    # if isinstance(user_elements, collections.Mapping):
    #    temp_elements = dict()
    #    for key, val in user_elements.items():
    #        temp_elements[str(key)] = val
    #    user_elements = temp_elements

    # unicode_elementsets = input_dict.get("elementsets")
    # if unicode_elementsets is not None:
    #    user_elementsets = list()
    #    for elementset in unicode_elementsets:
    #        user_elementsets.append(str(elementset))
    # else:
    #    user_elementsets = None

    # unicode_parts = input_dict.get("parts")
    # if unicode_parts is not None:
    #    user_parts = list()
    #    for part in unicode_parts:
    #        user_parts.append(str(part))
    # else:
    #    user_parts = None

    # unicode_steps = input_dict.get("steps")
    # if unicode_steps is not None:
    #    user_steps = list()
    #    for step in unicode_steps:
    #        user_steps.append(str(step))
    # else:
    #    user_steps = None

    # coord_key = str(input_dict.get("coord_key", "COORD"))
    num_cpus = int(input_dict.get("cpus", multiprocessing.cpu_count()))
    # time_step = int(input_dict.get("time_step", 1))
    target_part = input_dict.get("part")
    target_nodeset = input_dict.get("nodeset")
    target_elementset = input_dict.get("elementset")
    target_outputs = input_dict.get("outputs")
    defaults_for_outputs = input_dict.get("defaults_for_outputs", {})

    # result_name = convert_odb_to_npz(
    #    odb_path,
    #    user_nodesets,
    #    user_nodes,
    #    user_elementsets,
    #    user_elements,
    #    user_parts,
    #    user_steps,
    #    coord_key,
    #    num_cpus,
    #    time_step,
    # )
    result_name = convert_odb_to_npz(
        odb_path,
        num_cpus,
        target_part,
        target_nodeset,
        target_elementset,
        target_outputs,
        defaults_for_outputs,
    )
    try:
        result_file = open(result_path, "wb")
        pickle.dump(result_name, result_file)

    except Exception as e:
        print("Error with pickling:")
        print(e)

    finally:
        result_file.close()


# def convert_odb_to_npz(
#    odb_path,
#    user_nodesets,
#    user_nodes,
#    user_elementsets,
#    user_elements,
#    user_parts,
#    user_steps,
#    coord_key,
#    num_cpus,
#    time_step,
# ):
def convert_odb_to_npz(
    odb_path,
    num_cpus,
    target_part,
    target_nodeset,
    target_elementset,
    target_outputs,
    defaults_for_outputs,
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

        base_times = []
        total_idx = 0
        for step_key, step_val in steps.items():
            base_times.append((step_key, step_val.totalTime, total_idx))
            total_idx += len(step_val.frames)

        max_idx = base_times[-1][2] + len(steps[base_times[-1][0]].frames) - 1

        if target_outputs is None:
            target_outputs = list(steps[steps.keys()[0]].frames[0].fieldOutputs.keys())

        if "COORD" in target_outputs:
            target_outputs.remove("COORD")

        assembly = odb.rootAssembly

        if target_part is None:
            target_part = list(assembly.instances.keys())[0]

        # print target_part, list(assembly.instances.keys())

        if target_nodeset is None:
            # TODO!!!
            target_nodeset = list(assembly.instances[target_part].nodeSets.keys())[0]
            # target_nodeset = list(assembly.nodeSets.keys())[0]

        target_nodeset_length = len(
            assembly.instances[target_part].nodeSets[target_nodeset].nodes
        )

        # print target_nodeset, list(assembly.nodeSets.keys()), len(assembly.instances[target_part].nodeSets[target_nodeset].nodes)

        if target_elementset is None:
            # TODO!!!
            target_elementset = list(
                assembly.instances[target_part].elementSets.keys()
            )[0]
            # target_elementset = list(assembly.elementSets.keys())[0]

        target_elementset_length = len(
            assembly.instances[target_part].elementSets[target_elementset].elements
        )
        # print target_elementset, list(assembly.elementSets.keys()), len(assembly.instances[target_part].elementSets[target_elementset].elements)#, len(assembly.instances[target_part].elementSets["ALL_ELEMENTS"].elements)

        target_frames = dict()
        # if user_steps is not None:
        #    for step in user_steps:
        #        step_data = steps[step]
        #        target_frames[step] = set()
        #        for frame in step_data.frames:
        #            target_frames[step].add(frame.frameId)

        #    for step in target_frames:
        #        target_frames[step] = sorted(list(target_frames[step]))

        # else:
        for step, step_data in steps.items():
            target_frames[step] = list()
            for frame in step_data.frames:
                target_frames[step].append(frame.frameId)

        # for step in target_frames:
        #    target_frames[step] = target_frames[step][::time_step]

        # target_nodesets = set()
        # if user_nodes is not None:
        #    if isinstance(user_nodes, collections.Mapping):
        #        for key, val in user_nodes.items():
        #            assembly.NodeSetFromNodeLabels(name=key, nodeLabels=(val,))
        #            target_nodesets.add(key)
        #    elif isinstance(user_nodes, collections.Iterable):
        #        if isinstance(user_nodes[0], int):
        #            user_nodes = [user_nodes]
        #        for i, val in enumerate(user_nodes):
        #            assembly.NodeSetFromNodeLabels(name=i, nodeLabels=(val,))
        #            target_nodesets.add(i)

        # if user_parts is not None:
        #    for part in assembly.instances.keys():
        #        for nodeset in assembly.instances[part].nodeSets:
        #            target_nodesets.add(nodeset)

        # if user_nodesets is not None:
        #    for nodeset in user_nodesets:
        #        target_nodesets.add(nodeset)

        # target_nodesets = sorted(list(target_nodesets))

        # if len(target_nodesets) == 0:
        #    target_nodesets = [list(assembly.nodeSets.keys())[0]] # If none is specified, grab the first

        # target_elementsets = set()
        # if user_elements is not None:
        #    if isinstance(user_elements, collections.Mapping):
        #        for key, val in user_elements.items():
        #            assembly.ElementSetFromElementLabels(name=key, elementLabels=(val,))
        #            target_elementsets.add(key)
        #    elif isinstance(user_elements, collections.Iterable):
        #        if isinstance(user_elements[0], int):
        #            user_elements = [user_elements]
        #        for i, val in enumerate(user_elements):
        #            assembly.ElementSetFromElementLabels(name=i, elementLabels=(val,))
        #            target_elementsets.add(i)

        # if user_parts is not None:
        #    for part in assembly.instances.keys():
        #        for elementset in assembly.instances[part].elementSets:
        #            target_elementsets.add(elementset)

        # if user_elementsets is not None:
        #    for elementset in user_elementsets:
        #        target_elementsets.add(elementset)

        # target_elementsets = sorted(list(target_elementsets))

        # if len(target_elementsets) == 0:
        #    target_elementsets = [
        #        list(assembly.elementSets.keys())[0]
        #    ]  # If none is specified, grab the first

        # all_outputs = list(steps[steps.keys()[0]].frames[0].fieldOutputs.keys())
        # if coord_key not in all_outputs:
        #    raise ValueError(
        #        "Coordinate Key {} was no found in this .odb file".format(coord_key)
        #    )

        # first_step = list(steps.values())[0]
        # first_frame = first_step.frames[0]

        # if len(target_nodesets) > 0:
        #    first_nodeset = target_nodesets[0]
        #    first_nodeset_field = first_frame.fieldOutputs[coord_key].getSubset(
        #        region=assembly.nodeSets[first_nodeset]
        #    )
        #    nodeset_coords = len(first_nodeset_field.values) != 0
        # else:
        #    nodeset_coords = False

        # if len(target_elementsets) > 0:
        #    first_elset = target_elementsets[0]
        #    first_elset_field = first_frame.fieldOutputs[coord_key].getSubset(
        #        region=assembly.elementSets[first_elset]
        #    )
        #    elset_coords = len(first_elset_field.values) != 0
        # else:
        #    elset_coords = False

        # if not nodeset_coords and not elset_coords:
        #    print(
        #        "ERROR! {} not mapped to either Nodesets or Elementsets".format(
        #            coord_key
        #        )
        #    )
        #    sys.exit(-1)

        # TODO output these
        part_to_elementsets_mapping = {}
        elementsets_to_elements_mapping = {}
        part_to_nodesets_mapping = {}
        nodesets_to_nodes_mapping = {}
        node_coords = []
        element_connectivity = []

        connectivity_elementset = ("", "", 0)
        coordinate_nodeset = ("", "", 0)
        for part in assembly.instances.keys():
            for elementset in assembly.instances[part].elementSets.keys():
                if (
                    len(assembly.instances[part].elementSets[elementset].elements)
                    > connectivity_elementset[2]
                ):
                    connectivity_elementset = (
                        part,
                        elementset,
                        len(assembly.instances[part].elementSets[elementset].elements),
                    )

            for nodeset in assembly.instances[part].nodeSets.keys():
                if (
                    len(assembly.instances[part].nodeSets[nodeset].nodes)
                    > coordinate_nodeset[2]
                ):
                    coordinate_nodeset = (
                        part,
                        nodeset,
                        len(assembly.instances[part].nodeSets[nodeset].nodes),
                    )

        for part in assembly.instances.keys():
            part_to_elementsets_mapping[part] = []
            part_to_nodesets_mapping[part] = []

            for elementset in assembly.instances[part].elementSets.keys():
                part_to_elementsets_mapping[part].append(elementset)
                elementsets_to_elements_mapping[elementset] = []
                if (
                    part == connectivity_elementset[0]
                    and elementset == connectivity_elementset[1]
                ):
                    for element in (
                        assembly.instances[part].elementSets[elementset].elements
                    ):
                        elementsets_to_elements_mapping[elementset].append(
                            element.label
                        )
                        element_connectivity.append(element.connectivity)
                else:
                    for element in (
                        assembly.instances[part].elementSets[elementset].elements
                    ):
                        elementsets_to_elements_mapping[elementset].append(
                            element.label
                        )

            for nodeset in assembly.instances[part].nodeSets.keys():
                part_to_nodesets_mapping[part].append(nodeset)
                nodesets_to_nodes_mapping[nodeset] = []
                if part == coordinate_nodeset[0] and nodeset == coordinate_nodeset[1]:
                    for node in assembly.instances[part].nodeSets[nodeset].nodes:
                        nodesets_to_nodes_mapping[nodeset].append(node.label)
                        node_coords.append(node.coordinates)
                else:
                    for node in assembly.instances[part].nodeSets[nodeset].nodes:
                        nodesets_to_nodes_mapping[nodeset].append(node.label)

        output_python_values = [
            ("part_to_elementsets_mapping", part_to_elementsets_mapping),
            ("elementsets_to_elements_mapping", elementsets_to_elements_mapping),
            ("part_to_nodesets_mapping", part_to_nodesets_mapping),
            ("nodesets_to_nodes_mapping", nodesets_to_nodes_mapping),
        ]
        for name, output in output_python_values:
            try:
                result_file = open(os.path.join(data_dir, name + ".pickle"), "wb")
                pickle.dump(output, result_file)

            except Exception as e:
                print("Error with pickling:")
                print(e)

            finally:
                result_file.close()

        output_numpy_values = [
            ("node_coords", node_coords),
            ("element_connectivity", element_connectivity),
        ]
        for name, output in output_numpy_values:
            # print name, len(output)
            np.savez_compressed(
                os.path.join(data_dir, name),
                np.array(output),
            )

    finally:
        odb.close()

    for step_key, base_time, base_idx in base_times:
        read_step_data(
            odb_path,
            data_dir,
            step_key,
            base_time,
            base_idx,
            max_idx,
            target_frames[step_key],
            num_cpus,
            target_nodeset,
            target_nodeset_length,
            target_elementset,
            target_elementset_length,
            target_outputs,
            target_part,
            defaults_for_outputs,
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
    num_cpus,
    target_nodeset,
    target_nodeset_length,
    target_elementset,
    target_elementset_length,
    target_outputs,
    target_part,
    defaults_for_outputs,
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
                        target_nodeset,
                        target_nodeset_length,
                        target_elementset,
                        target_elementset_length,
                        target_outputs,
                        target_part,
                        defaults_for_outputs,
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
    target_nodeset,
    target_nodeset_length,
    target_elementset,
    target_elementset_length,
    target_outputs,
    target_part,
    defaults_for_outputs,
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

            nodal_elemental_format = [
                (target_nodeset, "Nodal", "nodeSets", target_nodeset_length, "nodes"),
                (
                    target_elementset,
                    "Elemental",
                    "elementSets",
                    target_elementset_length,
                    "elements",
                ),
            ]

            for (
                target_set,
                label,
                set_attr,
                target_length,
                val_list,
            ) in nodal_elemental_format:
                for output in target_outputs:
                    field = frame.fieldOutputs[output].getSubset(
                        region=getattr(assembly.instances[target_part], set_attr)[
                            target_set
                        ]
                    )

                    output_defined_only_on_subset = len(field.values) not in (
                        target_length,
                        0,
                    )
                    default = None
                    defined_set = None
                    if output_defined_only_on_subset:
                        defined_set = []
                        for val in field.values:
                            defined_set.append(
                                getattr(val, "{}Label".format(set_attr[:-4]))
                            )
                        if output not in defaults_for_outputs:
                            raise Exception(
                                "{} is not defined on all {}. The values are of type {}. Please set a default value for the undefined elements.".format(
                                    output, val_list, type(field.values[0].data)
                                )
                            )

                        default = defaults_for_outputs[output]

                    vals = []
                    if len(field.componentLabels) > 0:
                        if output_defined_only_on_subset:
                            field_val_ind = 0
                            for i in range(1, target_length + 1):
                                if i in defined_set:
                                    val = field.values[field_val_ind]
                                    val_dict = {}
                                    for item, lab in zip(
                                        val.data, field.componentLabels
                                    ):
                                        val_dict[lab] = item

                                    vals.append(val_dict)
                                    field_val_ind += 1

                                else:
                                    vals.append(
                                        {lab: default for lab in field.componentLabels}
                                    )

                        else:
                            for val in field.values:
                                val_dict = {}
                                for item, lab in zip(val.data, field.componentLabels):
                                    val_dict[lab] = item

                                vals.append(val_dict)

                    else:
                        if output_defined_only_on_subset:
                            field_val_ind = 0
                            for i in range(1, target_length + 1):
                                if i in defined_set:
                                    val = field.values[field_val_ind]
                                    vals.append(val.data)
                                    field_val_ind += 1
                                else:
                                    vals.append(default)

                        else:
                            for item in field.values:
                                val = item.data
                                vals.append(val)

                    if len(vals) > 0:
                        if type(vals[0]) is dict:
                            reordered_dict = {}
                            for k in vals[0].keys():
                                reordered_dict[k] = []

                            for output_dict in vals:
                                for k, v in output_dict.items():
                                    reordered_dict[k].append(v)

                            for k, v in reordered_dict.items():
                                np.savez_compressed(
                                    os.path.join(
                                        curr_step_dir,
                                        "{}_{}_{}_{}".format(label, output, k, num),
                                    ),
                                    np.array(v),
                                )
                        else:
                            np.savez_compressed(
                                os.path.join(
                                    curr_step_dir, "{}_{}_{}".format(label, output, num)
                                ),
                                np.array(vals),
                            )

                np.savez_compressed(
                    # Technically duplicated, but it makes restructuring easier
                    os.path.join(curr_step_dir, "{}_Time_{}".format(label, num)),
                    np.array([round(frame.frameValue + base_time, 5)]),
                )

    finally:
        odb.close()


if __name__ == "__main__":
    main()

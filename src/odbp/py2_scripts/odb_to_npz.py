# Author: CJ Nguyen
# Heavily based off of example ODB extraction script from the CMML Github written by Will Furr and Matthew Dantin
# This script takes an ODB and converts the nodeset data to npz files
# The data that is created is the following
# * npzs containing temperatures for each node at every frame, one dataframe per step of the program
# * npz containing coordinate data for each node, organized by nodelabel - xyz coordinate
# * npz containing the starting time of each frame

# Usage: python <script name> <odb file name> <inp file name>
# Where <inp file name> is the .inp file used to generate the ODB (needed for the timing of each frame)
# NOTE: This script makes three major assumptions:
# * The Odb has a part named "PART-1-1"
# * The part's first nodeset is the nodeset that references all nodes
# * The first frame of the sequence outputs the coordinates of all nodes
# Without these assumptions, the script will have unexpected behavior.

# This script can be configured by having a file of name `odb_to_npz_config.json` in the working directory.
# The format of this file is specified in the readme.

import os
from odbAccess import openOdb
from abaqusConstants import *
import numpy as np
import argparse
from multiprocessing import Process


def main(odb_path, frame_step):

    parent_dir = os.path.join(os.getcwd(), "tmp_npz")
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    #for part in parts:
    # Create directory to store npzs
    coord_file = os.path.join(parent_dir, "node_coords.npz")

    temps_dir = os.path.join(parent_dir, "temps")
    if not os.path.exists(temps_dir):
        os.mkdir(temps_dir)

    time_dir = os.path.join(parent_dir, "step_frame_times")
    if not os.path.exists(time_dir):
        os.mkdir(time_dir)

    # Create output directory for nodesets
    #nodeset_dir = os.path.join(parent_dir, "nodesets")
    #if not os.path.exists(nodeset_dir):
    #    os.mkdir(nodeset_dir)

    # Loop through all the nodesets and get the nodes that they cover
    ####
    # Get the list of nodeset keys to run the access
    ####

    odb = openOdb(odb_path, readOnly=True)
    #assembly = odb.rootAssembly
    steps = odb.steps

    base_times = [(step_key, step_val.totalTime) for step_key, step_val in steps.items()]

    odb.close()
    
    #if parts is not None:
    #    for part in parts:
    #        part_nodeset_dir = os.path.join(nodeset_dir, part)
    #        if not os.path.exists(part_nodeset_dir):
    #            os.mkdir(part_nodeset_dir)
    #
    #        nodesets = assembly.instances[part].nodeSets
    #        nodeset_keys = nodesets.keys()
    #        read_item_procs = list()
    #        for key in nodeset_keys:
    #            p = Process(target=read_nodeset_items, args=(part_nodeset_dir, key, filename, part))
    #            p.start()
    #            read_item_procs.append(p)

    #        for p in read_item_procs:
    #            p.join()

    #else:
    #    pass

    read_coords_procs = list()
    for step_key, _ in base_times:
        p = Process(target=read_frame_coords, args=(step_key, filename, frame_step, coord_file))
        p.start()
        read_coords_procs.append(p)

    for p in read_coords_procs:
        p.join()


    read_step_procs = list()
    for step_key, base_time in base_times:
        p = Process(target=read_step_data, args=(temps_dir, time_dir, step_key, base_time, frame_step, filename))
        p.start()
        read_step_procs.append(p)

    for p in read_step_procs:
        p.join()


def read_nodeset_items(nodeset_dir, nodeset_name, filename, part):
    odb = openOdb(filename, readOnly=True)
    assembly = odb.rootAssembly
    nodesets = assembly.instances[part].nodeSets # returns a dictionary of ODB objects
    out_nodeset_name = os.path.join(nodeset_dir, nodeset_name)
    out_nodeset_name += ".npz"
    np.savez_compressed(out_nodeset_name, np.array([node.label for node in nodesets[nodeset_name].nodes]))
    odb.close()


def read_step_data(temps_dir, time_dir, step_name, base_time, frame_step, filename):

    odb = openOdb(filename, readOnly=True)
    steps = odb.steps
    assembly = odb.rootAssembly

    curr_step_dir = os.path.join(temps_dir, step_name)
    if not os.path.exists(curr_step_dir):
        os.mkdir(curr_step_dir)

    frame_times = list()
    step_size = len(steps[step_name].frames)
    for num in range(0, step_size, frame_step):
        frame_pct = ((num + 1) * 100) / step_size
        print("\tGetting node temperatures for frame {0}, {1}% Complete".format(num, frame_pct))
        frame = steps[step_name].frames[num]
        field = frame.fieldOutputs['NT11'].getSubset(region=assembly.nodeSets["ALL NODES"])
        frame_times.append(float(format(round(frame.frameValue + base_time, 5), ".2f")))
        node_temps = list()
        for item in field.values:
            # e.g. for node in values
            temp = item.data
            node_temps.append(temp)
        np.savez_compressed(os.path.join(curr_step_dir, "frame_{}".format(num)), np.array(node_temps))
    np.savez_compressed("{}.npz".format(os.path.join(time_dir, step_name)), np.array(frame_times))

    odb.close()


def read_frame_coords(step_key, filename, frame_step, coord_file):
    odb = openOdb(filename, readOnly=True)
    steps = odb.steps
    assembly = odb.rootAssembly
    #nodesets = assembly.instances[config["first_part"]].nodeSets # returns a dictionary of ODB objects
    frame_list = steps[step_key].frames
    frame_size = len(frame_list)
    for num in range(0, frame_size, frame_step):
        frame = frame_list[num]
        # The below reference pulls from the nodeset representing all nodes
        coords = frame.fieldOutputs['COORD'].getSubset(region=assembly.nodeSets["ALL NODES"])
        frame_pct = ((num + 1) * 100) / frame_size
        print("\tGetting node coordinates for frame {0}, {1}% Complete".format(num, frame_pct))

        coord_arr = list()
        for item in coords.values:
            node = item.nodeLabel
            coord = item.data
            xyz = [node]
            for axis in coord:
                xyz.append(axis)
            coord_arr.append(xyz)
        np.savez_compressed(coord_file, np.array(coord_arr))

    odb.close()


if __name__ == "__main__":
    input_args = "input args"
    parser = argparse.ArgumentParser()
    parser.add_argument(input_args, nargs="*")
    filename, frame_step = vars(parser.parse_args())[input_args]
    frame_step = int(frame_step)

    main(filename, frame_step)

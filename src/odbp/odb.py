#!/usr/bin/env python3

"""
odb_plotter base classes. used for storing the state of a .odb file in the
interactive mode
"""

import os
import tomli_w
import subprocess
import shutil

import numpy as np
import pandas as pd

from typing import TextIO, Any, Union
from multiprocessing import Pool

# Needed only for typing, circular import
#from .state import UserOptions
from .npz_to_hdf import npz_to_hdf
from .read_hdf5 import get_coords, read_node_data


class Axis:
    """
    Named Axis class, stores extrema, discrete values along the axis
    Not to be used on its own, only a part of the Odb class
    """
    def __init__(self, name: str) -> None:
        """
        __init__(self, name: str) -> None
        
        """
        self.name: str = name

        # Setting default values
        self.low: float
        self.high: float

class Odb:
    """
    Stores Data from a .hdf5, implements extractor methods to transfer from .odb to .hdf5
    Implements abilities to resize the dimenisons or timeframe of the data
    """
    def __init__(self) -> None:
        """
        """

        self.odb_file_path: str
        self.hdf_file_path: str

        self.parts: list[str]
        self.nodes: dict[str, list[int]]
        self.nodesets: list[str]

        self.x: Axis = Axis("x")
        self.y: Axis = Axis("y")
        self.z: Axis = Axis("z")

        self.time_low: float = 0.0
        self.time_high: float

        self.meltpoint: float
        self.low_temp: float

        self.time_sample: int
        self.mesh_seed_size: float

        self.abaqus_program: str

        self.hdf_processed: bool = False
        self.loaded: bool = False

        self.bounded_nodes: Any

        self.out_nodes: Any
        self.out_nodes_low_time: Any


    def set_mesh_seed_size(self, seed: float) -> None:
        if not isinstance(seed, float):
            seed = float(seed)
        self.mesh_seed_size: float = seed


    def set_meltpoint(self, meltpoint: float) -> None:
        if not isinstance(meltpoint, float):
            meltpoint = float(meltpoint)
        self.meltpoint = meltpoint


    def set_low_temp(self, low_temp: float) -> None:
        if not isinstance(low_temp, float):
            low_temp = float(low_temp)
        self.low_temp = low_temp


    def set_x_high(self, high: float) -> None:
        if not isinstance(high, float):
            high = float(high)
        self.x.high = high


    def set_x_low(self, low: float) -> None:
        if not isinstance(low, float):
            low = float(low)
        self.x.low = low


    def set_y_high(self, high: float) -> None:
        if not isinstance(high, float):
            high = float(high)
        self.y.high = high


    def set_y_low(self, low: float) -> None:
        if not isinstance(low, float):
            low = float(low)
        self.y.low = low


    def set_z_high(self, high: float) -> None:
        if not isinstance(high, float):
            high = float(high)
        self.z.high = high


    def set_z_low(self, low: float) -> None:
        if not isinstance(low, float):
            low = float(low)
        self.z.low = low


    def set_time_high(self, high: float) -> None:
        if not isinstance(high, float):
            high = float(high)
        self.time_high = high


    def set_time_low(self, low: float) -> None:
        if not isinstance(low, float):
            low = float(low)
        if low < 0:
            raise ValueError("Lower Time Bound must not be less than 0")
        self.time_low = low


    def set_time_sample(self, value: int) -> None:
        if not isinstance(value, int):
            value = int(value)
        if value < 1:
            raise ValueError("Time Sample must be an integer greater than or equal to 1")
        self.time_sample = value


    def set_parts(self, parts: "list[str]") -> None:
        if not isinstance(parts, list):
            new_list: list[str] = [parts]
            self.parts = new_list

        else:
            self.parts = parts

    
    def set_nodes(self, nodes: "dict[str, list[int]]") -> None:
        if not isinstance(nodes, dict):
            nodes = dict(nodes)

        self.nodes = nodes

    
    def set_nodesets(self, nodesets: "list[str]") -> None:
        if not isinstance(nodesets, list):
            new_list: list[str] = [nodesets]
            self.nodesets = new_list

        else:
            self.nodesets = nodesets


    def set_abaqus(self, abq: str) -> None:
        self.abaqus_program = abq


    def select_odb(self, user_options: Any, given_odb_file_path: str) -> "Union[None, bool]":
        odb_file_path: str
        if not os.path.exists(os.path.join(user_options.odb_source_directory, given_odb_file_path)):
            if not os.path.exists(os.path.join(os.getcwd(), given_odb_file_path)):
                if not os.path.exists(given_odb_file_path):
                    return False

                else:
                    odb_file_path = given_odb_file_path
            else:
                odb_file_path = os.path.join(os.getcwd(), given_odb_file_path)
        else:
            odb_file_path = os.path.join(user_options.odb_source_directory, given_odb_file_path)

        self.odb_file_path = odb_file_path


    def select_hdf(self, user_options: Any, given_hdf_file_path: str) -> "Union[Any, bool]":
        hdf_file_path: str
        if not os.path.exists(os.path.join(user_options.hdf_source_directory, given_hdf_file_path)):
            if not os.path.exists(os.path.join(os.getcwd(), given_hdf_file_path)):
                if not os.path.exists(given_hdf_file_path):
                        return False

                else:
                    hdf_file_path = given_hdf_file_path
            else:
                hdf_file_path = os.path.join(os.getcwd(), given_hdf_file_path)
        else:
            hdf_file_path = os.path.join(user_options.hdf_source_directory, given_hdf_file_path)

        self.hdf_file_path = hdf_file_path

        config_file_path: str = self.hdf_file_path.split(".")[0] + ".toml"
        if os.path.exists(config_file_path):
            user_options.config_file_path = config_file_path

        else:
            user_options.config_file_path = None

        self.hdf_processed = True
        return user_options


    def _post_process_data(self) -> None:
        """
        "Private" method. Not to be used on its own, but called with process_hdf
        """
        self.out_nodes_low_time = self.out_nodes[self.out_nodes["Time"] == self.time_low]

        self.loaded = True


    def odb_to_hdf(self, hdf_file_path: str) -> None:
        
        assert self.odb_file_path != ""
        # Must run this script via abaqus python
        odb_to_npz_script_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "py2_scripts", "odb_to_npz.py")

        if len(self.nodesets) != 1:
            raise ValueError("You must have exactly one Nodeset specified to convert a .odb to a .hdf5")
        odb_to_npz_args: list[str]  = [self.abaqus_program, "python", odb_to_npz_script_path, self.odb_file_path, str(self.time_sample), str(self.nodesets[0])]
        subprocess.run(odb_to_npz_args, shell=True)

        npz_dir: str = os.path.join(os.getcwd(), "tmp_npz")
        npz_to_hdf(hdf_file_path, npz_dir)
        
        if os.path.exists(npz_dir):
            shutil.rmtree(npz_dir)

        self.hdf_processed = True
        self.hdf_file_path = hdf_file_path


    def dump_config_to_toml(self, toml_path: str) -> None:
        config: dict[str, Union[float, str]] = dict()
        if hasattr(self, "hdf_file_path"):
            config["hdf_file_path"] = self.hdf_file_path
        if hasattr(self, "mesh_seed_size"):
            config["mesh_seed_size"] = self.mesh_seed_size
        if hasattr(self, "meltpoint"):
            config["meltpoint"] = self.meltpoint
        if hasattr(self, "low_temp"):
            config["low_temp"] = self.low_temp
        if hasattr(self, "time_sample"):
            config["time_sample"] = self.time_sample

        toml_file: TextIO
        with open(toml_path, "wb") as toml_file:
            tomli_w.dump(config, toml_file)


    def process_hdf(self) -> None:

        # Ensure that all 6 dimension extrema are set
        #if not (hasattr(self.x, "low") and hasattr(self.x, "high") and hasattr(self.y, "low") and hasattr(self.y, "high") and hasattr(self.z, "low") and hasattr(self.z, "high")):
        #    raise RangeNotSetError

        ## Ensure that both time boundaries are set
        #if not (hasattr(self, "time_low") and hasattr(self, "time_high")):
        #    raise TimeNotSetError

        ## Ensure the mesh seed size is set
        #if not hasattr(self, "mesh_seed_size"):
        #    raise SeedNotSetError
        
        ## Ensure the hdf file is set
        #if not hasattr(self, "hdf_file_path"):
        #    raise HDFNotSetError

        # Adapted from CJ's read_hdf5.py
        coords_df: Any = get_coords(self.hdf_file_path)
        self.bounded_nodes = list(
            coords_df[
                (
                    ((coords_df["X"] == self.x.high) | (coords_df["X"] == self.x.low))
                    & (
                        (coords_df["Y"] >= self.y.low)
                        & (coords_df["Y"] <= self.y.high)
                        & (coords_df["Z"] >= self.z.low)
                        & (coords_df["Z"] <= self.z.high)
                    )
                )
                | (
                    ((coords_df["Y"] == self.y.high) | (coords_df["Y"] == self.y.low))
                    & (
                        (coords_df["X"] >= self.x.low)
                        & (coords_df["X"] <= self.x.high)
                        & (coords_df["Z"] >= self.z.low)
                        & (coords_df["Z"] <= self.z.high)
                    )
                )
                | (
                    ((coords_df["Z"] == self.z.high) | (coords_df["Z"] == self.z.low))
                    & (
                        (coords_df["X"] >= self.x.low)
                        & (coords_df["X"] <= self.x.high)
                        & (coords_df["Y"] >= self.y.low)
                        & (coords_df["Y"] <= self.y.high)
                    )
                )
            ]["Node Labels"]
        )

        pool: Any
        with Pool() as pool:
            # TODO can imap be used? starred imap?
            data: list[tuple[str, int, int]] = list()
            node: int
            for node in self.bounded_nodes:
                data.append((self.hdf_file_path, node, self.time_sample))
            results: Any = pool.starmap(read_node_data, data)

        self.out_nodes = pd.concat(results)
        self.out_nodes = self.out_nodes[(self.out_nodes["Time"] <= self.time_high) & (self.out_nodes["Time"] >= self.time_low)]

        self._post_process_data()

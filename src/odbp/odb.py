#!/usr/bin/env python3

"""
odb_plotter base classes. used for storing the state of a .odb file in the
interactive mode
"""

import os
import json
import tempfile
import subprocess
import shutil

import numpy as np
import pandas as pd

from typing import TextIO, Any
from multiprocessing import Pool

from .npz_to_hdf import npz_to_hdf
from .read_hdf5 import get_coords, read_node_data


# HDF Manger Exceptions
class RangeNotSetError(Exception):
    def __init__(self) -> None:
        self.message = "You must set the upper-and-lower bounds in the x, y, and z directions"
        super().__init__(self.message)


class TimeNotSetError(Exception):
    def __init__(self) -> None:
        self.message = "You must set the upper-and-lower bounds of time to extract. Lower time bound must not be less than zero. Pass float(\"inf\") or equivalent for upper time bound to get all times"
        super().__init__(self.message)


class SeedNotSetError(Exception):
    def __init__(self) -> None:
        self.message = "You must set the mesh-seed-size"
        super().__init__(self.message)


class HDFNotSetError(Exception):
    def __init__(self) -> None:
        self.message = "You must either provide the .hdf5 file you wish to extract from, or provide a corresponding .odb file and use the odb_to_hdf method"
        super().__init__(self.message)


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
        self.vals: Any
        self.size: int


class Odb:
    """
    Stores Data from a .hdf5, implements extractor methods to transfer from .odb to .hdf5
    Implements abilities to resize the dimenisons or timeframe of the data
    """
    def __init__(self, **kwargs) -> None:
        """
        __init__(self) -> None
        key-word arguments:
        odb_file (str): .odb file to extract from
        hdf_file (str): .hdf5 file from which to read or to which to write extracted data from a .odb file
        x_low (float): lower x_axis value to process
        x_high (float): upper x_axis value to process
        y_low (float): lower y_axis value to process
        y_high (float): upper y_axis value to process
        z_low (float): lower z_axis value to process
        z_high (float): upper z_axis value to process
        time_low (float): lower time value to process (Default 0)
        time_high (float): upper time value to process
        meltpoint (float): melting point of the sample
        time_sample (int): N for "extract from every Nth frame" (Default 1)
        abaqus_program (str): name of the version of abaqus (or path to that executable if it is not on your path) (Default "abaqus")
        """

        # Relative to package's data, not user's cwd, in order to access things like the abaqus python script or views.py
        self.cwd: str = os.getcwd()

        self.odb_file: str = kwargs.get("odb_file", "")
        self.hdf_file: str = kwargs.get("hdf_file", "")

        self.x: Axis = Axis("x")
        if "x_low" in kwargs:
            self.x.low = kwargs["x_low"]
        if "x_high" in kwargs:
            self.x.high = kwargs["x_high"]

        self.y: Axis = Axis("y")
        if "y_low" in kwargs:
            self.x.low = kwargs["y_low"]
        if "y_high" in kwargs:
            self.x.high = kwargs["y_high"]

        self.z: Axis = Axis("z")
        if "z_low" in kwargs:
            self.x.low = kwargs["z_low"]
        if "z_high" in kwargs:
            self.x.high = kwargs["z_high"]

        self.time_low: float = kwargs.get("time_low", 0)

        self.time_high: float
        if "time_high" in kwargs:
            self.time_low = kwargs["time_high"]

        self.mesh_seed_size: float
        if "mesh_seed_size" in kwargs:
            self.time_low = kwargs["mesh_seed_size"]

        self.show_plots: bool = kwargs.get("show_plots", True)

        self.meltpoint: float
        if "meltpoint" in kwargs:
            self.meltpoint = kwargs["meltpoint"]

        self.time_sample: int = kwargs.get("time_sample", 1)

        self.abaqus_program = kwargs.get("abaqus_program", "abaqus")

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


    def _post_process_data(self) -> None:
        """
        "Private" method. Not to be used on its own, but called with process_hdf
        """
        self.out_nodes_low_time = self.out_nodes[self.out_nodes["Time"] == self.time_low]
        temp_x_list: list[float] = list()
        temp_y_list: list[float] = list()
        temp_z_list: list[float] = list()
        for _, node in self.out_nodes_low_time.iterrows():
            x: float = round(node["X"], 5)
            y: float = round(node["Y"], 5)
            z: float = round(node["Z"], 5)
            if (x % self.mesh_seed_size == 0) and (y % self.mesh_seed_size == 0) and (z % self.mesh_seed_size == 0):
                temp_x_list.append(x)
                temp_y_list.append(y)
                temp_z_list.append(z)

        # Makes these in-order lists of unique values
        temp_x_list = list(dict.fromkeys(temp_x_list))
        temp_y_list = list(dict.fromkeys(temp_y_list))
        temp_z_list = list(dict.fromkeys(temp_z_list))

        temp_x_list.sort()
        temp_y_list.sort()
        temp_z_list.sort()

        self.x.vals = np.asarray(temp_x_list)
        self.y.vals = np.asarray(temp_y_list)
        self.z.vals = np.asarray(temp_z_list)

        self.x.size = len(self.x.vals)
        self.y.size = len(self.y.vals)
        self.z.size = len(self.z.vals)

        self.loaded = True


    def odb_to_hdf(self, hdf_file_path: str) -> None:
        
        assert self.odb_file != ""
        # Must run this script via abaqus python
        odb_to_npz_script_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "odb_to_npz.py")
        # Move to tempdir for this operation
        os.chdir(tempfile.gettempdir())

        odb_to_npz_args: list[str] = [self.abaqus_program, "python", odb_to_npz_script_path, self.odb_file, str(self.time_sample)]
        subprocess.run(odb_to_npz_args)

        # Back to pwd
        os.chdir(self.cwd)

        npz_dir: str = os.path.join(tempfile.gettempdir(), "tmp_npz")

        # Convert npz to hdf5
        npz_to_hdf(hdf_file_path, npz_dir)
        
        if os.path.exists(npz_dir):
            shutil.rmtree(npz_dir)

        self.hdf_file = hdf_file_path


    def dump_config_to_json(self, json_path: str) -> None:
        config = dict()
        if hasattr(self, "hdf_file"):
            config["hdf_file"] = self.hdf_file
        if hasattr(self, "mesh_seed_size"):
            config["mesh_seed_size"] = self.mesh_seed_size
        if hasattr(self, "meltpoint"):
            config["meltpoint"] = self.meltpoint
        if hasattr(self, "time_sample"):
            config["time_sample"] = self.time_sample

        json_file: TextIO
        with open(json_path, "w") as json_file:
            json.dump(config, json_file)

    
    def process_hdf(self) -> None:

        # Ensure that all 6 dimension extrema are set
        if not (hasattr(self.x, "low") and hasattr(self.x, "high") and hasattr(self.y, "low") and hasattr(self.y, "high") and hasattr(self.z, "low") and hasattr(self.z, "high")):
            raise RangeNotSetError

        # Ensure that both time boundaries are set
        if not (hasattr(self, "time_low") and hasattr(self, "time_high")):
            raise TimeNotSetError

        # Ensure the mesh seed size is set
        if not hasattr(self, "mesh_seed_size"):
            raise SeedNotSetError
        
        # Ensure the hdf file is set
        if not hasattr(self, "hdf_file"):
            raise HDFNotSetError

        # Adapted from CJ's read_hdf5.py
        coords_df: Any = get_coords(self.hdf_file)
        self.bounded_nodes = list(
                coords_df[
                    (((coords_df["X"] == self.x.high) | (coords_df["X"] == self.x.low)) & ((coords_df["Y"] >= self.y.low) & (coords_df["Y"] <= self.y.high) & (coords_df["Z"] >= self.z.low) & (coords_df["Z"] <= self.z.high))) |
                    (((coords_df["Y"] == self.y.high) | (coords_df["Y"] == self.y.low)) & ((coords_df["X"] >= self.x.low) & (coords_df["X"] <= self.x.high) & (coords_df["Z"] >= self.z.low) & (coords_df["Z"] <= self.z.high))) |
                    (((coords_df["Z"] == self.z.high) | (coords_df["Z"] == self.z.low)) & ((coords_df["X"] >= self.x.low) & (coords_df["X"] <= self.x.high) & (coords_df["Y"] >= self.y.low) & (coords_df["Y"] <= self.y.high)))
                    ]
                    ["Node Labels"]
                )

        pool: Any
        with Pool() as pool:
            # TODO can imap be used? starred imap?
            data: list[tuple[str, int, int]] = list()
            node: int
            for node in self.bounded_nodes:
                data.append((self.hdf_file, node, self.time_sample))
            results: Any = pool.starmap(read_node_data, data)

        self.out_nodes = pd.concat(results)
        self.out_nodes = self.out_nodes[(self.out_nodes["Time"] <= self.time_high) & (self.out_nodes["Time"] >= self.time_low)]

        self._post_process_data()

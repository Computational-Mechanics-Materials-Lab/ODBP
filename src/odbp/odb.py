#!/usr/bin/env python3

"""
ODBPlotter base_odb.py
ODBPlotter
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBPlotter
MIT License (c) 2023

Exposes the 
"""

import subprocess
import shutil
import pathlib
import pickle

import numpy as np
import pandas as pd

from typing import TextIO, Any, Union
from multiprocessing import Pool
from os import PathLike

from .base_odb import SpatialODB, ThermalODB, TemporalODB
from .npz_to_hdf import convert_npz_to_hdf
from .read_hdf5 import get_node_coords, get_node_times_temps
from .util import NullableIntListUnion, NullableStrListUnion, DataFrameType


class Odb(SpatialODB, ThermalODB, TemporalODB):
    """
    Stores Data from a .hdf5, implements extractor methods to transfer from .odb to .hdf5
    Implements abilities to resize the dimenisons or timeframe of the data
    """

    __slots__ = (
        "_odb_path",
        "_odb_source_dir",
        "_hdf_path",
        "_hdf_source_dir",
        "_abaqus_executable",
        "_nodesets",
        "_frames",
        "_odb_to_npz_script_path",
        "_odb_to_npz_conversion_pickle_path",
        "_npz_result_path"
        )

    def __init__(self) -> None:
        """
        """

        self._odb_path: PathLike
        self._odb_source_dir: Union[PathLike, None]

        self._hdf_path: PathLike
        self._hdf_source_dir: Union[PathLike, None]

        self._abaqus_executable: str = "abaqus"

        self._nodesets: NullableStrListUnion = None
        self._frames: NullableIntListUnion = None

        # Hardcoded paths for Python 3 - 2 communication
        self._odb_to_npz_script_path: PathLike = pathlib.Path(
            pathlib.Path(__file__).parent,
            "py2_scripts",
            "odb_to_npz.py"
        )
        self._odb_to_npz_conversion_pickle_path: PathLike = pathlib.Path(
            pathlib.Path.cwd(),
            "odb_to_npz_conversion.pickle"
        )
        self._npz_result_path: PathLike = pathlib.Path(
            pathlib.Path.cwd(),
            "npz_path.pickle"
        )

        # TODO
        """self._parts: NullableStrListUnion
        self._nodes: dict[str, list[int]]
        
        self._frame_sample: int

        self._hdf_processed: bool = False
        self._loaded: bool = False

        self.bounded_nodes: Any

        self.out_nodes: Any
        self.out_nodes_low_time: Any"""


    @property
    def odb_source_dir(self) -> Union[PathLike, None]:
        return self._odb_source_dir

    @odb_source_dir.setter
    def odb_source_dir(self, value: PathLike) -> None:
        if not isinstance(value, PathLike):
            value = pathlib.Path(value)

        if not value.exists():
            raise FileNotFoundError(f"Directory {value} does not exist")
        
        self._odb_source_dir = value

    @property
    def odb_path(self) -> PathLike:
        return self._odb_path

    
    @odb_path.setter
    def odb_path(self, value: PathLike) -> None:
        if not isinstance(value, PathLike):
            value = pathlib.Path(value)

        if value.exists():
            self._odb_path = value
            return

        if self._odb_source_dir is not None:
            if (self._odb_source_dir / value).exists():
                self._odb_path = self.odb_source_dir / value
                return

        if (pathlib.Path.cwd() / value).exists():
            self._odb_path = pathlib.Path.cwd() / value
            return

        raise FileNotFoundError(f"File {value} could not be found")

    @property
    def hdf_source_dir(self) -> Union[PathLike, None]:
        return self._hdf_source_dir

    @hdf_source_dir.setter
    def hdf_source_dir(self, value: PathLike) -> None:
        if not isinstance(value, PathLike):
            value = pathlib.Path(value)

        if not value.exists():
            raise FileNotFoundError(f"Directory {value} does not exist")
        
        self._hdf_source_dir = value

    @property
    def hdf_path(self) -> PathLike:
        return self._hdf_path

    
    @hdf_path.setter
    def hdf_path(self, value: PathLike) -> None:
        if not isinstance(value, PathLike):
            value = pathlib.Path(value)

        if value.exists():
            self._hdf_path = value
            return

        if self._hdf_source_dir is not None:
            if (self._hdf_source_dir / value).exists():
                self._hdf_path = self.hdf_source_dir / value
                return

        if (pathlib.Path.cwd() / value).exists():
            self._hdf_path = pathlib.Path.cwd() / value
            return

        raise FileNotFoundError(f"File {value} could not be found")


    @property
    def abaqus_executable(self) -> str:
        return self._abaqus_executable
    

    @abaqus_executable.setter
    def abaqus_executable(self, value: str) -> None:
        self._abaqus_executable = value

    
    """def set_parts(self, parts: "list[str]") -> None:
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
            self.nodesets = nodesets"""


    def convert_odb_to_hdf(self, hdf_path: Union[PathLike, None]) -> None:
        
        if not hasattr(self, "odb_path"):
            raise AttributeError("Path to target .odb file is not set")

        # If an hdf_path is passed in, update it on the user
        if hdf_path is not None:
            self.hdf_path = hdf_path

        odb_to_npz_pickle_input_dict: dict[
            str,Union[list[str], list[int], None]
            ] = {
                "nodesets": self._nodesets,
                "frames": self._frames
            }
        pickle_file: TextIO
        with open(
            self._odb_to_npz_conversion_pickle_path, "wb") as pickle_file:
            pickle.dump(odb_to_npz_pickle_input_dict, pickle_file, protocol=2)

        odb_convert_args: list[Union[PathLike, str]]  = [
            self._abaqus_executable,
            "python",
            self._odb_to_npz_script_path,
            self._odb_path,
            self._odb_to_npz_conversion_pickle_path,
            self._npz_result_path
        ]

        subprocess.run(odb_convert_args, shell=True)

        result_file: TextIO
        result_dir: PathLike
        with open(self._npz_result_path, "rb") as result_file:
            result_dir = pathlib.Path(pickle.load(result_file))
        
        pathlib.Path.unlink(self._npz_result_path)

        convert_npz_to_hdf(self._hdf_path, result_dir)
        
        if result_dir.exists():
            shutil.rmtree(result_dir)


    def align_hdf_to_dataframe(self) -> DataFrameType:

        # Adapted from CJ's read_hdf5.py
        coords_df: DataFrameType = get_node_coords(self._hdf__path)
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

        self.out_nodes_low_time = self.out_nodes[self.out_nodes["Time"] == self.time_low]

        self.loaded = True

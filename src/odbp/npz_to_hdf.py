#!/usr/bin/env python3

"""
ODBPlotter npz_to_hdf.py
ODBPlotter
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBPlotter
MIT License (c) 2023

This file exposes the npz_to_hdf() method, used to translate a hierarchical
directory of .npz files into a .hdf5 file.

The inputs must be:
    A Pathlike for the directory of the source files
    A Pathlike of the resulting output .hdf5 file

Originally written by CMML Member CJ Nguyen
"""


import h5py
import pathlib
import os
import numpy as np
from typing import Dict, List, Tuple
from .util import NDArrayType, NPZFileType, H5PYFileType


def convert_npz_to_hdf(
    hdf_path: pathlib.Path,
    npz_dir: pathlib.Path = pathlib.Path("tmp_npz")
    ) -> None:

    # Format of the npz_dir:
    # node_coords.npz (locations per node)
    # step_frame_times/<step>.npz (times per step)
    # temps/<step>/<frame>.npz (temperatures per node per frame)
    # All of these must exist (error if they do not)
    # They're the only things we care about

    hdf_path = pathlib.Path(hdf_path)
    npz_dir = pathlib.Path(npz_dir)

    step_frame_times_dir: pathlib.Path = npz_dir / pathlib.Path("step_frame_times")
    step_frame_times: Dict[str, NDArrayType] = dict()
    root: pathlib.Path
    files: List[pathlib.Path]
    for root, _, files in os.walk(step_frame_times_dir):
        root = pathlib.Path(root)
        file: pathlib.Path
        for file in files:
            file = pathlib.Path(file)
            key: str = str(file.stem)
            step_frame_times_file: NPZFileType
            with np.load(root / file) as step_frame_times_file:
                time_data: NDArrayType = step_frame_times_file[
                        step_frame_times_file.files[0]
                        ]

            step_frame_times[key] = time_data

    node_coords_path: pathlib.Path = npz_dir / pathlib.Path("node_coords.npz")
    node_coords_file: NPZFileType
    with np.load(node_coords_path) as node_coords_file:
        coordinate_data: NDArrayType = node_coords_file[
                node_coords_file.files[0]
                ]

    temps_dir: pathlib.Path = npz_dir / pathlib.Path("temps")
    temp_dict: Dict[pathlib.Path, Dict[str, NDArrayType]] = dict()
    for root, _, files in os.walk(temps_dir):
        root = pathlib.Path(root)
        step_name: str = root.stem
        temp_dict[step_name] = dict()
        files.sort()
        file: pathlib.Path
        for file in files:
            file = pathlib.Path(file)
            temps_file: NPZFileType
            with np.load(root / file) as temps_file:
                temps_data: NDArrayType = temps_file[
                        temps_file.files[0]
                        ]
                temp_dict[step_name][file] = np.hstack((
                    coordinate_data,
                    np.vstack(temps_data)
                    ))

    hdf5_file: H5PYFileType
    with h5py.File(hdf_path, "w") as hdf5_file:
        # Temps/Coords
        step: str
        node_data: NDArrayType
        for step in temp_dict:
            i: int
            items: Tuple[str, NDArrayType]
            for i, items in enumerate(temp_dict[step].items()):
                file: str
                node_data: NDArrayType
                file, node_data = items
                frame: str = pathlib.Path(file).stem
                target_len: int = len(node_data)
                hdf5_file.create_dataset(
                        f"nodes/{step}/{frame}",
                        data = np.hstack((
                            node_data,
                            np.vstack(
                                np.full(
                                    (target_len,),
                                    step_frame_times[step][i]
                                    )
                                )
                            )),
                        compression = "gzip"
                        )

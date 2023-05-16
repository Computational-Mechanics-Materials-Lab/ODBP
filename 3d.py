import pathlib
import numpy as np
import pandas as pd
import pyvista as pv
import h5py
import multiprocessing

from os import PathLike
from typing import Union, TypeAlias
#from multiprocessing import Pool

PathType: TypeAlias = Union[str, PathLike]
PoolType: TypeAlias = multiprocessing.Pool
DataFrameType: TypeAlias = pd.core.frame.DataFrame
H5PYGroupType: TypeAlias = h5py._hl.group.Group
NDArrayType: TypeAlias = np.ndarray

def extract(hdf5_filepath: PathType, node: int, frame_sample: int, x: float, y: float, z: float) -> DataFrameType:
    with h5py.File(hdf5_filepath, "r") as hdf5_file:
        temp_steps: H5PYGroupType = hdf5_file["temps"]
        time_steps: H5PYGroupType = hdf5_file["step_frame_times"]
        target_len: int = len(temp_steps[list(temp_steps.keys())[0]])
        temps: NDArrayType = np.zeros(target_len)
        times: NDArrayType = np.zeros(target_len)
        xs: NDArrayType = np.full(target_len, x)
        ys: NDArrayType = np.full(target_len, y)
        zs: NDArrayType = np.full(target_len, z)
        
        step: str
        for step in temp_steps:
            ind: int
            frame: str
            for ind, frame in enumerate(temp_steps[step]):
                temps[ind] = temp_steps[step][frame][node]
                times[ind] = time_steps[step][int(frame.replace("frame_", "")) // frame_sample]
    
    return pd.DataFrame(data=np.vstack((temps, times, xs, ys, zs), casting="no").T, columns=["Temp", "Time", "X", "Y", "Z"])


def main() -> None:
    frame_sample: int = 10
    x_high: float = 9
    x_low: float = -2.5
    y_high: float = 3.0
    y_low: float = -3.0
    z_high: float = 2.0
    z_low: float = 0
    time_high: float = 7.1
    time_low: float = 4.9
    temp_high: float = 1727.0
    temp_low: float = 300.0
    
    #x_high: float = np.inf
    #x_low: float = -1 * np.inf
    #y_high: float = np.inf
    #y_low: float = -1 * np.inf
    #z_high: float = np.inf
    #z_low: float = 0
    #z_low: float = -1 * np.inf
    
    hdf5_filepath: PathType = pathlib.Path(".", "hdfs", "test_old_2.hdf5")
    hdf5_file: h5py.File
    with h5py.File(hdf5_filepath, "r") as hdf5_file:
        coords: NDArrayType = hdf5_file["node_coords"][:]
        
    node_coords: DataFrameType = pd.DataFrame(data=coords, columns=["Node Labels", "X", "Y", "Z"]).astype({"Node Labels": int})
    
    bounded_node_coords: DataFrameType = node_coords[
        (node_coords["X"] >= x_low)
        & (node_coords["X"] <= x_high)
        & (node_coords["Y"] >= y_low)
        & (node_coords["Y"] <= y_high)
        & (node_coords["Z"] >= z_low)
        & (node_coords["Z"] <= z_high)
    ]
   
    node: int
    args: list[tuple[PathType, int, int, float, float, float]] = [(hdf5_filepath, node - 1, frame_sample, bounded_node_coords[bounded_node_coords["Node Labels"] == node]["X"], bounded_node_coords[bounded_node_coords["Node Labels"] == node]["Y"], bounded_node_coords[bounded_node_coords["Node Labels"] == node]["Z"]) for node in bounded_node_coords["Node Labels"]]
    
    pool: PoolType
    with multiprocessing.Pool() as pool:
        results: list[DataFrameType] = pool.starmap(extract, args)
        
    target_nodes: DataFrameType = pd.concat(results)
    target_nodes = target_nodes[
        (target_nodes["Time"] >= time_low) & (target_nodes["Time"] <= time_high)
    ]
    
    dims_columns: set[str] = {"X", "Y", "Z"}
    time: float
    for time in target_nodes["Time"].unique():
        plotter: pv.Plotter = pv.Plotter(window_size=(1920, 1080), lighting="three lights")
        instance_nodes: DataFrameType = target_nodes[target_nodes["Time"] == time]
        instance_nodes = instance_nodes[instance_nodes["Temp"] >= temp_high]
        points: pv.PolyData = pv.PolyData(instance_nodes.drop(columns=list(set(target_nodes.columns.values.tolist()) - dims_columns)).to_numpy())
        points["Temp"] = instance_nodes["Temp"].to_numpy()
        x_min, x_max, y_min, y_max, z_min, z_max = points.bounds
        mesh = points.delaunay_3d()
        plotter.add_mesh(mesh, scalars="Temp", cmap=pv.LookupTable(cmap="turbo", scalar_range=(temp_low, temp_high), above_range_color=(0.75, 0.75, 0.75, 1.0)))
        plotter.show_bounds(location="outer", ticks="both", font_size=14.0, font_family="courier", color="#000000", axes_ranges=[x_min, x_max, y_min, y_max, z_min, z_max])
        
        plotter.show()
    
    
if __name__ == "__main__":
    main()
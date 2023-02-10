# Author: CJ Nguyen
#

import h5py
import numpy as np
import pandas as pd
from typing import Any
from multiprocessing import Pool

def main():
    hdf5_filename: str = ""
    time_sample: int = 1
    target_csv: str = ""
    nodes: Any = get_coords(hdf5_filename)
    args: list[tuple[str, int, int]] = [(hdf5_filename, node, time_sample) for node in nodes]
    with Pool() as pool:
        results: Any = pool.starmap(read_node_data, args)

    final: Any = pd.concat(results)

    print(final)

    final.to_csv(target_csv)


def get_coords(hdf5_filename: str) -> Any:
    hdf5_file: h5py.File
    with h5py.File(hdf5_filename, "r") as hdf5_file:
        coords: Any = hdf5_file["node_coords"][:]
        # TODO type hints
        node_labels, x, y, z = np.transpose(coords)
    return pd.DataFrame.from_dict({"Node Labels": node_labels.astype(int), "X": x, "Y": y, "Z": z})


def read_node_data(hdf5_filename: str, node_label: int, time_sample: int) -> Any:
    hdf5_file: h5py.File
    with h5py.File(hdf5_filename, "r") as hdf5_file:
        coords: Any = hdf5_file["node_coords"][:]
        node_coords: Any = coords[np.where(coords[:, 0] == node_label)[0][0]][1:]
        temps: list[float] = list()
        times: list[int] = list()
        temp_steps: Any = hdf5_file["temps"]
        time_steps: Any = hdf5_file["step_frame_times"]
        for step in temp_steps:
            for frame in temp_steps[step]:
                # Nodes start at 1
                temps.append(temp_steps[step][frame][node_label - 1])
                times.append(time_steps[step][int(frame.replace("frame_", "")) // time_sample])
        data_dict: dict[str, Any] = {
                "Node Label": node_label,
                "X": node_coords[0],
                "Y": node_coords[1],
                "Z": node_coords[2],
                "Temp": temps,
                "Time": times
                }

    return pd.DataFrame(data_dict, index=None).sort_values("Time")

if __name__ == "__main__":
    main()

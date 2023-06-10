#!/usr/bin/env python3

"""
ODBPlotter odb_visualizer.py
ODBPlotter
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBPlotter
MIT License (c) 2023

A child class of the ODB class which implements two- and three-dimensional
plotting capabilities.
"""

import operator
import multiprocessing
import numpy as np
import pyvista as pv
from typing import Union, Tuple, List
from .odb import Odb
from .util import DataFrameType, MultiprocessingPoolType


class OdbVisualizer(Odb):
    """
    Serves exactly as a normal Odb instance, but includes
    plotting capabilities
    """

    __slots__ = (
        "_interactive",
        "_angle",
        "_colormap"
        )
    def __init__(self) -> None:
        """
        """
        
        super().__init__()

        self._interactive: bool = False
        self._colormap: str = "turbo"

        # TODO ???
        self._angle: Union[str, Tuple[float, float, float]]


    @property
    def colormap(self) -> str:
        return self._colormap
    

    @colormap.setter
    def colormap(self, value: str) -> None:
        self._colormap = value


    @property
    def interactive(self) -> bool:
        return self._interactive


    @interactive.setter
    def interactive(self, value: bool) -> None:
        self._interactive = value


    def plot_3d_all_times(self, label: str = "") -> "List[pv.Plotter]":
        """
        """

        times: DataFrameType = np.sort(self._filtered_nodes["Time"].unique())
        
        plotting_args: List[
            Tuple[
                float,
                str
                ]
            ] = [(time, label) for time in times]
        results: List[pv.Plotter] = list()
        time: float
        for time in times:
            results.append(self.plot_3d_single(time, label))
        # TODO Any way to make this work?
        """
        # TODO dataclass
        plotting_args: List[
            Tuple[
                float,
                str
                ]
            ] = [(time, label) for time in times]
        results: List[pv.Plotter] = list()
        pool: MultiprocessingPoolType
        with multiprocessing.Pool(processes=self.cpus) as pool:
            results: list[pv.Plotter] = pool.starmap(
                self.plot_3d_single,
                plotting_args
                )"""

        return results 


    def plot_3d_single(
        self,
        time: float,
        label: str
        )-> "Union[pv.Plotter, None]":
        """
        """

        dims_columns: set[str] = {"X", "Y", "Z"}
        combined_label: str = f"{label}-{round(time, 2):.2f}"

        plotter: pv.Plotter = pv.Plotter(
            off_screen=(not self._interactive),
            window_size=(1920, 1080),
            lighting="three lights"
            )

        plotter.add_text(
            combined_label,
            position="upper_edge",
            color="#000000",
            font="courier"
        )

        instance_nodes: DataFrameType = self.filter_nodes(
            "Time",
            time,
            operator.eq
        )

        if not instance_nodes.empty:
            points: pv.PolyData = pv.PolyData(
                instance_nodes.drop(
                    columns=list(
                        set(self._target_nodes.columns.values.tolist())
                        - dims_columns
                        )
                    ).to_numpy()
                )
            
            points["Temp"] = instance_nodes["Temp"].to_numpy()
            mesh: pv.PolyData = points.delaunay_3d()

            plotter.add_mesh(
                mesh,
                scalars="Temp",
                cmap = pv.LookupTable(
                    cmap=self._colormap,
                    scalar_range=(
                        self._temp_low,
                        self._temp_high
                        ),
                    above_range_color=(
                        0.75,
                        0.75,
                        0.75,
                        1.0
                    )
                ),
                scalar_bar_args={
                    "title": "Nodal Temperature (Kelvin)",
                    "font_family": "courier",
                    "color": "#000000",
                    "fmt": "%.2f",
                    "position_y": 0
                }
            )

            plotter.show_bounds(
                location="outer",
                ticks="both",
                font_size=14.0,
                font_family="courier",
                color="#000000",
                axes_ranges=points.bounds
                )

            plotter.set_background(color="#FFFFFF")

            # TODO
            plotter.camera.elevation = 0
            plotter.camera.azimuth = 270
            plotter.camera.roll = 300
            plotter.camera_set = True

            return plotter

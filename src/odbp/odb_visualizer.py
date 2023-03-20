#!/usr/bin/env python3


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv
from typing import TextIO, Union, Any
from .odb import Odb


class MeltpointNotSetError(Exception):
    def __init__(self) -> None:
        self.message = "You must enter the meltpoint and, optionally, the string name of a plt colormap (default \"turbo\") in order to set the colormap"


class OdbVisualizer(Odb):
    """

    """
    def __init__(self) -> None:
        Odb.__init__(self)

        self.interactive: bool

        self.angle: str
        self.x_rot: int
        self.y_rot: int
        self.z_rot: int

        self.colormap_name: str
        self.colormap: pv.LookupTable


    def select_colormap(self) -> None:
        if not (hasattr(self, "meltpoint") or hasattr(self, "colormap_name")):
            raise MeltpointNotSetError

        # TODO different melt color? Don't plot from as low as 0?
        self.colormap = pv.LookupTable(cmap=self.colormap_name, scalar_range=(self.low_temp, self.meltpoint), above_range_color=(0.25, 0.25, 0.25, 1.0))


    # Overload parent's set_meltpoint method to always select colormap
    def set_meltpoint(self, meltpoint: float) -> None:
        if not isinstance(meltpoint, float):
            meltpoint = float(meltpoint)
        self.meltpoint = meltpoint

        if hasattr(self, "low_temp"):
            self.select_colormap()


    # Overload parent's set_low_temp method to always select colormap
    def set_low_temp(self, low_temp: float) -> None:
        if not isinstance(low_temp, float):
            low_temp = float(low_temp)
        self.low_temp = low_temp

        if hasattr(self, "meltpoint"):
            self.select_colormap()


    def plot_time_3d(self, time: float, label: str, interactive: bool)-> Any:
        curr_nodes: Any = self.out_nodes[self.out_nodes["Time"] == time]

        formatted_time: str = format(round(time, 2), ".2f")
        combined_label = f"{label}-{formatted_time}"

        off_screen: bool = not interactive
        plotter = pv.Plotter(off_screen=off_screen, window_size=[1920, 1080])
        plotter.add_text(combined_label, position="upper_edge", color="white", font="courier", )
        faces: list[str] = ["x_low", "x_high", "y_low", "y_high", "z_low", "z_high"]
        face: str
        for face in faces:
            selected_nodes: Any

            # TODO This whole idea could be parameterized, but it might be less readable
            if "x" in face:
                if "low" in face:
                    selected_nodes = curr_nodes[curr_nodes["X"] == self.x.low]
                else: # if "high" in face:
                    selected_nodes = curr_nodes[curr_nodes["X"] == self.x.high]

            elif "y" in face:
                if "low" in face:
                    selected_nodes = curr_nodes[curr_nodes["Y"] == self.y.low]
                else: # if "high" in face:
                    selected_nodes = curr_nodes[curr_nodes["Y"] == self.y.high]

            else: # if "z" in face
                if "low" in face:
                    selected_nodes = curr_nodes[curr_nodes["Z"] == self.z.low]
                else: # if "high" in face:
                    selected_nodes = curr_nodes[curr_nodes["Z"] == self.z.high]

            dims_columns: set[str] = set(["X", "Y", "Z"])
            points: Any = pv.PolyData(selected_nodes.drop(columns=list(set(selected_nodes.columns.values.tolist()) - dims_columns)).to_numpy())
            points["Temp"] = selected_nodes["Temp"].to_numpy()
            surface: Any = points.delaunay_2d()

            # For whatever reason, the input data is rotated 180 degrees about the y axis. This fixes that.
            #surface = surface.rotate_z(180)

            plotter.add_mesh(surface, scalars="Temp", cmap=self.colormap, scalar_bar_args={"title": "Nodal Temperature (Kelvins)"})

        plotter.show_bounds(location="outer", ticks="both", font_size=14.0, font_family="courier", color="#FFFFFF", axes_ranges=[self.x.low, self.x.high, self.y.low, self.y.high, self.z.low, self.z.high])
        plotter.set_background(color="#000000")

        #plotter.camera.focal_point = ((self.x.high + self.x.low)/2, (self.y.high + self.y.low)/2, (self.z.high + self.z.low)/2)
        plotter.camera.elevation = 0
        plotter.camera.azimuth = 270
        plotter.camera.roll = 300
        plotter.camera_set = True

        return plotter

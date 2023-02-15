#!/usr/bin/env python3


import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv
from typing import TypeAlias, TextIO, Union, Any
from .odb import Odb


ViewsDict: TypeAlias = dict[str, tuple[int, int, int]]


class MeltpointNotSetError(Exception):
    def __init__(self) -> None:
        self.message = "You must enter the meltpoint and, optionally, the string name of a plt colormap (default \"turbo\") in order to set the colormap"


class OdbVisualizer(Odb):
    """

    """
    def __init__(self, **kwargs) -> None:
        Odb.__init__(self, **kwargs)

        self.results_dir: str = kwargs.get("results_dir", os.getcwd())
        self.interactive: bool = kwargs.get("interactive", True)

        self.views: ViewsDict = self.load_views_dict(os.path.join(os.path.dirname(os.path.abspath(__file__)), "views.json"))

        self.views_list: list[str] = list(self.views.keys())

        #self.angle: str = self.views_list[49]
        self.angle: str = self.views_list[0]
        self.elev: int = self.views[self.angle][0]
        self.azim: int = self.views[self.angle][1]
        self.roll: int = self.views[self.angle][2]

        self.colormap_name: str = kwargs.get("colormap_name", "turbo")
        self.colormap: pv.LookupTable


    def select_colormap(self) -> None:
        if not (hasattr(self, "meltpoint") or hasattr(self, "colormap_name")):
            raise MeltpointNotSetError

        # TODO different melt color? Don't plot from as low as 0?
        self.colormap = pv.LookupTable(cmap=self.colormap_name, scalar_range=(0, self.meltpoint), above_range_color=(0.25, 0.25, 0.25, 1.0))


    # Overload parent's set_meltpoint method to always select colormap
    def set_meltpoint(self, meltpoint: float) -> None:
        if not isinstance(meltpoint, float):
            meltpoint = float(meltpoint)
        self.meltpoint = meltpoint
        self.select_colormap()


    def load_views_dict(self, file) -> ViewsDict:
        o_file: TextIO
        with open(file, "r") as o_file:
            return json.load(o_file)


    def select_views(self, view: Union[int, tuple[int, int, int]]) -> None:
        if isinstance(view, int):
            self.angle = self.views_list[view]
            self.elev = self.views[self.angle][0]
            self.azim = self.views[self.angle][1]
            self.roll = self.views[self.angle][2]

        else:
            self.angle = "custom"
            self.elev = view[0]
            self.azim = view[1]
            self.roll = view[2]


    def plot_time_3d(self, time: float, label: str, interactive: bool)-> Any:
        curr_nodes: Any = self.out_nodes[self.out_nodes["Time"] == time]

        formatted_time: str = format(round(time, 2), ".2f")
        combined_label = f"{label}-{formatted_time}"

        off_screen: bool = not interactive
        plotter = pv.Plotter(off_screen=off_screen)
        plotter.add_text(combined_label, position="upper_edge", color="white", font="courier", )
        faces: list[str] = ["x_low", "x_high", "y_low", "y_high", "z_low", "z_high"]
        face: str
        for face in faces:
            selected_nodes: Any

            # TODO This whole idea could be parameterized, but it might be less readable
            if "x" in face:
                if "low" in face:
                    selected_nodes = curr_nodes[curr_nodes["X"] == self.x.vals[0]]
                else: # if "high" in face:
                    selected_nodes = curr_nodes[curr_nodes["X"] == self.x.vals[-1]]

            elif "y" in face:
                if "low" in face:
                    selected_nodes = curr_nodes[curr_nodes["Y"] == self.y.vals[0]]
                else: # if "high" in face:
                    selected_nodes = curr_nodes[curr_nodes["Y"] == self.y.vals[-1]]

            else: # if "z" in face
                if "low" in face:
                    selected_nodes = curr_nodes[curr_nodes["Z"] == self.z.vals[0]]
                else: # if "high" in face:
                    selected_nodes = curr_nodes[curr_nodes["Z"] == self.z.vals[-1]]

            dims_columns: list[str] = ["X", "Y", "Z"]
            points: Any = pv.PolyData(selected_nodes.drop(columns=list(set(selected_nodes.columns.values.tolist()) - set(dims_columns))).to_numpy())
            points["Temp"] = selected_nodes["Temp"].to_numpy()
            surface: Any = points.delaunay_2d()
            plotter.add_mesh(surface, scalars="Temp", cmap=self.colormap, scalar_bar_args={"title": "Nodal Temperature (Kelvins)"})
            plotter.view_vector((self.elev, self.azim, self.roll))

        return plotter


        

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

        self.angle: str = self.views_list[49]
        self.elev: int = self.views[self.angle][0]
        self.azim: int = self.views[self.angle][1]
        self.roll: int = self.views[self.angle][2]

        self.colormap_name: str = kwargs.get("colormap_name", "turbo")
        self.colormap: plt.cm.ScalarMappable


    def select_colormap(self) -> None:
        if not (hasattr(self, "meltpoint") or hasattr(self, "colormap_name")):
            raise MeltpointNotSetError

        norm: mcolors.Normalize = mcolors.Normalize(0, self.meltpoint)
        self.colormap = plt.cm.ScalarMappable(norm=norm, cmap=self.colormap_name)
        self.colormap.set_array([])


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


    def plot_time_3d(self, time: float, label: str, title: str)-> Any:
        curr_nodes: Any = self.out_nodes[self.out_nodes["Time"] == time]

        formatted_time: str = format(round(time, 2), ".2f")
        combined_label = f"{label}-{formatted_time}"

        # PLT does not play nice with type hints, we use "any"
        # fig: Any = plt.figure(figsize=(19.2, 10.8))
        # ax: Any = plt.axes(projection="3d", label=combined_label)
        #
        # ax.set_xlabel(self.x.name)
        # ax.set_ylabel(self.y.name)
        # ax.set_zlabel(self.z.name)
        #
        # ax.set_box_aspect((self.x.size, self.y.size, self.z.size))
        # ax.view_init(elev=self.elev, azim=self.azim, roll=self.roll)
        #
        # ax.set_title(f"{title}, time step: {formatted_time}")
        # fig.add_axes(ax, label=ax.title)

        plotter = pv.Plotter()
        faces: list[str] = ["x_low", "x_high", "y_low", "y_high", "z_low", "z_high"]
        face: str
        for face in faces:
            # X: Any
            # Y: Any
            # Z: Any
            temp_mask: Any
            indices: list[str] = ["X", "Y", "Z"]
            direction: tuple[int, int, int]
            center: tuple[float, float, float]
            i_size: int
            j_size: int
            i_resolution: int
            j_resolution: int

            # TODO This whole idea could be parameterized, but it might be less readable
            if "x" in face:
                indices.remove("X")
                direction = (1, 0, 0)
                i_size = self.z.size
                i_resolution = self.z.size
                j_size = self.y.size
                j_resolution = self.y.size
                if "low" in face:
                    temp_mask = curr_nodes["X"] == self.x.vals[0]
                    center = (0, self.y.size / 2, self.z.size / 2)
                else: # if "high" in face:
                    temp_mask = curr_nodes["X"] == self.x.vals[-1]
                    center = (self.x.size, self.y.size / 2, self.z.size / 2)

            elif "y" in face:
                indices.remove("Y")
                direction = (0, 1, 0)
                i_size = self.x.size
                i_resolution = self.x.size
                j_size = self.z.size
                j_resolution = self.z.size
                if "low" in face:
                    temp_mask = curr_nodes["Y"] == self.y.vals[0]
                    center = (self.x.size / 2, 0, self.z.size / 2)
                else: # if "high" in face:
                    temp_mask = curr_nodes["Y"] == self.y.vals[-1]
                    center = (self.x.size / 2, self.y.size, self.z.size / 2)

            else: # if "z" in face
                indices.remove("Z")
                direction = (0, 0, 1)
                i_size = self.x.size
                i_resolution = self.x.size
                j_size = self.y.size
                j_resolution = self.y.size
                if "low" in face:
                    temp_mask = curr_nodes["Z"] == self.z.vals[0]
                    center = (self.x.size / 2, self.y.size / 2, 0)
                else: # if "high" in face:
                    temp_mask = curr_nodes["Z"] == self.z.vals[-1]
                    center = (self.x.size / 2, self.y.size / 2, self.z.size)

            temp_nodes: Any = curr_nodes[temp_mask]
            # TODO
            #first_offset: float = temp_nodes[indices[0]].min()
            #second_offset: float = temp_nodes[indices[1]].min()

            colors: Any = np.zeros((i_size * j_size, 3))
            face_xs: list[float] = list()
            face_ys: list[float] = list()
            face_zs: list[float] = list()
            node: Any
            i = 0
            for _, node in temp_nodes.iterrows():
                print(i)
                temp = node["Temp"]
                face_xs.append(node["X"])
                face_ys.append(node["Y"])
                face_zs.append(node["Z"])
                print(temp)
                # TODO
                #second_index: int = int((node[indices[0]] - first_offset) / self.mesh_seed_size)
                #first_index: int = int((node[indices[1]] - second_offset) / self.mesh_seed_size)

                if temp >= self.meltpoint:
                    colors[i] = (0.25, 0.25, 0.25)
                else:
                    colors[i] = self.colormap.to_rgba(temp)[:3]

                i += 1

            curr_surface = np.vstack((face_xs, face_ys, face_zs)).T
            surface: Any = pv.Plane(center=center, direction=direction, i_size=i_size, j_size=j_size, i_resolution=i_resolution, j_resolution=j_resolution)
            #surface.cell_data["colors"] = colors
            surface = surface.interpolate(pv.PolyData(curr_surface))
            #plotter.add_mesh(surface, scalars="colors", rgb=True)
            plotter.add_mesh(surface, rgb=True)
            plotter.view_isometric()

        return plotter


        

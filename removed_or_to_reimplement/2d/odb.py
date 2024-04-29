    # 2D Plotting

    # Structure of the dataframe:
    # Times are repeated
    # We'll assume, by default, that you want (one of) min, max, mean of duplicated time

    def plot_key_versus_time(
        self,
        target_output: str,
        mean_max_both: str = "both",
        title: str | None = None,
    ) -> pathlib.Path | None:
        # TODO What if I want to 2d-plot only 1 nodeset, but I extractor more stuff
        # or DIDN'T extract the nodeset at all. Same w/ 3D. Metadata?

        if not PYVISTA_AVAILABLE:
            raise Exception(
                "Plotting cabailities are not included."
                ' Please install pyvista via pip install odb-plotter["plot"]'
                ' or odb-plotter["all"] rather than pip install odb-plotter'
                " Or export the data from Odb.extract() to another tool,"
                " such as matplotlib, plotly, or bokeh."
            )

        if not hasattr(self, "_extracted_data") and self._extracted_data is not None:
            _ = self.extract()

        target_data = self._extracted_data[
            (self.time_low <= self._extracted_data["Time_mean"])
            & (self._extracted_data["Time_mean"] <= self.time_high)
        ]
        time_data: list[float] = list(target_data.index)

        title = (
            title
            if title is not None
            else (
                self.h5_path.stem
                if (hasattr(self, "h5_path") and self.h5_path is not None)
                else self.odb_path.stem
            )
        )
        title += f" {target_output} versus Time"

        temp_v_time: pv.Chart2D = pv.Chart2D()
        #    x_label="Time (seconds)", y_label="Temperature (Kelvin)"
        # )
        # temp_v_time.title = title

        if mean_max_both.lower() in ("mean", "both"):
            temp_v_time.line(
                time_data,
                target_data[f"{target_output}_mean"].values,
                color="#0000FF",  # TODO param
                # label=f"Mean {target_output}",
                width=5.0,
            )

        if mean_max_both.lower() in ("max", "both"):
            temp_v_time.line(
                time_data,
                target_data[f"{target_output}_max"].values,
                color="#FF0000",  # TODO param
                # label=f"Max {target_output}",
                width=5.0,
            )

        screenshot: bool | pathlib.Path = (
            self.result_dir / f"{title}.png" if self.save else False
        )
        if self.save:
            if not self.result_dir.exists():
                self.result_dir.mkdir()

        save_path: pathlib.Path = (
            self.result_dir / f"{mean_max_both + '_'}{target_output + '_'}{title}.png"
        )
        temp_v_time.show(interactive=True, off_screen=False, screenshot=screenshot)

        if self.save:
            return save_path

    def plot_single_node(
        self, target_output: str, node: int, title: str | None = None
    ) -> pathlib.Path | None:
        if not PYVISTA_AVAILABLE:
            raise Exception(
                "Plotting cabailities are not included."
                ' Please install pyvista via pip install odb-plotter["plot"]'
                ' or odb-plotter["all"] rather than pip install odb-plotter'
                " Or export the data from Odb.extract() to another tool,"
                " such as matplotlib, plotly, or bokeh."
            )

        if not hasattr(self, "data") or self.data is None:
            self.load_h5()

        node_vals = self.data[self.data["Node Label"] == node]

        title = (
            title
            if title is not None
            else (
                self.h5_path.stem
                if (hasattr(self, "h5_path") and self.h5_path is not None)
                else self.odb_path.stem
            )
        )
        title += f" {target_output} versus Time for Node {node}"

        temp_v_time: pv.Chart2D = pv.Chart2D()
        #    x_label="Time (seconds)", y_label="Temperature (Kelvin)"
        # )
        # temp_v_time.title = title

        data_to_plot = node_vals.drop(
            columns=list(set(node_vals.keys()) - set(("Time", target_output)))
        )
        data_to_plot = data_to_plot[
            (self.time_low <= data_to_plot["Time"])
            & (data_to_plot["Time"] <= self.time_high)
        ]
        data_to_plot = data_to_plot.sort_values(by="Time", ascending=True)
        temp_v_time.line(
            data_to_plot["Time"],
            data_to_plot[target_output],
            color="#FF0000",  # TODO param
            # label=f"{target_output} per time for Node {node}",
            width=5.0,
        )

        screenshot: bool | pathlib.Path = (
            self.result_dir / f"{target_output}_Node_{node}_{title}.png"
            if self.save
            else False
        )
        if self.save:
            if not self.result_dir.exists():
                self.result_dir.mkdir()

        save_path: pathlib.Path = self.result_dir / f"{title}.png"
        temp_v_time.show(interactive=True, off_screen=False, screenshot=screenshot)

        if self.save:
            return save_path

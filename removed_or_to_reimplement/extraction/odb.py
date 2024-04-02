    # TODO!!! Extract needs a rework. Or remove it?
    @classmethod
    def extract_by_path(
        cls,
        path: pathlib.Path,
    ) -> pd.DataFrame:
        if ensure_magic(path, H5_MAGIC_NUM):
            # Extract from .hdf5
            return cls().extract_from_h5(path)

        elif ensure_magic(path, ODB_MAGIC_NUM):
            # extract from .odb
            return cls().extract_from_odb(path)

        else:
            raise ValueError(f"{path} is not a valid .hdf5 or .odb file!")

    def extract(self) -> pd.DataFrame:
        result: pd.DataFrame
        if (hasattr(self, "h5_path") and self.h5_path is not None) or (
            hasattr(self, "data") and self.data is not None
        ):
            result = self.extract_from_h5()
            self._extracted_data = result
            return result

        elif hasattr(self, "odb_path"):
            result = self.extract_from_odb()
            self._extracted_data = result
            return result

        else:
            raise AttributeError(
                "This Odb object does not have a .odb file or a .hdf5 file from which to extract"
            )

    def extract_from_odb(
        self, target_file: pathlib.Path | None = None
    ) -> pd.DataFrame:
        if target_file is None:
            target_file = self.odb_path

        if target_file is None:
            raise ValueError("odb_path must be set to extract from .odb file")

        # extract_odb_pickle_input_dict: dict[
        #    str, list[str] | list[int] | None
        # ] = {
        # TODO
        extract_odb_pickle_input_dict: dict[str, Any] = {
            "cpus": self.cpus,
            "nodes": self.nodes,
            "nodesets": self.nodesets,
            "time_step": self.time_step,
            "parts": self.parts,
            "steps": self.steps,
            "coord_key": self.coord_key,
            "target_outputs": self.target_outputs,
        }

        send_pickle_file: BinaryIO
        with open(self._extract_pickle_path, "wb") as send_pickle_file:
            pickle.dump(extract_odb_pickle_input_dict, send_pickle_file, protocol=2)

        args_list: list[str | pathlib.Path] = [
            self.abaqus_executable,
            "python",
            self._extract_script_path,
            target_file,
            self._extract_pickle_path,
            self._extract_result_path,
        ]

        # TODO
        # shell=True is bad practice, but abaqus python will not run without it.
        subprocess.run(args_list, shell=True)

        result_file: TextIO
        try:
            with open(self._extract_result_path, "rb") as result_file:
                # From the Pickle spec, decoding python 2 numpy arrays must use
                # "latin-1" encoding
                results: list[dict[str, float]]
                results = pickle.load(result_file, encoding="latin-1")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"File {self._extract_result_path} was not found. See previous Python 2 errors"
            )

        results = sorted(results, key=lambda d: d["time"])

        results_df_list: list[pd.DataFrame] = list()
        for result in results:
            time = result.pop("time")
            results_df_list.append(
                pd.DataFrame.from_dict({time: result}, orient="index")
            )

        result_df: pd.DataFrame = pd.concat(results_df_list)

        if self._extract_result_path.exists():
            self._extract_result_path.unlink()

        return result_df

    def extract_from_h5(self, target_file: pathlib.Path | None = None) -> pd.DataFrame:
        if target_file is not None:
            if hasattr(self, "data"):
                raise AttributeError(
                    "Do not pass in a new path to an "
                    "existing Odb() object for extracting. Use the classmethod "
                    "instead."
                )

            else:
                self.h5_path = target_file
                self.load_h5()

        else:
            if not hasattr(self, "data"):
                self.load_h5()

        results: list[pd.DataFrame] = []
        frame: pd.DataFrame
        for frame in self:
            time: float = frame["Time"].values[0]
            output: str
            frame_dict: dict[int, dict[str, float]] = {time: {}}
            chosen_outputs = (
                self.target_outputs
                if (hasattr(self, "target_outputs") and self.target_outputs is not None)
                else frame.keys()
            )
            for output in chosen_outputs:
                output_data: pd.DataFrame = frame[output].values
                if output in ("NT11",):
                    output_data = output_data[output_data != 300.0]
                    output_data = output_data[output_data != 0.0]
                output_data = output_data[output_data != np.nan]
                min_val: float = np.min(output_data) if len(output_data) > 0 else np.nan
                max_val: float = np.max(output_data) if len(output_data) > 0 else np.nan
                mean_val: float = (
                    np.mean(output_data) if len(output_data) > 0 else np.nan
                )
                frame_dict[time][f"{output}_min"] = min_val
                frame_dict[time][f"{output}_max"] = max_val
                frame_dict[time][f"{output}_mean"] = mean_val

            results.append(pd.DataFrame.from_dict(frame_dict, orient="index"))

        results_df: pd.DataFrame = pd.concat(results)

        return results_df


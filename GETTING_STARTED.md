# How to use ODBPlotter

## Installation
ODBPlotter is available via `pip` on [pypi.org](https://pypi.org)
```sh
python -m pip install odb-plotter
```
`pip` will handle all dependencies EXCEPT Abaqus. In order to convert a .odb to a .hdf5, you will need to have Abaqus installed. If you do not use the conversion, only the plotting, this is no required.

## API usage
ODBPlotter can be used as a Python module in Python code.

Generally, processing a .odb file comes from the Odbp object type.
Assuming you have an Abaqus .odb called `example.odb`, loading it with ODBPlotter looks like this:
```py
from odbp import Odbp

odb = Odbp()
odb.odb_path = <"path/to/example.odb">

# Give the resulting .hdf5 file a path and name
odb.h5_path = <"path/to/example.hdf5">

# Assuming you have a working Abaqus installation, you can automatically convert from a .odb to a .hdf5
odb.convert()
```

This will result in the `example.hdf5` file being created in the desired directory.

Opening a .hdf5 reads the data into memory and tidies the data into a useable [pandas](https://pandas.pydata.org/) DataFrame.
```py
from odbp import Odbp
odb = Odbp()
odb.h5_path = <"path/to/example.hdf5"> # Assuming we've already created this like above
odb.load_hdf()
odb.data # This is now a dataclass with five simple arrays and two pandas DataFrames with the .odb data
```
As such, the .odb data can now be used just as any other Python 3 pandas DataFrame.

Before loading data in, it may be necessary to define bounds, such as for not loading substrate, or only loading desired frames:
```py
from odbp import Odbp
odb = Odbp()
odb.h5_path = <"path/to/example.hdf5"> # Assuming we've already created this

# Spatial constraints use the same coordinate axes as in Abaqus
odb.x_low = -1
odb.x_high = 1
odb.y_low = -3.5
odb.y_high - 3.0
odb.z_low = 0.0
# You can leave some of these unset, and the  n_low will default to -inf, and x_high will default to inf, thus grabbing all data in the .hdf5
# So, the above starts from z = 0 (i.e., above the substrate) and captures everythin above.

# Time constraints use the seconds given in Abaqus
odb.time_low = 1.0
odb.time_high = 5.5
# So, the above shows the simulation from 1.0 to 5.5 seconds

# Temperature constraints don't control which temperatures are loaded, but instead define the ranges for the colormap and legend
odb.temp_low = 300.0
odb.temp_high = 1727.0
# Temperature, by default, are in Kevlin
```
Thus, to show a cut view from any plane, simply update the spatial constraints, even after loading. To view different times, simply update the time range.

3D Plotting is as follows:
```py
from odbp import Odbp
odb = Odbp()
odb.h5_path = <"path/to/example.hdf5"> # Assuming we've already created this

# Choose whether to open the interactive PyVista plotter,
# allowing for panning and zooming (interactive = True)
# Or to save plots automatically (interactive = False)
odb.interative = <boolean>

# Choose a default view (especially if non-interactive)
odb.view = "UFR-U" # The Up/Front/Right Corner with the "Up" face on top
# All views are found in views.toml, but follow this format, view views for all Faces, Edges, and Vertices.

# Plot the currently set time range:
odb.plot_3d_all_times("<target_output>")
# The "target output" is the name given to the output. For example, "NT11" or "Temp"

# Plot the meltpools for the current time range:
# Specific to arc-DED simulations

# Ensure that the melting point is set
odb.temp_high = 1727.0
meltpool_only = odb.data[odb.data["Temp"] >= odb.temp_high]
odb.plot_3d_all_times(
"Temp",
target_nodes=meltpool_only
)
```

Full Documentation can be found at: **COMING SOON!**
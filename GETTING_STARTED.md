# How to use ODBP

## Installation
ODBP is available via `pip` on [pypi.org](https://pypi.org)
```sh
python -m pip install odbp
```
`pip` will handle all dependencies EXCEPT Abaqus. In order to convert a .odb to a .hdf5, you will need to have Abaqus installed. If you do not have Abaqus, you will still be able to interface with, manipulate, and visualize ODBP .hdf5 files, just not convert from .odb to .hdf5.

## API usage
Generally, processing a .odb file comes from the Odbp object type.
Assuming you have an Abaqus .odb called `example.odb`, loading it with ODBP looks like this:
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

Opening a .hdf5 reads the data into memory and tidies the data into a small data structure with metadata and a number of [pandas](https://pandas.pydata.org/) DataFrames.
```py
from odbp import Odbp
odb = Odbp()
odb.h5_path = <"path/to/example.hdf5"> # Assuming we've already created this like above
odb.load_h5()
odb.data # A data strucure with .odb metadata and dataframes
```

Any given output can be filtered, showing only a desired subset of the data.
Before loading data in, it may be necessary to define bounds, such as for not loading substrate, or only loading desired frames:
```py
from odbp import Odbp
odb = Odbp()
odb.h5_path = <"path/to/example.hdf5"> # Assuming we've already created this
odb.load_h5()

print(odb.outputs.outputs_by_names)
# A dictionary of string names to which output field they correspond to

# A lower bound for STATUS, i.e., don't show STATUS = 0, un-activated nodes
odb.outputs.output_by_name["STATUS"].bound_min = 1

# Only timestep 2.5, i.e., both upper- and lower- are the same value
odb.outputs.outputs_by_name["Time"].set_bounds(2.5)

# Set both upper- and lower- bounds
odb.outputs.outputs_by_name["Temp"].set_bounds(300.0, 1727.0) 

# Set a cut-plane along the X axis (i.e., to not show the substrate)
odb.outputs.outputs_by_name["Z"].bound_min = 0.0
```
Thus, to show a cut view from any plane, simply update the spatial constraints, even after loading. To view different times, simply update the time range.

3D Plotting is as follows:
```py
from odbp import Odbp
odb = Odbp()
odb.h5_path = <"path/to/example.hdf5"> # Assuming we've already created this

# Choose a default view (especially if non-interactive)
odb.view = "PxPyPz-Pz" # The PositiveX-PositiveY-PositiveZ Corner with the "PositiveZ" face on top
# All views are found in views.toml, but follow this format.

# Set your bounds as above

# Plot the currently selected subset
odb.plot()

# Plot a colormap of a given output:
odb.plot("Temp")

# Plot a colormap of a given output with desired colormap bounds:
odb.plot("Temp", target_key_lower_bound=300.0, target_key_upper_bound=1727.0)

# Plot the meltpool:
# Specific to arc-DED simulations

odb.outputs.outputs_by_name["Temp"].bound_min = 1727.0
odb.plot("Temp", target_key_lower_bound=300.0, target_key_upper_bound=1727.0)
```

Full Documentation can be found at: **COMING SOON!**
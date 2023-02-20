# ODB Plotter

## Constraints:
ODB Plotter is being developed by [CMML](https://www.cmml.me.msstate.edu), and as such has a focus on Additive Manufacturing and Temperature Data.

## Install with pip
```shell
pip install odb-plotter
```

## Run the cli with python
```shell
python -m odbp
```

## Or import to use the api
```python
from odbp import Odb
...
```

## ODB Plotter Design Goals

### I intend for this project to serve two purposes:
- First, implement an extensible, flexible api for accessing data within .odb files or .hdf5 files with odb data
- Second, implement a user-friendly, sane-defaults cli to allow for quick data extraction, manipulation, and visualization with no hassle

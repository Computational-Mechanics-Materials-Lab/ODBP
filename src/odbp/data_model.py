#!/usr/bin/env python3
from typing import Final

# One place to store the names of the data columns in the .hdf5:
DATA_COLUMNS: Final[tuple[str, str, str, str, str]] = (
    "Element Connectivity",
    "Element Sets to Elements",
    "Node Sets to Nodes",
    "Part to Element Set",
    "Part to Node Set",
)
element_connectivity_name: Final[str] = DATA_COLUMNS[0]
element_sets_to_elements_name: Final[str] = DATA_COLUMNS[1]
node_sets_to_nodes_name: Final[str] = DATA_COLUMNS[2]
part_to_element_set_name: Final[str] = DATA_COLUMNS[3]
part_to_node_set_name: Final[str] = DATA_COLUMNS[4]

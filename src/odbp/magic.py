import pathlib
from typing import BinaryIO

# Magic # Constants
ODB_MAGIC_NUM: bytes = b"HKSRD0"
HDF_MAGIC_NUM: bytes = b"\x89HDF\r\n"


def ensure_magic(file_path: pathlib.Path, magic: bytes) -> bool:
    file: BinaryIO
    first_line: bytes
    with open(file_path, "rb") as file:
        first_line = file.readline()

    return first_line.startswith(magic)

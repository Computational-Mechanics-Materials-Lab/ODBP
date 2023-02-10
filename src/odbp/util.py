#!/usr/bin/env python3

"""
Utility methods for odb_plotter
"""

from typing import Union

DefaultType = Union[None, str]
AnswerType = Union[tuple[str, str], tuple[str, str, str]]

def confirm(message: str, default: DefaultType = None) -> bool:
    check_str: str = "Is this correct (y/n)? "
    yes_vals: AnswerType = ("yes", "y")
    no_vals: AnswerType = ("no", "n")
    if isinstance(default, str):
        if default.lower() in yes_vals:
            yes_vals = ("yes", "y", "")
            check_str = "Is this correct (Y/n)? "
        elif default.lower() in no_vals:
            no_vals = ("no", "n", "")
            check_str = "Is this correct (y/N)? "

    while True:
        print(message)
        user_input: str = input(check_str).lower()
        if user_input in yes_vals:
            return True
        elif user_input in no_vals:
            return False
        else:
            print("Error: invalid input")



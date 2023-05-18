#!/usr/bin/env python3

"""
ODBPlotter base_odb.py
ODBPlotter
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBPlotter
MIT License (c) 2023

Base classes for measuring .odb data. Spatial, Temporal, Thermal components
"""

class SpatialODB:
    """
    Base Spatial class for .odb data
    """

    __slots__ = ("_x_low", "_x_high", "_y_low", "_y_high", "_z_low", "_z_high")


    def __init__(self) -> None:

        self._x_low: float
        self._x_high: float
        self._y_low: float
        self._y_high: float
        self._z_low: float
        self._z_high: float


    @property
    def x_low(self) -> float:
        return self._x_low
    
    @x_low.setter
    def x_low(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("x_low must be a float")

        # Don't let users set improper dimensions
        if hasattr(self, "_x_high"): # If high is set
            if value > self.x_high:
                raise ValueError(f"The value for x_low ({value}) must not be greater than the value for")
        
        self._x_low = value

    @property
    def x_high(self) -> float:
        return self._x_high
    
    @x_high.setter
    def x_high(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("x_high must be a float")
        
        self._x_high = value

    @property
    def y_low(self) -> float:
        return self._y_low
    
    @y_low.setter
    def y_low(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("y_low must be a float")
        
        self._y_low = value

    @property
    def y_high(self) -> float:
        return self._y_high
    
    @y_high.setter
    def y_high(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("y_high must be a float")
        
        self._y_high = value

    @property
    def z_low(self) -> float:
        return self._z_low
    
    @z_low.setter
    def z_low(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("z_low must be a float")
        
        self._z_low = value

    @property
    def z_high(self) -> float:
        return self._z_high
    
    @z_high.setter
    def z_high(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("z_high must be a float")
        
        self._z_high = value


class ThermalODB:
    """
    Base Thermal class of .odb data
    """

    __slots__ = ("_temp_low", "_temp_high")

    def __init__(self) -> None:
        self._temp_low: float
        self._temp_high: float


    @property
    def temp_low(self) -> float:
        return self._temp_low
    
    @temp_low.setter
    def temp_low(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise Exception("temp_low must be a float")
        
        if value < 0:
            raise Exception("temp_low must be greater than or equal to 0 (Kelvins)")
        
        self._temp_low = value

    @property
    def temp_high(self) -> float:
        return self._temp_high
    
    @temp_high.setter
    def temp_high(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise Exception("temp_high must be a float")
        
        self._temp_high = value


class TemporalODB:
    """
    Base Temporal class of .odb data
    """

    __slots__ = ("_time_low", "_time_high")

    def __init__(self) -> None:
        self._time_low: float
        self._time_high: float


    @property
    def time_low(self) -> float:
        return self._time_low
    
    @time_low.setter
    def time_low(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise Exception("time_low must be a float")
        
        if value < 0:
            raise Exception("time_low must be greater than or equal to 0")
        
        self._time_low = value

    @property
    def time_high(self) -> float:
        return self._time_high
    
    @time_high.setter
    def time_high(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise Exception("time_high must be a float")
        
        self._time_high = value
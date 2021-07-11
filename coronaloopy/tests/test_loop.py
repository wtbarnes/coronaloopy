"""
Tests for Loop class
"""
import pytest
import astropy.units as u
import sunpy.sun.constants

from coronaloopy import Loop
from coronaloopy.geometry import semi_circular_loop


@pytest.fixture
def coord():
    return semi_circular_loop(100*u.Mm)


@pytest.fixture
def loop(coord):
    B = 1e3 * u.G * (sunpy.sun.constants.radius / coord.radius)**2
    return Loop(coord, B)


def test_loop_attributes(loop):
    ...

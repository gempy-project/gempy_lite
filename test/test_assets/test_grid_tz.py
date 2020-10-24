# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")

# Importing GemPy
import gempy_lite as gp
from gempy_lite.assets.geophysics import GravityPreprocessing
from gempy_lite.core.grid_modules.grid_types import CenteredGrid

# Importing auxiliary libraries
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import pytest


def test_irregular_grid():
    g = CenteredGrid()
    g.set_centered_grid(np.array([0, 0, 0]), resolution=[5, 5, 5], radius = [100,100,100])

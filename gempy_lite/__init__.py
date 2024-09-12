"""
Module initialisation for GemPy
Created on 24/10/2020

@author: Miguel de la Varga
"""
import sys
import os

import warnings

try:
    import faulthandler
    faulthandler.enable()
except Exception as e:  # pragma: no cover
    warnings.warn('Unable to enable faulthandler:\n%s' % str(e))


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from gempy_lite.gempy_api import *
from gempy_lite.api_modules.getters import *
from gempy_lite.api_modules.setters import *
from gempy_lite.api_modules.io import *
from gempy_lite.core.model import Project, ImplicitCoKriging, Faults, Grid, \
    Orientations, Series, SurfacePoints
from gempy_lite.core.kernel_data import Surfaces, Structure, KrigingParameters
from gempy_lite.core.model_data import Options, RescaledData, AdditionalData

from gempy_lite.core.predictor.solution import Solution
from gempy_lite.addons.gempy_to_rexfile import geomodel_to_rex


assert sys.version_info[0] >= 3, "GemPy requires Python 3.X"  # sys.version_info[1] for minor e.g. 6
__version__ = '0.1'

if __name__ == '__main__':
    pass

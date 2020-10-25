"""Unstructured data module, i.e.:
- Faults
- Series/Features
- Surfaces
- SurfacePoints
- Orientations

===
DERIVATIVE DATA
===
- Rescaled Data
- Additional Data
    - Structure
    - Kriging Parameters
    - Options
===
- Interpolation


 """

from dataclasses import dataclass
import xarray as xr
import pandas as pd
import numpy as np


@dataclass
class UnstructGemPy:
        """Primary structure definition for unstructured data

        Attributes:


        Notes:


        """
        data: xr.Dataset



        # vertex: np.ndarray
        # edges: np.ndarray
        # attributes: Optional[pd.DataFrame] = None

        # def __init__(self, vertex: np.ndarray, edges: np.ndarray,
        #              attributes: Optional[pd.DataFrame] = None,
        #              points_attributes: Optional[pd.DataFrame] = None):
        #     v = xr.DataArray(vertex, dims=['points', 'XYZ'])
        #     e = xr.DataArray(edges, dims=['edge', 'nodes'])
        #
        #     if attributes is None:
        #         attributes = pd.DataFrame(np.zeros((edges.shape[0], 0)))
        #
        #     if points_attributes is None:
        #         points_attributes = pd.DataFrame(np.zeros((vertex.shape[0], 0)))
        #
        #     a = xr.DataArray(attributes, dims=['edge', 'attribute'])
        #     pa = xr.DataArray(points_attributes, dims=['points', 'points_attribute'])
        #
        #     c = xr.Dataset({'vertex': v, 'edges': e,
        #                     'attributes': a, 'points_attributes': pa})
        #     self.data = c.reset_index('edge')
        #
        #     self.validate()

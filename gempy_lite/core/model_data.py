import warnings
from typing import Union

import numpy as np
import pandas as pn

from gempy_lite.core.kernel_data.geometric_data import SurfacePoints, Orientations, Surfaces
from gempy_lite.core.kernel_data import Structure, KrigingParameters
from gempy_lite.core.predictor.structured_data import Grid
from gempy_lite.utils import docstring as ds
from gempy_lite.utils.meta import _setdoc_pro


class MetaData(object):
    """Class containing metadata of the project.

    Set of attributes and methods that are not related directly with the geological model but more with the project

    Args:
        project_name (str): Name of the project. This is use as default value for some I/O actions

    Attributes:
        date (str): Time of the creations of the project
        project_name (str): Name of the project. This is use as default value for some I/O actions

    """

    def __init__(self, project_name='default_project'):
        import datetime
        now = datetime.datetime.now()
        self.date = now.strftime(" %Y-%m-%d %H:%M")

        if project_name == 'default_project':
            project_name += self.date

        self.project_name = project_name


class Options(object):
    """The class options contains the auxiliary user editable flags mainly independent to the model.

     Attributes:
        df (:class:`pn.DataFrame`): df containing the flags. All fields are pandas categories allowing the user to
         change among those categories.

     """

    def __init__(self):
        df_ = pn.DataFrame(np.array(['float32', 'geology', 'fast_compile', 'cpu', None]).reshape(1, -1),
                           index=['values'],
                           columns=['dtype', 'output', 'theano_optimizer', 'device', 'verbosity'])

        self.df = df_.astype({'dtype': 'category', 'output': 'category',
                              'theano_optimizer': 'category', 'device': 'category',
                              'verbosity': object})

        self.df['dtype'].cat.set_categories(['float32', 'float64'], inplace=True)
        self.df['theano_optimizer'].cat.set_categories(['fast_run', 'fast_compile'], inplace=True)
        self.df['device'].cat.set_categories(['cpu', 'cuda'], inplace=True)

        self.default_options()

    def __repr__(self):
        return self.df.T.to_string()

    def _repr_html_(self):
        return self.df.T.to_html()

    def modify_options(self, attribute, value):
        """Method to modify a given field

        Args:
            attribute (str): Name of the field to modify
            value: new value of the field. It will have to exist in the category in order for pandas to modify it.

        Returns:
            :class:`pandas.DataFrame`: df where options data is stored
        """

        assert np.isin(attribute, self.df.columns).all(), 'Valid properties are: ' + np.array2string(self.df.columns)
        self.df.loc['values', attribute] = value
        return self.df

    def default_options(self):
        """Set default options.

        Returns:
            bool: True
        """

        # We want to have an infer function here
        self.df.loc['values', 'device'] = 'cpu'

        if self.df.loc['values', 'device'] == 'cpu':
            self.df.loc['values', 'dtype'] = 'float64'
        else:
            self.df.loc['values', 'dtype'] = 'float32'

        self.df.loc['values', 'theano_optimizer'] = 'fast_compile'
        return True


@_setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, Grid.__doc__])
class RescaledData(object):
    """
    Auxiliary class to rescale the coordinates between 0 and 1 to increase float stability.

    Attributes:
        df (:class:`pn.DataFrame`): Data frame containing the rescaling factor and centers
        surface_points (:class:`SurfacePoints`): [s0]
        orientations (:class:`Orientations`): [s1]
        grid (:class:`Grid`): [s2]

    Args:
        surface_points (:class:`SurfacePoints`):
        orientations (:class:`Orientations`):
        grid (:class:`Grid`):
        rescaling_factor (float): value which divide all coordinates
        centers (list[float]): New center of the coordinates after shifting
    """

    def __init__(self, surface_points: SurfacePoints, orientations: Orientations, grid: Grid,
                 rescaling_factor: float = None, centers: Union[list, pn.DataFrame] = None):

        self.surface_points = surface_points
        self.orientations = orientations
        self.grid = grid

        self.df = pn.DataFrame(np.array([rescaling_factor, centers]).reshape(1, -1),
                               index=['values'],
                               columns=['rescaling factor', 'centers'])

        self.rescale_data(rescaling_factor=rescaling_factor, centers=centers)

    def __repr__(self):
        return self.df.T.to_string()

    def _repr_html_(self):
        return self.df.T.to_html()

    @_setdoc_pro([ds.centers, ds.rescaling_factor])
    def modify_rescaling_parameters(self, attribute, value):
        """
        Modify the parameters used to rescale data

        Args:
            attribute (str): Attribute to be modified. It can be: centers, rescaling factor
                * centers: [s0]
                * rescaling factor: [s1]
            value (float, list[float])


        Returns:
            :class:`gempy_lite.core.kernel_data.geometric_data.Rescaling`

        """
        assert np.isin(attribute, self.df.columns).all(), 'Valid attributes are: ' + np.array2string(self.df.columns)

        if attribute == 'centers':
            try:
                assert value.shape[0] == 3

                self.df.loc['values', attribute] = value

            except AssertionError:
                print('centers length must be 3: XYZ')

        else:
            self.df.loc['values', attribute] = value

        return self

    @_setdoc_pro([ds.centers, ds.rescaling_factor])
    def rescale_data(self, rescaling_factor=None, centers=None):
        """
        Rescale inplace: surface_points, orientations---adding columns in the categories_df---and grid---adding values_r
        attributes. The rescaled values will get stored on the linked objects.

        Args:
            rescaling_factor: [s1]
            centers: [s0]

        Returns:

        """
        max_coord, min_coord = self.max_min_coord(self.surface_points, self.orientations)
        if rescaling_factor is None:
            self.df['rescaling factor'] = self.compute_rescaling_factor(self.surface_points, self.orientations,
                                                                        max_coord, min_coord)
        else:
            self.df['rescaling factor'] = rescaling_factor
        if centers is None:
            self.df.at['values', 'centers'] = self.compute_data_center(self.surface_points, self.orientations,
                                                                       max_coord, min_coord)
        else:
            self.df.at['values', 'centers'] = centers

        self.set_rescaled_surface_points()
        self.set_rescaled_orientations()
        self.set_rescaled_grid()
        return True

    def get_rescaled_surface_points(self):
        """
        Get the rescaled coordinates. return an image of the interface and orientations categories_df with the X_r..
         columns

        Returns:
            :attr:`SurfacePoints.df[['X_r', 'Y_r', 'Z_r']]`
        """
        return self.surface_points.df[['X_r', 'Y_r', 'Z_r']],

    def get_rescaled_orientations(self):
        """
        Get the rescaled coordinates. return an image of the interface and orientations categories_df with the X_r..
         columns.

        Returns:
            :attr:`Orientations.df[['X_r', 'Y_r', 'Z_r']]`
        """
        return self.orientations.df[['X_r', 'Y_r', 'Z_r']]

    @staticmethod
    @_setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__])
    def max_min_coord(surface_points=None, orientations=None):
        """
        Find the maximum and minimum location of any input data in each cartesian coordinate

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            orientations (:class:`Orientations`): [s1]

        Returns:
            tuple: max[XYZ], min[XYZ]
        """
        if surface_points is None:
            if orientations is None:
                raise AttributeError('You must pass at least one Data object')
            else:
                df = orientations.df
        else:
            if orientations is None:
                df = surface_points.df
            else:
                df = pn.concat([orientations.df, surface_points.df], sort=False)

        max_coord = df.max()[['X', 'Y', 'Z']]
        min_coord = df.min()[['X', 'Y', 'Z']]
        return max_coord, min_coord

    @_setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, ds.centers])
    def compute_data_center(self, surface_points=None, orientations=None,
                            max_coord=None, min_coord=None, inplace=True):
        """
        Calculate the center of the data once it is shifted between 0 and 1.

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            orientations (:class:`Orientations`): [s1]
            max_coord (float): Max XYZ coordinates of all GeometricData
            min_coord (float): Min XYZ coordinates of all GeometricData
            inplace (bool): if True modify the self.df rescaling factor attribute

        Returns:
            np.array: [s2]
        """

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(surface_points, orientations)

        # Get the centers of every axis
        centers = ((max_coord + min_coord) / 2).astype(float).values
        if inplace is True:
            self.df.at['values', 'centers'] = centers
        return centers

    @_setdoc_pro([SurfacePoints.__doc__, Orientations.__doc__, ds.rescaling_factor])
    def compute_rescaling_factor(self, surface_points=None, orientations=None,
                                 max_coord=None, min_coord=None, inplace=True):
        """
        Calculate the rescaling factor of the data to keep all coordinates between 0 and 1

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            orientations (:class:`Orientations`): [s1]
            max_coord (float): Max XYZ coordinates of all GeometricData
            min_coord (float): Min XYZ coordinates of all GeometricData
            inplace (bool): if True modify the self.df rescaling factor attribute

        Returns:
            float: [s2]
        """

        if max_coord is None or min_coord is None:
            max_coord, min_coord = self.max_min_coord(surface_points, orientations)
        rescaling_factor_val = (2 * np.max(max_coord - min_coord))
        if inplace is True:
            self.df['rescaling factor'] = rescaling_factor_val
        return rescaling_factor_val

    # def update_rescaling_factor(self, surface_points=None, orientations=None,
    #                             max_coord=None, min_coord=None):
    #     self.compute_rescaling_factor(surface_points, orientations, max_coord, min_coord, inplace=True)

    @staticmethod
    @_setdoc_pro([SurfacePoints.__doc__, compute_data_center.__doc__, compute_rescaling_factor.__doc__, ds.idx_sp])
    def rescale_surface_points(surface_points, rescaling_factor, centers, idx: list = None):
        """
        Rescale inplace: surface_points. The rescaled values will get stored on the linked objects.

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            rescaling_factor: [s2]
            centers: [s1]
            idx (int, list of int): [s3]

        Returns:

        """

        if idx is None:
            idx = surface_points.df.index

        # Change the coordinates of surface_points
        new_coord_surface_points = (surface_points.df.loc[idx, ['X', 'Y', 'Z']] -
                                    centers) / rescaling_factor + 0.5001

        new_coord_surface_points.rename(columns={"X": "X_r", "Y": "Y_r", "Z": 'Z_r'}, inplace=True)
        return new_coord_surface_points

    @_setdoc_pro(ds.idx_sp)
    def set_rescaled_surface_points(self, idx: Union[list, np.ndarray] = None):
        """
        Set the rescaled coordinates into the surface_points categories_df

        Args:
            idx (int, list of int): [s0]

        Returns:

        """
        if idx is None:
            idx = self.surface_points.df.index
        idx = np.atleast_1d(idx)

        self.surface_points.df.loc[idx, ['X_r', 'Y_r', 'Z_r']] = self.rescale_surface_points(
            self.surface_points, self.df.loc['values', 'rescaling factor'], self.df.loc['values', 'centers'], idx=idx)

        return self.surface_points

    def rescale_data_point(self, data_points: np.ndarray, rescaling_factor=None, centers=None):
        """This method now is very similar to set_rescaled_surface_points passing an index"""
        if rescaling_factor is None:
            rescaling_factor = self.df.loc['values', 'rescaling factor']
        if centers is None:
            centers = self.df.loc['values', 'centers']

        rescaled_data_point = (data_points - centers) / rescaling_factor + 0.5001

        return rescaled_data_point

    @staticmethod
    @_setdoc_pro([Orientations.__doc__, compute_data_center.__doc__, compute_rescaling_factor.__doc__, ds.idx_sp])
    def rescale_orientations(orientations, rescaling_factor, centers, idx: list = None):
        """
        Rescale inplace: surface_points. The rescaled values will get stored on the linked objects.

        Args:
            orientations (:class:`Orientations`): [s0]
            rescaling_factor: [s2]
            centers: [s1]
            idx (int, list of int): [s3]

        Returns:

        """
        if idx is None:
            idx = orientations.df.index

        # Change the coordinates of orientations
        new_coord_orientations = (orientations.df.loc[idx, ['X', 'Y', 'Z']] -
                                  centers) / rescaling_factor + 0.5001

        new_coord_orientations.rename(columns={"X": "X_r", "Y": "Y_r", "Z": 'Z_r'}, inplace=True)

        return new_coord_orientations

    @_setdoc_pro(ds.idx_sp)
    def set_rescaled_orientations(self, idx: Union[list, np.ndarray] = None):
        """
        Set the rescaled coordinates into the surface_points categories_df

        Args:
            idx (int, list of int): [s0]

        Returns:

        """
        if idx is None:
            idx = self.orientations.df.index
        idx = np.atleast_1d(idx)

        self.orientations.df.loc[idx, ['X_r', 'Y_r', 'Z_r']] = self.rescale_orientations(
            self.orientations, self.df.loc['values', 'rescaling factor'], self.df.loc['values', 'centers'], idx=idx)
        return True

    @staticmethod
    def rescale_grid(grid, rescaling_factor, centers: pn.DataFrame):
        new_grid_extent = (grid.regular_grid.extent - np.repeat(centers, 2)) / rescaling_factor + 0.5001
        new_grid_values = (grid.values - centers) / rescaling_factor + 0.5001
        return new_grid_extent, new_grid_values,

    def set_rescaled_grid(self):
        """
        Set the rescaled coordinates and extent into a grid object
        """

        self.grid.extent_r, self.grid.values_r = self.rescale_grid(
            self.grid, self.df.loc['values', 'rescaling factor'], self.df.loc['values', 'centers'])

        self.grid.regular_grid.extent_r, self.grid.regular_grid.values_r = self.grid.extent_r, self.grid.values_r


class AdditionalData(object):
    """
    Container class that encapsulate :class:`Structure`, :class:`KrigingParameters`, :class:`Options` and
     rescaling parameters

    Args:
        surface_points (:class:`SurfacePoints`): [s0]
        orientations (:class:`Orientations`): [s1]
        grid (:class:`Grid`): [s2]
        faults (:class:`Faults`): [s4]
        surfaces (:class:`Surfaces`): [s3]
        rescaling (:class:`RescaledData`): [s5]

    Attributes:
        structure_data (:class:`Structure`): [s6]
        options (:class:`gempy_lite.Options`): [s8]
        kriging_data (:class:`Structure`): [s7]
        rescaling_data (:class:`RescaledData`):

    """

    def __init__(self, surface_points, orientations, grid: Grid,
                 faults, surfaces: Surfaces, rescaling):
        self.structure_data = Structure(surface_points, orientations, surfaces, faults)
        self.options = Options()
        self.kriging_data = KrigingParameters(self.structure_data, grid)
        self.rescaling_data = rescaling

    def __repr__(self):
        concat_ = self.get_additional_data()
        return concat_.to_string()

    def _repr_html_(self):
        concat_ = self.get_additional_data()
        return concat_.to_html()

    def get_additional_data(self):
        """
        Concatenate all linked data frames and transpose them for a nice visualization.

        Returns:
            pn.DataFrame: concatenated and transposed dataframe
        """
        concat_ = pn.concat([self.structure_data.df, self.options.df, self.kriging_data.df, self.rescaling_data.df],
                            axis=1, keys=['Structure', 'Options', 'Kriging', 'Rescaling'])
        return concat_.T

    def update_default_kriging(self):
        """
        Update default kriging values.
        """
        self.kriging_data.set_default_range()
        self.kriging_data.set_default_c_o()
        self.kriging_data.set_u_grade()

    def update_structure(self):
        """
        Update fields dependent on input data sucha as structure and universal kriging grade
        """
        warnings.warn('structured is not used anymore', DeprecationWarning)
        pass
        # self.structure_data.update_structure_from_input()
        # if len(self.kriging_data.df.loc['values', 'drift equations']) < \
        #         self.structure_data.df.loc['values', 'number series']:
        #     self.kriging_data.set_u_grade()
import sys
import warnings
from typing import Union

import numpy as np
import pandas as pn

from gempy_lite.core.kernel_data import Surfaces, Stack

from gempy_lite.core.checkers import check_for_nans
from gempy_lite.utils import docstring as ds
from gempy_lite.utils.meta import _setdoc_pro, _setdoc


@_setdoc_pro(Surfaces.__doc__)
class GeometricData(object):
    """
    Parent class of the objects which containing the input parameters: surface_points and orientations. This class
     contain the common methods for both types of data sets.

    Args:
        surfaces (:class:`Surfaces`): [s0]

    Attributes:
        surfaces (:class:`Surfaces`)
        df (:class:`pn.DataFrame`): Pandas DataFrame containing all the properties of each individual data point i.e.
        surface points and orientations
    """

    def __init__(self, surfaces: Surfaces):

        self.surfaces = surfaces
        self.df = pn.DataFrame()

    def __repr__(self):
        c_ = self._columns_rend
        return self.df[c_].to_string()

    def _repr_html_(self):
        c_ = self._columns_rend
        return self.df[c_].to_html()

    def init_dependent_properties(self):
        """Set the defaults values to the columns before gets mapped with the the :class:`Surfaces` attribute. This
        method will get invoked for example when we add a new point."""

        # series
        self.df['Feature'] = 'Default series'
        self.df['Feature'] = self.df['Feature'].astype('category', copy=True)
        self.df['Feature'].cat.set_categories(self.surfaces.df['Feature'].cat.categories, inplace=True)

        # id
        self.df['id'] = np.nan

        # order_series
        self.df['OrderFeature'] = 1

        #
        self.df['isActive'] = True
        return self

    @staticmethod
    @_setdoc(pn.read_csv.__doc__, indent=False)
    def read_data(file_path, **kwargs):
        """"""
        if 'sep' not in kwargs:
            kwargs['sep'] = ','

        table = pn.read_csv(file_path, **kwargs)
        return table

    def sort_table(self):
        """
        First we sort the dataframes by the series age. Then we set a unique number for every surface and resort
        the surfaces. All inplace
        """

        # We order the pandas table by surface (also by series in case something weird happened)
        self.df.sort_values(by=['OrderFeature', 'id'],
                            ascending=True, kind='mergesort',
                            inplace=True)
        return self.df

    @_setdoc_pro(Stack.__doc__)
    def set_series_categories_from_series(self, series):
        """set the series categorical columns with the series index of the passed :class:`Series`

        Args:
            series (:class:`Series`): [s0]
        """
        self.df['Feature'].cat.set_categories(series.df.index, inplace=True)
        return True

    def update_series_category(self):
        """Update the series categorical columns with the series categories of the :class:`Surfaces` attribute."""
        self.df['Feature'].cat.set_categories(self.surfaces.df['Feature'].cat.categories, inplace=True)

        return True

    @_setdoc_pro(Surfaces.__doc__)
    def set_surface_categories_from_surfaces(self, surfaces: Surfaces):
        """set the series categorical columns with the series index of the passed :class:`Series`.

        Args:
            surfaces (:class:`Surfaces`): [s0]

        """

        self.df['Surface'].cat.set_categories(surfaces.df['Surface'], inplace=True)
        return True

    @_setdoc_pro(Stack.__doc__)
    def map_data_from_series(self, series, attribute: str, idx=None):
        """
        Map columns from the :class:`Series` data frame to a :class:`GeometricData` data frame.

        Args:
            series (:class:`Series`): [s0]
            attribute (str): column to be mapped from the :class:`Series` to the :class:`GeometricData`.
            idx (Optional[int, list[int]): If passed, list of indices of the :class:`GeometricData` that will be mapped.

        Returns:
            :class:GeometricData
        """
        if idx is None:
            idx = self.df.index

        idx = np.atleast_1d(idx)
        if attribute in ['id', 'OrderFeature']:
            self.df.loc[idx, attribute] = self.df['Feature'].map(series.df[attribute]).astype(int)

        else:
            self.df.loc[idx, attribute] = self.df['Feature'].map(series.df[attribute])

        if type(self.df['OrderFeature'].dtype) is pn.CategoricalDtype:

            self.df['OrderFeature'].cat.remove_unused_categories(inplace=True)
        return self

    @_setdoc_pro(Surfaces.__doc__)
    def map_data_from_surfaces(self, surfaces, attribute: str, idx=None):
        """
        Map columns from the :class:`Series` data frame to a :class:`GeometricData` data frame.
        Properties of surfaces: series, id, values.

        Args:
            surfaces (:class:`Surfaces`): [s0]
            attribute (str): column to be mapped from the :class:`Series` to the :class:`GeometricData`.
            idx (Optional[int, list[int]): If passed, list of indices of the :class:`GeometricData` that will be mapped.

        Returns:
            :class:GeometricData
        """

        if idx is None:
            idx = self.df.index
        idx = np.atleast_1d(idx)
        if attribute == 'Feature':
            if surfaces.df.loc[~surfaces.df['isBasement']]['Feature'].isna().sum() != 0:
                raise AttributeError('Surfaces does not have the correspondent series assigned. See'
                                     'Surfaces.map_series_from_series.')
            self.df.loc[idx, attribute] = self.df.loc[idx, 'Surface'].map(surfaces.df.set_index('Surface')[attribute])

        elif attribute in ['id', 'OrderFeature']:
            self.df.loc[idx, attribute] = (self.df.loc[idx, 'Surface'].map(surfaces.df.set_index('Surface')[attribute])).astype(int)
        else:

            self.df.loc[idx, attribute] = self.df.loc[idx, 'Surface'].map(surfaces.df.set_index('Surface')[attribute])


@_setdoc_pro([Surfaces.__doc__, ds.coord, ds.surface_sp])
class SurfacePoints(GeometricData):
    """
    Data child with specific methods to manipulate interface data. It is initialize without arguments to give
    flexibility to the origin of the data.

    Args:
        surfaces (:class:`Surfaces`): [s0]
        coord (np.ndarray): [s1]
        surface (list[str]): [s2]


    Attributes:
          df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the necessary information respect
          the surface points of the model
    """

    def __init__(self, surfaces: Surfaces, coord=None, surface=None):

        super().__init__(surfaces)
        self._columns_i_all = ['X', 'Y', 'Z', 'Surface', 'Feature', 'X_std', 'Y_std', 'Z_std',
                               'OrderFeature', 'surface_number']

        self._columns_i_1 = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'Surface', 'Feature', 'id',
                             'OrderFeature', 'isFault', 'smooth']

        self._columns_rep = ['X', 'Y', 'Z', 'Surface', 'Feature']
        self._columns_i_num = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r']
        self._columns_rend = ['X', 'Y', 'Z', 'smooth', 'Surface']

        self._private_attr = ['X_r', 'Y_r', 'Z_r', 'Features', 'id', 'OrderFeature']
        self._public_attr = ['X', 'Y', 'Z', 'smooth', 'Surface', 'isActive']

        if (np.array(sys.version_info[:2]) <= np.array([3, 6])).all():
            self.df: pn.DataFrame

        self.set_surface_points(coord, surface)

    @property
    def n_sp_per_feature(self):

        """
        Set the length of each **series** on `SurfacePoints` i.e. how many data points are for each series. Also
        sets the number of series itself.

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored

        """
        #len_series = np.zeros(self.surfaces.stack.n_features, dtype=int)
        # Array containing the size of every series. SurfacePoints.
        points_count = self.df['OrderFeature'].value_counts(sort=False)
        len_series_i = np.zeros(self.surfaces.stack.n_features, dtype=int)
        len_series_i[points_count.index - 1] = points_count.values

        return len_series_i

    @_setdoc_pro([ds.coord, ds.surface_sp])
    def set_surface_points(self, coord: np.ndarray = None, surface: list = None):
        """
        Set coordinates and surface columns on the df.

        Args:
            coord (np.ndarray): [s0]
            surface (list[str]): [s1]

        Returns:
            :class:`SurfacePoints`
        """
        self.df = pn.DataFrame(columns=['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'Surface'], dtype=float)

        if coord is not None and surface is not None:
            self.df[['X', 'Y', 'Z']] = pn.DataFrame(coord)
            self.df['Surface'] = surface

        self.df['Surface'] = self.df['Surface'].astype('category', copy=True)
        self.df['Surface'].cat.set_categories(self.surfaces.df['Surface'].values, inplace=True)

        # Choose types
        self.init_dependent_properties()

        # Add nugget columns
        self.df['smooth'] = 2e-6

        assert ~self.df['Surface'].isna().any(), 'Some of the surface passed does not exist in the Formation' \
                                                 'object. %s' % self.df['Surface'][self.df['Surface'].isna()]

        return self

    @_setdoc_pro([ds.x, ds.y, ds.z, ds.surface_sp, ds.idx_sp])
    def add_surface_points(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray], z: Union[float, np.ndarray],
                           surface: Union[list, np.ndarray], idx: Union[int, list, np.ndarray] = None):
        """
        Add surface points.

        Args:
            x (float, np.ndarray): [s0]
            y (float, np.ndarray): [s1]
            z (float, np.ndarray): [s2]
            surface (list[str]): [s3]
            idx (Optional[int, list[int]): [s4]

        Returns:
           :class:`gempy_lite.core.kernel_data.geometric_data.SurfacePoints`

        """
        max_idx = self.df.index.max()

        if idx is None:
            idx = max_idx
            if idx is np.nan:
                idx = 0
            else:
                idx += 1

        if max_idx is not np.nan:
            self.df.loc[idx] = self.df.loc[max_idx]

        coord_array = np.array([x, y, z])
        assert coord_array.ndim == 1, 'Adding an interface only works one by one.'

        try:
            if self.surfaces.df.groupby('isBasement').get_group(True)['Surface'].isin(surface).any():
                warnings.warn('Surface Points for the basement will not be used. Maybe you are missing an extra'
                              'layer at the bottom of the pile.')

            self.df.loc[idx, ['X', 'Y', 'Z']] = coord_array.astype('float64')
            self.df.loc[idx, 'Surface'] = surface
        # ToDO test this
        except ValueError as error:
            self.del_surface_points(idx)
            print('The surface passed does not exist in the pandas categories. This may imply that'
                  'does not exist in the surface object either.')
            raise ValueError(error)

        self.df.loc[idx, ['smooth']] = 1e-6

        self.df['Surface'] = self.df['Surface'].astype('category', copy=True)
        self.df['Surface'].cat.set_categories(self.surfaces.df['Surface'].values, inplace=True)

        self.df['Feature'] = self.df['Feature'].astype('category', copy=True)
        self.df['Feature'].cat.set_categories(self.surfaces.df['Feature'].cat.categories, inplace=True)

        self.map_data_from_surfaces(self.surfaces, 'Feature', idx=idx)
        self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
        self.map_data_from_series(self.surfaces.stack, 'OrderFeature', idx=idx)

        self.sort_table()
        return self, idx

    @_setdoc_pro([ds.idx_sp])
    def del_surface_points(self, idx: Union[int, list, np.ndarray]):
        """Delete surface points.

        Args:
            idx (int, list[int]): [s0]

        Returns:
            :class:`gempy_lite.core.kernel_data.geometric_data.SurfacePoints`

        """
        self.df.drop(idx, inplace=True)
        return self

    @_setdoc_pro([ds.idx_sp, ds.x, ds.y, ds.z, ds.surface_sp])
    def modify_surface_points(self, idx: Union[int, list, np.ndarray], **kwargs):
        """Allows modification of the x,y and/or z-coordinates of an interface at specified dataframe index.

         Args:
             idx (int, list, np.ndarray): [s0]
             **kwargs:
                * X: [s1]
                * Y: [s2]
                * Z: [s3]
                * surface: [s4]

         Returns:
            :class:`gempy_lite.core.kernel_data.geometric_data.SurfacePoints`

         """
        idx = np.array(idx, ndmin=1)
        try:
            surface_names = kwargs.pop('Surface')
            self.df.loc[idx, ['Surface']] = surface_names
            self.map_data_from_surfaces(self.surfaces, 'Feature', idx=idx)
            self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
            self.map_data_from_series(self.surfaces.stack, 'OrderFeature', idx=idx)
            self.sort_table()
        except KeyError:
            pass

        # keys = list(kwargs.keys())
    #    is_surface = np.isin('Surface', keys).all()

        # Check idx exist in the df
        assert np.isin(np.atleast_1d(idx), self.df.index).all(), 'Indices must exist in the' \
                                                                 ' dataframe to be modified.'

        # Check the properties are valid
        assert np.isin(list(kwargs.keys()), ['X', 'Y', 'Z', 'Surface', 'smooth']).all(),\
            'Properties must be one or more of the following: \'X\', \'Y\', \'Z\', ' '\'surface\''
        # stack properties values
        values = np.array(list(kwargs.values()))

        # If we pass multiple index we need to transpose the numpy array
        if type(idx) is list or type(idx) is np.ndarray:
            values = values.T

        # Selecting the properties passed to be modified
        if values.shape[0] == 1:
            values = np.repeat(values, idx.shape[0])

        self.df.loc[idx, list(kwargs.keys())] = values

        return self

    @_setdoc_pro([ds.file_path, ds.debug, ds.inplace])
    def read_surface_points(self, table_source, debug=False, inplace=False,
                            kwargs_pandas: dict = None, **kwargs, ):
        """
        Read tabular using pandas tools and if inplace set it properly to the surface points object.

        Parameters:
            table_source (str, path object, file-like object or direct pandas data frame): [s0]
            debug (bool): [s1]
            inplace (bool): [s2]
            kwargs_pandas: kwargs for the panda function :func:`pn.read_csv`
            **kwargs:
                * update_surfaces (bool): If True add to the linked `Surfaces` object unique surface names read on
                  the csv file
                * coord_x_name (str): Name of the header on the csv for this attribute, e.g for coord_x. Default X
                * coord_y_name (str): Name of the header on the csv for this attribute. Default Y.
                * coord_z_name (str): Name of the header on the csv for this attribute. Default Z.
                * surface_name (str): Name of the header on the csv for this attribute. Default formation

        Returns:

        See Also:
            :meth:`GeometricData.read_data`
        """
        # TODO read by default either formation or surface

        if 'sep' not in kwargs:
            kwargs['sep'] = ','

        coord_x_name = kwargs.get('coord_x_name', "X")
        coord_y_name = kwargs.get('coord_y_name', "Y")
        coord_z_name = kwargs.get('coord_z_name', "Z")
        surface_name = kwargs.get('surface_name', "formation")

        if kwargs_pandas is None:
            kwargs_pandas = {}

        if 'sep' not in kwargs_pandas:
            kwargs_pandas['sep'] = ','

        if isinstance(table_source, pn.DataFrame):
            table = table_source
        else:
            table = pn.read_csv(table_source, **kwargs_pandas)

        if 'update_surfaces' in kwargs:
            if kwargs['update_surfaces'] is True:
                self.surfaces.add_surface(table[surface_name].unique())

        if debug is True:
            print('Debugging activated. Changes won\'t be saved.')
            return table
        else:
            assert {coord_x_name, coord_y_name, coord_z_name, surface_name}.issubset(table.columns), \
                "One or more columns do not match with the expected values " + str(table.columns)

            if inplace:
                c = np.array(self._columns_i_1)
                surface_points_read = table.assign(**dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
                self.set_surface_points(surface_points_read[[coord_x_name, coord_y_name, coord_z_name]],
                                        surface=surface_points_read[surface_name])
            else:
                return table

    def set_default_surface_points(self):
        """
        Set a default point at the middle of the extent area to be able to start making the model
        """
        if self.df.shape[0] == 0:
            self.add_surface_points(0.00001, 0.00001, 0.00001, self.surfaces.df['Surface'].iloc[0])
        return True

    def update_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:
            :class:`SurfacePoints`
        """
        point_num = self.df.groupby('id').cumcount()
        point_l = [r'${\bf{x}}_{\alpha \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                   for p, f in zip(point_num, self.df['id'])]

        self.df['annotations'] = point_l
        return self


@_setdoc_pro([Surfaces.__doc__, ds.coord_ori, ds.surface_sp, ds.pole_vector, ds.orientations])
class Orientations(GeometricData):
    """
    Data child with specific methods to manipulate orientation data. It is initialize without arguments to give
    flexibility to the origin of the data.

    Args:
        surfaces (:class:`Surfaces`): [s0]
        coord (np.ndarray): [s1]
        pole_vector (np.ndarray): [s3]
        orientation (np.ndarray): [s4]
        surface (list[str]): [s2]
    Attributes:
        df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the necessary information respect
         the orientations of the model
    """

    def __init__(self, surfaces: Surfaces, coord=None, pole_vector=None, orientation=None, surface=None):
        super().__init__(surfaces)
        self._columns_o_all = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity',
                               'Surface', 'Feature', 'id', 'OrderFeature', 'surface_number']
        self._columns_o_1 = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity',
                             'Surface', 'Feature', 'id', 'OrderFeature', 'isFault']
        self._columns_o_num = ['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r',
                               'G_x', 'G_y', 'G_z', 'dip', 'azimuth', 'polarity']
        self._columns_rend = ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'smooth', 'Surface']

        self._private_attr = ['X_r', 'Y_r', 'Z_r', 'Features', 'id', 'OrderFeature']
        self._public_attr = ['X', 'Y', 'Z', 'smooth', 'Surface', 'dip',
                             'azimuth', 'polarity', 'isActive']

        if (np.array(sys.version_info[:2]) <= np.array([3, 6])).all():
            self.df: pn.DataFrame

        self.set_orientations(coord, pole_vector, orientation, surface)

    @property
    def n_orientations_per_feature(self):

        """
        Set the length of each **series** on `Orientations` i.e. how many orientations are for each series.

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored

        """
        # Array containing the size of every series. orientations.

        len_series_o = np.zeros(self.surfaces.stack.n_features, dtype=int)
        ori_count = self.df['OrderFeature'].value_counts(sort=False)
        len_series_o[ori_count.index - 1] = ori_count.values

        return len_series_o


    @_setdoc_pro([ds.coord_ori, ds.surface_sp, ds.pole_vector, ds.orientations])
    def set_orientations(self, coord: np.ndarray = None, pole_vector: np.ndarray = None,
                         orientation: np.ndarray = None, surface: list = None):
        """
        Set coordinates, surface and orientation data.

        If both are passed pole vector has priority over orientation

        Args:
            coord (np.ndarray): [s0]
            pole_vector (np.ndarray): [s2]
            orientation (np.ndarray): [s3]
            surface (list[str]): [s1]

        Returns:

        """
        self.df = pn.DataFrame(columns=['X', 'Y', 'Z', 'X_r', 'Y_r', 'Z_r', 'G_x', 'G_y', 'G_z', 'dip',
                                        'azimuth', 'polarity', 'Surface'], dtype=float)

        self.df['Surface'] = self.df['Surface'].astype('category', copy=True)
        self.df['Surface'].cat.set_categories(self.surfaces.df['Surface'].values, inplace=True)

        pole_vector = check_for_nans(pole_vector)
        orientation = check_for_nans(orientation)

        if coord is not None and ((pole_vector is not None) or (orientation is not None)) and surface is not None:

            self.df[['X', 'Y', 'Z']] = pn.DataFrame(coord)
            self.df['Surface'] = surface
            if pole_vector is not None:
                self.df['G_x'] = pole_vector[:, 0]
                self.df['G_y'] = pole_vector[:, 1]
                self.df['G_z'] = pole_vector[:, 2]
                self.calculate_orientations()

                if orientation is not None:
                    warnings.warn('If pole_vector and orientation are passed pole_vector is used/')
            else:
                if orientation is not None:
                    self.df['azimuth'] = orientation[:, 0]
                    self.df['dip'] = orientation[:, 1]
                    self.df['polarity'] = orientation[:, 2]
                    self.calculate_gradient()
                else:
                    raise AttributeError('At least pole_vector or orientation should have been passed to reach'
                                         'this point. Check previous condition')

        self.df['Surface'] = self.df['Surface'].astype('category', copy=True)
        self.df['Surface'].cat.set_categories(self.surfaces.df['Surface'].values, inplace=True)

        self.init_dependent_properties()

        # Add nugget effect
        self.df['smooth'] = 0.01
        assert ~self.df['Surface'].isna().any(), 'Some of the surface passed does not exist in the Formation' \
                                                 'object. %s' % self.df['Surface'][self.df['Surface'].isna()]

    @_setdoc_pro([ds.x, ds.y, ds.z, ds.surface_sp, ds.pole_vector, ds.orientations, ds.idx_sp])
    def add_orientation(self, x, y, z, surface, pole_vector: Union[list, tuple, np.ndarray] = None,
                        orientation: Union[list, np.ndarray] = None, idx=None):
        """
        Add orientation.

        Args:
            x (float, np.ndarray): [s0]
            y (float, np.ndarray): [s1]
            z (float, np.ndarray): [s2]
            surface (list[str], str): [s3]
            pole_vector (np.ndarray): [s4]
            orientation (np.ndarray): [s5]
            idx (Optional[int, list[int]): [s6]

        Returns:
            Orientations
        """
        if pole_vector is None and orientation is None:
            raise AttributeError('Either pole_vector or orientation must have a value. If both are passed pole_vector'
                                 'has preference')

        max_idx = self.df.index.max()

        if idx is None:
            idx = max_idx
            if idx is np.nan:
                idx = 0
            else:
                idx += 1

        if max_idx is not np.nan:
            self.df.loc[idx] = self.df.loc[max_idx]

        if pole_vector is not None:
            self.df.loc[idx, ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z']] = np.array([x, y, z, *pole_vector], dtype=float)
            self.df.loc[idx, 'Surface'] = surface

            self.calculate_orientations(idx)

            if orientation is not None:
                warnings.warn('If pole_vector and orientation are passed pole_vector is used/')
        else:
            if orientation is not None:
                self.df.loc[idx, ['X', 'Y', 'Z', ]] = np.array([x, y, z], dtype=float)
                self.df.loc[idx, ['azimuth', 'dip', 'polarity']] = np.array(orientation, dtype=float)
                self.df.loc[idx, 'Surface'] = surface

                self.calculate_gradient(idx)
            else:
                raise AttributeError('At least pole_vector or orientation should have been passed to reach'
                                     'this point. Check previous condition')
        self.df.loc[idx, ['smooth']] = 0.01
        self.df['Surface'] = self.df['Surface'].astype('category', copy=True)
        self.df['Surface'].cat.set_categories(self.surfaces.df['Surface'].values, inplace=True)

        self.df['Feature'] = self.df['Feature'].astype('category', copy=True)
        self.df['Feature'].cat.set_categories(self.surfaces.df['Feature'].cat.categories, inplace=True)

        self.map_data_from_surfaces(self.surfaces, 'Feature', idx=idx)
        self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
        self.map_data_from_series(self.surfaces.stack, 'OrderFeature', idx=idx)

        self.sort_table()
        return self

    @_setdoc_pro()
    def del_orientation(self, idx):
        """Delete orientation

        Args:
            idx: [s_idx_sp]

        Returns:
            :class:`gempy_lite.core.kernel_data.geometric_data.Orientations`

        """
        self.df.drop(idx, inplace=True)

    @_setdoc_pro([ds.idx_sp, ds.surface_sp])
    def modify_orientations(self, idx, **kwargs):
        """Allows modification of any of an orientation column at a given index.

        Args:
            idx (int, list[int]): [s0]

        Keyword Args:
                * X
                * Y
                * Z
                * G_x
                * G_y
                * G_z
                * dip
                * azimuth
                * polarity
                * surface (str): [s1]

         Returns:
            :class:`gempy_lite.core.kernel_data.geometric_data.Orientations`

         """

        idx = np.array(idx, ndmin=1)
        try:
            surface_names = kwargs.pop('Surface')
            self.df.loc[idx, ['Surface']] = surface_names
            self.map_data_from_surfaces(self.surfaces, 'Feature', idx=idx)
            self.map_data_from_surfaces(self.surfaces, 'id', idx=idx)
            self.map_data_from_series(self.surfaces.stack, 'OrderFeature', idx=idx)
            self.sort_table()
        except KeyError:
            pass

        # TODO Dep
        keys = list(kwargs.keys())

        # Check idx exist in the df
        assert np.isin(np.atleast_1d(idx), self.df.index).all(), 'Indices must exist in the dataframe to be modified.'

        # Check the properties are valid
        assert np.isin(list(kwargs.keys()), ['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z', 'dip',
                                             'azimuth', 'polarity', 'Surface', 'smooth']).all(),\
            'Properties must be one or more of the following: \'X\', \'Y\', \'Z\', \'G_x\', \'G_y\', \'G_z\', \'dip,\''\
            '\'azimuth\', \'polarity\', \'surface\''

        # stack properties values
        values = np.atleast_1d(list(kwargs.values()))

        # If we pass multiple index we need to transpose the numpy array
        if type(idx) is list or type(idx) is np.ndarray:
            values = values.T

        if values.shape[0] == 1:
            values = np.repeat(values, idx.shape[0])

        # Selecting the properties passed to be modified
        self.df.loc[idx, list(kwargs.keys())] = values.astype('float64')

        if np.isin(list(kwargs.keys()), ['G_x', 'G_y', 'G_z']).any():
            self.calculate_orientations(idx)
        else:
            if np.isin(list(kwargs.keys()), ['azimuth', 'dip', 'polarity']).any():
                self.calculate_gradient(idx)
        return self

    def calculate_gradient(self, idx=None):
        """
        Calculate the gradient vector of module 1 given dip and azimuth to be able to plot the orientations
        """
        if idx is None:
            self.df['G_x'] = np.sin(np.deg2rad(self.df["dip"].astype('float'))) * \
                np.sin(np.deg2rad(self.df["azimuth"].astype('float'))) * \
                self.df["polarity"].astype('float') + 1e-12
            self.df['G_y'] = np.sin(np.deg2rad(self.df["dip"].astype('float'))) * \
                np.cos(np.deg2rad(self.df["azimuth"].astype('float'))) * \
                self.df["polarity"].astype('float') + 1e-12
            self.df['G_z'] = np.cos(np.deg2rad(self.df["dip"].astype('float'))) * \
                self.df["polarity"].astype('float') + 1e-12
        else:
            self.df.loc[idx, 'G_x'] = np.sin(np.deg2rad(self.df.loc[idx, "dip"].astype('float'))) * \
                                      np.sin(np.deg2rad(self.df.loc[idx, "azimuth"].astype('float'))) * \
                                      self.df.loc[idx, "polarity"].astype('float') + 1e-12
            self.df.loc[idx, 'G_y'] = np.sin(np.deg2rad(self.df.loc[idx, "dip"].astype('float'))) * \
                np.cos(np.deg2rad(self.df.loc[idx, "azimuth"].astype('float'))) * \
                self.df.loc[idx, "polarity"].astype('float') + 1e-12
            self.df.loc[idx, 'G_z'] = np.cos(np.deg2rad(self.df.loc[idx, "dip"].astype('float'))) * \
                self.df.loc[idx, "polarity"].astype('float') + 1e-12
        return True

    def calculate_orientations(self, idx=None):
        """
        Calculate and update the orientation data (azimuth and dip) from gradients in the data frame.

        Authors: Elisa Heim, Miguel de la Varga
        """
        if idx is None:
            self.df['polarity'] = 1
            self.df["dip"] = np.rad2deg(np.nan_to_num(np.arccos(self.df["G_z"] / self.df["polarity"])))

            self.df["azimuth"] = np.rad2deg(np.nan_to_num(np.arctan2(self.df["G_x"] / self.df["polarity"],
                                                                     self.df["G_y"] / self.df["polarity"])))
            self.df["azimuth"][self.df["azimuth"] < 0] += 360  # shift values from [-pi, 0] to [pi,2*pi]
            self.df["azimuth"][self.df["dip"] < 0.001] = 0  # because if dip is zero azimuth is undefined

        else:

            self.df.loc[idx, 'polarity'] = 1
            self.df.loc[idx, "dip"] = np.rad2deg(np.nan_to_num(np.arccos(self.df.loc[idx, "G_z"] /
                                                                         self.df.loc[idx, "polarity"])))

            self.df.loc[idx, "azimuth"] = np.rad2deg(np.nan_to_num(
                np.arctan2(self.df.loc[idx, "G_x"] / self.df.loc[idx, "polarity"],
                           self.df.loc[idx, "G_y"] / self.df.loc[idx, "polarity"])))

            self.df["azimuth"][self.df["azimuth"] < 0] += 360  # shift values from [-pi, 0] to [pi,2*pi]
            self.df["azimuth"][self.df["dip"] < 0.001] = 0  # because if dip is zero azimuth is undefined

        return True

    @_setdoc_pro([SurfacePoints.__doc__])
    def create_orientation_from_surface_points(self, surface_points: SurfacePoints, indices):
        # TODO test!!!!
        """
        Create and set orientations from at least 3 points categories_df

        Args:
            surface_points (:class:`SurfacePoints`): [s0]
            indices (list[int]): indices of the surface point used to generate the orientation. At least
             3 independent points will need to be passed.
        """
        selected_points = surface_points.df[['X', 'Y', 'Z']].loc[indices].values.T

        center, normal = self.plane_fit(selected_points)
        orientation = self.get_orientation(normal)

        return np.array([*center, *orientation, *normal])

    def set_default_orientation(self):
        """
        Set a default point at the middle of the extent area to be able to start making the model
        """
        if self.df.shape[0] == 0:
            self.add_orientation(.00001, .00001, .00001,
                                 self.surfaces.df['Surface'].iloc[0],
                                 [0, 0, 1],
                                 )

    @_setdoc_pro([ds.file_path, ds.debug, ds.inplace])
    def read_orientations(self, table_source, debug=False, inplace=True, kwargs_pandas: dict = None, **kwargs):
        """
        Read tabular using pandas tools and if inplace set it properly to the surface points object.

        Args:
            table_source (str, path object, file-like object, or direct data frame): [s0]
            debug (bool): [s1]
            inplace (bool): [s2]
            kwargs_pandas: kwargs for the panda function :func:`pn.read_csv`
            **kwargs:
                * update_surfaces (bool): If True add to the linked `Surfaces` object unique surface names read on
                  the csv file
                * coord_x_name (str): Name of the header on the csv for this attribute, e.g for coord_x. Default X
                * coord_y_name (str): Name of the header on the csv for this attribute. Default Y
                * coord_z_name (str): Name of the header on the csv for this attribute. Default Z
                * coord_x_name (str): Name of the header on the csv for this attribute. Default G_x
                * coord_y_name (str): Name of the header on the csv for this attribute. Default G_y
                * coord_z_name (str): Name of the header on the csv for this attribute. Default G_z
                * azimuth_name (str): Name of the header on the csv for this attribute. Default azimuth
                * dip_name     (str): Name of the header on the csv for this attribute. Default dip
                * polarity_name (str): Name of the header on the csv for this attribute. Default polarity
                * surface_name (str): Name of the header on the csv for this attribute. Default formation


        Returns:

        See Also:
            :meth:`GeometricData.read_data`
        """
        coord_x_name = kwargs.get('coord_x_name', "X")
        coord_y_name = kwargs.get('coord_y_name', "Y")
        coord_z_name = kwargs.get('coord_z_name', "Z")
        g_x_name = kwargs.get('G_x_name', 'G_x')
        g_y_name = kwargs.get('G_y_name', 'G_y')
        g_z_name = kwargs.get('G_z_name', 'G_z')
        azimuth_name = kwargs.get('azimuth_name', 'azimuth')
        dip_name = kwargs.get('dip_name', 'dip')
        polarity_name = kwargs.get('polarity_name', 'polarity')
        surface_name = kwargs.get('surface_name', "formation")

        if kwargs_pandas is None:
            kwargs_pandas = {}

        if 'sep' not in kwargs_pandas:
            kwargs_pandas['sep'] = ','

        if isinstance(table_source, pn.DataFrame):
            table = table_source
        else:
            table = pn.read_csv(table_source, **kwargs_pandas)

        if 'update_surfaces' in kwargs:
            if kwargs['update_surfaces'] is True:
                self.surfaces.add_surface(table[surface_name].unique())

        if debug is True:
            print('Debugging activated. Changes won\'t be saved.')
            return table

        else:
            assert np.logical_or({coord_x_name, coord_y_name, coord_z_name, dip_name, azimuth_name,
                    polarity_name, surface_name}.issubset(table.columns),
                 {coord_x_name, coord_y_name, coord_z_name, g_x_name, g_y_name, g_z_name,
                  polarity_name, surface_name}.issubset(table.columns)), \
                "One or more columns do not match with the expected values, which are: \n" +\
                "- the locations of the measurement points '{}','{}' and '{}' \n".format(coord_x_name,coord_y_name,
                                                                                         coord_z_name)+ \
                "- EITHER '{}' (trend direction indicated by an angle between 0 and 360 with North at 0 AND " \
                "'{}' (inclination angle, measured from horizontal plane downwards, between 0 and 90 degrees) \n".format(
                azimuth_name, dip_name) +\
                "- OR the pole vectors of the orientation in a cartesian system '{}','{}' and '{}' \n".format(g_x_name,
                                                                                                              g_y_name,
                                                                                                              g_z_name)+\
                "- the '{}' of the orientation, can be normal (1) or reversed (-1) \n".format(polarity_name)+\
                "- the name of the surface: '{}'\n".format(surface_name)+\
                "Your headers are "+str(list(table.columns))

            if inplace:
                # self.categories_df[table.columns] = table
                c = np.array(self._columns_o_1)
                orientations_read = table.assign(**dict.fromkeys(c[~np.in1d(c, table.columns)], np.nan))
                self.set_orientations(coord=orientations_read[[coord_x_name, coord_y_name, coord_z_name]],
                                      pole_vector=orientations_read[[g_x_name, g_y_name, g_z_name]].values,
                                      orientation=orientations_read[[azimuth_name, dip_name, polarity_name]].values,
                                      surface=orientations_read[surface_name])
            else:
                return table

    def update_annotations(self):
        """
        Add a column in the Dataframes with latex names for each input_data paramenter.

        Returns:

        """
        orientation_num = self.df.groupby('id').cumcount()
        foli_l = [r'${\bf{x}}_{\beta \,{\bf{' + str(f) + '}},' + str(p) + '}$'
                  for p, f in zip(orientation_num, self.df['id'])]

        self.df['annotations'] = foli_l
        return self

    @staticmethod
    def get_orientation(normal):
        """Get orientation (dip, azimuth, polarity ) for points in all point set"""

        # calculate dip
        dip = np.arccos(normal[2]) / np.pi * 180.

        # calculate dip direction
        # +/+
        if normal[0] >= 0 and normal[1] > 0:
            dip_direction = np.arctan(normal[0] / normal[1]) / np.pi * 180.
        # border cases where arctan not defined:
        elif normal[0] > 0 and normal[1] == 0:
            dip_direction = 90
        elif normal[0] < 0 and normal[1] == 0:
            dip_direction = 270
        # +-/-
        elif normal[1] < 0:
            dip_direction = 180 + np.arctan(normal[0] / normal[1]) / np.pi * 180.
        # -/-
        elif normal[0] < 0 >= normal[1]:
            dip_direction = 360 + np.arctan(normal[0] / normal[1]) / np.pi * 180.
        # if dip is just straight up vertical
        elif normal[0] == 0 and normal[1] == 0:
            dip_direction = 0

        else:
            raise ValueError('The values of normal are not valid.')

        if -90 < dip < 90:
            polarity = 1
        else:
            polarity = -1

        return dip, dip_direction, polarity

    @staticmethod
    def plane_fit(point_list):
        """
        Fit plane to points in PointSet
        Fit an d-dimensional plane to the points in a point set.
        adjusted from: http://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points

        Args:
            point_list (array_like): array of points XYZ

        Returns:
            Return a point, p, on the plane (the point-cloud centroid),
            and the normal, n.
        """

        points = point_list

        from numpy.linalg import svd
        points = np.reshape(points, (np.shape(points)[0], -1))  # Collapse trialing dimensions
        assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                       points.shape[0])
        ctr = points.mean(axis=1)
        x = points - ctr[:, np.newaxis]
        M = np.dot(x, x.T)  # Could also use np.cov(x) here.

        # ctr = Point(x=ctr[0], y=ctr[1], z=ctr[2], type='utm', zone=self.points[0].zone)
        normal = svd(M)[0][:, -1]
        # return ctr, svd(M)[0][:, -1]
        if normal[2] < 0:
            normal = - normal

        return ctr, normal



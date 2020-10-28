import re
import sys
import warnings
from typing import Union, List

import numpy as np
import pandas as pn
import seaborn as sns

from gempy_lite.utils.meta import _setdoc_pro
from .stack import Stack


class Surfaces(object):
    """
    Class that contains the surfaces of the model and the values of each of them.

    Args:
        surface_names (list or np.ndarray): list containing the names of the surfaces
        series (:class:`Series`): [s0]
        values_array (np.ndarray): 2D array with the values of each surface
        properties names (list or np.ndarray): list containing the names of each properties

    Attributes:
        df (:class:`pn.core.frame.DataFrames`): Pandas data frame containing the surfaces names mapped to series and
            the value used for each voxel in the final model.
        series (:class:`Series`)
        colors (:class:`Colors`)

    """

    def __repr__(self):
        return self.df[self._public_attr].to_string()

    def _repr_html_(self):
        return self.df[self._public_attr].style.applymap(self.background_color,
                                                         subset=['color']).render()

    def __init__(self, series, surface_names=None, values_array=None, properties_names=None):

        # Other data objects
        self.stack = series

        # Dataframe views
        self._columns = ['Surface', 'Feature', 'OrderSurface',
                         'isBasement', 'isFault', 'isActive', 'hasData', 'color',
                         'vertices', 'edges', 'sfai', 'id']

        #self.properties_val = []
        self._private_attr = [
            'vertices', 'edges', 'sfai', 'isBasement', 'hasData', 'isFault']

        # ----

        # Init df
        df_ = pn.DataFrame(columns=self._columns)
        self.df = df_.astype({'Surface': str, 'Feature': 'category',
                              'OrderSurface': int,
                              'isBasement': bool, 'isFault': bool, 'isActive': bool, 'hasData': bool,
                              'color': bool, 'id': int, 'vertices': object, 'edges': object})

        self.df['Feature'].cat.add_categories(['Default series'], inplace=True)

        # Set initial values
        if surface_names is not None:
            self.set_surfaces_names(surface_names)

        if values_array is not None:
            self.set_surfaces_values(values_array=values_array,
                                     properties_names=properties_names)

        # Initialize aux objects
        self.colors = Colors(self)

    @property
    def _public_attr(self):
        """Properties values are arbitrary given by the user. e.g. porosity"""
        fixed = ['Surface', 'Feature', 'OrderSurface', 'isActive', 'color', 'id', *self.properties_val]
        return fixed

    @property
    def properties_val(self):
        all_col = self.df.columns
        return all_col.drop(self._columns)

    @property
    def series(self):
        warnings.warn('Series will be deprecated use stack intead', DeprecationWarning)
        return self.stack

    @property
    def n_surfaces(self):
        return self.df['Surface'].nunique()

    @property
    def n_surfaces_per_feature(self):
        """
        Set number of surfaces for each series

        Returns:
            :class:`pn.DataFrame`: df where Structural data is stored

        """

        # DEP 27.10
        # len_sps = np.zeros(self.stack.number_features, dtype=int)
        # surf_count = self.df.groupby('OrderFeature').surface.nunique()
        # len_sps[surf_count.index - 1] = surf_count.values
        #
        # self.df.at['values', 'number surfaces per series'] = len_sps
        return self.df.groupby('Feature').count()['Surface'].values

    def update_id(self, id_list: list = None):
        """
        Set id of the layers (1 based)
        Args:
            id_list (list):

        Returns:
             :class:`Surfaces`:

        """
        self.map_faults()
        if id_list is None:
            # This id is necessary for the faults
            id_unique = self.df.reset_index().index + 1

        self.df['id'] = id_unique

        return self

    def map_faults(self):
        self.df['isFault'] = self.df['Feature'].map(self.stack.faults.df['isFault'])

    @staticmethod
    def background_color(value):
        if isinstance(value, str):
            return "background-color: %s" % value

    # region set formation names
    def set_surfaces_names(self, surfaces_list: list, update_df=True):
        """
         Method to set the names of the surfaces in order. This applies in the surface column of the df

         Args:
             surfaces_list (list[str]):  list of names of surfaces. They are ordered.
             update_df (bool): Update Surfaces.df columns with the default values

         Returns:
             :class:`Surfaces`:
         """
        if isinstance(surfaces_list, (list, np.ndarray)):
            surfaces_list = np.asarray(surfaces_list)

        else:
            raise AttributeError('list_names must be either array_like type')

        # Deleting all columns if they exist
        # TODO check if some of the names are in the df and not deleting them?
        self.df.drop(self.df.index, inplace=True)
        self.df['Surface'] = surfaces_list

        # Changing the name of the series is the only way to mutate the series object from surfaces
        if update_df is True:
            self.map_stack()
            self.update_id()
            self.set_basement()
            self.reset_order_surfaces()
            self.colors.update_colors()
        return self

    def set_default_surface_name(self):
        """
        Set the minimum number of surfaces to compute a model i.e. surfaces_names: surface1 and basement

        Returns:
             :class:`Surfaces`:

        """

        if self.df.shape[0] == 0:
            # TODO DEBUG: I am not sure that surfaces always has at least one entry. Check it
            self.set_surfaces_names(['surface1', 'surface2'])
        return self

    def set_surfaces_names_from_surface_points(self, surface_points):
        """
        Set surfaces names from a :class:`Surface_points` object. This can be useful if the surface points are imported
        from a table.

        Args:
            surface_points (:class:`Surface_points`):

        Returns:

        """
        self.set_surfaces_names(surface_points.df['Surface'].unique())
        return self

    def add_surface(self, surface_list: Union[str, list], update_df=True):
        """ Add surface to the df.

        Args:
            surface_list (str, list): name or list of names of the surfaces to apply the functionality
            update_df (bool): Update Surfaces.df columns with the default values

        Returns:
            :class:`gempy_lite.core.data.Surfaces`
        """

        surface_list = np.atleast_1d(surface_list)

        # Remove from the list categories that already exist
        surface_list = surface_list[~np.in1d(surface_list, self.df['Surface'].values)]

        for c in surface_list:
            idx = self.df.index.max()
            if idx is np.nan:
                idx = -1
            self.df.loc[idx + 1, 'Surface'] = c

        if update_df is True:
            self.map_stack()
            self.update_id()
            self.set_basement()
            self.reset_order_surfaces()
            self.colors.update_colors()
        return self

    @_setdoc_pro([update_id.__doc__, pn.DataFrame.drop.__doc__])
    def delete_surface(self, indices: Union[int, str, list, np.ndarray], update_id=True):
        """[s1]

        Args:
            indices (str, list): name or list of names of the series to apply the functionality
            update_id (bool): if true [s0]

        Returns:
             :class:`Surfaces`:

        """
        indices = np.atleast_1d(indices)
        if indices.dtype == int:
            self.df.drop(indices, inplace=True)
        else:
            self.df.drop(self.df.index[self.df['Surface'].isin(indices)], inplace=True)

        if update_id is True:
            self.update_id()
            self.set_basement()
            self.reset_order_surfaces()
        return self

    def rename_surfaces(self, to_replace: Union[str, list, dict], **kwargs):
        """Replace values given in to_replace with value.

        Args:
            to_replace (str, regex, list, dict, Series, int, float, or None) –
             How to find the values that will be replaced.
            **kwargs:

        Returns:
            :class:`gempy_lite.core.data.Surfaces`

        See Also:
            :any:`pandas.Series.replace`

        """
        if np.isin(to_replace, self.df['Surface']).any():
            print('Two surfaces cannot have the same name.')
        else:
            self.df['Surface'].replace(to_replace, inplace=True, **kwargs)
        return self

    def reset_order_surfaces(self):
        self.df['OrderSurface'] = self.df.groupby('Feature').cumcount() + 1

    def modify_order_surfaces(self, new_value: int, idx: int, series_name: str = None):
        """
        Replace to the new location the old series

        Args:
            new_value (int): New location
            idx (int): Index of the surface to be moved
            series_name (str): name of the series to be moved

        Returns:
            :class:`gempy_lite.core.data.Surfaces`

        """
        if series_name is None:
            series_name = self.df.loc[idx, 'Feature']

        group = self.df.groupby('Feature').get_group(series_name)['OrderSurface']
        assert np.isin(new_value, group), 'new_value must exist already in the order_surfaces group.'
        old_value = group[idx]
        self.df.loc[group.index, 'OrderSurface'] = group.replace([new_value, old_value], [old_value, new_value])
        self.sort_surfaces()
        self.set_basement()
        return self

    def sort_surfaces(self):
        """Sort surfaces by series and order_surfaces"""

        self.df.sort_values(by=['Feature', 'OrderSurface'], inplace=True)
        self.update_id()
        return self.df

    def set_basement(self):
        """
        Set isBasement property to true to the last series of the df.

        Returns:
             :class:`Surfaces`:

        """

        self.df['isBasement'] = False
        idx = self.df.last_valid_index()
        if idx is not None:
            self.df.loc[idx, 'isBasement'] = True

        # TODO add functionality of passing the basement and calling reorder to push basement surface to the bottom
        #  of the data frame
        assert self.df['isBasement'].values.astype(bool).sum() <= 1, 'Only one surface can be basement'
        return self

    # endregion

    # set_series
    def map_stack(self, mapping_object: Union[dict, pn.DataFrame] = None):
        """
        Method to map to which series every surface belongs to. This step is
         necessary to assign differenct tectonics such as unconformities or faults.

        Args:
            mapping_object (dict, :class:`pn.DataFrame`):
                * dict: keys are the series and values the surfaces belonging to
                 that series

                * pn.DataFrame: Dataframe with surfaces as index and a column
                series with the correspondent series name of each surface

        Returns:
             :class:`Surfaces`

        """

        # Updating surfaces['Feature'] categories
        self.df['Feature'].cat.set_categories(self.stack.df.index, inplace=True)
        # TODO Fixing this. It is overriding the formations already mapped
        if mapping_object is not None:
            # If none is passed and series exist we will take the name of the
            # first series as a default

            if type(mapping_object) is dict:

                s = []
                f = []
                for k, v in mapping_object.items():
                    for form in np.atleast_1d(v):
                        s.append(k)
                        f.append(form)

                new_series_mapping = pn.DataFrame([pn.Categorical(s, self.stack.df.index)],
                                                  f, columns=['Feature'])

            elif isinstance(mapping_object, pn.Categorical):
                # This condition is for the case we have surface on the index and in 'Feature' the category
                # TODO Test this
                new_series_mapping = mapping_object

            else:
                raise AttributeError(str(type(mapping_object)) + ' is not the right attribute type.')

            # Checking which surfaces are on the list to be mapped
            b = self.df['Surface'].isin(new_series_mapping.index)
            idx = self.df.index[b]

            # Mapping
            self.df.loc[idx, 'Feature'] = self.df.loc[idx, 'Surface'].map(new_series_mapping['Feature'])

        # Fill nans
        self.df['Feature'].fillna(self.stack.df.index.values[-1], inplace=True)

        # Reorganize the pile
        self.reset_order_surfaces()
        self.sort_surfaces()
        self.set_basement()
        return self

    # endregion

    # region update_id

    # endregion

    def add_surfaces_values(self, values_array: Union[np.ndarray, list], properties_names: list = np.empty(0)):
        """Add values to be interpolated for each surfaces.

        Args:
            values_array (np.ndarray, list): array-like of the same length as number of surfaces. This functionality
             can be used to assign different geophysical properties to each layer
            properties_names (list): list of names for each values_array columns. This must be of same size as
             values_array axis 1. By default properties will take the column name: 'value_X'.

        Returns:
            :class:`gempy_lite.core.data.Surfaces`

        """
        values_array = np.atleast_2d(values_array)
        properties_names = np.atleast_1d(properties_names)
        if properties_names.shape[0] != values_array.shape[0]:
            for i in range(values_array.shape[0]):
                properties_names = np.append(properties_names, 'value_' + str(i))

        for e, p_name in enumerate(properties_names):
            try:
                self.df.loc[:, p_name] = values_array[e]
            except ValueError:
                raise ValueError('value_array must have the same length in axis 0 as the number of surfaces')
        return self

    def delete_surface_values(self, properties_names: Union[str, list]):
        """Delete a property or several properties column.

        Args:
            properties_names (str, list[str]): Name of the property to delete

        Returns:
             :class:`gempy_lite.core.data.Surfaces`

        """

        properties_names = np.asarray(properties_names)
        self.df.drop(properties_names, axis=1, inplace=True)
        return True

    def set_surfaces_values(self, values_array: Union[np.ndarray, list], properties_names: list = np.empty(0)):
        """Set values to be interpolated for each surfaces. This method will delete the previous values.

        Args:
            values_array (np.ndarray, list): array-like of the same length as number of surfaces. This functionality
             can be used to assign different geophysical properties to each layer
            properties_names (list): list of names for each values_array columns. This must be of same size as
             values_array axis 1. By default properties will take the column name: 'value_X'.

        Returns:
             :class:`gempy_lite.core.data.Surfaces`

        """
        # Check if there are values columns already
        old_prop_names = self.df.columns[~self.df.columns.isin(self._columns)]
        # Delete old
        self.delete_surface_values(old_prop_names)

        # Create new
        self.add_surfaces_values(values_array, properties_names)
        return self

    def modify_surface_values(self, idx, properties_names, values):
        """Method to modify values using loc of pandas.

        Args:
            idx (int, list[int]):
            properties_names (str, list[str]):
            values (float, np.ndarray):

        Returns:
             :class:`gempy_lite.core.data.Surfaces`

        """
        properties_names = np.atleast_1d(properties_names)
        assert ~np.isin(properties_names, ['Surface', 'Feature', 'OrderSurface', 'id', 'isBasement', 'color']), \
            'only property names can be modified with this method'

        self.df.loc[idx, properties_names] = values
        return self


class Colors:
    """
    Object that handles the color management in the model.
    """
    def __init__(self, surfaces):
        self.surfaces = surfaces
        self.colordict = None
        self._hexcolors_soft = [
            '#015482', '#9f0052', '#ffbe00', '#728f02', '#443988',
            '#ff3f20', '#5DA629', '#b271d0', '#72e54a', '#583bd1',
            '#d0e63d', '#b949e2', '#95ce4b', '#6d2b9f', '#60eb91',
            '#d746be', '#52a22e', '#5e63d8', '#e5c339', '#371970',
            '#d3dc76', '#4d478e', '#43b665', '#d14897', '#59e5b8',
            '#e5421d', '#62dedb', '#df344e', '#9ce4a9', '#d94077',
            '#99c573', '#842f74', '#578131', '#708de7', '#df872f',
            '#5a73b1', '#ab912b', '#321f4d', '#e4bd7c', '#142932',
            '#cd4f30', '#69aedd', '#892a23', '#aad6de', '#5c1a34',
            '#cfddb4', '#381d29', '#5da37c', '#d8676e', '#52a2a3',
            '#9b405c', '#346542', '#de91c9', '#555719', '#bbaed6',
            '#945624', '#517c91', '#de8a68', '#3c4b64', '#9d8a4d',
            '#825f7e', '#2c3821', '#ddadaa', '#5e3524', '#a3a68e',
            '#a2706b', '#686d56'
        ]  # source: https://medialab.github.io/iwanthue/

    def generate_colordict(
            self,
            hex_colors: Union[List[str], str] = 'palettes',
            palettes: List[str] = 'default',
    ):
        """Generates and sets color dictionary.

        Args:
           hex_colors (list[str], str): List of hex color values. In the future this could
           accommodate the actual geological palettes. For example striplog has a quite
           good set of palettes.
                * palettes: If hexcolors='palettes' the colors will be chosen from the
                   palettes arg
                * soft: https://medialab.github.io/iwanthue/
           palettes (list[str], optional): list with name of seaborn palettes. Defaults to 'default'.
        """
        if hex_colors == 'palettes':
            hex_colors = []
            if palettes == 'default':
                # we predefine some 7 colors manually
                hex_colors = ['#015482', '#9f0052', '#ffbe00', '#728f02', '#443988', '#ff3f20', '#5DA629']
                # then we create a list of seaborn color palette names, as the user didn't provide any
                palettes = ['muted', 'pastel', 'deep', 'bright', 'dark', 'colorblind']
            for palette in palettes:  # for each palette
                hex_colors += sns.color_palette(palette).as_hex()  # get all colors in palette and add to list
                if len(hex_colors) >= len(self.surfaces.df):
                    break
        elif hex_colors == 'soft':
            hex_colors = self._hexcolors_soft

        surface_names = self.surfaces.df['Surface'].values
        n_surfaces = len(surface_names)

        while n_surfaces > len(hex_colors):
            hex_colors.append(self._random_hexcolor())

        self.colordict = dict(
            zip(surface_names, hex_colors[:n_surfaces])
        )

    @staticmethod
    def _random_hexcolor() -> str:
        """Generates a random hex color string."""
        return "#"+str(hex(np.random.randint(0, 16777215))).lstrip("0x")

    def change_colors(self, colordict: dict = None):
        """Change the model colors either by providing a color dictionary or,
        if not, by using a color pick widget.

        Args:
            colordict (dict, optional): dict with surface names mapped to hex color codes, e.g. {'layer1':'#6b0318'}
            if None: opens jupyter widget to change colors interactively. Defaults to None.
        """
        try:
            from gempy_lite.core.kernel_data import ipywidgets_import
            from IPython.display import display
            import ipywidgets as widgets

        except ImportError:
            raise ImportError('You need to install IPython and for interactive color picking.')

        assert ipywidgets_import, 'ipywidgets not imported. Make sure the library is installed.'

        if colordict:
            self.update_colors(colordict)
        else:

            items = [
                widgets.ColorPicker(description=surface, value=color)
                for surface, color in self.colordict.items()
            ]
            colbox = widgets.VBox(items)
            print('Click to select new colors.')
            display(colbox)

            def on_change(v):
                self.colordict[v['owner'].description] = v['new']  # update colordict
                self._set_colors()

            for cols in colbox.children:
                cols.observe(on_change, 'value')

    def update_colors(self, colordict: dict = None):
        """ Updates the colors in self.colordict and in surfaces_df.

        Args:
            colordict (dict, optional): dict with surface names mapped to hex
                color codes, e.g. {'layer1':'#6b0318'}. Defaults to None.
        """
        if colordict is None:
            self.generate_colordict()
        else:
            for surf, color in colordict.items():  # map new colors to surfaces
                # assert this because user can set it manually
                assert surf in list(self.surfaces.df['surface']), str(surf) + ' is not a model surface'
                assert re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color), str(color) + ' is not a HEX color code'
                self.colordict[surf] = color

        self._set_colors()

    def _add_colors(self):
        """Assign a color to the last entry of surfaces df or check isnull and assign color there"""
        self.generate_colordict()

    def _set_colors(self):
        """sets colordict in surfaces dataframe"""
        for surf, color in self.colordict.items():
            self.surfaces.df.loc[self.surfaces.df['Surface'] == surf, 'color'] = color

    def set_default_colors(self, surfaces=None):
        if surfaces is not None:
            self.colordict[surfaces] = self.colordict[surfaces]
        self._set_colors()

    def delete_colors(self, surfaces):
        for surface in surfaces:
            self.colordict.pop(surface, None)
        self._set_colors()

    def make_faults_black(self, series_fault):
        faults_list = list(self.surfaces.df[self.surfaces.df.Feature.isin(series_fault)]['Surface'])
        for fault in faults_list:
            if self.colordict[fault] == '#527682':
                self.set_default_colors(fault)
            else:
                self.colordict[fault] = '#527682'
                self._set_colors()

    def reset_default_colors(self):
        self.generate_colordict()
        self._set_colors()
        return self.surfaces


class Structure(object):
    """
    The structure_data class analyse the different lengths of subset in the interface and orientations categories_df
    to pass them to the theano function.

    Attributes:
        surface_points (:class:`SurfacePoints`): [s0]
        orientations (:class:`Orientations`): [s1]
        surfaces (:class:`Surfaces`): [s2]
        faults (:class:`Faults`): [s3]
        df (:class:`pn.DataFrame`):
            * len surfaces surface_points (list): length of each surface/fault in surface_points
            * len series surface_points (list) : length of each series in surface_points
            * len series orientations (list) : length of each series in orientations
            * number surfaces per series (list): number of surfaces per series
            * ...
    Args:
        surface_points (:class:`SurfacePoints`): [s0]
        orientations (:class:`Orientations`): [s1]
        surfaces (:class:`Surfaces`): [s2]
        faults (:class:`Faults`): [s3]

    """

    def __init__(self, surface_points, orientations, surfaces: Surfaces, faults):
        self.surface_points = surface_points
        self.orientations = orientations
        self.surfaces = surfaces
        self.faults = faults

        df_ = pn.DataFrame(np.array(['False', 'False', -1, -1, -1, -1, -1, -1, -1], ).reshape(1, -1),
                           index=['values'],
                           columns=['isLith', 'isFault',
                                    'number faults', 'number surfaces', 'number series',
                                    'number surfaces per series',
                                    'len surfaces surface_points', 'len series surface_points',
                                    'len series orientations'])

        self.df = df_.astype({'isLith': bool, 'isFault': bool, 'number faults': int,
                              'number surfaces': int, 'number series': int})
    #
    #     self.update_structure_from_input()
    #
    # def __repr__(self):
    #     return self.df.T.to_string()
    #
    # def _repr_html_(self):
    #     return self.df.T.to_html()
    #
    # def update_structure_from_input(self):
    #     """
    #     Update all fields dependent on the linked Data objects.
    #
    #     Returns:
    #         bool: True
    #     """
    #     self.set_length_surfaces_i()
    #     self.set_series_and_length_series_i()
    #     self.set_length_series_o()
    #     self.set_number_of_surfaces_per_series()
    #     self.set_number_of_faults()
    #     self.set_number_of_surfaces()
    #     self.set_is_lith_is_fault()
    #     return True
    #
    # def set_length_surfaces_i(self):
    #     """
    #     Set the length of each **surface** on `SurfacePoints` i.e. how many data points are for each surface
    #
    #     Returns:
    #         :class:`pn.DataFrame`: df where Structural data is stored
    #
    #     """
    #     # ==================
    #     # Extracting lengths
    #     # ==================
    #     # Array containing the size of every surface. SurfacePoints
    #     lssp = self.surface_points.df.groupby('id')['OrderFeature'].count().values
    #     lssp_nonzero = lssp[np.nonzero(lssp)]
    #
    #     self.df.at['values', 'len surfaces surface_points'] = lssp_nonzero
    #
    #     return self.df
    #
    # def set_series_and_length_series_i(self):
    #     """
    #     Set the length of each **series** on `SurfacePoints` i.e. how many data points are for each series. Also
    #     sets the number of series itself.
    #
    #     Returns:
    #         :class:`pn.DataFrame`: df where Structural data is stored
    #
    #     """
    #     len_series = self.surfaces.stack.df.shape[0]
    #
    #     # Array containing the size of every series. SurfacePoints.
    #     points_count = self.surface_points.df['OrderFeature'].value_counts(sort=False)
    #     len_series_i = np.zeros(len_series, dtype=int)
    #     len_series_i[points_count.index - 1] = points_count.values
    #
    #     if len_series_i.shape[0] == 0:
    #         len_series_i = np.insert(len_series_i, 0, 0)
    #
    #     self.df.at['values', 'len series surface_points'] = len_series_i
    #     self.df['number series'] = len(len_series_i)
    #     return self.df
    #
    # def set_length_series_o(self):
    #     """
    #     Set the length of each **series** on `Orientations` i.e. how many orientations are for each series.
    #
    #     Returns:
    #         :class:`pn.DataFrame`: df where Structural data is stored
    #
    #     """
    #     # Array containing the size of every series. orientations.
    #
    #     len_series_o = np.zeros(self.surfaces.stack.df.shape[0], dtype=int)
    #     ori_count = self.orientations.df['OrderFeature'].value_counts(sort=False)
    #     len_series_o[ori_count.index - 1] = ori_count.values
    #
    #     self.df.at['values', 'len series orientations'] = len_series_o
    #
    #     return self.df
    #
    # def set_number_of_surfaces_per_series(self):
    #     """
    #     Set number of surfaces for each series
    #
    #     Returns:
    #         :class:`pn.DataFrame`: df where Structural data is stored
    #
    #     """
    #     len_sps = np.zeros(self.surfaces.stack.df.shape[0], dtype=int)
    #     surf_count = self.surface_points.df.groupby('OrderFeature'). \
    #         surface.nunique()
    #
    #     len_sps[surf_count.index - 1] = surf_count.values
    #
    #     self.df.at['values', 'number surfaces per series'] = len_sps
    #     return self.df
    #
    # def set_number_of_faults(self):
    #     """
    #     Set number of faults series. This method in gempy_lite v2 is simply informative
    #
    #     Returns:
    #         :class:`pn.DataFrame`: df where Structural data is stored
    #
    #     """
    #     # Number of faults existing in the surface_points df
    #     self.df.at['values', 'number faults'] = self.faults.df['isFault'].sum()
    #     return self.df
    #
    # def set_number_of_surfaces(self):
    #     """
    #     Set the number of total surfaces
    #
    #     Returns:
    #         :class:`pn.DataFrame`: df where Structural data is stored
    #
    #     """
    #     # Number of surfaces existing in the surface_points df
    #     self.df.at['values', 'number surfaces'] = self.surface_points.df['Surface'].nunique()
    #
    #     return self.df
    #
    # def set_is_lith_is_fault(self):
    #     """
    #     Check if there is lithologies in the data and/or df. This method in gempy_lite v2 is simply informative
    #
    #     Returns:
    #         :class:`pn.DataFrame`: df where Structural data is stored
    #     """
    #     self.df['isLith'] = True if self.df.loc['values', 'number series'] >= self.df.loc['values', 'number faults'] \
    #         else False
    #     self.df['isFault'] = True if self.df.loc['values', 'number faults'] > 0 else False
    #
    #     return self.df


@_setdoc_pro([Structure.__doc__])
class KrigingParameters(object):
    """
    Class that stores and computes the default values for the kriging parameters used during the interpolation.
    The default values will be computed from the :class:`Grid` and :class:`Structure` linked objects

    Attributes:
        grid (:class:`Grid`): [s0]
        structure (:class:`Structure`): [s1]
        df (:class:`pn.DataFrame`): df containing the kriging parameters.

    Args:
        grid (:class:`Grid`): [s0]
        structure (:class:`Structure`): [s1]
    """

    def __init__(self, structure: Structure, grid=None):
        self.structure = structure
        self.grid = grid

        df_ = pn.DataFrame(np.array([np.nan, np.nan, 3]).reshape(1, -1),
                           index=['values'],
                           columns=['range', '$C_o$', 'drift equations',
                                    ])

        self.df = df_.astype({'drift equations': object})
        self.set_default_range()
        self.set_default_c_o()
        self.set_u_grade()

    def __repr__(self):
        return self.df.T.to_string()

    def _repr_html_(self):
        return self.df.T.to_html()

    def modify_kriging_parameters(self, attribute: str, value, **kwargs):
        """Method to modify a given field

        Args:
            attribute (str): Name of the field to modify
            value: new value of the field. It will have to exist in the category in order for pandas to modify it.
            kwargs:
                * u_grade_sep (str): If drift equations values are `str`, symbol that separates the values.

        Returns:
            :class:`pandas.DataFrame`: df where options data is stored
        """

        u_grade_sep = kwargs.get('u_grade_sep', ',')

        assert np.isin(attribute, self.df.columns).all(), 'Valid properties are: ' + np.array2string(self.df.columns)

        if attribute == 'drift equations':
            value = np.asarray(value)
            print(value)

            if type(value) is str:
                value = np.fromstring(value[1:-1], sep=u_grade_sep, dtype=int)
            try:
                assert value.shape[0] is self.structure.df.loc['values', 'len series surface_points'].shape[0]
                print(value, attribute)
                self.df.at['values', attribute] = value
                print(self.df)

            except AssertionError:
                print('u_grade length must be the same as the number of series')

        else:
            self.df.loc['values', attribute] = value

    def str2int_u_grade(self, **kwargs):
        """
        Convert u_grade to ints

        Args:
            **kwargs:
                * u_grade_sep (str): If drift equations values are `str`, symbol that separates the values.

        Returns:

        """
        u_grade_sep = kwargs.get('u_grade_sep', ',')
        value = self.df.loc['values', 'drift equations']
        if type(value) is str:
            value = np.fromstring(value[1:-1], sep=u_grade_sep, dtype=int)
        try:
            assert value.shape[0] is self.structure.df.loc['values', 'len series surface_points'].shape[0]
            self.df.at['values', 'drift equations'] = value

        except AssertionError:
            print('u_grade length must be the same as the number of series')

        return self.df

    def set_default_range(self, extent=None):
        """
        Set default kriging_data range

        Args:
            extent (Optional[float, np.array]): extent used to compute the default range--i.e. largest diagonal. If None
             extent of the linked :class:`Grid` will be used.

        Returns:

        """
        if extent is None:
            extent = self.grid.regular_grid.extent
            if np.sum(extent) == 0 and self.grid.values.shape[0] > 1:
                extent = np.concatenate((np.min(self.grid.values, axis=0),
                                         np.max(self.grid.values, axis=0)))[[0, 3, 1, 4, 2, 5]]

        try:
            range_var = np.sqrt(
                (extent[0] - extent[1]) ** 2 +
                (extent[2] - extent[3]) ** 2 +
                (extent[4] - extent[5]) ** 2)
        except TypeError:
            warnings.warn('The extent passed or if None the extent of the grid object has some type of problem',
                          TypeError)
            range_var = np.nan

        self.df['range'] = range_var

        return range_var

    def set_default_c_o(self, range_var=None):
        """
        Set default covariance at 0.

        Args:
            range_var (Optional[float, np.array]): range used to compute the default c_0--i.e. largest diagonal. If None
             the already computed range will be used.

        Returns:

        """
        if range_var is None:
            range_var = self.df['range']

        self.df['$C_o$'] = range_var ** 2 / 14 / 3
        return self.df['$C_o$']

    def set_u_grade(self, u_grade: list = None):
        """
         Set default universal grade. Transform polynomial grades to number of equations

         Args:

             u_grade (list):

         Returns:

         """
        # =========================
        # Choosing Universal drifts
        # =========================
        if u_grade is None:

            len_series_i = self.structure.df.loc['values', 'len series surface_points']
            u_grade = np.ones_like(len_series_i)
            # u_grade[(len_series_i > 1)] = 1

        else:
            u_grade = np.array(u_grade)

        # Transforming grade to number of equations
        n_universal_eq = np.zeros_like(u_grade)
        n_universal_eq[u_grade == 0] = 0
        n_universal_eq[u_grade == 1] = 3
        n_universal_eq[u_grade == 2] = 9

        self.df.at['values', 'drift equations'] = n_universal_eq
        return self.df['drift equations']



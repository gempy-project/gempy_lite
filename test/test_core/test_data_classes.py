# Importing GemPy
import gempy as gp


# Importing auxiliary libraries
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
import pytest

import gempy.core.data_modules.geometric_data
import gempy.core.data_modules.stack


@pytest.fixture(scope='module')
def create_faults():
    faults = gempy.core.data_modules.stack.Faults()
    return faults


@pytest.fixture(scope='module')
def create_series(create_faults):
    faults = create_faults

    series = gempy.core.data_modules.stack.Series(faults)
    series.set_series_index(['foo', 'foo2', 'foo5', 'foo7'])
    series.add_series('foo3')
    series.delete_series('foo2')
    series.rename_series({'foo': 'boo'})
    series.reorder_series(['foo3', 'boo', 'foo7', 'foo5'])

    faults.set_is_fault(['boo'])

    fr = np.zeros((4, 4))
    fr[2, 2] = True
    faults.set_fault_relation(fr)

    series.add_series('foo20')

    return series


@pytest.fixture(scope='module')
def create_surfaces(create_series):
    series = create_series
    surfaces = gp.Surfaces(series)
    surfaces.set_surfaces_names(['foo', 'foo2', 'foo5'])

    print(series)

    # We can add new surfaces:
    surfaces.add_surface(['feeeee'])
    print(surfaces)

    # The column surface is also a pandas.Categories.
    # This will be important for the Data clases (SurfacePoints and Orientations)

    print(surfaces.df['surface'])

    ### Set values

    # To set the values we do it with the following method
    surfaces.set_surfaces_values([2, 2, 2, 5])

    print(surfaces)

    # #### Set values with a given name:

    # We can give specific names to the properties (i.e. density)
    surfaces.add_surfaces_values([[2, 2, 2, 6], [2, 2, 1, 8]], ['val_foo', 'val2_foo'])
    print(surfaces)

    ### Delete surfaces values
    #
    # To delete a full propery:
    surfaces.delete_surface_values(['val_foo', 'value_0'])

    # #### One of the surfaces must be set be the basement:

    surfaces.set_basement()
    print(surfaces)

    # #### Set surface values
    #
    # We can also use set values instead adding. This will delete the previous properties and add the new one

    surfaces.set_surfaces_values([[2, 2, 2, 6], [2, 2, 1, 8]], ['val_foo', 'val2_foo'])
    print(surfaces)

    # The last property is the correspondant series that each surface belong to. `series` and `surface`
    # are pandas categories. To get a overview of what this mean
    # check https://pandas.pydata.org/pandas-docs/stable/categorical.html.

    print(surfaces.df['series'])

    print(surfaces.df['surface'])

    # ### Map series to surface

    # To map a series to a surface we can do it by passing a dict:
    # If a series does not exist in the `Series` object, we rise a warning and we set those surfaces to nans

    d = {"foo7": 'foo', "booX": ('foo2', 'foo5', 'fee')}

    surfaces.map_series(d)
    surfaces.map_series({"foo7": 'foo', "boo": ('foo2', 'foo5', 'fee')})

    print(surfaces)

    # An advantage of categories is that they are order so no we can tidy the df by series and surface

    surfaces.df.sort_values(by='series', inplace=True)

    # If we change the basement:

    surfaces.set_basement()

    # Only one surface can be the basement:

    print(surfaces)

    # ### Modify surface name

    surfaces.rename_surfaces({'foo2': 'lala'})

    print(surfaces)

    surfaces.df.loc[2, 'val_foo'] = 22

    print(surfaces)

    # We can use `set_is_fault` to choose which of our series are faults:
    return surfaces


@pytest.fixture(scope='module')
def create_surface_points(create_surfaces, create_series):
    # # Data
    # #### SurfacePoints
    # These two DataFrames (df from now on) will contain the individual information of each point at an interface or
    # orientation. Some properties of this table are mapped from the *df* below.
    surfaces = create_surfaces
    surface_points = gempy.core.data_modules.geometric_data.SurfacePoints(surfaces)

    print(surface_points)

    surface_points.set_surface_points(pn.DataFrame(np.random.rand(6, 3)), ['foo', 'foo5', 'lala', 'foo5', 'lala', 'feeeee'])

    print(surface_points)

    surface_points.map_data_from_surfaces(surfaces, 'series')
    print(surface_points)


    surface_points.map_data_from_surfaces(surfaces, 'id')
    print(surface_points)


    surface_points.map_data_from_series(create_series, 'order_series')
    print(surface_points)

    surface_points.sort_table()
    print(surface_points)
    return surface_points


@pytest.fixture(scope='module')
def create_orientations(create_surfaces, create_series):
    surfaces = create_surfaces

    # ### Orientations
    orientations = gempy.core.data_modules.geometric_data.Orientations(surfaces)

    print(orientations)

    # ### Set values passing pole vectors:

    orientations.set_orientations(np.random.rand(6, 3) * 10,
                                  np.random.rand(6, 3),
                                  surface=['foo', 'foo5', 'lala', 'foo5', 'lala', 'feeeee'])

    print(orientations)

    # ### Set values pasing orientation data: azimuth, dip, pole (dip direction)

    orientations.set_orientations(np.random.rand(6, 3) * 10,
                                  orientation=np.random.rand(6, 3) * 20,
                                  surface=['foo', 'foo5', 'lala', 'foo5', 'lala', 'feeeee'])

    print(orientations)

    # ### Mapping data from the other df
    orientations.map_data_from_surfaces(surfaces, 'series')
    print(orientations)

    orientations.map_data_from_surfaces(surfaces, 'id')
    print(orientations)

    orientations.map_data_from_series(create_series, 'order_series')
    print(orientations)

    orientations.update_annotations()
    return orientations


def test_add_orientation_with_pole(create_surfaces):
    orientations = gempy.core.data_modules.geometric_data.Orientations(create_surfaces)
    orientations.add_orientation(1, 1, 1, 'foo', pole_vector=(1, 0, 1))
    orientations.add_orientation(2, 2, 2, 'foo', orientation=(0, 0, 1))
    orientations.add_orientation(1, 1, 1, 'foo', pole_vector=(.45, 0, .45))
    orientations.modify_orientations(2, G_x=1, G_z=1)

    print(orientations)


@pytest.fixture(scope='module')
def create_grid():
    # Test creating an empty list
    grid = gp.Grid()
    # Test set regular grid by hand
    grid.create_regular_grid([0, 2000, 0, 2000, -2000, 0], [50, 50, 50])
    return grid


@pytest.fixture(scope='module')
def create_rescaling(create_surface_points, create_orientations, create_grid):
    rescaling = gempy.core.data_modules.geometric_data.RescaledData(create_surface_points, create_orientations, create_grid)
    return rescaling


@pytest.fixture(scope='module')
def create_additional_data(create_surface_points, create_orientations, create_grid, create_faults,
                           create_surfaces, create_rescaling):

    ad = gp.AdditionalData(create_surface_points, create_orientations, create_grid, create_faults,
                           create_surfaces, create_rescaling)
    return ad


class TestDataManipulation:

    def test_series(self, create_series):
        return create_series

    def test_surfaces(self, create_surfaces):
        return create_surfaces

    def test_surface_points(self, create_surface_points):
        return create_surface_points

    def test_orientations(self, create_orientations):
        return create_orientations

    def test_rescaling(self, create_rescaling):
        return create_rescaling

    def test_additional_data(self, create_additional_data):
        return create_additional_data


def test_stack():
    stack = gempy.core.data_modules.stack.Stack()
    stack.set_series_index(['foo', 'foo2', 'foo5', 'foo7'])
    stack.add_series('foo3')
    stack.delete_series('foo2')
    stack.rename_series({'foo': 'boo'})
    stack.reorder_series(['foo3', 'boo', 'foo7', 'foo5'])
    stack.set_is_fault(['boo'])

    faults = stack
    faults.set_is_fault(['boo'])

    fr = np.zeros((4, 4))
    fr[2, 2] = True
    faults.set_fault_relation(fr)

    stack.add_series('foo20')

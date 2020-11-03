import gempy_lite as gp
import matplotlib.pyplot as plt
import pandas as pn
import numpy as np
import os
import pytest
input_path = os.path.dirname(__file__)+'/../input_data'

# ## Preparing the Python environment
#
# For modeling with GemPy, we first need to import it. We should also import any other packages we want to
# utilize in our Python environment.Typically, we will also require `NumPy` and `Matplotlib` when working
# with GemPy. At this point, we can further customize some settings as desired, e.g. the size of figures or,
# as we do here, the way that `Matplotlib` figures are displayed in our notebook (`%matplotlib inline`).


# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../..")


@pytest.fixture(scope="module")
def load_model():
    verbose = False
    geo_model = gp.create_model('Model_Tuto1-1')

    # Importing the data from CSV-files and setting extent and resolution
    gp.init_data(geo_model, [0, 2000., 0, 2000., 0, 2000.], [50 ,50 ,50],
                 path_o=input_path+"/simple_fault_model_orientations.csv",
                 path_i=input_path+"/simple_fault_model_points.csv", default_values=True)

    df_cmp_i = gp.get_data(geo_model, 'surface_points')
    df_cmp_o = gp.get_data(geo_model, 'orientations')

    df_o = pn.read_csv(input_path + "/simple_fault_model_orientations.csv")
    df_i = pn.read_csv(input_path + "/simple_fault_model_points.csv")

    assert not df_cmp_i.empty, 'data was not set to dataframe'
    assert not df_cmp_o.empty, 'data was not set to dataframe'
    assert df_cmp_i.shape[0] == df_i.shape[0], 'data was not set to dataframe'
    assert df_cmp_o.shape[0] == df_o.shape[0], 'data was not set to dataframe'

    if verbose:
        gp.get_data(geo_model, 'surface_points').head()

    return geo_model


def test_load_model_df():

    verbose = True
    df_i = pn.DataFrame(np.random.randn(6,3), columns='X Y Z'.split())
    df_i['formation'] = ['surface_1' for _ in range(3)] + ['surface_2' for _ in range(3)]

    df_o = pn.DataFrame(np.random.randn(6,6), columns='X Y Z azimuth dip polarity'.split())
    df_o['formation'] = ['surface_1' for _ in range(3)] + ['surface_2' for _ in range(3)]

    geo_model = gp.create_model('test')
    # Importing the data directly from the dataframes
    gp.init_data(geo_model, [0, 2000., 0, 2000., 0, 2000.], [50, 50, 50],
                 surface_points_df=df_i, orientations_df=df_o, default_values=True)

    df_cmp_i = gp.get_data(geo_model, 'surface_points')
    df_cmp_o = gp.get_data(geo_model, 'orientations')

    if verbose:
        print(df_cmp_i.head())
        print(df_cmp_o.head())

    assert not df_cmp_i.empty, 'data was not set to dataframe'
    assert not df_cmp_o.empty, 'data was not set to dataframe'
    assert df_cmp_i.shape[0] == 6, 'data was not set to dataframe'
    assert df_cmp_o.shape[0] == 6, 'data was not set to dataframe'

    # try without the default_values command

    geo_model = gp.create_model('test')
    # Importing the data directly from the dataframes
    gp.init_data(geo_model, [0, 2000., 0, 2000., 0, 2000.], [50 ,50 ,50],
                 surface_points_df=df_i, orientations_df=df_o)

    df_cmp_i2 = gp.get_data(geo_model, 'surface_points')
    df_cmp_o2 = gp.get_data(geo_model, 'orientations')

    if verbose:
        print(df_cmp_i2.head())
        print(df_cmp_o2.head())

    assert not df_cmp_i2.empty, 'data was not set to dataframe'
    assert not df_cmp_o2.empty, 'data was not set to dataframe'
    assert df_cmp_i2.shape[0] == 6, 'data was not set to dataframe'
    assert df_cmp_o2.shape[0] == 6, 'data was not set to dataframe'

    return geo_model


@pytest.fixture(scope='module')
def map_sequential_pile(load_model):
    geo_model = load_model

    # TODO decide what I do with the layer order

    gp.map_stack_to_surfaces(geo_model, {"Fault_Series": 'Main_Fault',
                                          "Strat_Series": ('Sandstone_2', 'Siltstone',
                                                           'Shale', 'Sandstone_1', 'basement')},
                             remove_unused_series=True)

    geo_model.set_is_fault(['Fault_Series'])
    return geo_model


def test_get_data(load_model):
    geo_model = load_model
    return gp.get_data(geo_model, 'orientations').head()


def test_define_sequential_pile(map_sequential_pile):
    print(map_sequential_pile._surfaces)


def test_kriging_mutation(map_sequential_pile):
    geo_model = map_sequential_pile
    geo_model.modify_kriging_parameters('Range', 1)
    geo_model.modify_kriging_parameters('DriftDegree', [0, 1])
    print(geo_model.kriging)
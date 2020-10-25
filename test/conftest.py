import pytest
import gempy_lite.gempy_api as gempy
import gempy_lite as gp

import os

input_path = os.path.dirname(__file__)+'/input_data'
input_path2 = os.path.dirname(__file__)+'/../examples/data/input_data/'
import numpy as np
np.random.seed(1234)


@pytest.fixture(scope='session')
def one_fault_model_no_interp():
    """This only makes sense for running small test fast"""
    model = gp.create_data('one_fault_model', [0, 2000, 0, 2000, 0, 2000], [50, 50, 50],
                           path_o=input_path2 + 'tut_chapter1/simple_fault_model_orientations.csv',
                           path_i=input_path2 + 'tut_chapter1/simple_fault_model_points.csv')

    # Assigning series to surface as well as their order (timewise)
    gp.map_stack_to_surfaces(model, {"Fault_Series": 'Main_Fault',
                                      "Strat_Series": ('Sandstone_2', 'Siltstone',
                                                       'Shale', 'Sandstone_1')},
                             )
    model.set_is_fault(['Fault_Series'])
    return model


@pytest.fixture(scope='session')
def one_fault_model():
    """This only makes sense for running small test fast"""
    model = gp.create_data('one_fault_model', [0, 2000, 0, 2000, 0, 2000], [50, 50, 50],
                           path_o=input_path2 + 'tut_chapter1/simple_fault_model_orientations.csv',
                           path_i=input_path2 + 'tut_chapter1/simple_fault_model_points.csv')

    # Assigning series to surface as well as their order (timewise)
    gp.map_stack_to_surfaces(model, {"Fault_Series": 'Main_Fault',
                                      "Strat_Series": ('Sandstone_2', 'Siltstone',
                                                       'Shale', 'Sandstone_1')},
                             )
    model.set_is_fault(['Fault_Series'])

    return model


@pytest.fixture(scope='session')
def one_fault_model_topo(one_fault_model):

    one_fault_model.update_additional_data()
    one_fault_model.update_to_interpolator()
    one_fault_model.set_topography(d_z = (800, 1800))

    return one_fault_model


@pytest.fixture(scope='session')
def unconformity_model():
    geo_model = gp.create_data('unconformity_model',
        [0, 1000, 0, 1000, 0, 1000],
        resolution=[50, 42, 33],
        path_o=input_path2 + "jan_models/model6_orientations.csv",
        path_i=input_path2 + "jan_models/model6_surface_points.csv"
    )
    gp.map_stack_to_surfaces(
        geo_model,
        {"Strat_Series1": ('rock3'),
         "Strat_Series2": ('rock2', 'rock1'),
         "Basement_Series": ('basement')}
    )

    return geo_model


@pytest.fixture(scope='session')
def unconformity_model_topo():
    geo_model = gp.create_data('unconformity_model',
        [0, 1000, 0, 1000, 0, 1000],
        resolution=[50, 42, 38],
        path_o=input_path2 + "jan_models/model6_orientations.csv",
        path_i=input_path2 + "jan_models/model6_surface_points.csv"
    )

    geo_model.set_topography('random', d_z=(200, 920))
    gp.map_stack_to_surfaces(
        geo_model,
        {"Strat_Series1": ('rock3'),
         "Strat_Series2": ('rock2', 'rock1'),
         "Basement_Series": ('basement')}
    )

    # with open("input_data/geomodel_jan_sol.p", "rb") as f:
    # geo_model.solutions = load(f)
    return geo_model


@pytest.fixture(scope='session')
def model_horizontal_two_layers():

    geo_model = gp.create_model('interpolator')

    # Importing the data from csv files and settign extent and resolution
    gp.init_data(geo_model, [0, 10, 0, 10, -10, 0], [50, 50, 50],
                 path_o=input_path + "/GeoModeller/test_a/test_a_Foliations.csv",
                 path_i=input_path + "/GeoModeller/test_a/test_a_Points.csv")

    return geo_model


@pytest.fixture(scope='session')
def model_complex():

    model = gempy.create_data(extent=[0, 2500, 0, 1000, 0, 1000], resolution = [50, 20, 20],
                                 path_o=input_path2 + "jan_models/fixture_model_orientations.csv",
                                 path_i=input_path2 + "jan_models/fixture_model_surfaces.csv")

    # Assigning series to surface as well as their order (timewise)
    gp.map_stack_to_surfaces(model, {"Fault_Series": ('fault'), "Strat_Series1": ('rock3'),
                                            "Strat_Series2": ('rock2', 'rock1'),
                                            "Basement_Series": ('basement')})

    model.set_is_fault(['Fault_Series'])

    return model

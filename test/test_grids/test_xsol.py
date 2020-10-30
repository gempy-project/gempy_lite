import pytest
import numpy as np
import gempy_lite as gp
from gempy_lite.core.kernel_data import Stack, Surfaces
from gempy_lite.core.predictor.solution import XSolution


@pytest.fixture(scope='module')
def a_grid():
    # Or we can init one of the default grids since the beginning by passing
    # the correspondant attributes
    grid = gp.Grid(extent=[0, 2000, 0, 2000, -2000, 0],
                   resolution=[50, 50, 50])
    grid.set_active('regular')

    grid.create_custom_grid(np.arange(12).reshape(-1, 3))
    grid.set_active('custom')

    grid.create_topography()
    grid.set_active('topography')

    section_dict = {'section_SW-NE': ([250, 250], [1750, 1750], [100, 100]),
                    'section_NW-SE': ([250, 1750], [1750, 250], [100, 100])}
    grid.create_section_grid(section_dict)
    grid.set_active('sections')

    return grid


@pytest.fixture(scope='module')
def stack_eg():
    series = Stack()
    series.set_series_index(['foo', 'foo2', 'foo5', 'foo7'])
    series.add_series('foo3')
    series.delete_series('foo2')
    series.rename_series({'foo': 'boo'})
    series.reorder_series(['foo3', 'boo', 'foo7', 'foo5'])

    series.set_is_fault(['boo'])

    fr = np.zeros((4, 4))
    fr[2, 2] = True
    series.set_fault_relation(fr)

    series.add_series('foo20')

    # Mock
    series.df['isActive'] = True
    return series


@pytest.fixture(scope='module')
def surface_eg(stack_eg):
    surfaces = Surfaces(stack_eg)
    surfaces.set_surfaces_names(['foo', 'foo2', 'foo5', 'fee'])
    surfaces.add_surfaces_values([[2, 2, 2, 6], [2, 2, 1, 8]],
                                 ['val_foo', 'val2_foo'])
    return surfaces


@pytest.fixture(scope='module')
def sol_values(a_grid):
    rg_s = a_grid.values.shape[0]
    n_input = 100
    len_x = rg_s + n_input

    n_features = 5
    n_properties = 3
    # Generate random solution
    values = list()
    values_matrix = np.random.random_integers(0, 10, (n_properties, len_x))
    block_matrix = np.random.random_integers(
        0, 10, (n_features, n_properties, len_x)
    )

    fault_block = np.random.random_integers(40, 50, (n_features, len_x))
    weights = None
    scalar_field = np.random.random_integers(20, 30, (n_features, len_x))
    unknows = None
    mask_matrix = None
    fault_mask = None

    values.append(values_matrix)
    values.append(block_matrix)
    for i in [fault_block, weights, scalar_field, unknows, mask_matrix, fault_mask]:
        values.append(i)
    return values


def test_xsol(sol_values, a_grid, stack_eg, surface_eg):
    sol = XSolution(a_grid, stack=stack_eg, surfaces=surface_eg)
    sol.set_values(sol_values)
    print('\n regular', sol.s_regular_grid)
    print('\n custom', sol.s_custom_grid)
    print('\n topo', sol.s_topography)
    print('\n sect', sol.s_sections['section_SW-NE'])
    print('\n sect', sol.s_sections['section_NW-SE'])

    sol.set_values(sol_values, attach_xyz=False)
    print('\n custom2', sol.s_custom_grid)


def test_property(sol_values, a_grid, stack_eg, surface_eg):
    sol = XSolution(a_grid, stack=stack_eg, surfaces=surface_eg)
    sol.set_values(sol_values)
    print('scalar', sol.scalar_field_matrix)
    print('lith', sol.lith_block)
    print('values', sol.values_matrix)

    print('scalar_asp', sol.scalar_field_at_surface_points)



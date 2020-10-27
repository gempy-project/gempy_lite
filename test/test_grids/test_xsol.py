import pytest
import gempy_lite as gp
import numpy as np

from gempy_lite.core.kernel_data import Stack
from gempy_lite.core.predictor.solution import XSolution


@pytest.fixture(scope='module')
def regular_grid():
    # Or we can init one of the default grids since the beginning by passing
    # the correspondant attributes
    grid = gp.Grid(extent=[0, 2000, 0, 2000, -2000, 0],
                   resolution=[50, 50, 50])
    grid.set_active('regular')
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
def sol_values(regular_grid):
    rg_s = regular_grid.values.shape[0]
    n_input = 100
    len_x = rg_s + n_input

    n_features = 5
    n_properties = 2
    # Generate random solution
    values = list()
    values_matrix = np.random.random_integers(0, 10, (n_features, len_x))
    block_matrix = np.random.random_integers(
        0, 10, (n_properties, n_features, len_x)
    )
    values.append(values_matrix)
    values.append(block_matrix)
    return values


def test_xsol(sol_values, regular_grid, stack_eg):
    sol = XSolution(regular_grid, stack=stack_eg)
    sol.set_values(sol_values)
    print(sol.s_regular_grid)
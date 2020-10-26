import xarray as xr
import pytest


def test_all_running(model_horizontal_two_layers):
    print(model_horizontal_two_layers.surfaces)


@pytest.mark.skip
def test_combine_stack_surfaces(model_horizontal_two_layers):
    surf = model_horizontal_two_layers.surfaces.df
    stack = model_horizontal_two_layers.stack.df
    surfaces = xr.DataArray(surf,
                            dims=['surface', 'surface_attributes'],\
                            name='surfaces',
                            )
    #stack = xr.DataArray(stack, dims=['feature', 'feature_attributes'], name='series')

    m = xr.merge([surfaces, stack])
    print(surfaces)
    print(stack)
    print(m)


def test_combine_stack_surfaces_df(model_horizontal_two_layers):
    df1 = model_horizontal_two_layers.surfaces.df
    df2 = model_horizontal_two_layers.stack.df
    df3 = df1.join(df2, on='series', lsuffix='foo')
    df4 = df3.set_index(['series', 'surface'])
    print(df3)

    x = xr.DataArray(df4)
    print(x)

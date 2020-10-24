from .visualization_3d import GemPyvtkInteract
from gempy.plot.vista import GemPyToVista
from functools import wraps


def plot_add_surface_points(func):
    @wraps(func)
    def pasp(*args, **kwargs):
        plot_object = kwargs.pop('plot_object') if 'plot_object' in kwargs else None
        surface_points, idx = func(*args, **kwargs)
        if plot_object is not None:
            if isinstance(plot_object, GemPyToVista):
                if plot_object.live_updating is True:
                    plot_object.render_add_surface_points(idx)
            else:
                raise AttributeError('plot_object must be one GemPy compatible plot')
        return surface_points
    return pasp


def plot_delete_surface_points(func):
    @wraps(func)
    def pdsp(*args, **kwargs):
        plot_object = kwargs.pop('plot_object') if 'plot_object' in kwargs else None
        surface_points = func(*args, **kwargs)
        if plot_object is not None:
            if isinstance(plot_object, GemPyToVista):
                if plot_object.live_updating is True:
                    plot_object.render_delete_surface_points(args[1])
            else:
                raise AttributeError('plot_object must be one GemPy compatible plot')
        return surface_points
    return pdsp


def plot_move_surface_points(func):
    @wraps(func)
    def pmsp(*args, **kwargs):
        # TODO: Add this thing with the indices to all the decorators of this
        # module
        if 'indices' not in kwargs:
            indices = args[1]
        else:
            indices = kwargs['indices']

        plot_object = kwargs.pop('plot_object') if 'plot_object' in kwargs else None
        surface_points = func(*args, **kwargs)
        if plot_object is not None:
            if isinstance(plot_object, GemPyToVista):
                if plot_object.live_updating is True:
                    plot_object.render_move_surface_points(indices)
            else:
                raise AttributeError('plot_object must be one GemPy compatible plot')
        return surface_points
    return pmsp


def plot_add_orientation(func):
    @wraps(func)
    def pao(*args, **kwargs):
        plot_object = kwargs.pop('plot_object') if 'plot_object' in kwargs else None
        orientation, idx = func(*args, **kwargs)
        if plot_object is not None:
            if isinstance(plot_object, GemPyToVista):
                if plot_object.live_updating is True:
                    plot_object.render_add_orientations(idx)
            else:
                raise AttributeError('plot_object must be one GemPy compatible plot')
        return orientation
    return pao


def plot_delete_orientations(func):
    @wraps(func)
    def pdo(*args, **kwargs):
        plot_object = kwargs.pop('plot_object') if 'plot_object' in kwargs else None
        orientations = func(*args, **kwargs)
        if plot_object is not None:
            if isinstance(plot_object, GemPyToVista):
                if plot_object.live_updating is True:
                    plot_object.render_delete_orientations(args[1])
            else:
                raise AttributeError('plot_object must be one GemPy compatible plot')
        return orientations
    return pdo


def plot_move_orientations(func):
    @wraps(func)
    def pmo(*args, **kwargs):
        plot_object = kwargs.pop('plot_object') if 'plot_object' in kwargs else None
        orientations = func(*args, **kwargs)

        if plot_object is not None:
            if isinstance(plot_object, GemPyToVista):
                if plot_object.live_updating is True:
                    plot_object.render_move_orientations(args[1])
            else:
                raise AttributeError('plot_object must be one GemPy compatible plot')
        return orientations
    return pmo


def plot_set_topography(func):
    @wraps(func)
    def pst(*args, **kwargs):
        plot_object = kwargs.pop('plot_object') if 'plot_object' in kwargs else None
        topography = func(*args, **kwargs)

        if plot_object is not None:
            if isinstance(plot_object, GemPyToVista):
                if plot_object.live_updating is True:
                    plot_object.render_topography()
            else:
                raise AttributeError('plot_object must be one GemPy compatible plot')
        return topography
    return pst

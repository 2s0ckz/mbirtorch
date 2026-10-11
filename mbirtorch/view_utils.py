"""This module provides the slice viewers to mbirtorch users.  The functions in this module
convert torch tensors to numpy arrays.  They also convert data dicts, such as a recon_dict, to
strings that the viewers display.
"""

import pprint

import numpy as np

from .utilities import save_volume_as_gif
from .viewers.slice_figure import SliceViewer, VolumeStack
from .viewers.slice_figure import slice_viewer as _slice_viewer
from .viewers.slice_figure4d import SliceViewer4D, VolumeStack4D
from .viewers.slice_figure4d import slice_viewer4d as _slice_viewer4d
from .viewers.geometry_scene import GeometryScene
from .viewers.geometry_figure import GeometryFigure, geometry_viewer

__all__ = ['SliceViewer', 'VolumeStack', 'convert_subdicts_to_strings',
           'slice_viewer', 'GeometryScene', 'GeometryFigure',
           'geometry_viewer', 'SliceViewer4D', 'VolumeStack4D', 'slice_viewer4d']


def _to_numpy(dataset):
    """This function converts one dataset to a numpy array.  If the dataset is None, the
    function returns None."""
    if dataset is None:
        return None
    if hasattr(dataset, 'detach'):
        # This is a torch tensor.  np.asarray cannot convert a tensor on a GPU.
        return dataset.detach().cpu().numpy()
    return np.asarray(dataset)


def convert_subdicts_to_strings(data_dict):
    """This function converts the entries of a data dict to strings for display.  If the
    input is not a dict, the function returns the input unchanged.
    """
    if not isinstance(data_dict, dict):
        return data_dict
    string_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            string_dict[key] = pprint.pformat(value, width=100,
                                              sort_dicts=False)
        elif isinstance(value, str):
            string_dict[key] = value
        else:
            string_dict[key] = str(value)
    return string_dict


def slice_viewer(*datasets, data_dicts=None, title='', vmin=None, vmax=None,
                 slice_label=None, slice_axis=None, cmap='gray',
                 show_instructions=True, block=True, save_fn=None):
    """
    This function opens an interactive viewer for one or more 2D or 3D arrays.  The viewer
    moves through the slices of all arrays together.  It also shows ROI statistics and
    difference images, transposes axes, loads and saves files, and sets the intensity range.
    Each image may have a data dict, such as a recon_dict, which the viewer shows as text.

    Args:
        *datasets (ndarray, tensor, or None): One or more 2D or 3D arrays to
            display.  The function converts torch tensors to numpy arrays.
            The viewer adds a last axis of length 1 to each 2D array.
            The viewer replaces each None with an array of zeros.

        data_dicts (None or dict or list of None or dicts, optional):
            A dictionary of entries that describe the data, such as the
            recon_dict from :meth:`TomographyModel.recon <mbirtorch.TomographyModel.recon>`.
        title (str, optional): The figure title.  Defaults to an empty string.
        vmin (float, optional): The minimum intensity value for display.
            Defaults to the minimum over all datasets.
        vmax (float, optional): The maximum intensity value for display.
            Defaults to the maximum over all datasets.
        slice_label (str or list of str, optional): The label for the current
            slice.  Defaults to "Slice".
        slice_axis (int or list of int, optional): The axis along which to slice:
            0, 1, or 2.  Defaults to the last axis, 2.
        cmap (str, optional): The colormap.  Defaults to "gray".
        show_instructions (bool, optional): If True, the figure shows a short help
            message.  Defaults to True.
        block (bool, optional): If True, the function returns only after the window
            is closed.  If False, the function returns at once and leaves the window
            open.  The window responds to input when the next call with block=True
            runs, and that call returns when all open windows are closed.
            Defaults to True.
        save_fn (callable, optional): A function that replaces the built-in HDF5
            writer for the viewer's Save action.  The viewer calls ``save_fn`` as
            ``save_fn(file_path, array, array_name, attributes_dict)``.

    Returns:
        SliceViewer: The viewer object.

    Notes:
        - Right-click an image for a menu of actions on that image.
        - Right-click the intensity slider, or press Set range, to enter
          exact display limits.
        - Press 'h' to show help.  Press Esc to clear the help and other
          displayed items, or to remove the ROI.

    Example:
        >>> import mbirtorch
        >>> recon, recon_dict = ct_model.recon(sinogram, weights=weights)
        >>> mbirtorch.slice_viewer(phantom, recon, data_dicts=[None, recon_dict],
        ...                        title='ground truth phantom and recon')
    """
    datasets = [_to_numpy(dataset) for dataset in datasets]
    if isinstance(data_dicts, dict):
        data_dicts = convert_subdicts_to_strings(data_dicts)
    elif isinstance(data_dicts, (list, tuple)):
        data_dicts = [convert_subdicts_to_strings(d) for d in data_dicts]
    return _slice_viewer(*datasets, data_dicts=data_dicts, title=title,
                         vmin=vmin, vmax=vmax, slice_label=slice_label,
                         slice_axis=slice_axis, cmap=cmap,
                         show_instructions=show_instructions, block=block,
                         save_fn=save_fn)


def slice_viewer4d(*datasets, data_dicts=None, title='', vmin=None, vmax=None,
                   slice_label=None, slice_axis=None, cmap='gray',
                   show_instructions=True, block=True, save_fn=None, fps=5):
    """
    This function opens an interactive viewer for one or more 4D volumes, such as
    the output of :meth:`MACE4DModel.recon <mbirtorch.MACE4DModel.recon>`.  The viewer has
    the features of :func:`slice_viewer`.  It also has a time frame slider, images in planes
    that include the time axis, and a plot of the ROI mean against time frame.  Each slider
    has Play and GIF buttons.

    Args:
        *datasets (ndarray, tensor, or None): One or more 2D, 3D, or 4D arrays to
            display.  A 4D array has axes ``(t, x, y, z)``.  The viewer shows a 2D or 3D
            array as the same image at every time.  The function converts torch tensors
            to numpy arrays.
        data_dicts (None or dict or list of None or dicts, optional): A dictionary of
            entries that describe the data, such as the recon_dict from
            :meth:`MACE4DModel.recon <mbirtorch.MACE4DModel.recon>`.
        title (str, optional): The figure title.  Defaults to an empty string.
        vmin (float, optional): The minimum intensity value for display.  Defaults to the
            minimum over every voxel of every volume.
        vmax (float, optional): The maximum intensity value for display.  Defaults to the
            maximum over every voxel of every volume.
        slice_label (str or list of str, optional): The label at the start of each
            panel title.  Defaults to no label.
        slice_axis (int or list of int, optional): The axis along which to slice, counted
            in the axes of each array.  The axis is 1, 2, or 3 for a 4D array, and 0, 1, or 2
            for a 3D array.  Defaults to the z axis.  If you display 3D and 4D arrays
            together, give a list.
        cmap (str, optional): The colormap.  Defaults to "gray".
        show_instructions (bool, optional): If True, the figure shows a short help
            message.  Defaults to True.
        block (bool, optional): If True, the function returns only after the window is
            closed.  If False, the function returns at once, as in :func:`slice_viewer`.
            Defaults to True.
        save_fn (callable, optional): A function that replaces the built-in HDF5 writer
            for the viewer's Save action.  The viewer calls ``save_fn`` as
            ``save_fn(file_path, array, array_name, attributes_dict)``.
        fps (float, optional): The playback speed in frames per second.  Defaults to 5.

    Returns:
        SliceViewer4D: The viewer object.

    Example:
        >>> import mbirtorch
        >>> recon_4d, recon_dict = mace4d_model.recon(sinogram)
        >>> mbirtorch.slice_viewer4d(recon_4d, data_dicts=recon_dict, vmin=0, vmax=0.06)
    """
    datasets = [_to_numpy(dataset) for dataset in datasets]
    if isinstance(data_dicts, dict):
        data_dicts = convert_subdicts_to_strings(data_dicts)
    elif isinstance(data_dicts, (list, tuple)):
        data_dicts = [convert_subdicts_to_strings(d) for d in data_dicts]
    return _slice_viewer4d(*datasets, data_dicts=data_dicts, title=title,
                           vmin=vmin, vmax=vmax, slice_label=slice_label,
                           slice_axis=slice_axis, cmap=cmap,
                           show_instructions=show_instructions, block=block,
                           save_fn=save_fn, fps=fps, movie_fn=save_volume_as_gif)

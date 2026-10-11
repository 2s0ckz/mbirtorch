"""This package holds three viewers: the slice viewer, the 4D slice viewer, and the
geometry viewer.

The viewers are defined in the modules ``slice_figure``, ``slice_figure4d``, ``geometry_scene``,
and ``geometry_figure``.  These modules import only numpy and the base matplotlib package.
As a result, the package loads without torch and without a GUI toolkit.  The functions
``mbirtorch.slice_viewer`` and ``mbirtorch.slice_viewer4d`` in ``mbirtorch.view_utils``
convert tensors and data dicts and then call these viewers.
"""

from .slice_figure import SliceViewer, VolumeStack, slice_viewer
from .slice_figure4d import SliceViewer4D, VolumeStack4D, slice_viewer4d
from .geometry_scene import GeometryScene
from .geometry_figure import GeometryFigure, geometry_viewer

__all__ = ['SliceViewer', 'VolumeStack', 'slice_viewer', 'GeometryScene',
           'GeometryFigure', 'geometry_viewer', 'SliceViewer4D', 'VolumeStack4D',
           'slice_viewer4d']

"""This module draws an mbirtorch scan geometry with matplotlib, in five panels.

The panels are a 3D view, a top view of the xy plane, a side view of the yz plane, the detector
face, and a text panel of derived numbers.  A slider steps through the views.  Toggles show or
hide a sinogram, a phantom, and a second geometry for comparison.  The figure draws negative z
at the top, so the volume appears in the same order as its array index.  This module calls the
point where the central ray meets the detector the detector iso.
"""

import os
import textwrap

import numpy as np
import matplotlib  # This imports only the base package, so no GUI backend is loaded.

from .geometry_scene import GeometryScene, required_parameter_names

__all__ = ['GeometryFigure', 'geometry_viewer', 'COLORS',
           'DEFAULT_ELEVATION_DEG', 'DEFAULT_AZIMUTH_DEG',
           'VOLUME_BOX_EDGES', 'ZOOM_MODES', 'BLIT_BACKENDS',
           'Z_UP_SIGN', 'TOP_PANEL_COLUMNS', 'SIDE_PANEL_COLUMNS',
           'ISO_NAME', 'CENTER_NAME', 'PHANTOM_NAME']


# --- names used in every panel ---

#: The name of the point where the central ray meets the detector.
ISO_NAME = 'detector iso'

#: The name of the detector grid's center.  The detector offsets move the center away from the iso.
CENTER_NAME = 'detector center'

#: The widget row and the 3D panel's legend use this name for the ``recon`` array.
PHANTOM_NAME = 'phantom'


# --- the display convention ---

#: The sign of the object-frame z that points up the screen.  The value -1 draws
#: -z at the top of every panel, and +1 draws +z up.  The sign changes only the
#: display, not the geometry.
Z_UP_SIGN = -1


def _z_up_is_negative():
    """This function returns True when -z points up the screen.  See :data:`Z_UP_SIGN`."""
    return Z_UP_SIGN < 0


def _top_view_title():
    """This function returns the top panel's title, which names the viewing side."""
    side = 'left' if _z_up_is_negative() else 'right'
    return ('Top view, the xy plane, seen from -z (the top)\n'
            f'y increases to the {side}')


def _side_view_title():
    """This function returns the side panel's title, which names the viewing side."""
    return 'Side view, the yz plane, seen from +x'


def _detector_view_title():
    """This function returns the first line of the detector panel's title, which names the row
    order.
    """
    if _z_up_is_negative():
        return 'Detector face, seen from the source, row 0 at the top'
    return 'Detector face, row 0 at the bottom'


def _camera_roll_deg():
    """This function returns the 3D camera roll, in degrees, that puts -z at the top of the
    panel.
    """
    return CAMERA_ROLL_DEG if _z_up_is_negative() else 0.0


def _convention_note():
    """This function returns the text that names the display convention in the text panel."""
    return '-z up.' if _z_up_is_negative() else '+z up.'


def _screen_step(delta):
    """This function returns 1 when a step ``delta`` along a projected panel's axis points right
    or up the screen, and -1 otherwise.
    """
    forward = 1 if float(delta) >= 0.0 else -1
    return -forward if _z_up_is_negative() else forward


def _screen_bottom(values):
    """This function returns the value in ``values`` that a panel draws lowest on the screen.
    The figure uses this value to place a label below an object.
    """
    values = np.asarray(values, dtype=np.float64)
    return float(np.max(values) if _z_up_is_negative() else np.min(values))


def _screen_pair(low, high):
    """This function returns the limits ``(low, high)`` of one axis.  The limits are reversed
    when -z points up the screen.
    """
    if _z_up_is_negative():
        return (float(high), float(low))
    return (float(low), float(high))


#: The top view's horizontal and vertical axes, as indices into (x, y, z).  The
#: top view draws y across the screen and x down it.
TOP_PANEL_COLUMNS = (1, 0)

#: The side view's horizontal and vertical axes.  The side view draws y across the
#: screen and z down it.
SIDE_PANEL_COLUMNS = (1, 2)

#: The top view draws these two of a view's four corner rays darker.  These two rays
#: bound the plane that the top view shows.  They lie at the two ends of the channel direction.
TOP_EDGE_RAYS = (0, 1)

#: The side view draws these two corner rays darker.  These two rays bound the plane
#: that the side view shows.  They lie at the two ends of the row direction.
SIDE_EDGE_RAYS = (0, 3)


# --- appearance ---

#: Each element has the same color in every panel.
COLORS = {
    'source': '#ff7f0e',        # orange
    'detector': '#1f77b4',      # blue
    'rays': '#c0c0c0',          # light gray
    'central_ray': '#111111',   # near black
    'volume': '#9467bd',        # purple
    'ror': '#17becf',           # cyan
    'axis': '#555555',          # gray
    'pixel0': '#e7298a',        # magenta
    'voxel0': '#2ca02c',        # green
    'trajectory': '#8c564b',    # brown
    'overshoot': '#d62728',     # red
    'compare': '#6b8e00',       # dark yellow-green
}

TITLE_FONT_SIZE = 9
LABEL_FONT_SIZE = 8
TICK_FONT_SIZE = 7
ANNOTATION_FONT_SIZE = 7
LEGEND_FONT_SIZE = 6.5
TEXT_PANEL_FONT_SIZE = 7.5
WIDGET_FONT_SIZE = 8

#: The text panel wraps its sentences at this many characters per line.
TEXT_PANEL_WRAP_WIDTH = 52

#: A label sits this far to the side of the point it names, in points.
LABEL_GAP_POINTS = 6

#: The labels at the detector iso sit this far above or below that point, in points.
ISO_LABEL_GAP_POINTS = 6.0
LABEL_LINE_POINTS = 11.0

#: The top view's pixel-0 label sits this far from its marker, in points.
PIXEL0_LABEL_GAP_POINTS = 14.0

#: The top view's "source travel" label sits this far from the end of the
#: rotation arc, in points.
ARC_LABEL_GAP_POINTS = 8.0

#: The projected panels use these short forms of ``det_channel_offset`` and
#: ``det_row_offset``.  The text panel prints the full parameter names.
CHANNEL_OFFSET_LABEL = 'chan offset'
ROW_OFFSET_LABEL = 'row offset'

SOURCE_MARKER_SIZE = 13
INDEX_MARKER_SIZE = 8
INDEX_MARKER_WIDTH = 1.8

#: A 2D panel draws the two corner rays that bound its plane in this color.
EMPHASIZED_RAY_COLOR = '#8a8a8a'

#: Every comparison line uses this dash pattern and line width.
COMPARE_DASHES = (0, (5.0, 2.0))
COMPARE_LINEWIDTH = 1.4

#: The angle-0 reference uses this alpha, dash pattern, and line width.
REFERENCE_ALPHA = 0.45
REFERENCE_DOTS = (0, (1.0, 2.0))
REFERENCE_LINEWIDTH = 1.2

#: The arrowhead of the reference central ray spans this fraction of it.
REFERENCE_ARROW_FRACTION = 0.06

#: The angle-0 reference's caption sits here in the top view, in that panel's
#: axes coordinates.
REFERENCE_LABEL_CORNER = (0.015, 0.985)

# --- the two data overlays ---

#: The figure paints the sinogram on the detector face with this colormap.
SINOGRAM_COLORMAP = 'gray'

#: The painted sinogram keeps at most this many detector pixels along each
#: detector axis.  A feature narrower than the sampling stride may not appear.
SINOGRAM_DISPLAY_PIXELS = 128

#: The fill of the reconstruction silhouette uses this alpha.
RECON_FILL_ALPHA = 0.35

#: The phantom outline uses these line widths in a projected panel, the 3D panel, and the
#: detector face.
RECON_OUTLINE_LINEWIDTH = 1.6
RECON_OUTLINE_3D_LINEWIDTH = 1.4
RECON_OUTLINE_DETECTOR_LINEWIDTH = 0.9

#: This is the drawing order of the phantom outline.  The outline lies above the
#: silhouette's fill and below the panel's lines.
RECON_OUTLINE_ZORDER = 1.8

#: A voxel belongs to the reconstruction's support when its absolute value exceeds
#: this fraction of the largest absolute value.  The caller may give its own threshold.
DEFAULT_RECON_THRESHOLD_FRACTION = 0.1

#: The 3D panel draws at most this many sections of the phantom.  The panel spaces
#: the sections evenly along the axis on which the support is thinnest.
PHANTOM_SECTION_COUNT = 9

#: This setting chooses what one section shows.  'section' outlines the support in
#: one plane.  'slab' outlines the support over a range of planes, so every support
#: voxel appears.
PHANTOM_SECTION_KIND = 'section'
PHANTOM_SECTION_KINDS = ('section', 'slab')

#: When two axes have the same thinnest extent, the sections lie across the first
#: of these axes.
PHANTOM_SECTION_AXIS_PREFERENCE = ('z', 'y', 'x')

#: The phantom's section outlines hold at most this many points in total.  Each
#: section gets an equal part of this total.  When a section's outline has more
#: points than its part, the figure draws it from a coarsened mask.
PHANTOM_OUTLINE_POINT_BUDGET = 8000

#: One chunk of the support computation holds about this many reconstruction elements.
OVERLAY_CHUNK_ELEMENTS = 2 ** 24

#: This is the drawing order of the two overlays.  Both lie under every line, which matplotlib
#: draws at order 2.
SINOGRAM_ZORDER = 1.5
SILHOUETTE_ZORDER = 1.0

#: This keyword hides the parts of a 3D artist that lie outside the axes box.
#: matplotlib added ``axlim_clip`` in release 3.10.  An older release omits it.
AXLIM_CLIP = ({'axlim_clip': True}
              if matplotlib.__version_info__ >= (3, 10) else {})

#: These are the default 3D camera angles, in degrees.
DEFAULT_ELEVATION_DEG = -25.0
DEFAULT_AZIMUTH_DEG = -40.0

#: The 3D camera is rolled this far about its own axis, in degrees, to put -z
#: at the top of the panel.
CAMERA_ROLL_DEG = 180.0

#: An ellipse of the region of reconstruction is drawn with this many points.
ELLIPSE_SAMPLES = 65

#: The 3D panel samples each ray at this many points.  The 3D panel clips a line
#: point by point, so a line of only two points disappears when it leaves the limits.
RAY_SAMPLES = 33

#: The figure samples a projected volume edge that crosses the detector's boundary at this
#: many points.
EDGE_CLIP_SAMPLES = 64

#: A 2D panel adds this fractional margin around its data.
PANEL_MARGIN = 0.10

#: These are the two states of the 3D panel's zoom control.  ``'scan'`` puts the source, the
#: detector, and the volume in one cube.  ``'volume'`` encloses the volume alone.
ZOOM_MODES = ('scan', 'volume')
DEFAULT_ZOOM = 'scan'

#: The width of the ``'volume'`` zoom cube is this multiple of the volume's
#: largest extent.
ZOOM_VOLUME_WIDTH_FACTOR = 3.0

#: The figure computes the panel limits from this many sampled views.  When a
#: view's content falls outside the limits, the figure enlarges the limits and
#: redraws the whole figure once.
LIMIT_SAMPLE_VIEWS = 16

#: The text panel's comparison section lists at most this many differing
#: parameters.  The comparison window lists every difference.
MAX_COMPARISON_ENTRIES = 6

#: These are the comparison window's width, row height, title height, margin, and
#: largest height, all in inches.  The window's height grows with its number of rows.
COMPARE_WINDOW_WIDTH_IN = 7.5
COMPARE_ROW_HEIGHT_IN = 0.28
COMPARE_WINDOW_TITLE_IN = 0.55
COMPARE_WINDOW_MARGIN_IN = 0.25
COMPARE_WINDOW_MAX_HEIGHT_IN = 10.0

#: These are the comparison window's font size and the width of one DejaVu Sans Mono
#: character.  The width is a fraction of the font size.  The figure uses this width to shrink
#: the font until the longest line fits the window.
COMPARE_WINDOW_FONT_SIZE = 9.5
MONOSPACE_ADVANCE = 0.602

#: These are the comparison window's title and its desktop window name.
COMPARE_WINDOW_TITLE = ('Comparison: primary (solid) against '
                        'comparison (dashed)')
COMPARE_WINDOW_NAME = 'Geometry comparison'

#: These are the three column headings of the comparison window's table.
COMPARE_TABLE_HEADING = ('name', 'primary (solid)', 'comparison (dashed)')

#: Two columns of the comparison window's table are this many characters apart.
COMPARE_COLUMN_GAP = 2

#: The text panel uses this font size while a comparison is drawn.
COMPARING_FONT_SIZE = 6.0

#: The text panel's three blocks may fill at most this fraction of its height.
TEXT_PANEL_FILL = 0.97

#: The text panel does not shrink its font below this size.
TEXT_PANEL_FONT_MINIMUM = 5.0

#: The widgets sit at these rectangles in figure coordinates, given as (left,
#: bottom, width, height).  The six toggles sit right of the slider in two rows.
SLIDER_RECT = (0.07, 0.045, 0.38, 0.025)
TRAJECTORY_CHECK_RECT = (0.50, 0.052, 0.12, 0.038)
ZOOM_CHECK_RECT = (0.645, 0.052, 0.15, 0.038)
REFERENCE_CHECK_RECT = (0.815, 0.052, 0.15, 0.038)
SINOGRAM_CHECK_RECT = (0.50, 0.010, 0.12, 0.038)
RECON_CHECK_RECT = (0.645, 0.010, 0.15, 0.038)
COMPARE_CHECK_RECT = (0.815, 0.010, 0.15, 0.038)

#: The bottom of the panel grid lies here, as a fraction of the figure's height.
#: The space below it holds the widget row.
GRID_BOTTOM = 0.115

#: The bottom of the detector-face panel lies here, as a fraction of the figure's
#: height.  The space under the panel holds the panel's x label and its legend.
DETECTOR_PANEL_BOTTOM = 0.19

#: The bottom edge of the detector-face panel's legend lies here, as a fraction of
#: the figure's height.  The legend has this many columns.
DETECTOR_LEGEND_BOTTOM = 0.098
DETECTOR_LEGEND_COLUMNS = 3


def _corner_edges():
    """This function returns the volume box's twelve edges as pairs of corner indices.  Two
    corners share an edge when their indices differ in one bit.
    """
    edges = []
    for first in range(8):
        for second in range(first + 1, 8):
            if bin(first ^ second).count('1') == 1:
                edges.append((first, second))
    return tuple(edges)


#: These are the twelve edges of the volume box.
VOLUME_BOX_EDGES = _corner_edges()

#: These corner indices go around the volume box's footprint in the xy plane.
#: The four corners all have the low z value.
_XY_FOOTPRINT_WALK = (0, 2, 6, 4, 0)

#: These corner indices go around the volume box's face in the yz plane.  The
#: four corners all have the low x value.
_YZ_FACE_WALK = (0, 1, 3, 2, 0)

#: On these backends, a view change redraws only the artists that moved (blitting).
#: On other backends, a view change redraws the whole figure.
BLIT_BACKENDS = {'agg', 'tkagg'}

# These backends have no interactive window.  On these backends,
# GeometryFigure.show prints a message and returns.
NONINTERACTIVE_BACKENDS = {'agg', 'pdf', 'ps', 'svg', 'template', 'cairo'}


# _load_pyplot sets these names when the first figure is constructed.  As a
# result, importing this module does not select a matplotlib backend.
plt = None
Poly3DCollection = None
Rectangle = None
FancyArrowPatch = None
Slider = None
CheckButtons = None
IdentityTransform = None
Bbox = None
ListedColormap = None


def _load_pyplot():
    """This function imports pyplot, the widgets, and the drawing classes on first use."""
    global plt, Poly3DCollection, Rectangle, FancyArrowPatch
    global Slider, CheckButtons, IdentityTransform, Bbox, ListedColormap
    if plt is not None:
        return
    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d.art3d import (
        Poly3DCollection as _Poly3DCollection)
    from matplotlib.patches import (Rectangle as _Rectangle,
                                    FancyArrowPatch as _FancyArrowPatch)
    from matplotlib.widgets import (Slider as _Slider,
                                    CheckButtons as _CheckButtons)
    from matplotlib.transforms import (IdentityTransform as _IdentityTransform,
                                       Bbox as _Bbox)
    from matplotlib.colors import ListedColormap as _ListedColormap
    plt = _plt
    Poly3DCollection = _Poly3DCollection
    Rectangle = _Rectangle
    FancyArrowPatch = _FancyArrowPatch
    Slider = _Slider
    CheckButtons = _CheckButtons
    IdentityTransform = _IdentityTransform
    Bbox = _Bbox
    ListedColormap = _ListedColormap


# --- small formatting and geometry-free drawing helpers ---

def _three_figures(value):
    """This function formats a number to three significant figures.  It writes negative zero
    as ``0``."""
    return f'{float(value) + 0.0:.3g}'


def _is_tensor(values):
    """This function returns True when an array-like is a torch tensor, without importing torch."""
    return hasattr(values, 'detach')


def _to_host(values):
    """This function returns an array-like as a numpy array.  A tensor is moved to the host, and
    a numpy array is not copied.
    """
    if _is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _absolute(values):
    """This function computes the absolute value of an array on the array's device."""
    if _is_tensor(values):
        return values.abs()
    return np.abs(values)


def _any_along(mask, axis):
    """This function returns whether any entry of a boolean array is True along one axis.
    The function computes the result on the array's device.
    """
    if _is_tensor(mask):
        return mask.any(dim=axis)
    return mask.any(axis=axis)


def _shape_of(values):
    """This function returns the shape of an array-like without copying it to the host."""
    return tuple(np.shape(values))


def _reject_divided(name, values):
    """This function raises a TypeError for a ``Shards`` container, which holds one tensor per
    device and has no shape.
    """
    if hasattr(values, 'tensors') and hasattr(values, 'placement'):
        raise TypeError(
            f'geometry_viewer does not accept a {name} in the divided device '
            'form.  Gather it to the host first with shards.gather(), then '
            'pass the host array.')


def _as_scene(model_or_scene, **scene_kwargs):
    """This function returns a ``GeometryScene``.  It builds the scene from a model when one is
    given."""
    if isinstance(model_or_scene, GeometryScene):
        if scene_kwargs:
            raise TypeError('Scene options apply only when a model is given; '
                            'a GeometryScene is already built.')
        return model_or_scene
    return GeometryScene.from_model(model_or_scene, **scene_kwargs)


def _ellipse_points(center, semi_axis_x, semi_axis_y, height):
    """This function returns points on an axis-aligned ellipse at one height, as an (N, 3)
    array.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, ELLIPSE_SAMPLES)
    x = center[0] + semi_axis_x * np.cos(angle)
    y = center[1] + semi_axis_y * np.sin(angle)
    z = np.full_like(angle, float(height))
    return np.stack([x, y, z], axis=1)


def _point_3d(point):
    """This function returns one point as three one-element arrays for ``set_data_3d``."""
    point = np.asarray(point, dtype=np.float64).reshape(3)
    return point[0:1], point[1:2], point[2:3]


def _sampled_segment(start, end, count=RAY_SAMPLES):
    """This function returns a straight segment of ``count`` points from ``start`` to ``end``."""
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    fraction = np.linspace(0.0, 1.0, count)[:, None]
    return start[None, :] + fraction * (end - start)[None, :]


def _joined(parts, width=3):
    """This function joins polylines of ``width`` columns into one array with a row of NaN
    between them, so that one matplotlib line draws them all.
    """
    pieces = []
    separator = np.full((1, width), np.nan)
    for index, part in enumerate(parts):
        if index:
            pieces.append(separator)
        pieces.append(np.asarray(part, dtype=np.float64).reshape(-1, width))
    if not pieces:
        return np.zeros((0, width))
    return np.concatenate(pieces, axis=0)


def _runs_of_equal_flags(flags):
    """This function returns (slice, value) for each run of equal entries in a boolean array.
    Neighboring runs overlap by one entry, so polylines drawn from them join up.
    """
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0:
        return []
    breaks = np.flatnonzero(flags[1:] != flags[:-1]) + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [flags.size]])
    runs = []
    for start, stop in zip(starts, stops):
        # Extend by one so that this run reaches the next run's first point.
        runs.append((slice(start, min(stop + 1, flags.size)),
                     bool(flags[start])))
    return runs


def _true_runs(flags):
    """This function returns the half-open (start, stop) index pairs of the runs of True in a
    boolean array.
    """
    flags = np.asarray(flags, dtype=bool)
    # A False on each end turns every run into one rise and one fall, so the
    # changes come in pairs and each pair is one run.
    padded = np.concatenate([[False], flags, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop))
            for start, stop in zip(changes[::2], changes[1::2])]


def _mask_outline(mask, across, down):
    """This function returns the boundary of an (R, C) boolean mask as an (N, 2) array of
    (horizontal, vertical) points.  A row of NaN separates the segments.  ``across`` holds the
    positions of the C+1 grid lines between mask columns.  ``down`` holds the positions of the
    R+1 grid lines between mask rows.  The function treats every entry outside the array as False.
    """
    mask = np.asarray(mask, dtype=bool)
    # A ring of False gives an entry at the array's edge a neighbor to differ from.
    padded = np.pad(mask, 1)
    parts = []
    # Grid line c lies between mask columns c - 1 and c.  The outline follows it where they differ.
    for column in range(mask.shape[1] + 1):
        differs = padded[1:-1, column] != padded[1:-1, column + 1]
        for start, stop in _true_runs(differs):
            parts.append(np.array([[across[column], down[start]],
                                   [across[column], down[stop]]]))
    # This loop does the same for the grid lines between rows.
    for row in range(mask.shape[0] + 1):
        differs = padded[row, 1:-1] != padded[row + 1, 1:-1]
        for start, stop in _true_runs(differs):
            parts.append(np.array([[across[start], down[row]],
                                   [across[stop], down[row]]]))
    return _joined(parts, 2)


# --- reading the reconstruction's support a chunk at a time ---

def _slice_chunks(shape):
    """This function splits an array of shape (rows, cols, slices) into chunks along the slice
    axis.  Each chunk holds about :data:`OVERLAY_CHUNK_ELEMENTS` elements.  The function yields
    the half-open (first, last) slice range of each chunk.
    """
    per_slice = max(int(shape[0]) * int(shape[1]), 1)
    step = max(1, OVERLAY_CHUNK_ELEMENTS // per_slice)
    for start in range(0, int(shape[2]), step):
        yield start, min(start + step, int(shape[2]))


def _largest_magnitude(values):
    """This function returns the largest absolute value of an array as a float.  It returns zero
    for an empty array.  The function reads the array one chunk at a time on the device that
    holds the array.
    """
    shape = _shape_of(values)
    if 0 in shape:
        return 0.0
    largest = 0.0
    for first, last in _slice_chunks(shape):
        largest = max(largest,
                      float(_absolute(values[:, :, first:last]).max()))
    return largest


def _support_projections(values, level):
    """This function projects the support of ``values`` along each axis.  It returns a dict of
    boolean arrays with the keys ``'xy'``, ``'yz'``, and ``'xz'``.  A voxel is in the
    support when its absolute value is above ``level``.
    """
    rows, cols, slices = _shape_of(values)
    projections = dict(xy=np.zeros((rows, cols), dtype=bool),
                       yz=np.zeros((rows, slices), dtype=bool),
                       xz=np.zeros((cols, slices), dtype=bool))
    for first, last in _slice_chunks((rows, cols, slices)):
        mask = _absolute(values[:, :, first:last]) > level
        projections['xy'] |= _to_host(_any_along(mask, 2))
        projections['yz'][:, first:last] = _to_host(_any_along(mask, 1))
        projections['xz'][:, first:last] = _to_host(_any_along(mask, 0))
    return projections


def _support_extent(flags):
    """This function returns ``(lo, hi)``, the indices of the first and the last True entry of a
    1-D boolean array.  It returns None when no entry is True.
    """
    indices = np.flatnonzero(np.asarray(flags, dtype=bool))
    if indices.size == 0:
        return None
    return int(indices[0]), int(indices[-1])


def _section_extents(projections):
    """This function returns the range ``(lo, hi)`` of planes that the support spans along each of
    ``'x'``, ``'y'``, and ``'z'``.  It returns None when the support is empty.  The row index is
    y, the column index is x, and the slice index is z.
    """
    extents = dict(y=_support_extent(projections['xy'].any(axis=1)),
                   x=_support_extent(projections['xy'].any(axis=0)),
                   z=_support_extent(projections['yz'].any(axis=0)))
    if any(extent is None for extent in extents.values()):
        return None
    return extents


def _section_axis(projections):
    """This function returns the axis along which the support is thinnest.  It returns None when
    the support is empty.  A tie is broken by :data:`PHANTOM_SECTION_AXIS_PREFERENCE`.
    """
    extents = _section_extents(projections)
    if extents is None:
        return None
    counts = {axis: high - low + 1 for axis, (low, high) in extents.items()}
    smallest = min(counts.values())
    for axis in PHANTOM_SECTION_AXIS_PREFERENCE:
        if counts[axis] == smallest:
            return axis
    return None


def _section_positions(lo, hi, count, kind):
    """This function spreads up to ``count`` sections over the planes ``lo`` to ``hi``.  It
    returns ``(position, (first_plane, last_plane))`` for each section.  The position of a
    slab is the middle of its planes.
    """
    if kind == 'slab':
        count = min(int(count), hi - lo + 1)
        bounds = np.round(np.linspace(lo, hi + 1, count + 1)).astype(int)
        return [(0.5 * (float(bounds[index]) + float(bounds[index + 1]) - 1.0),
                 (int(bounds[index]), int(bounds[index + 1]) - 1))
                for index in range(count)]
    if hi - lo + 1 <= count:
        planes = np.arange(lo, hi + 1)
    else:
        planes = np.unique(np.round(np.linspace(lo, hi, count)).astype(int))
    return [(int(plane), (int(plane), int(plane))) for plane in planes]


def _section_mask(values, axis, planes, level):
    """This function returns one section's support as a (down, across) boolean mask on the host.
    The function reads only the planes in ``planes``.  The mask axes are the row and the column
    for axis ``'z'``.  They are the column and the slice for axis ``'y'``.  They are the row and
    the slice for axis ``'x'``.
    """
    first, last = int(planes[0]), int(planes[1])
    if axis == 'z':
        block, reduced = values[:, :, first:last + 1], 2
    elif axis == 'y':
        block, reduced = values[first:last + 1, :, :], 0
    else:
        block, reduced = values[:, first:last + 1, :], 1
    return _to_host(_any_along(_absolute(block) > level, reduced))


def _coarsen(mask, factor):
    """This function reduces a boolean mask by blocks of ``factor`` by ``factor`` entries.  A
    block is True when any of its entries is True.
    """
    factor = int(factor)
    if factor <= 1:
        return mask
    down = -(-mask.shape[0] // factor)
    across = -(-mask.shape[1] // factor)
    padded = np.zeros((down * factor, across * factor), dtype=bool)
    padded[:mask.shape[0], :mask.shape[1]] = mask
    return padded.reshape(down, factor, across, factor).any(axis=(1, 3))


def _section_outline(mask, factor):
    """This function coarsens a section's mask by ``factor`` and returns the outline of the
    result.  The outline is an (N, 2) array of (across, down) fractional voxel indices.  A row of
    NaN separates the segments.
    """
    coarse = _coarsen(mask, factor)
    across = np.minimum(-0.5 + factor * np.arange(coarse.shape[1] + 1),
                        mask.shape[1] - 0.5)
    down = np.minimum(-0.5 + factor * np.arange(coarse.shape[0] + 1),
                      mask.shape[0] - 0.5)
    return _mask_outline(coarse, across, down)


def _budgeted_outline(mask, share):
    """This function returns ``(outline, factor)`` for a section's mask.  It uses the smallest
    coarsening factor that gives an outline of at most ``share`` points.  If the coarsened mask has
    only one entry, the function returns the outline of that mask even when the outline has more
    points.
    """
    factor = 1
    while True:
        outline = _section_outline(mask, factor)
        finite = int(np.isfinite(outline).all(axis=1).sum())
        coarse_down = -(-mask.shape[0] // factor)
        coarse_across = -(-mask.shape[1] // factor)
        if finite <= share or (coarse_down <= 1 and coarse_across <= 1):
            return outline, factor
        factor += 1


def _bounds(points, margin=PANEL_MARGIN):
    """This function returns the (low, high) range of each column of ``points``.  It extends each
    range on both sides by ``margin`` times the largest extent.
    """
    points = np.asarray(points, dtype=np.float64)
    low = np.nanmin(points, axis=0)
    high = np.nanmax(points, axis=0)
    pad = margin * float(np.max(high - low))
    if pad <= 0.0:
        pad = 1.0
    return tuple((float(low[index] - pad), float(high[index] + pad))
                 for index in range(points.shape[1]))


def _cube_bounds(points, margin=PANEL_MARGIN):
    """This function returns three (low, high) pairs that form a cube holding ``points``.  With
    a cube, each angle in the drawing equals the angle in the geometry.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    low = np.nanmin(points, axis=0)
    high = np.nanmax(points, axis=0)
    center = 0.5 * (low + high)
    extent = float(np.max(high - low)) * (1.0 + 2.0 * margin)
    if extent <= 0.0:
        extent = 1.0
    return tuple((float(center[index] - 0.5 * extent),
                  float(center[index] + 0.5 * extent))
                 for index in range(3))


def _far_corner(view, columns):
    """This function returns the detector corner that is farthest from the pixel-0 marker in the
    panel that draws ``columns``.  The corner is a (3,) array.  The figure puts the
    detector's label at this corner.
    """
    corners = np.asarray(view.detector_corners, dtype=np.float64)
    columns = list(columns)
    pixel0 = np.asarray(view.detector_pixel0, dtype=np.float64)[columns]
    distance = np.linalg.norm(corners[:, columns] - pixel0[None, :], axis=1)
    return corners[int(np.argmax(distance))]


def _place_beside(label, point, side=None, vertical=None):
    """This function moves a label beside a point in a 2D panel.  ``side`` is 1 for the right
    and -1 for the left, and the default puts the label toward the middle of the panel.
    ``vertical`` is the vertical offset in points.
    """
    if side is None:
        low, high = label.axes.get_xlim()
        past_middle = float(point[0]) > 0.5 * (low + high)
        # An inverted horizontal axis draws a point past the middle on the left
        # half of the screen, so its label must read right to stay in the panel.
        if label.axes.xaxis_inverted():
            past_middle = not past_middle
        side = -1 if past_middle else 1
    label.set_ha('right' if side < 0 else 'left')
    if vertical is None:
        vertical = float(label.xyann[1])
    else:
        vertical = float(vertical)
        label.set_va('bottom' if vertical >= 0.0 else 'top')
    label.xyann = (side * LABEL_GAP_POINTS, vertical)
    label.xy = (float(point[0]), float(point[1]))


def _within(points, limits):
    """This function returns True when every point lies inside the per-column limits.
    Each pair of limits may come in either order.
    """
    points = np.asarray(points, dtype=np.float64)
    for index, pair in enumerate(limits):
        low, high = min(pair), max(pair)
        column = points[:, index]
        column = column[np.isfinite(column)]
        if column.size == 0:
            continue
        if float(np.min(column)) < low or float(np.max(column)) > high:
            return False
    return True


def _lateral_fit_text(quantities):
    """This function returns the lateral half of the fit statement, which says whether the shape
    tested stays inside the detector's channel range in every view.
    """
    shape = quantities['fit_shape']
    if quantities['fits_laterally']:
        return f'yes ({shape})'
    over = _three_figures(quantities['worst_channel_overshoot_pixels'])
    return f'no ({shape}, {over} px over)'


def _axial_fit_text(quantities):
    """This function returns the axial half of the fit statement.  This half says whether the
    shape stays inside the detector's rows.  For a helical scan, it says whether the shape stays
    inside the z range the detector sweeps.  See ``GeometryScene.fit_report``.
    """
    if quantities['helical_fit_rule']:
        swept = (f"{_three_figures(quantities['swept_z_min'])} to "
                 f"{_three_figures(quantities['swept_z_max'])} ALU swept")
        if quantities['fits_axially']:
            return f'yes ({swept})'
        return f'no (z extent not in {swept})'
    if quantities['fits_axially']:
        return 'yes'
    over = _three_figures(quantities['worst_row_overshoot_pixels'])
    return f'no ({over} px over)'


#: A comparison line prints numbers to this many significant figures.  The line
#: uses ``COMPARISON_LONG_FIGURES`` when the short form makes the two values look equal.
COMPARISON_FIGURES = 3
COMPARISON_LONG_FIGURES = 7


def _value_text(value, figures=COMPARISON_FIGURES):
    """This function formats one parameter or derived value as text.  The function writes a
    long array as its shape.
    """
    if value is None:
        return 'not set'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, str):
        return value if len(value) <= 28 else value[:25] + '...'
    array = np.asarray(value)
    if array.ndim == 0 or array.size == 1:
        return f'{float(array.reshape(-1)[0]) + 0.0:.{figures}g}'
    if array.ndim == 1 and array.size <= 4:
        return ('(' + ', '.join(f'{float(entry) + 0.0:.{figures}g}'
                                for entry in array) + ')')
    return f'array{tuple(int(n) for n in array.shape)}'


def _difference_values(mine, theirs):
    """This function returns the primary and the comparison value of one difference as text.
    The function uses more significant figures when the short form makes the two look equal.
    """
    left, right = _value_text(mine), _value_text(theirs)
    if left == right:
        left = _value_text(mine, COMPARISON_LONG_FIGURES)
        right = _value_text(theirs, COMPARISON_LONG_FIGURES)
    return left, right


def _difference_text(name, mine, theirs):
    """This function returns one comparison line, as ``name: primary -> comparison``."""
    left, right = _difference_values(mine, theirs)
    return f'{name}: {left} -> {right}'


def _derived_difference_sentence(count):
    """This function returns the sentence in the text panel that gives the number ``count`` of
    derived quantities that differ.
    """
    if count == 0:
        return 'no derived quantity differs'
    noun = 'quantity differs' if count == 1 else 'quantities differ'
    return f'{count} derived {noun}; see the comparison window'


def _companion_path(path):
    """This function returns the file path of the comparison window for a saved figure.  The path
    is in the same directory as the figure.  For example, ``scan.png`` gives
    ``scan_comparison.png``.
    """
    stem, suffix = os.path.splitext(os.fspath(path))
    return stem + '_comparison' + suffix


class GeometryFigure:
    """This class is used to draw one scan geometry as a five-panel matplotlib figure.

    The method :meth:`set_view` moves to another view.  The other ``set_`` methods add, remove,
    show, or hide parts of the drawing.  Constructing the figure opens no window.  The method
    :meth:`save` writes a file, and the method :meth:`show` opens a window.

    Args:
        model_or_scene: a ``GeometryScene``, or a ``TomographyModel`` from
            which one is built with ``GeometryScene.from_model``.
        view_index (int, optional): the view to draw.  Defaults to 0.
        show_trajectory (bool, optional): whether to draw the source's path
            over all views.  Defaults to False.
        figsize (tuple, optional): the figure size in inches.
        title (str, optional): the figure's suptitle.  Defaults to a line
            naming the geometry kind and the model's shapes.
        elevation_deg, azimuth_deg (float, optional): the initial 3D camera
            angles, in degrees.
        compare (optional): a second geometry to draw over the first.  It may be a
            ``GeometryScene`` or a ``TomographyModel``.  It may also be a dictionary
            of parameter overrides, such as ``dict(det_channel_offset=12.5)``.  The
            figure applies the overrides to a copy of this scene's parameters.  None
            draws no comparison.
        zoom (str, optional): the 3D panel's zoom, one of ``ZOOM_MODES``.
        show_reference (bool, optional): whether to draw the source and the
            detector at their angle-0 position as a fixed reference.  Defaults
            to True.
        widgets (bool, optional): whether to build the slider and the six
            toggles.  Defaults to True.
        blit (bool, optional): whether a view change may redraw only the
            artists that moved.  Defaults to True.  False redraws the whole
            figure on every change.
        sinogram (array_like, optional): a sinogram of shape
            ``(num_views, num_det_rows, num_det_channels)``.  The figure paints
            the current view of the sinogram on the detector face.  None
            (default) paints no sinogram.  When the detector has more than
            ``SINOGRAM_DISPLAY_PIXELS`` pixels along an axis, the figure
            subsamples the sinogram for display.  The subsampling runs on the
            device that holds the array.
        vmin, vmax (float, optional): the ends of the gray scale.  None
            (default) uses the smallest and the largest displayed value of the
            sinogram.
        recon (array_like, optional): a reconstruction or a phantom of shape
            ``recon_shape``.  The figure draws it as a silhouette in the top
            view and the side view, and as up to ``PHANTOM_SECTION_COUNT``
            outlines in the 3D panel.  The figure reads the array one chunk at a
            time on the device that holds the array.  None (default) draws no
            phantom.
        recon_threshold (float, optional): the absolute value above which a
            voxel belongs to the silhouette.  None (default) uses
            ``DEFAULT_RECON_THRESHOLD_FRACTION`` of the largest absolute
            value in ``recon``.
        show_sinogram (bool, optional): whether the painted sinogram is drawn.
            Defaults to True.
        show_recon (bool, optional): whether the phantom's silhouette and
            outlines are drawn.  Defaults to True.
        show_compare (bool, optional): whether the comparison's artists are
            drawn in the four drawing panels.  Defaults to True.

    Attributes:
        scene: the ``GeometryScene`` drawn.
        compare_scene: the second ``GeometryScene`` drawn, or None when no
            comparison is drawn.
        figure: the matplotlib ``Figure``.
        compare_figure: the second matplotlib ``Figure``, which lists every
            difference between the two geometries in a table.  It is None when
            no comparison is drawn.
        panel_axes (tuple): the five axes, in the order 3D view, top view, side
            view, detector face, text panel.
        view_slider: the view ``Slider``, or None when the scan has one view or
            ``widgets`` is False.
        trajectory_check, zoom_check, reference_check, sinogram_check,
            recon_check, compare_check: the six ``CheckButtons``, or None when
            ``widgets`` is False.
        detector_volume_edges (numpy.ndarray): the twelve projected volume edges
            that the detector-face panel drew.  The array has shape (12, 2, 2) and
            holds (row, channel) pairs.  The values come from
            ``ViewScene.volume_outline_on_detector``.
    """

    def __init__(self, model_or_scene, view_index=0, show_trajectory=False,
                 figsize=(15.0, 9.0), title=None,
                 elevation_deg=DEFAULT_ELEVATION_DEG,
                 azimuth_deg=DEFAULT_AZIMUTH_DEG,
                 compare=None, zoom=DEFAULT_ZOOM, show_reference=True,
                 widgets=True, blit=True, sinogram=None, vmin=None, vmax=None,
                 recon=None, recon_threshold=None, show_sinogram=True,
                 show_recon=True, show_compare=True):
        _load_pyplot()
        self.scene = _as_scene(model_or_scene)
        self.compare_scene = None
        self.compare_figure = None
        self._view_index = self._checked_view_index(view_index)
        self._show_trajectory = bool(show_trajectory)
        self._show_reference = bool(show_reference)
        # The overlay install methods read these three flags, so the flags must
        # be set before anything is built.
        self._show_sinogram = bool(show_sinogram)
        self._show_recon = bool(show_recon)
        self._show_compare = bool(show_compare)
        self._zoom = self._checked_zoom(zoom)
        self._elevation_deg = float(elevation_deg)
        self._azimuth_deg = float(azimuth_deg)

        self.enable_blit = bool(blit)
        self._background = None
        self._suspend_blit = False
        self._syncing_widgets = False
        self._slider_region = None

        self._trajectory = None
        self._compare_trajectory = None
        self._reference = None
        self._limits = {}
        self._limit_views = set()
        self.detector_volume_edges = None

        self._quantities = self.scene.derived_quantities()
        self._compare_quantities = None
        self._compare_differences = None
        self._compare_line_limit = None
        self._text_font_size = TEXT_PANEL_FONT_SIZE

        # _moving holds the (axes, artist) pairs that a view change updates.  This
        # module calls these artists the moving artists.
        self._moving = []
        self._compare_moving = []
        self._compare_static = []
        self._reference_artists = []
        self._arrow_3d = None

        # These attributes hold the sinogram and phantom overlays.  The phantom's
        # outline on the detector face is also listed in _moving.
        self._sinogram = None
        self._sinogram_stride = 1
        self._sinogram_image = None
        self._recon_projections = None
        self._recon_section_axis = None
        self._recon_section_indices = []
        self._recon_section_total = 0
        self._recon_section_factors = []
        self._recon_images = []
        self._recon_outlines = []
        self._recon_outline_parts = []
        self._recon_detector_line = None
        self._recon_threshold = None
        self._recon_threshold_given = None

        self._build_panels(figsize, title)
        self._create_widgets(widgets)
        self._create_artists()
        self._install_sinogram(sinogram, vmin=vmin, vmax=vmax)
        self._install_recon(recon, recon_threshold)
        if compare is not None:
            self._install_compare(compare)
        self._create_legends()
        self.figure.canvas.mpl_connect('draw_event', self._on_draw_event)
        self._refresh(rebuild_limits=True)

    @classmethod
    def from_model(cls, model, scene_kwargs=None, **kwargs):
        """This method builds a figure from an mbirtorch model.

        Args:
            model: a ``TomographyModel`` subclass instance.
            scene_kwargs (dict, optional): keyword arguments passed to
                ``GeometryScene.from_model``.
            **kwargs: keyword arguments passed to the constructor.

        Returns:
            GeometryFigure
        """
        scene = GeometryScene.from_model(model, **(scene_kwargs or {}))
        return cls(scene, **kwargs)

    # Public state

    @property
    def view_index(self):
        """This property holds the index of the view currently drawn."""
        return self._view_index

    @property
    def show_trajectory(self):
        """This property is True when the source's path over all views is drawn."""
        return self._show_trajectory

    @property
    def zoom(self):
        """This property holds the 3D panel's zoom, one of :data:`ZOOM_MODES`."""
        return self._zoom

    @property
    def show_reference(self):
        """This property is True when the angle-0 reference is drawn."""
        return self._show_reference

    @property
    def show_sinogram(self):
        """This property is True when a sinogram given to the figure is painted."""
        return self._show_sinogram

    @property
    def show_recon(self):
        """This property is True when the phantom's silhouette and outlines are drawn."""
        return self._show_recon

    @property
    def show_compare(self):
        """This property is True when the comparison is drawn in the four panels."""
        return self._show_compare

    def set_view(self, view_index):
        """This method draws another view.  The 3D camera and the panel limits do not change.

        Args:
            view_index (int): the view to draw.
        """
        self._view_index = self._checked_view_index(view_index)
        self._sync_slider()
        self._refresh()

    def set_show_trajectory(self, flag):
        """This method turns the source's path over all views on or off.

        Args:
            flag (bool): whether to draw the path.
        """
        flag = bool(flag)
        if flag == self._show_trajectory:
            return
        self._show_trajectory = flag
        self._sync_trajectory_check()
        for _, artist in self._trajectory_artists():
            artist.set_visible(flag)
        # The comparison's path is drawn only when the comparison toggle is also on.
        self._apply_compare_visibility()
        # The path changes the panel limits, so the whole figure is redrawn.
        self._refresh(rebuild_limits=True)

    def set_zoom(self, mode):
        """This method zooms the 3D panel to the whole scan, or to a cube about
        ``ZOOM_VOLUME_WIDTH_FACTOR`` volume extents wide.

        Args:
            mode (str): one of ``ZOOM_MODES``.
        """
        mode = self._checked_zoom(mode)
        if mode == self._zoom:
            return
        self._zoom = mode
        self._sync_zoom_check()
        self._refresh(rebuild_limits=True)

    def set_show_reference(self, flag):
        """This method turns the angle-0 reference on or off.

        Args:
            flag (bool): whether to draw the reference.
        """
        flag = bool(flag)
        if flag == self._show_reference:
            return
        self._show_reference = flag
        self._sync_reference_check()
        for _, artist in self._reference_artists:
            artist.set_visible(flag)
        self._refresh(rebuild_limits=True)

    def set_show_sinogram(self, flag):
        """This method shows or hides the painted sinogram.  The figure keeps the array.  If the
        figure has no sinogram, the call has no visible effect.

        Args:
            flag (bool): whether to paint the sinogram.
        """
        flag = bool(flag)
        if flag == self._show_sinogram:
            return
        self._show_sinogram = flag
        self._sync_sinogram_check()
        self._apply_sinogram_visibility()
        self._refresh(rebuild_limits=True)

    def set_show_recon(self, flag):
        """This method shows or hides the phantom's silhouette and outlines.  The figure keeps the
        array and the legend entries.

        Args:
            flag (bool): whether to draw the phantom.
        """
        flag = bool(flag)
        if flag == self._show_recon:
            return
        self._show_recon = flag
        self._sync_recon_check()
        self._apply_recon_visibility()
        # The silhouette is static, so it belongs to the background.
        self._refresh(rebuild_limits=True)

    def set_show_compare(self, flag):
        """This method shows or hides the drawn comparison.  The comparison window and the
        comparison block in the text panel remain.

        Args:
            flag (bool): whether to draw the comparison.
        """
        flag = bool(flag)
        if flag == self._show_compare:
            return
        self._show_compare = flag
        self._sync_compare_check()
        self._apply_compare_visibility()
        self._update_static_text()
        self._refresh(rebuild_limits=True)

    def set_compare(self, compare):
        """This method draws a second geometry over the first.  It also opens a window with a
        table of every difference.  Passing None removes the comparison and closes the window.

        Args:
            compare: a ``GeometryScene``, a ``TomographyModel``, a dictionary
                of parameter overrides applied to a copy of this scene's
                parameters, or None to remove the comparison.
        """
        self._remove_compare_artists()
        if self.compare_figure is not None:
            plt.close(self.compare_figure)
            self.compare_figure = None
        self.compare_scene = None
        self._compare_quantities = None
        self._compare_differences = None
        self._compare_trajectory = None
        # A new comparison starts with no limit on its text panel lines.
        self._compare_line_limit = None
        if compare is not None:
            self._install_compare(compare)
        self._create_legends()
        self._update_static_text()
        self._refresh(rebuild_limits=True)

    def set_sinogram(self, sinogram, vmin=None, vmax=None):
        """This method paints a sinogram on the detector face, or removes it.

        The detector panel shows the sinogram for the current view.  The gray scale is the
        same for every view.

        Args:
            sinogram (array_like): an array of shape
                ``(num_views, num_det_rows, num_det_channels)``, or None to
                remove the sinogram.  When the detector has more than
                ``SINOGRAM_DISPLAY_PIXELS`` pixels along an axis, the figure
                subsamples the sinogram for display.  The subsampling runs on the
                device that holds the array.
            vmin, vmax (float, optional): the ends of the gray scale.  None
                (default) uses the smallest and the largest displayed value of
                the sinogram.

        Raises:
            ValueError: if the array's shape is not the scan's sinogram shape.
        """
        self._install_sinogram(sinogram, vmin=vmin, vmax=vmax)
        # The image and the titles belong to the background, so the whole
        # figure is redrawn.
        self._refresh(rebuild_limits=True)

    def set_recon(self, recon, threshold=None):
        """This method draws a reconstruction's silhouette in the top and side views and up to
        ``PHANTOM_SECTION_COUNT`` outlines in the 3D panel, or removes them.  The silhouette is
        the set of voxels whose absolute value is above the threshold.

        Args:
            recon (array_like): an array of shape ``recon_shape``, as
                (rows, columns, slices), or None to remove the silhouette.
            threshold (float, optional): the absolute value a voxel must exceed
                to belong to the support.  None (default) uses
                ``DEFAULT_RECON_THRESHOLD_FRACTION`` of the largest
                absolute value in ``recon``.

        Raises:
            ValueError: if the array's shape is not the scan's recon shape.
        """
        self._install_recon(recon, threshold)
        # The silhouette is static, so it is part of the background.
        self._refresh(rebuild_limits=True)

    def save(self, path, dpi=110):
        """This method writes the figure to an image file.

        When a comparison is drawn, this method also writes the comparison window to a
        second file whose name ends in ``_comparison``.  For example, saving ``scan.png``
        also writes ``scan_comparison.png``.

        Args:
            path (str): the file to write.  The extension chooses the format.
            dpi (int, optional): dots per inch.

        Returns:
            str: the path of the main figure, which is the argument given.
        """
        self._suspend_blit = True
        try:
            self._set_animated(False)
            self.figure.savefig(path, dpi=dpi, facecolor='white')
        finally:
            self._set_animated(self._animate_moving())
            self._suspend_blit = False
            self._background = None
        if self.compare_figure is not None:
            self.compare_figure.savefig(_companion_path(path), dpi=dpi,
                                        facecolor='white')
        return path

    def show(self, block=True):
        """This method opens a window on the figure, and on the comparison window if there is
        one.  On a backend without windows, this method prints a message and returns.

        Args:
            block (bool, optional): whether ``show`` waits for the window to
                close.
        """
        backend = matplotlib.get_backend().lower()
        if backend in NONINTERACTIVE_BACKENDS:
            print(f'The {backend} backend has no window; nothing was shown.  '
                  'Use GeometryFigure.save to write a file.')
            return
        plt.show(block=block)

    # Checked arguments and small lookups

    def _checked_view_index(self, view_index):
        view_index = int(view_index)
        if not 0 <= view_index < self.scene.num_views:
            raise IndexError(f'view_index {view_index} is outside '
                             f'[0, {self.scene.num_views}).')
        return view_index

    @staticmethod
    def _checked_zoom(mode):
        mode = str(mode)
        if mode not in ZOOM_MODES:
            raise ValueError(f'Unknown zoom {mode!r}; expected one of '
                             f'{ZOOM_MODES}.')
        return mode

    def _compare_index(self, view_index):
        """This method returns the comparison's view index for a primary view index.  When the
        comparison has fewer views, this method returns the comparison's last view.
        """
        return min(int(view_index), self.compare_scene.num_views - 1)

    def _compare_view(self):
        """This method returns the comparison's ``ViewScene`` for the current view.  It returns
        None when there is no comparison."""
        if self.compare_scene is None:
            return None
        return self.compare_scene.view(self._compare_index(self._view_index))

    def _source_trajectory(self):
        """This method returns the source's path over all views.  It computes the path once and
        keeps it.
        """
        if self._trajectory is None:
            sources, _ = self.scene.trajectory()
            self._trajectory = sources
        return self._trajectory

    def _reference_view(self):
        """This method returns the angle-0 reference view.  It computes the view once and keeps
        it.
        """
        if self._reference is None:
            self._reference = self.scene.reference_view()
        return self._reference

    def _compare_source_trajectory(self):
        """This method returns the comparison's source path over all views.  It computes the
        path once and keeps it.
        """
        if self._compare_trajectory is None:
            sources, _ = self.compare_scene.trajectory()
            self._compare_trajectory = sources
        return self._compare_trajectory

    def _trajectory_artists(self):
        """This method returns the (axes, artist) pairs that draw a source path."""
        pairs = [(axes, artist) for axes, artist in self._trajectory_lines]
        pairs.extend(self._compare_trajectory_lines)
        return pairs

    def _view_label(self, scene=None, view_index=None):
        """This method returns a short phrase that names a view by its angle.  For the
        translation geometry, the phrase names the view by its translation vector.
        """
        scene = self.scene if scene is None else scene
        index = self._view_index if view_index is None else view_index
        if scene.kind == 'translation':
            vector = scene.translation_vectors[index]
            return ('translation ('
                    + ', '.join(_three_figures(value) for value in vector)
                    + ') ALU')
        angle = float(scene.angles[index])
        label = (f'angle {angle:.3f} rad '
                 f'({np.degrees(angle):.1f} deg)')
        if scene.kind == 'multiaxis':
            elevation = float(scene.elevations[index])
            label += f', elevation {elevation:.3f} rad'
        elif scene.kind == 'cone':
            shift = float(scene.z_shifts[index])
            if shift != 0.0:
                label += f', z shift {_three_figures(shift)} ALU'
        return label

    # Layout and widgets

    def _build_panels(self, figsize, title):
        """This method creates the figure and its five axes."""
        self.figure = plt.figure(figsize=figsize)
        grid = self.figure.add_gridspec(
            2, 3, width_ratios=(1.35, 1.0, 1.0), left=0.04, right=0.985,
            bottom=GRID_BOTTOM, top=0.90, wspace=0.30, hspace=0.28)
        self.ax_3d = self.figure.add_subplot(grid[:, 0], projection='3d')
        self.ax_top = self.figure.add_subplot(grid[0, 1])
        self.ax_side = self.figure.add_subplot(grid[0, 2])
        self.ax_detector = self.figure.add_subplot(grid[1, 1])
        # This raises the bottom of the detector panel to leave room for its legend.
        cell = self.ax_detector.get_subplotspec().get_position(self.figure)
        self.ax_detector.set_position(
            (cell.x0, DETECTOR_PANEL_BOTTOM, cell.width,
             cell.y1 - DETECTOR_PANEL_BOTTOM))
        self.ax_text = self.figure.add_subplot(grid[1, 2])
        self.panel_axes = (self.ax_3d, self.ax_top, self.ax_side,
                           self.ax_detector, self.ax_text)

        if title is None:
            title = (f'{self._quantities["geometry_kind"]} geometry, '
                     f'sinogram {self._quantities["sinogram_shape_text"]}, '
                     f'recon {self._quantities["recon_shape_text"]}')
        self.figure.suptitle(title, fontsize=11)

        # When only the moved artists are redrawn, this opaque rectangle is painted
        # over the slider row first.  The animated flag keeps the rectangle out of
        # full redraws.
        self._clear_rect = Rectangle((0, 0), 1, 1,
                                     facecolor=self.figure.get_facecolor(),
                                     edgecolor='none', animated=True,
                                     transform=IdentityTransform())
        self.figure.add_artist(self._clear_rect)

    def _create_widgets(self, wanted):
        """This method creates the view slider and the six toggles.  An overlay toggle has no
        effect until the figure has the overlay's array.
        """
        self.view_slider = None
        self.trajectory_check = None
        self.zoom_check = None
        self.reference_check = None
        self.sinogram_check = None
        self.recon_check = None
        self.compare_check = None
        self._slider_axes = None
        if not wanted:
            return

        self._slider_axes = self.figure.add_axes(SLIDER_RECT)
        if self.scene.num_views > 1:
            self.view_slider = Slider(
                self._slider_axes, label='View', valmin=0,
                valmax=self.scene.num_views - 1, valinit=self._view_index,
                valstep=1, valfmt='%0.0f')
            self.view_slider.drawon = False
            self.view_slider.label.set_fontsize(WIDGET_FONT_SIZE)
            self.view_slider.valtext.set_fontsize(WIDGET_FONT_SIZE)
            self.view_slider.on_changed(self._on_slider)
        else:
            self._slider_axes.set_visible(False)

        self.trajectory_check = self._add_check(
            TRAJECTORY_CHECK_RECT, 'source path', self._show_trajectory,
            self._on_trajectory_check)
        self.zoom_check = self._add_check(
            ZOOM_CHECK_RECT, '3D zoom to volume', self._zoom == 'volume',
            self._on_zoom_check)
        self.reference_check = self._add_check(
            REFERENCE_CHECK_RECT, 'angle-0 reference', self._show_reference,
            self._on_reference_check)
        self.sinogram_check = self._add_check(
            SINOGRAM_CHECK_RECT, 'sinogram', self._show_sinogram,
            self._on_sinogram_check)
        self.recon_check = self._add_check(
            RECON_CHECK_RECT, PHANTOM_NAME, self._show_recon,
            self._on_recon_check)
        self.compare_check = self._add_check(
            COMPARE_CHECK_RECT, 'comparison', self._show_compare,
            self._on_compare_check)

    def _add_check(self, rect, label, state, handler):
        """This method creates and returns one ``CheckButtons`` toggle at ``rect`` in figure
        coordinates.  The toggle has one label and a click handler.
        """
        axes = self.figure.add_axes(rect)
        axes.set_frame_on(False)
        check = CheckButtons(axes, [label], [bool(state)])
        for text in check.labels:
            text.set_fontsize(WIDGET_FONT_SIZE)
        check.on_clicked(handler)
        return check

    def _on_slider(self, value):
        """This method handles a move of the view slider."""
        if self._syncing_widgets:
            return
        index = int(round(float(value)))
        if index == self._view_index:
            return
        self.set_view(index)

    def _on_trajectory_check(self, _label):
        """This method handles a click on the source-path toggle."""
        if self._syncing_widgets:
            return
        self.set_show_trajectory(self.trajectory_check.get_status()[0])

    def _on_zoom_check(self, _label):
        """This method handles a click on the 3D zoom toggle."""
        if self._syncing_widgets:
            return
        self.set_zoom('volume' if self.zoom_check.get_status()[0] else 'scan')

    def _on_reference_check(self, _label):
        """This method handles a click on the angle-0 reference toggle."""
        if self._syncing_widgets:
            return
        self.set_show_reference(self.reference_check.get_status()[0])

    def _on_sinogram_check(self, _label):
        """This method handles a click on the sinogram toggle."""
        if self._syncing_widgets:
            return
        self.set_show_sinogram(self.sinogram_check.get_status()[0])

    def _on_recon_check(self, _label):
        """This method handles a click on the phantom toggle."""
        if self._syncing_widgets:
            return
        self.set_show_recon(self.recon_check.get_status()[0])

    def _on_compare_check(self, _label):
        """This method handles a click on the comparison toggle."""
        if self._syncing_widgets:
            return
        self.set_show_compare(self.compare_check.get_status()[0])

    def _sync_slider(self):
        """This method moves the slider to the current view without calling its handler."""
        if self.view_slider is None:
            return
        if int(round(float(self.view_slider.val))) == self._view_index:
            return
        self._syncing_widgets = True
        try:
            self.view_slider.set_val(self._view_index)
        finally:
            self._syncing_widgets = False

    def _set_check(self, check, state):
        """This method sets one toggle to ``state`` without calling its handler.  The argument
        ``check`` is None when the figure has no widgets.
        """
        if check is None or check.get_status()[0] == bool(state):
            return
        self._syncing_widgets = True
        try:
            check.set_active(0)
        finally:
            self._syncing_widgets = False

    def _sync_trajectory_check(self):
        """This method sets the source-path toggle to match the figure's state.  It does not
        call the toggle's handler.
        """
        self._set_check(self.trajectory_check, self._show_trajectory)

    def _sync_zoom_check(self):
        """This method sets the zoom toggle to match the figure's state.  It does not
        call the toggle's handler.
        """
        self._set_check(self.zoom_check, self._zoom == 'volume')

    def _sync_reference_check(self):
        """This method sets the reference toggle to match the figure's state.  It does not
        call the toggle's handler.
        """
        self._set_check(self.reference_check, self._show_reference)

    def _sync_sinogram_check(self):
        """This method sets the sinogram toggle to match the figure's state.  It does not
        call the toggle's handler.
        """
        self._set_check(self.sinogram_check, self._show_sinogram)

    def _sync_recon_check(self):
        """This method sets the phantom toggle to match the figure's state.  It does not
        call the toggle's handler.
        """
        self._set_check(self.recon_check, self._show_recon)

    def _sync_compare_check(self):
        """This method sets the comparison toggle to match the figure's state.  It does not
        call the toggle's handler.
        """
        self._set_check(self.compare_check, self._show_compare)

    # Creating the artists

    def _create_artists(self):
        """This method creates every artist of the primary geometry once.  The methods
        :meth:`_install_sinogram` and :meth:`_install_recon` add the data overlays.
        """
        view = self.scene.view(self._view_index)
        self._create_3d_artists(view)
        self._create_top_artists(view)
        self._create_side_artists(view)
        self._create_detector_artists(view)
        self._create_reference_artists()
        self._create_text_artists()
        self._trajectory_lines = [(self.ax_3d, self._path_3d),
                                  (self.ax_top, self._path_top),
                                  (self.ax_side, self._path_side)]
        self._compare_trajectory_lines = []
        self._set_animated(self._animate_moving())

    def _set_animated(self, flag):
        """This method sets or clears the animated flag of every moving artist."""
        for _, artist in self._moving + self._compare_moving:
            artist.set_animated(bool(flag))
        if self._arrow_3d is not None:
            self._arrow_3d.set_animated(bool(flag))

    def _moving_line(self, axes, color, **kwargs):
        """This method creates one empty line artist that a view change will fill in."""
        line, = axes.plot([], [], color=color, **kwargs)
        self._moving.append((axes, line))
        return line

    def _moving_line_3d(self, axes, color, **kwargs):
        """This method creates one empty 3D line artist that a view change will fill in."""
        line, = axes.plot(np.zeros(0), np.zeros(0), np.zeros(0),
                          color=color, **AXLIM_CLIP, **kwargs)
        self._moving.append((axes, line))
        return line

    def _moving_text(self, axes, text, color, vertical):
        """This method creates and returns a small label.  A view change moves the label to the
        point it names.  The argument ``vertical`` is the label's vertical offset in points.  A
        positive offset puts the label above the point.
        """
        label = axes.annotate(text, xy=(0.0, 0.0),
                              textcoords='offset points',
                              xytext=(LABEL_GAP_POINTS, vertical),
                              fontsize=ANNOTATION_FONT_SIZE, color=color,
                              ha='left',
                              va='bottom' if vertical >= 0 else 'top')
        self._moving.append((axes, label))
        return label

    def _moving_text_3d(self, axes, text, color, vertical='baseline'):
        """This method creates a small 3D label.  A view change moves the label to the point it
        names.  The argument ``vertical`` is the label's vertical alignment.
        """
        label = axes.text(0.0, 0.0, 0.0, text, color=color,
                          fontsize=ANNOTATION_FONT_SIZE,
                          verticalalignment=vertical)
        self._moving.append((axes, label))
        return label

    # --- the 3D panel ---

    def _create_3d_artists(self, view):
        """This method creates the 3D panel's artists and labels its axes."""
        axes = self.ax_3d

        # These artists are static.  The object and its axis do not move with
        # the view.
        box = _joined([view.volume_corners[[first, second]]
                       for first, second in VOLUME_BOX_EDGES])
        axes.plot(box[:, 0], box[:, 1], box[:, 2], color=COLORS['volume'],
                  linewidth=1.0, label='volume box', **AXLIM_CLIP)
        self._create_3d_ror(view)
        if view.rotation_axis is not None:
            segment = _sampled_segment(view.rotation_axis[0],
                                       view.rotation_axis[1])
            axes.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                      color=COLORS['axis'], linewidth=1.2, linestyle='--',
                      label='rotation axis', **AXLIM_CLIP)
        if view.translation_path is not None:
            path = view.translation_path
            axes.plot(path[:, 0], path[:, 1], path[:, 2], color=COLORS['axis'],
                      linewidth=1.0, marker='o', markersize=2.5,
                      label='translation path', **AXLIM_CLIP)
        axes.plot([view.voxel0_center[0]], [view.voxel0_center[1]],
                  [view.voxel0_center[2]], marker='x', linestyle='none',
                  markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['voxel0'],
                  label='voxel (0, 0, 0)', **AXLIM_CLIP, zorder=6)
        self._path_3d, = axes.plot(np.zeros(0), np.zeros(0), np.zeros(0),
                                   color=COLORS['trajectory'],
                                   linewidth=1.0, linestyle='-.',
                                   label='source path', **AXLIM_CLIP)
        self._path_3d.set_visible(self._show_trajectory)

        # These artists move with the view.
        source_label = ('source' if view.source is not None
                        else 'source (drawn)')
        self._source_3d = self._moving_line_3d(
            axes, COLORS['source'], marker='*', markersize=SOURCE_MARKER_SIZE,
            linestyle='none', label=source_label, zorder=6)
        self._detector_3d = self._moving_line_3d(
            axes, COLORS['detector'], linewidth=1.6, label='detector')
        self._face_3d = None
        if not self.scene.use_curved_detector:
            # A curved detector's face is not a polygon, so the panel draws only its
            # outline.  The panel fills a flat detector's face lightly.
            self._face_3d = Poly3DCollection(
                [view.detector_corners], facecolor=COLORS['detector'],
                alpha=0.12, edgecolor='none', **AXLIM_CLIP)
            axes.add_collection3d(self._face_3d)
            self._moving.append((axes, self._face_3d))
        self._rays_3d = self._moving_line_3d(axes, COLORS['rays'],
                                             linewidth=0.7)
        self._central_3d = self._moving_line_3d(
            axes, COLORS['central_ray'], linewidth=1.1, label='central ray')
        self._pixel0_3d = self._moving_line_3d(
            axes, COLORS['pixel0'], marker='x', linestyle='none',
            markersize=INDEX_MARKER_SIZE,
            markeredgewidth=INDEX_MARKER_WIDTH,
            label='detector pixel (0, 0)', zorder=6)
        self._arc_3d = self._moving_line_3d(axes, COLORS['axis'],
                                            linewidth=1.4)
        # The arc's label goes below the end of the arc, so it does not overlap
        # the source's label.
        self._arc_text_3d = self._moving_text_3d(axes, ' source travel',
                                                 COLORS['axis'],
                                                 vertical='top')
        # The label starts with two spaces.  The source's star marker is wide
        # enough to overlap a label that starts with one space.
        self._source_text_3d = self._moving_text_3d(axes, '  source',
                                                    COLORS['source'])
        self._detector_text_3d = self._moving_text_3d(axes, ' detector',
                                                      COLORS['detector'])
        # The iso label goes below its point, so it does not overlap the angle-0
        # reference's label.
        self._iso_text_3d = self._moving_text_3d(axes, ' ' + ISO_NAME,
                                                 COLORS['central_ray'],
                                                 vertical='top')

        axes.set_xlabel('x (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('y (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_zlabel('z (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.tick_params(labelsize=TICK_FONT_SIZE)
        axes.view_init(elev=self._elevation_deg, azim=self._azimuth_deg,
                       roll=_camera_roll_deg())
        axes.set_box_aspect((1.0, 1.0, 1.0))
        self._title_3d = axes.set_title('', fontsize=TITLE_FONT_SIZE)
        self._moving.append((axes, self._title_3d))

    def _create_3d_ror(self, view):
        """This method draws the region of reconstruction in the 3D panel, when there is one.
        The drawing has two rings and four vertical lines.
        """
        cylinder = view.ror_cylinder
        if cylinder is None:
            return
        lower = _ellipse_points(cylinder['center'], cylinder['semi_axis_x'],
                                cylinder['semi_axis_y'], cylinder['z_min'])
        upper = _ellipse_points(cylinder['center'], cylinder['semi_axis_x'],
                                cylinder['semi_axis_y'], cylinder['z_max'])
        parts = [lower, upper]
        for step in range(0, ELLIPSE_SAMPLES - 1, (ELLIPSE_SAMPLES - 1) // 4):
            parts.append(np.stack([lower[step], upper[step]]))
        rings = _joined(parts)
        self.ax_3d.plot(rings[:, 0], rings[:, 1], rings[:, 2],
                        color=COLORS['ror'], linewidth=0.9,
                        label='region of reconstruction', **AXLIM_CLIP)

    # --- the two projected panels ---

    def _create_top_artists(self, view):
        """This method creates the top panel's artists.  The top panel shows the xy plane seen
        from -z, with y to the left and x downward.
        """
        axes = self.ax_top
        first, second = TOP_PANEL_COLUMNS

        walk = view.volume_corners[list(_XY_FOOTPRINT_WALK)]
        axes.plot(walk[:, first], walk[:, second], color=COLORS['volume'],
                  linewidth=1.2)
        cylinder = view.ror_cylinder
        if cylinder is not None:
            ring = _ellipse_points(cylinder['center'], cylinder['semi_axis_x'],
                                   cylinder['semi_axis_y'], cylinder['z_min'])
            axes.plot(ring[:, first], ring[:, second], color=COLORS['ror'],
                      linewidth=0.9)
        if view.rotation_axis is not None:
            # The rotation axis is a point in this plane.
            axes.plot([0.0], [0.0], marker='+', markersize=9,
                      markeredgewidth=1.4, color=COLORS['axis'],
                      linestyle='none')
        if view.translation_path is not None:
            path = view.translation_path
            axes.plot(path[:, first], path[:, second], color=COLORS['axis'],
                      linewidth=1.2, marker='o', markersize=3.0)
            # The label goes below the path on the screen, so it does not overlap
            # the angle-0 reference's label above the path.
            axes.annotate('translation path',
                          xy=(float(np.mean(path[:, first])),
                              _screen_bottom(path[:, second])),
                          textcoords='offset points', xytext=(0, -5),
                          ha='center', va='top',
                          fontsize=ANNOTATION_FONT_SIZE, color=COLORS['axis'])
        axes.plot([view.voxel0_center[first]], [view.voxel0_center[second]],
                  marker='x', linestyle='none', markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['voxel0'],
                  zorder=6)
        self._path_top, = axes.plot([], [], color=COLORS['trajectory'],
                                    linewidth=1.0, linestyle='-.')
        self._path_top.set_visible(self._show_trajectory)

        self._top = self._create_projected_moving_artists(axes)
        # Only the top view names the pixel-0 marker.  In the side view the
        # detector's own label already sits at that end.
        self._top['pixel0_label'] = self._moving_text(
            axes, 'pixel (0,0)', COLORS['pixel0'], PIXEL0_LABEL_GAP_POINTS)
        self._arc_top = self._moving_line(axes, COLORS['axis'], linewidth=1.4)
        self._arrow_top = FancyArrowPatch(
            (0.0, 0.0), (0.0, 0.0), arrowstyle='-|>', mutation_scale=9,
            linewidth=1.4, color=COLORS['axis'], shrinkA=0.0, shrinkB=0.0)
        axes.add_patch(self._arrow_top)
        self._moving.append((axes, self._arrow_top))
        # The method _update_arc_and_trajectory chooses which side of the arc's
        # end this label sits on.
        self._arc_text_top = axes.annotate(
            'source travel', xy=(0.0, 0.0), textcoords='offset points',
            xytext=(3, -ARC_LABEL_GAP_POINTS), fontsize=ANNOTATION_FONT_SIZE,
            color=COLORS['axis'])
        self._moving.append((axes, self._arc_text_top))
        axes.set_xlabel('y (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('x (ALU)', fontsize=LABEL_FONT_SIZE)
        self._title_top = axes.set_title(_top_view_title(),
                                         fontsize=TITLE_FONT_SIZE)
        _finish_2d_panel(axes)

    def _create_side_artists(self, view):
        """This method creates the side panel's artists.  The side panel shows the yz plane seen
        from +x, with y to the left and z downward.
        """
        axes = self.ax_side
        first, second = SIDE_PANEL_COLUMNS

        walk = view.volume_corners[list(_YZ_FACE_WALK)]
        axes.plot(walk[:, first], walk[:, second], color=COLORS['volume'],
                  linewidth=1.2)
        axes.axhline(0.0, color=COLORS['axis'], linewidth=0.7, linestyle=':')

        # The volume's z center is recon_slice_offset.  The panel draws this offset
        # as a segment from z = 0 to the z center, at the volume's high y edge.
        z_min, z_max = self.scene.volume_z_range()
        z_center = 0.5 * (z_min + z_max)
        self._slice_offset_label = None
        self._slice_offset_point = None
        if z_center != 0.0:
            y_at = float(np.max(view.volume_corners[:, 1]))
            axes.plot([y_at, y_at], [0.0, z_center], color=COLORS['volume'],
                      linewidth=2.6, solid_capstyle='butt')
            # The label extends from the segment away from the volume box.  The
            # label's side changes with the view, so the label is a moving artist.
            self._slice_offset_point = (y_at, z_center)
            self._slice_offset_label = self._moving_text(
                axes, f'recon_slice_offset {_three_figures(z_center)}',
                COLORS['volume'], LABEL_GAP_POINTS)
        if view.rotation_axis is not None:
            segment = view.rotation_axis
            axes.plot(segment[:, first], segment[:, second],
                      color=COLORS['axis'], linewidth=1.2, linestyle='--')
        if view.translation_path is not None:
            path = view.translation_path
            axes.plot(path[:, first], path[:, second], color=COLORS['axis'],
                      linewidth=1.0, marker='o', markersize=2.5)
        axes.plot([view.voxel0_center[first]], [view.voxel0_center[second]],
                  marker='x', linestyle='none', markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['voxel0'],
                  zorder=6)
        self._path_side, = axes.plot([], [], color=COLORS['trajectory'],
                                     linewidth=1.0, linestyle='-.')
        self._path_side.set_visible(self._show_trajectory)
        self._path_note_side = axes.annotate(
            '', xy=(0.98, 0.04), xycoords='axes fraction', va='bottom',
            ha='right', fontsize=ANNOTATION_FONT_SIZE,
            color=COLORS['trajectory'])
        self._path_note_side.set_visible(self._show_trajectory)

        # The source's label goes below the source in this panel.  The
        # recon_slice_offset label sits above it.
        self._side = self._create_projected_moving_artists(
            axes, source_vertical=-7, name_iso=False)
        axes.set_xlabel('y (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('z (ALU)', fontsize=LABEL_FONT_SIZE)
        self._title_side = axes.set_title(_side_view_title(),
                                          fontsize=TITLE_FONT_SIZE)
        _finish_2d_panel(axes)

    def _create_projected_moving_artists(self, axes, source_vertical=7,
                                         name_iso=True):
        """This method creates the moving artists that the top and side panels both use.  It
        returns them in a dict keyed by name.  The argument ``source_vertical`` is the source
        label's offset in points.  The argument ``name_iso`` says whether to label the detector
        iso.
        """
        artists = dict(
            rays=self._moving_line(axes, COLORS['rays'], linewidth=0.7),
            edge_rays=self._moving_line(axes, EMPHASIZED_RAY_COLOR,
                                        linewidth=1.1),
            detector=self._moving_line(axes, COLORS['detector'],
                                       linewidth=1.6),
            central=self._moving_line(axes, COLORS['central_ray'],
                                      linewidth=1.1),
            offset=self._moving_line(axes, COLORS['detector'], linewidth=2.6,
                                     solid_capstyle='butt'),
            source=self._moving_line(axes, COLORS['source'], marker='*',
                                     markersize=SOURCE_MARKER_SIZE,
                                     linestyle='none', zorder=6),
            pixel0=self._moving_line(axes, COLORS['pixel0'], marker='x',
                                     linestyle='none',
                                     markersize=INDEX_MARKER_SIZE,
                                     markeredgewidth=INDEX_MARKER_WIDTH,
                                     zorder=6),
        )
        # The source's label is offset farther than the others, because the
        # central ray ends at the source.
        artists['source_label'] = self._moving_text(
            axes, 'source', COLORS['source'], source_vertical)
        artists['detector_label'] = self._moving_text(
            axes, 'detector', COLORS['detector'], 4)
        # Only the top view labels the detector iso.  The side view's detector
        # already has two labels at the same end.
        if name_iso:
            artists['iso_label'] = self._moving_text(
                axes, ISO_NAME, COLORS['central_ray'],
                ISO_LABEL_GAP_POINTS)
        # This label names the offset segment, which runs from the detector iso to
        # the detector center.  The label sits at the iso, one text line farther from
        # the point than the iso label.
        artists['offset_label'] = self._moving_text(
            axes, '', COLORS['detector'],
            ISO_LABEL_GAP_POINTS + LABEL_LINE_POINTS)
        return artists

    # --- the detector-face panel ---

    def _create_detector_artists(self, view):
        """This method creates the detector-face panel's artists.  The panel draws the channel
        index increasing to the right and row 0 at the top.
        """
        axes = self.ax_detector
        num_rows = self.scene.num_det_rows
        num_channels = self.scene.num_det_channels

        axes.add_patch(Rectangle((-0.5, -0.5), num_channels, num_rows,
                                 facecolor=COLORS['detector'], alpha=0.10,
                                 edgecolor=COLORS['detector'], linewidth=1.4))
        ((landing_row, landing_channel),
         (center_row, center_channel)) = \
            self.scene.detector_iso_and_center_indices()
        axes.plot([float(landing_channel)], [float(landing_row)], marker='o',
                  markersize=6, linestyle='none', markerfacecolor='none',
                  markeredgewidth=1.4, color=COLORS['central_ray'],
                  label=ISO_NAME)
        axes.plot([float(center_channel)], [float(center_row)], marker='s',
                  markersize=5, linestyle='none', markerfacecolor='none',
                  markeredgewidth=1.4, color=COLORS['detector'],
                  label=CENTER_NAME)
        axes.plot([0.0], [0.0], marker='x', linestyle='none',
                  markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['pixel0'],
                  label='pixel (0,0) = sino[v, 0, 0]')

        self._edges_inside = self._moving_line(axes, COLORS['volume'],
                                               linewidth=1.2,
                                               label='volume box')
        # This line has no legend label.  The line exists in every view, but it has
        # points only where the volume projects past the detector's edge.
        self._edges_outside = self._moving_line(axes, COLORS['overshoot'],
                                                linewidth=1.6)
        # The panel splits the two rims of the region of reconstruction the same
        # way.  A scan with no mask has no cylinder, so the panel draws neither rim.
        self._ror_inside = None
        self._ror_outside = None
        if view.ror_outline_on_detector is not None:
            self._ror_inside = self._moving_line(
                axes, COLORS['ror'], linewidth=1.0,
                label='region of reconstruction')
            self._ror_outside = self._moving_line(axes, COLORS['overshoot'],
                                                  linewidth=1.4)
        axes.set_aspect('equal', adjustable='box')
        axes.set_xlabel('channel index', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('row index', fontsize=LABEL_FONT_SIZE)
        axes.tick_params(labelsize=TICK_FONT_SIZE)
        self._title_detector = axes.set_title('', fontsize=TITLE_FONT_SIZE)
        self._moving.append((axes, self._title_detector))

    def _create_recon_detector_outline(self):
        """This method creates the phantom's projected outline on the detector face, which moves
        with the view.
        """
        self._recon_detector_line = self._moving_line(
            self.ax_detector, COLORS['volume'],
            linewidth=RECON_OUTLINE_DETECTOR_LINEWIDTH, linestyle='--',
            label=PHANTOM_NAME)
        self._recon_outlines.append((self.ax_detector,
                                     self._recon_detector_line))

    # --- the angle-0 reference ---

    def _reference_label(self):
        """This method returns the two-line name of the angle-0 reference arrow in the top view.
        For the translation geometry, the name says that the reference has no translation.
        """
        if self.scene.kind == 'translation':
            return 'projection direction,\nno translation'
        return 'projection direction,\nangle 0'

    def _reference_short_label(self):
        """This method returns a short form of the reference's name for the 3D panel."""
        return ('no translation' if self.scene.kind == 'translation'
                else 'angle 0')

    def _create_reference_artists(self):
        """This method draws the source and the detector at their angle-0 position in the 3D
        view and the top view.  These artists do not move with the view.
        """
        reference = self._reference_view()
        self._reference_artists = []
        dotted = dict(linestyle=REFERENCE_DOTS, linewidth=REFERENCE_LINEWIDTH,
                      alpha=REFERENCE_ALPHA)
        hollow_star = dict(marker='*', linestyle='none',
                           markersize=SOURCE_MARKER_SIZE,
                           markerfacecolor='none',
                           markeredgecolor=COLORS['source'],
                           markeredgewidth=1.3, alpha=REFERENCE_ALPHA)
        source = reference.source_draw
        origin = reference.detector_origin
        outline = reference.detector_outline
        # The arrowhead is drawn on a short segment at the detector end of the
        # central ray, so that the arrow points from the source to the detector.
        head_start = origin + REFERENCE_ARROW_FRACTION * (source - origin)

        def keep(axes, artist):
            """This function adds one artist to the reference list and sets its visibility."""
            artist.set_visible(self._show_reference)
            self._reference_artists.append((axes, artist))
            return artist

        axes = self.ax_3d
        keep(axes, axes.plot([source[0]], [source[1]], [source[2]],
                             label='angle-0 reference', **AXLIM_CLIP,
                             **hollow_star)[0])
        keep(axes, axes.plot(outline[:, 0], outline[:, 1], outline[:, 2],
                             color=COLORS['detector'], **AXLIM_CLIP,
                             **dotted)[0])
        central = _sampled_segment(source, origin)
        keep(axes, axes.plot(central[:, 0], central[:, 1], central[:, 2],
                             color=COLORS['central_ray'], **AXLIM_CLIP,
                             **dotted)[0])
        direction = origin - head_start
        self._reference_arrow_3d = keep(axes, axes.quiver(
            head_start[0], head_start[1], head_start[2],
            direction[0], direction[1], direction[2],
            color=COLORS['central_ray'], alpha=REFERENCE_ALPHA,
            arrow_length_ratio=3.0, linewidth=REFERENCE_LINEWIDTH))
        self._reference_arrow_point = head_start
        # The label goes at the middle of the reference detector's low-row
        # edge.
        corners = reference.detector_corners
        label_at = 0.5 * (corners[0] + corners[1])
        self._reference_text_3d = keep(axes, axes.text(
            label_at[0], label_at[1], label_at[2],
            ' ' + self._reference_short_label(),
            color=COLORS['central_ray'], alpha=REFERENCE_ALPHA,
            fontsize=ANNOTATION_FONT_SIZE))

        axes = self.ax_top
        across, down = TOP_PANEL_COLUMNS
        keep(axes, axes.plot([source[across]], [source[down]],
                             **hollow_star)[0])
        keep(axes, axes.plot(outline[:, across], outline[:, down],
                             color=COLORS['detector'], **dotted)[0])
        keep(axes, axes.plot([source[across], origin[across]],
                             [source[down], origin[down]],
                             color=COLORS['central_ray'], **dotted)[0])
        head = FancyArrowPatch(
            (head_start[across], head_start[down]),
            (origin[across], origin[down]),
            arrowstyle='-|>', mutation_scale=10,
            linewidth=REFERENCE_LINEWIDTH, color=COLORS['central_ray'],
            alpha=REFERENCE_ALPHA, shrinkA=0.0, shrinkB=0.0)
        axes.add_patch(head)
        keep(axes, head)
        # The caption sits in the panel's upper left corner, not on the ray it names.
        keep(axes, axes.annotate(
            self._reference_label(), xy=REFERENCE_LABEL_CORNER,
            xycoords='axes fraction', fontsize=ANNOTATION_FONT_SIZE,
            color=COLORS['central_ray'], alpha=REFERENCE_ALPHA, ha='left',
            va='top'))

    # --- the text panel ---

    def _create_text_artists(self):
        """This method creates the text panel's static block, comparison block, and footer.
        Only the footer changes with the view.
        """
        axes = self.ax_text
        axes.set_axis_off()
        self._static_text = axes.text(
            0.0, 1.0, '', transform=axes.transAxes,
            fontsize=TEXT_PANEL_FONT_SIZE, family='monospace', va='top',
            ha='left')
        # The method _place_text_blocks moves the comparison block to where the
        # static block ends.
        self._compare_text = axes.text(
            0.0, 0.30, '', transform=axes.transAxes,
            fontsize=TEXT_PANEL_FONT_SIZE, family='monospace', va='top',
            ha='left', color=COLORS['compare'])
        self._footer_text = axes.text(
            0.0, 0.10, '', transform=axes.transAxes,
            fontsize=TEXT_PANEL_FONT_SIZE, family='monospace', va='top',
            ha='left')
        self._moving.append((axes, self._footer_text))
        axes.set_title('Derived quantities', fontsize=TITLE_FONT_SIZE)
        self._update_static_text()

    def _update_static_text(self, keep_font=False):
        """This method fills in the static text block and the comparison block.  When
        ``keep_font`` is True, the method keeps the panel's current font size.
        """
        quantities = self._quantities

        def triple(*keys):
            return ', '.join(_three_figures(quantities[key]) for key in keys)

        rows = [
            ('geometry', quantities['geometry_kind'], ''),
            ('views', str(quantities['num_views']), ''),
            ('sinogram (v, r, c)', quantities['sinogram_shape_text'], ''),
            ('recon (r, c, s)', quantities['recon_shape_text'], ''),
            ('magnification', _three_figures(quantities['magnification']), ''),
            ('fan angle', _three_figures(quantities['fan_angle_deg']), 'deg'),
            ('cone angle', _three_figures(quantities['cone_angle_deg']),
             'deg'),
            ('lateral FoV', _three_figures(quantities['lateral_fov_alu']),
             'ALU'),
            ('axial FoV', _three_figures(quantities['axial_fov_alu']), 'ALU'),
            ('voxel pitch x, y, z',
             triple('delta_voxel_x', 'delta_voxel_y', 'delta_voxel_z'), 'ALU'),
            ('volume extent x,y,z',
             triple('volume_extent_x', 'volume_extent_y', 'volume_extent_z'),
             'ALU'),
            ('volume z center', _three_figures(quantities['volume_z_center']),
             'ALU'),
            ('detector w by h',
             _three_figures(quantities['detector_width']) + ' by '
             + _three_figures(quantities['detector_height']), 'ALU'),
            ('det pitch chan, row',
             triple('detector_channel_pitch', 'detector_row_pitch'), 'ALU'),
            ('det center du, dv',
             triple('detector_center_u', 'detector_center_v'), 'ALU'),
            ('helical travel',
             _three_figures(quantities['helical_travel_alu']), 'ALU'),
            # The fit statement has two halves.  The first half is about the channels.
            # The second half is about the rows, or the swept z range of a helical scan.
            ('lateral fit', _lateral_fit_text(quantities), ''),
            ('axial fit', _axial_fit_text(quantities), ''),
            # This row counts the views in which the fit statement's shape extends past
            # the detector.  In a helical scan, this happens in every view by design.
            ('leaves det in views',
             f'{quantities["views_leaving_detector"]} of '
             f'{quantities["num_views"]}', ''),
        ]
        width = max(len(name) for name, _, _ in rows)
        lines = [f'{name:<{width}} : {value}{" " + unit if unit else ""}'
                 for name, value, unit in rows]
        lines.append('')
        # These two notes describe the drawing.  The first note says what the
        # detector center offsets are measured from and which way z is drawn.
        for note in (f'(du, dv): {ISO_NAME} to {CENTER_NAME}; '
                     f'{_convention_note()}', quantities['drawing_note']):
            lines.extend(textwrap.wrap(note, width=TEXT_PANEL_WRAP_WIDTH))
        self._static_text.set_text('\n'.join(lines))

        compare_lines = self._comparison_lines()
        self._compare_text.set_text('\n'.join(compare_lines))
        self._compare_text.set_visible(bool(compare_lines))
        # New text starts at the panel's default font size.  The method
        # _fit_text_font reduces the size when the blocks are too tall.
        if not keep_font:
            self._text_font_size = (COMPARING_FONT_SIZE
                                    if self.compare_scene is not None
                                    else TEXT_PANEL_FONT_SIZE)
        self._set_text_font(self._text_font_size)

    def _set_text_font(self, size):
        """This method sets one font size on the text panel's three blocks."""
        for artist in (self._static_text, self._compare_text,
                       self._footer_text):
            artist.set_fontsize(float(size))

    def _fit_text_font(self, renderer, panel):
        """This method shrinks the text panel's font toward a size at which its blocks fit.  It
        returns True when the font changed.  In that case, the caller must draw the figure again.
        """
        blocks = [self._static_text]
        if self._compare_text.get_visible():
            blocks.append(self._compare_text)
        blocks.append(self._footer_text)
        counts = [artist.get_text().count('\n') + 1 for artist in blocks]
        # One blank line separates one block from the next.
        lines = sum(counts) + len(blocks) - 1
        extent = self._static_text.get_window_extent(renderer)
        line_height = extent.height / max(counts[0], 1)
        needed = lines * line_height
        room = TEXT_PANEL_FILL * panel.height
        # A font's line height does not scale exactly with its size, so this test
        # allows a tolerance of half a line.  Without it, the fitting never stops.
        if needed <= room + 0.5 * line_height or line_height <= 0.0:
            return False
        size = max(TEXT_PANEL_FONT_MINIMUM,
                   self._text_font_size * room / needed)
        if size >= self._text_font_size - 0.05:
            return False
        self._text_font_size = size
        self._set_text_font(size)
        return True

    def _difference_groups(self):
        """This method returns the parameter differences and the derived-quantity differences.
        Each is a list of ``(name, primary, comparison)``.  This method counts a name as a
        parameter when either geometry kind requires it.
        """
        if self._compare_differences is None:
            names = set(required_parameter_names(self.scene.kind))
            names.update(required_parameter_names(self.compare_scene.kind))
            parameters, derived = [], []
            for row in self.scene.differences(self.compare_scene):
                (parameters if row[0] in names else derived).append(row)
            self._compare_differences = (parameters, derived)
        return self._compare_differences

    def _comparison_lines(self):
        """This method returns the text panel's comparison section as a list of lines.  It lists
        up to :data:`MAX_COMPARISON_ENTRIES` differing parameters and counts the differing
        derived quantities.
        """
        if self.compare_scene is None:
            return []
        parameters, derived = self._difference_groups()
        heading = ('Comparison (dashed):' if self._show_compare
                   else 'Comparison (dashed, hidden):')
        lines = [heading]
        if not parameters and not derived:
            lines.append('  nothing differs')
            return lines

        blocks = [textwrap.wrap(_difference_text(name, mine, theirs),
                                width=TEXT_PANEL_WRAP_WIDTH,
                                initial_indent='  ',
                                subsequent_indent='      ')
                  for name, mine, theirs in
                  parameters[:MAX_COMPARISON_ENTRIES]]
        limit = self._compare_line_limit
        shown = 0
        for block in blocks:
            if limit is not None and len(lines) + len(block) + 1 > limit:
                break
            lines.extend(block)
            shown += 1
        left_out = len(parameters) - shown
        if left_out:
            noun = 'parameter' if left_out == 1 else 'parameters'
            lines.append(f'  and {left_out} more {noun}')
        sentence = textwrap.wrap(_derived_difference_sentence(len(derived)),
                                 width=TEXT_PANEL_WRAP_WIDTH,
                                 initial_indent='  ', subsequent_indent='  ')
        if limit is None or len(lines) + len(sentence) <= limit:
            lines.extend(sentence)
        return lines

    def _footer_lines(self):
        """This method returns the text panel's footer, as a list of lines."""
        lines = [f'view drawn : {self._view_index}']
        lines.extend(textwrap.wrap('view: ' + self._view_label(),
                                   width=TEXT_PANEL_WRAP_WIDTH))
        if self.compare_scene is not None:
            index = self._compare_index(self._view_index)
            if index != self._view_index:
                lines.extend(textwrap.wrap(
                    f'comparison holds view {index}: '
                    + self._view_label(self.compare_scene, index),
                    width=TEXT_PANEL_WRAP_WIDTH))
        # This line names the data overlays that are drawn, because neither
        # overlay has its own legend entry.
        overlays = self._overlay_names()
        if overlays:
            lines.extend(textwrap.wrap('overlays : ' + '; '.join(overlays),
                                       width=TEXT_PANEL_WRAP_WIDTH))
        return lines

    # --- the legends ---

    def _create_legends(self):
        """This method builds the legends of the 3D panel and of the detector face."""
        self.ax_3d.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE,
                          framealpha=0.85, borderpad=0.3, labelspacing=0.25)
        cell = self.ax_detector.get_subplotspec().get_position(self.figure)
        # The keyword borderaxespad sets the gap between a legend and its anchor.
        # The value here is zero, so the anchor is the legend's bottom edge.
        self.ax_detector.legend(
            loc='lower center',
            bbox_to_anchor=(0.5 * (cell.x0 + cell.x1),
                            DETECTOR_LEGEND_BOTTOM),
            bbox_transform=self.figure.transFigure,
            ncol=DETECTOR_LEGEND_COLUMNS, borderaxespad=0.0,
            fontsize=LEGEND_FONT_SIZE, framealpha=0.85, borderpad=0.3,
            labelspacing=0.25)

    def _place_text_blocks(self):
        """This method stacks the text panel's three blocks from the top, one blank line apart.
        It returns True when a block moved.  In that case, the caller must draw the figure
        again.
        """
        canvas = self.figure.canvas
        if not hasattr(canvas, 'get_renderer'):
            return False
        renderer = canvas.get_renderer()
        panel = self.ax_text.get_window_extent(renderer)
        if panel.height <= 0.0:
            return False

        def line_height(artist):
            extent = artist.get_window_extent(renderer)
            count = artist.get_text().count('\n') + 1
            return extent, extent.height / max(count, 1)

        def place(artist, top):
            """This function moves an artist so that its top is at the display height ``top``.
            It returns True when the artist moved.
            """
            target = float((top - panel.y0) / panel.height)
            if abs(target - float(artist.get_position()[1])) < 0.004:
                return False
            artist.set_position((0.0, target))
            return True

        # This chooses the font size first, because the block positions depend on
        # the line height.
        font_changed = self._fit_text_font(renderer, panel)
        moved = font_changed
        extent, height = line_height(self._static_text)
        next_top = extent.y0 - height
        if self._compare_text.get_visible():
            moved |= place(self._compare_text, next_top)
            extent, compare_height = line_height(self._compare_text)
            next_top = extent.y0 - compare_height
        moved |= place(self._footer_text, next_top)
        # This shortens the comparison section only in a pass that did not change
        # the font.  A smaller font fits more lines, and a shortened section stays short.
        if self._compare_text.get_visible() and not font_changed:
            moved |= self._limit_comparison_lines(renderer, panel,
                                                  compare_height)
        return moved

    def _limit_comparison_lines(self, renderer, panel, line_height):
        """This method shortens the comparison section to the number of lines that fit in the
        panel.  It returns True when it removed lines.  In that case, the caller must draw the
        figure again.
        """
        compare = self._compare_text.get_window_extent(renderer)
        footer = self._footer_text.get_window_extent(renderer)
        room = compare.y1 - panel.y0 - footer.height - line_height
        fits = int(max(room, 0.0) // max(line_height, 1.0))
        current = self._compare_text.get_text().count('\n') + 1
        if fits >= current:
            return False
        limit = fits if self._compare_line_limit is None else min(
            fits, self._compare_line_limit)
        if limit == self._compare_line_limit:
            return False
        self._compare_line_limit = limit
        self._update_static_text(keep_font=True)
        return True

    # The comparison overlay

    def _install_compare(self, compare):
        """This method builds the comparison scene, its artists, and its window."""
        self.compare_scene = self._as_compare_scene(compare)
        self._compare_quantities = self.compare_scene.derived_quantities()
        self._compare_differences = None
        self._compare_trajectory = None
        self._create_compare_artists()
        self._create_compare_window()
        self._update_static_text()

    def _create_compare_window(self):
        """This method builds the comparison window as a second figure.  The window holds a table
        of every difference, with columns for the name, the primary value, and the comparison
        value.
        """
        lines = self._compare_table_lines()
        height = min(COMPARE_WINDOW_MAX_HEIGHT_IN,
                     COMPARE_WINDOW_TITLE_IN + COMPARE_WINDOW_MARGIN_IN
                     + len(lines) * COMPARE_ROW_HEIGHT_IN)
        self.compare_figure = plt.figure(
            figsize=(COMPARE_WINDOW_WIDTH_IN, height))
        self.compare_figure.suptitle(COMPARE_WINDOW_TITLE,
                                     fontsize=TITLE_FONT_SIZE + 1)
        # A backend without windows has no window manager, so there is no title to set.
        manager = getattr(self.compare_figure.canvas, 'manager', None)
        if hasattr(manager, 'set_window_title'):
            manager.set_window_title(COMPARE_WINDOW_NAME)

        side = COMPARE_WINDOW_MARGIN_IN / COMPARE_WINDOW_WIDTH_IN
        bottom = COMPARE_WINDOW_MARGIN_IN / height
        top = 1.0 - COMPARE_WINDOW_TITLE_IN / height
        axes = self.compare_figure.add_axes(
            (side, bottom, 1.0 - 2.0 * side, top - bottom))
        axes.set_axis_off()

        room_points = 72.0 * (COMPARE_WINDOW_WIDTH_IN
                              - 2.0 * COMPARE_WINDOW_MARGIN_IN)
        longest = max(len(line) for line in lines)
        size = min(COMPARE_WINDOW_FONT_SIZE,
                   room_points / (MONOSPACE_ADVANCE * longest))
        for index, line in enumerate(lines):
            axes.text(0.0, 1.0 - (index + 0.5) / len(lines), line,
                      transform=axes.transAxes, family='monospace',
                      fontsize=size, va='center', ha='left',
                      # The first row is the column headings.
                      fontweight='bold' if index == 0 else 'normal')

    def _compare_table_lines(self):
        """This method returns the comparison window's table as a list of monospace lines.  The
        lines hold the parameters, then a line of dashes, then the derived quantities.
        """
        parameters, derived = self._difference_groups()
        rows = [COMPARE_TABLE_HEADING]
        for name, mine, theirs in parameters + derived:
            rows.append((name, *_difference_values(mine, theirs)))
        if not parameters and not derived:
            rows.append(('nothing differs', '', ''))
        widths = [max(len(row[column]) for row in rows)
                  for column in range(len(COMPARE_TABLE_HEADING))]
        # The line of dashes follows the parameters only when both kinds of
        # differences exist.
        if parameters and derived:
            rows.insert(1 + len(parameters),
                        tuple('-' * width for width in widths))

        gap = ' ' * COMPARE_COLUMN_GAP
        lines = [gap.join(entry.ljust(width)
                          for entry, width in zip(row, widths)).rstrip()
                 for row in rows]
        fits = int((COMPARE_WINDOW_MAX_HEIGHT_IN - COMPARE_WINDOW_TITLE_IN
                    - COMPARE_WINDOW_MARGIN_IN) // COMPARE_ROW_HEIGHT_IN)
        if len(lines) > fits:
            left_out = len(lines) - (fits - 1)
            lines = lines[:fits - 1] + [f'and {left_out} more']
        return lines

    def _as_compare_scene(self, compare):
        """This method returns a ``GeometryScene`` for the comparison.  When ``compare`` is a
        dictionary of parameter overrides, this method applies the overrides to a copy of this
        scene's parameters.
        """
        if isinstance(compare, GeometryScene):
            return compare
        if isinstance(compare, dict):
            return self.scene.with_parameters(compare)
        return GeometryScene.from_model(compare,
                                        **self.scene.drawing_options())

    def _create_compare_artists(self):
        """This method creates the comparison's dashed artists in the four drawing panels."""
        color = COLORS['compare']
        dashed = dict(linestyle=COMPARE_DASHES, linewidth=COMPARE_LINEWIDTH)

        def moving(axes, three_d=False, label=None, **kwargs):
            if three_d:
                line, = axes.plot(np.zeros(0), np.zeros(0), np.zeros(0),
                                  color=color, **AXLIM_CLIP, label=label,
                                  **kwargs)
            else:
                line, = axes.plot([], [], color=color, label=label, **kwargs)
            self._compare_moving.append((axes, line))
            return line

        self._compare = {}
        self._compare['detector_3d'] = moving(
            self.ax_3d, three_d=True, label='comparison', **dashed)
        self._compare['central_3d'] = moving(self.ax_3d, three_d=True,
                                             **dashed)
        self._compare['source_3d'] = moving(
            self.ax_3d, three_d=True, marker='*',
            markersize=SOURCE_MARKER_SIZE - 3, linestyle='none',
            markerfacecolor='none', markeredgewidth=1.4)
        self._compare['pixel0_3d'] = moving(
            self.ax_3d, three_d=True, marker='+', linestyle='none',
            markersize=INDEX_MARKER_SIZE, markeredgewidth=INDEX_MARKER_WIDTH)
        for name, axes in (('top', self.ax_top), ('side', self.ax_side)):
            self._compare[f'detector_{name}'] = moving(axes, **dashed)
            self._compare[f'central_{name}'] = moving(axes, **dashed)
            self._compare[f'source_{name}'] = moving(
                axes, marker='*', markersize=SOURCE_MARKER_SIZE - 3,
                linestyle='none', markerfacecolor='none', markeredgewidth=1.4)
            self._compare[f'pixel0_{name}'] = moving(
                axes, marker='+', linestyle='none',
                markersize=INDEX_MARKER_SIZE,
                markeredgewidth=INDEX_MARKER_WIDTH)
        # The comparison's name appears once, beside its detector in the top view.
        # The name goes below the primary's labels, away from the channel-offset label.
        label = self.ax_top.annotate(
            'comparison', xy=(0.0, 0.0), textcoords='offset points',
            xytext=(LABEL_GAP_POINTS, -14), fontsize=ANNOTATION_FONT_SIZE,
            color=color, ha='left', va='top')
        self._compare_moving.append((self.ax_top, label))
        self._compare['label_top'] = label
        self._compare['edges_detector'] = moving(
            self.ax_detector, label='comparison', **dashed)
        self._compare['landing_detector'] = moving(
            self.ax_detector, marker='o', markersize=6, linestyle='none',
            markerfacecolor='none', markeredgewidth=1.4)
        self._compare['center_detector'] = moving(
            self.ax_detector, marker='s', markersize=5, linestyle='none',
            markerfacecolor='none', markeredgewidth=1.4)

        # The figure draws a separate volume box for the comparison only when its
        # recon parameters differ.  The box does not move with the view.
        compare_corners = self.compare_scene.volume_corners()
        if not np.allclose(compare_corners, self.scene.volume_corners()):
            box = _joined([compare_corners[[first, second]]
                           for first, second in VOLUME_BOX_EDGES])
            line, = self.ax_3d.plot(box[:, 0], box[:, 1], box[:, 2],
                                    color=color, **AXLIM_CLIP, **dashed)
            self._compare_static.append((self.ax_3d, line))
            for axes, walk, columns in (
                    (self.ax_top, _XY_FOOTPRINT_WALK, TOP_PANEL_COLUMNS),
                    (self.ax_side, _YZ_FACE_WALK, SIDE_PANEL_COLUMNS)):
                points = compare_corners[list(walk)]
                line, = axes.plot(points[:, columns[0]],
                                  points[:, columns[1]], color=color,
                                  **dashed)
                self._compare_static.append((axes, line))

        # The comparison draws its own detector grid when its detector has a
        # different shape.
        if (self.compare_scene.num_det_rows != self.scene.num_det_rows
                or self.compare_scene.num_det_channels
                != self.scene.num_det_channels):
            grid = self.ax_detector.add_patch(Rectangle(
                (-0.5, -0.5), self.compare_scene.num_det_channels,
                self.compare_scene.num_det_rows, facecolor='none',
                edgecolor=color, linewidth=COMPARE_LINEWIDTH,
                linestyle='--'))
            self._compare_static.append((self.ax_detector, grid))

        # The comparison's source path is visible only when both the source-path
        # toggle and the comparison toggle are on.
        self._compare_trajectory_lines = []
        for axes in (self.ax_3d, self.ax_top, self.ax_side):
            if axes is self.ax_3d:
                line, = axes.plot(np.zeros(0), np.zeros(0), np.zeros(0),
                                  color=color, linewidth=1.0, linestyle='-.',
                                  **AXLIM_CLIP)
            else:
                line, = axes.plot([], [], color=color, linewidth=1.0,
                                  linestyle='-.')
            self._compare_static.append((axes, line))
            self._compare_trajectory_lines.append((axes, line))
        self._apply_compare_visibility()
        self._set_animated(self._animate_moving())

    def _apply_compare_visibility(self):
        """This method shows or hides the comparison's artists according to the toggles."""
        for _, artist in self._compare_moving + self._compare_static:
            artist.set_visible(self._show_compare)
        for _, line in self._compare_trajectory_lines:
            line.set_visible(self._show_compare and self._show_trajectory)

    def _remove_compare_artists(self):
        """This method removes every comparison artist from its axes."""
        for _, artist in self._compare_moving + self._compare_static:
            artist.remove()
        self._compare_moving = []
        self._compare_static = []
        self._compare_trajectory_lines = []
        self._compare = {}

    # The data overlays

    def _install_sinogram(self, sinogram, vmin=None, vmax=None):
        """This method creates, replaces, or removes the sinogram image on the detector face.
        When ``vmin`` or ``vmax`` is None, the method uses the smallest or the largest
        displayed value.
        """
        _reject_divided('sinogram', sinogram)
        if self._sinogram_image is not None:
            self._sinogram_image.remove()
            self._moving = [pair for pair in self._moving
                            if pair[1] is not self._sinogram_image]
            self._sinogram_image = None
        self._sinogram = None
        self._sinogram_stride = 1
        if sinogram is None:
            return
        shape = _shape_of(sinogram)
        expected = tuple(self.scene.sinogram_shape)
        if shape != expected:
            raise ValueError(f'The sinogram has shape {shape}, and '
                             f"this scan's sinogram shape is {expected}.")
        # One stride for both axes keeps the displayed pixels square.  The slicing
        # runs on the device that holds the array.
        largest = max(self.scene.num_det_rows, self.scene.num_det_channels)
        stride = max(1, -(-largest // SINOGRAM_DISPLAY_PIXELS))
        values = _to_host(sinogram[:, ::stride, ::stride])
        self._sinogram = values
        self._sinogram_stride = stride
        axes = self.ax_detector
        # The extent puts displayed sample (a, b) at channel b * stride and row
        # a * stride.  That position is the detector pixel the sample came from.
        extent = (-0.5 * stride, (values.shape[2] - 0.5) * stride,
                  (values.shape[1] - 0.5) * stride, -0.5 * stride)
        low = float(np.min(values)) if vmin is None else float(vmin)
        high = float(np.max(values)) if vmax is None else float(vmax)
        self._sinogram_image = axes.imshow(
            values[self._view_index], cmap=SINOGRAM_COLORMAP,
            interpolation='nearest', origin='upper', extent=extent,
            vmin=low, vmax=high, zorder=SINOGRAM_ZORDER)
        self._moving.append((axes, self._sinogram_image))
        self._apply_sinogram_visibility()
        self._set_animated(self._animate_moving())

    def _apply_sinogram_visibility(self):
        """This method shows or hides the painted sinogram according to its toggle."""
        if self._sinogram_image is not None:
            self._sinogram_image.set_visible(self._show_sinogram)

    def _install_recon(self, recon, threshold=None):
        """This method creates, replaces, or removes the reconstruction silhouette and the
        phantom's outlines.  When ``threshold`` is None, the method uses
        :data:`DEFAULT_RECON_THRESHOLD_FRACTION` of the largest absolute value.
        """
        _reject_divided('recon', recon)
        for _, artist in self._recon_images + self._recon_outlines:
            artist.remove()
        # The projected outline is a moving artist, so it must also be removed
        # from _moving.
        self._moving = [pair for pair in self._moving
                        if pair[1] is not self._recon_detector_line]
        self._recon_detector_line = None
        self._recon_images = []
        self._recon_outlines = []
        self._recon_outline_parts = []
        self._recon_projections = None
        self._recon_section_axis = None
        self._recon_section_indices = []
        self._recon_section_total = 0
        self._recon_section_factors = []
        self._recon_threshold = None
        self._recon_threshold_given = None
        if recon is None:
            self._create_legends()
            return
        shape = _shape_of(recon)
        expected = tuple(self.scene.recon_shape)
        if shape != expected:
            raise ValueError(f'The reconstruction has shape {shape}, '
                             f"and this scan's recon shape is {expected}.")
        if threshold is None:
            level = (DEFAULT_RECON_THRESHOLD_FRACTION
                     * _largest_magnitude(recon))
        else:
            level = float(threshold)
            self._recon_threshold_given = level
        self._recon_threshold = level
        self._recon_projections = _support_projections(recon, level)
        # Each projection is indexed (y, x) or (y, z).  Each panel draws y across
        # the screen, so the transpose puts the other coordinate on the rows.
        self._create_silhouette(self.ax_top, self._recon_projections['xy'].T,
                                TOP_PANEL_COLUMNS)
        self._create_silhouette(self.ax_side, self._recon_projections['yz'].T,
                                SIDE_PANEL_COLUMNS)
        self._recon_outline_parts = self._support_outline_parts(recon, level)
        if self._recon_outline_parts:
            self._create_support_outline_3d()
            self._create_recon_detector_outline()
        self._apply_recon_visibility()
        self._create_legends()
        self._set_animated(self._animate_moving())

    def _create_silhouette(self, axes, support, columns):
        """This method draws one panel's silhouette and its outline from the projected
        ``support``.  The array ``support`` is indexed (down, across) on the screen.
        """
        horizontal, vertical = columns
        low, high = self._volume_box_corners()
        # The extent of imshow is (left, right, bottom, top) in data coordinates,
        # and origin='upper' puts the array's first row at the top value.
        extent = (low[horizontal], high[horizontal],
                  high[vertical], low[vertical])
        filled = np.ma.masked_where(~support, np.ones(support.shape))
        image = axes.imshow(filled, cmap=ListedColormap([COLORS['volume']]),
                            interpolation='nearest', origin='upper',
                            extent=extent, alpha=RECON_FILL_ALPHA,
                            vmin=0.0, vmax=1.0, zorder=SILHOUETTE_ZORDER)
        self._recon_images.append((axes, image))

        # These are the image's pixel edges in panel coordinates.  One image pixel
        # is one voxel.
        down = np.linspace(low[vertical], high[vertical],
                           support.shape[0] + 1)
        across = np.linspace(low[horizontal], high[horizontal],
                             support.shape[1] + 1)
        outline = _mask_outline(support, across, down)
        if outline.shape[0] == 0:
            return
        line, = axes.plot(outline[:, 0], outline[:, 1],
                          color=COLORS['volume'],
                          linewidth=RECON_OUTLINE_LINEWIDTH,
                          zorder=RECON_OUTLINE_ZORDER)
        self._recon_outlines.append((axes, line))

    def _support_outline_parts(self, values, level):
        """This method returns the phantom's 3D outline as a list of up to
        :data:`PHANTOM_SECTION_COUNT` sections.  Each section is an (N, 3) array of (x, y, z)
        points with NaN rows between segments.  The sections lie across the axis on which the
        support is thinnest.
        """
        axis = _section_axis(self._recon_projections)
        if axis is None:
            return []
        low, high = _section_extents(self._recon_projections)[axis]
        kind = PHANTOM_SECTION_KIND
        sections = _section_positions(low, high, PHANTOM_SECTION_COUNT, kind)
        share = PHANTOM_OUTLINE_POINT_BUDGET // len(sections)
        parts, factors = [], []
        for position, planes in sections:
            mask = _section_mask(values, axis, planes, level)
            outline, factor = _budgeted_outline(mask, share)
            factors.append(factor)
            parts.append(self._section_points(axis, position, outline))
        self._recon_section_axis = axis
        self._recon_section_indices = [position for position, _ in sections]
        self._recon_section_total = high - low + 1
        self._recon_section_factors = factors
        return parts

    def _section_points(self, axis, position, outline):
        """This method converts one section's (across, down) outline to (N, 3) object
        coordinates as (x, y, z).  Rows of NaN stay NaN.
        """
        across, down = outline[:, 0], outline[:, 1]
        plane = np.full(outline.shape[0], float(position))
        if axis == 'z':
            ijk = np.column_stack([down, across, plane])
        elif axis == 'y':
            ijk = np.column_stack([plane, down, across])
        else:
            ijk = np.column_stack([down, plane, across])
        points = self.scene.voxel_centers(ijk)
        points[~np.isfinite(ijk).all(axis=1)] = np.nan
        return points

    def _section_label(self):
        """This method returns the 3D legend's name for the phantom's outline.  The name gives
        the number of sections drawn and the number of planes the support spans.
        """
        word = 'slabs' if PHANTOM_SECTION_KIND == 'slab' else 'sections'
        drawn = len(self._recon_section_indices)
        total = self._recon_section_total
        axis = self._recon_section_axis
        if drawn < total:
            return f'{PHANTOM_NAME} ({drawn} of {total} {word} across {axis})'
        return f'{PHANTOM_NAME} ({total} {word} across {axis})'

    def _create_support_outline_3d(self):
        """This method draws the phantom's outline in the 3D panel as a dashed line."""
        points = _joined(self._recon_outline_parts)
        line, = self.ax_3d.plot(points[:, 0], points[:, 1], points[:, 2],
                                color=COLORS['volume'],
                                linewidth=RECON_OUTLINE_3D_LINEWIDTH,
                                linestyle='--', label=self._section_label(),
                                **AXLIM_CLIP)
        self._recon_outlines.append((self.ax_3d, line))

    def _apply_recon_visibility(self):
        """This method shows or hides the phantom's fills and outlines according to its toggle."""
        for _, artist in self._recon_images + self._recon_outlines:
            artist.set_visible(self._show_recon)

    def _volume_box_corners(self):
        """This method returns the volume box's (x, y, z) corner at voxel (0, 0, 0) and the
        opposite corner.  Each corner lies half a voxel outside the volume.
        """
        rows, cols, slices = self.scene.recon_shape
        corners = self.scene.voxel_centers(
            [[-0.5, -0.5, -0.5],
             [rows - 0.5, cols - 0.5, slices - 0.5]])
        return corners[0], corners[1]

    def _overlay_names(self):
        """This method returns the names of the drawn overlays for the footer.  It leaves out each
        overlay whose toggle is off.
        """
        names = []
        if self._sinogram_image is not None and self._show_sinogram:
            names.append('sinogram')
        if self._recon_images and self._show_recon:
            if self._recon_threshold_given is None:
                fraction = _three_figures(DEFAULT_RECON_THRESHOLD_FRACTION)
                entry = f'recon above {fraction} max'
            else:
                entry = ('recon above '
                         + _three_figures(self._recon_threshold_given))
            coarsest = max(self._recon_section_factors, default=1)
            if coarsest > 1:
                entry += f', outline coarsened by {coarsest}'
            names.append(entry)
        return names

    # Updating the artists for one view

    def _refresh(self, rebuild_limits=False):
        """This method updates every moving artist for the current view and redraws.  It
        rebuilds the panel limits when ``rebuild_limits`` is True or when the view does not fit
        the current limits.
        """
        view = self.scene.view(self._view_index)
        compare_view = self._compare_view()
        if rebuild_limits or not self._limits_hold(view, compare_view):
            self._compute_limits(view, compare_view)
            self._apply_limits()
            rebuild_limits = True

        self._update_moving_artists(view, compare_view)
        if rebuild_limits:
            self._full_redraw()
        else:
            self._fast_redraw()

    def _update_moving_artists(self, view, compare_view):
        """This method replaces the data of every moving artist, in place."""
        self._update_3d_panel(view)
        self._update_projected_panel(self._top, view, TOP_PANEL_COLUMNS,
                                     TOP_EDGE_RAYS,
                                     self.scene.det_channel_offset,
                                     CHANNEL_OFFSET_LABEL)
        self._update_projected_panel(self._side, view, SIDE_PANEL_COLUMNS,
                                     SIDE_EDGE_RAYS, self.scene.row_offset,
                                     ROW_OFFSET_LABEL)
        self._place_slice_offset_label(view)
        self._update_arc_and_trajectory(view)
        self._update_detector_panel(view)
        self._clip_3d_artists(view)
        self._footer_text.set_text('\n'.join(self._footer_lines()))
        self._update_titles()
        if compare_view is not None:
            self._update_compare_artists(compare_view)

    def _clip_3d_artists(self, view):
        """This method hides the 3D labels and arrowheads whose point lies outside the 3D
        panel's cube.
        """
        cube = (self.ax_3d.get_xlim(), self.ax_3d.get_ylim(),
                self.ax_3d.get_zlim())
        arc = view.rotation_direction_arc
        artists = [(self._source_text_3d, view.source_draw, True),
                   (self._detector_text_3d, _far_corner(view, (0, 1, 2)),
                    True),
                   (self._iso_text_3d, view.detector_origin, True),
                   (self._arc_text_3d, None if arc is None else arc[-1],
                    arc is not None),
                   (self._arrow_3d, None if arc is None else arc[-2],
                    arc is not None and self._arrow_3d is not None),
                   (self._reference_text_3d,
                    self._reference_text_3d.get_position_3d(),
                    self._show_reference),
                   (self._reference_arrow_3d, self._reference_arrow_point,
                    self._show_reference)]
        for artist, point, drawn in artists:
            if artist is None:
                continue
            if not drawn:
                artist.set_visible(False)
                continue
            point = np.asarray(point, dtype=np.float64).reshape(1, 3)
            artist.set_visible(_within(point, cube))

    def _update_titles(self):
        """This method puts the view drawn into the titles of the drawing panels."""
        label = self._view_label()
        self._title_3d.set_text(f'3D view, view {self._view_index}, {label}')
        # A painted sinogram gets no legend entry, so the title says that it is
        # drawn.
        painted = self._sinogram_image is not None and self._show_sinogram
        note = ', with sinogram' if painted else ''
        # The row order is a display choice, so the title names it.
        self._title_detector.set_text(
            f'{_detector_view_title()}\n'
            f'view {self._view_index}, {label}{note}')

    def _update_3d_panel(self, view):
        """This method updates the 3D panel's moving artists."""
        self._source_3d.set_data_3d(*_point_3d(view.source_draw))
        outline = view.detector_outline
        self._detector_3d.set_data_3d(outline[:, 0], outline[:, 1],
                                      outline[:, 2])
        if self._face_3d is not None:
            self._face_3d.set_verts([view.detector_corners])
            # matplotlib projects a 3D collection's vertices only when it draws the
            # whole axes.  A blitted redraw draws the artist alone, so this forces
            # the projection.
            if self.ax_3d.M is not None:
                self._face_3d.do_3d_projection()
        rays = _joined([_sampled_segment(ray[0], ray[1])
                        for ray in view.corner_rays])
        self._rays_3d.set_data_3d(rays[:, 0], rays[:, 1], rays[:, 2])
        central = _sampled_segment(view.source_draw, view.detector_origin)
        self._central_3d.set_data_3d(central[:, 0], central[:, 1],
                                     central[:, 2])
        self._pixel0_3d.set_data_3d(*_point_3d(view.detector_pixel0))
        self._iso_text_3d.set_position_3d(tuple(view.detector_origin))
        self._source_text_3d.set_position_3d(tuple(view.source_draw))
        self._detector_text_3d.set_position_3d(
            tuple(_far_corner(view, (0, 1, 2))))
        if self._show_trajectory:
            path = self._source_trajectory()
            self._path_3d.set_data_3d(path[:, 0], path[:, 1], path[:, 2])

    def _update_projected_panel(self, artists, view, columns, edge_pair,
                                offset, offset_name):
        """This method updates the moving artists of the top or the side panel.  The arguments
        ``columns`` and ``edge_pair`` select the plane drawn.  The arguments ``offset`` and
        ``offset_name`` give the detector offset that the panel shows.
        """
        first, second = columns

        def flat(points):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            return points[:, first], points[:, second]

        rays = view.corner_rays
        # The two corner rays that bound this plane are drawn darker.
        other_pair = tuple(index for index in range(4)
                           if index not in edge_pair)
        edge = _joined([rays[index] for index in edge_pair])
        other = _joined([rays[index] for index in other_pair])
        artists['edge_rays'].set_data(*flat(edge))
        artists['rays'].set_data(*flat(other))

        artists['detector'].set_data(*flat(view.detector_outline))
        artists['central'].set_data(*flat(np.stack([view.source_draw,
                                                    view.detector_origin])))
        artists['source'].set_data(*flat(view.source_draw))
        artists['pixel0'].set_data(*flat(view.detector_pixel0))

        # The detector's label goes at the end farthest from the pixel-0 marker.
        anchor = _far_corner(view, columns)
        pixel0 = view.detector_pixel0
        _place_beside(artists['source_label'],
                      (view.source_draw[first], view.source_draw[second]))
        _place_beside(artists['detector_label'],
                      (anchor[first], anchor[second]))

        # The labels at the detector iso go on the side away from the detector's
        # far corner, which holds the detector's own label.
        iso_point = (view.detector_origin[first], view.detector_origin[second])
        away = -_screen_step(anchor[second] - view.detector_origin[second])
        iso_label = artists.get('iso_label')
        gap = ISO_LABEL_GAP_POINTS
        if iso_label is not None:
            _place_beside(iso_label, iso_point, vertical=away * gap)
            gap += LABEL_LINE_POINTS
        offset_label = artists['offset_label']
        offset_label.set_visible(float(offset) != 0.0)
        if float(offset) != 0.0:
            offset_label.set_text(f'{offset_name} {_three_figures(offset)}')
            _place_beside(offset_label, iso_point, vertical=away * gap)

        # The labels at the detector iso are stacked toward the pixel-0 marker.  The
        # pixel-0 label therefore goes on the side of its marker away from the iso.
        pixel0_label = artists.get('pixel0_label')
        if pixel0_label is not None:
            outward = _screen_step(pixel0[second]
                                   - view.detector_origin[second])
            _place_beside(pixel0_label, (pixel0[first], pixel0[second]),
                          vertical=outward * PIXEL0_LABEL_GAP_POINTS)

        # The offset segment runs from the detector iso to the center of the grid.
        # The other offset is perpendicular to this panel and does not show here.
        if float(offset) == 0.0:
            artists['offset'].set_data([], [])
        else:
            artists['offset'].set_data(
                *flat(np.stack([view.detector_origin, view.detector_center])))

    def _place_slice_offset_label(self, view):
        """This method puts the side view's recon_slice_offset label at the end of its segment,
        on the side away from the source.
        """
        label = self._slice_offset_label
        if label is None:
            return
        _, vertical_column = SIDE_PANEL_COLUMNS
        end_z = self._slice_offset_point[1]
        source_z = float(view.source_draw[vertical_column])
        # This is +1 when the source is drawn above the segment's end.  The
        # label then goes below the end.
        above = _screen_step(source_z - end_z)
        # The gap is a whole line wide.  In a wide and short side view the source's
        # marker and the detector's row-offset label sit at nearly the same height.
        gap = LABEL_GAP_POINTS + LABEL_LINE_POINTS
        # The label goes on the horizontal side away from the volume box, which is
        # the side of larger y.
        _place_beside(label, self._slice_offset_point,
                      side=_screen_step(1.0), vertical=-above * gap)

    def _update_arc_and_trajectory(self, view):
        """This method updates the rotation arc, its arrowhead, and the source path for the
        current view.
        """
        arc = view.rotation_direction_arc
        if self._arrow_3d is not None:
            self._arrow_3d.remove()
            self._arrow_3d = None
        for artist in (self._arc_3d, self._arc_text_3d, self._arc_top,
                       self._arrow_top, self._arc_text_top):
            artist.set_visible(arc is not None)
        if arc is not None:
            self._arc_3d.set_data_3d(arc[:, 0], arc[:, 1], arc[:, 2])
            self._arc_text_3d.set_position_3d(tuple(arc[-1]))
            start, end = arc[-2], arc[-1]
            direction = end - start
            self._arrow_3d = self.ax_3d.quiver(
                start[0], start[1], start[2],
                direction[0], direction[1], direction[2],
                color=COLORS['axis'], arrow_length_ratio=3.0, linewidth=1.4)
            self._arrow_3d.set_animated(self._animate_moving())
            # The call to quiver enlarges the axes limits to hold its arrow, so
            # this restores the panel's limits afterward.
            self._apply_3d_limits()

            top_first, top_second = TOP_PANEL_COLUMNS
            self._arc_top.set_data(arc[:, top_first], arc[:, top_second])
            self._arrow_top.set_positions(
                (start[top_first], start[top_second]),
                (end[top_first], end[top_second]))
            # The arc's label goes on the side of the arc's end away from the
            # source, so it does not overlap the source's label.
            ahead = _screen_step(end[top_second]
                                 - view.source_draw[top_second])
            _place_beside(self._arc_text_top,
                          (end[top_first], end[top_second]),
                          vertical=ahead * ARC_LABEL_GAP_POINTS)

        if self._show_trajectory:
            path = self._source_trajectory()
            self._path_top.set_data(path[:, TOP_PANEL_COLUMNS[0]],
                                    path[:, TOP_PANEL_COLUMNS[1]])
            self._path_side.set_data(path[:, SIDE_PANEL_COLUMNS[0]],
                                     path[:, SIDE_PANEL_COLUMNS[1]])
            travel = self._quantities['helical_travel_alu']
            self._path_note_side.set_text(
                f'source z range {_three_figures(travel)} ALU')
        for artist in (self._path_top, self._path_side, self._path_3d,
                       self._path_note_side):
            artist.set_visible(self._show_trajectory)

    def _update_detector_panel(self, view):
        """This method updates the projected outlines on the detector face.  It draws the parts
        that fall past the detector's edge in a separate color.
        """
        if self._sinogram_image is not None:
            self._sinogram_image.set_data(self._sinogram[self._view_index])
        self._update_recon_detector_outline()

        outline = np.asarray(view.volume_outline_on_detector, dtype=np.float64)
        edges = np.stack([outline[[first, second]]
                          for first, second in VOLUME_BOX_EDGES])
        self.detector_volume_edges = edges
        inside, outside = self._split_edges(edges)
        # The panel's axes are (channel, row), and the scene reports
        # (row, channel).
        self._edges_inside.set_data(inside[:, 1], inside[:, 0])
        self._edges_outside.set_data(outside[:, 1], outside[:, 0])

        if self._ror_inside is None:
            return
        # Each rim is a closed polyline, so its segments are its consecutive
        # pairs of points.  This splits the two rims together as one set.
        rims = np.asarray(view.ror_outline_on_detector, dtype=np.float64)
        segments = np.concatenate([np.stack([rim[:-1], rim[1:]], axis=1)
                                   for rim in rims])
        inside, outside = self._split_edges(segments)
        self._ror_inside.set_data(inside[:, 1], inside[:, 0])
        self._ror_outside.set_data(outside[:, 1], outside[:, 0])

    def _update_recon_detector_outline(self):
        """This method projects the phantom's outline onto the detector face for this view."""
        if self._recon_detector_line is None:
            return
        parts = self._recon_outline_parts
        points = np.concatenate(parts)
        finite = np.isfinite(points).all(axis=1)
        pairs = np.full((points.shape[0], 2), np.nan)
        if finite.any():
            rows, channels = self.scene.project_points(points[finite],
                                                       self._view_index)
            pairs[finite, 0] = channels
            pairs[finite, 1] = rows
        breaks = np.cumsum([part.shape[0] for part in parts])[:-1]
        drawn = _joined(np.split(pairs, breaks), 2)
        self._recon_detector_line.set_data(drawn[:, 0], drawn[:, 1])

    def _split_edges(self, edges):
        """This method splits an (M, 2, 2) array of projected segments into the parts on and off
        the detector grid.  Each part is a polyline of (row, channel) pairs with NaN rows between
        segments.
        """
        inside_parts, outside_parts = [], []
        for edge in edges:
            if self._inside_detector(edge).all():
                inside_parts.append(edge)
                continue
            fraction = np.linspace(0.0, 1.0, EDGE_CLIP_SAMPLES)[:, None]
            points = edge[0][None, :] + fraction * (edge[1] - edge[0])[None, :]
            inside = self._inside_detector(points)
            for span, is_inside in _runs_of_equal_flags(inside):
                piece = points[span]
                if piece.shape[0] < 2:
                    continue
                (inside_parts if is_inside else outside_parts).append(piece)
        return (_joined(inside_parts, 2), _joined(outside_parts, 2))

    def _inside_detector(self, row_channel):
        """This method returns whether each (row, channel) pair lies on the detector grid."""
        pairs = np.asarray(row_channel, dtype=np.float64).reshape(-1, 2)
        return ((pairs[:, 0] >= -0.5)
                & (pairs[:, 0] <= self.scene.num_det_rows - 0.5)
                & (pairs[:, 1] >= -0.5)
                & (pairs[:, 1] <= self.scene.num_det_channels - 0.5))

    def _update_compare_artists(self, view):
        """This method updates the comparison overlay from ``view``, which is the comparison's
        ``ViewScene`` for the current view."""
        compare = self._compare
        outline = view.detector_outline
        compare['detector_3d'].set_data_3d(outline[:, 0], outline[:, 1],
                                           outline[:, 2])
        central = np.stack([view.source_draw, view.detector_origin])
        compare['central_3d'].set_data_3d(central[:, 0], central[:, 1],
                                          central[:, 2])
        source = view.source_draw
        pixel0 = view.detector_pixel0
        compare['source_3d'].set_data_3d(*_point_3d(source))
        compare['pixel0_3d'].set_data_3d(*_point_3d(pixel0))
        for name, (first, second) in (('top', TOP_PANEL_COLUMNS),
                                      ('side', SIDE_PANEL_COLUMNS)):
            compare[f'detector_{name}'].set_data(outline[:, first],
                                                 outline[:, second])
            compare[f'central_{name}'].set_data(central[:, first],
                                                central[:, second])
            compare[f'source_{name}'].set_data([source[first]],
                                               [source[second]])
            compare[f'pixel0_{name}'].set_data([pixel0[first]],
                                               [pixel0[second]])

        # The comparison's label sits beside the end of its detector that is
        # farthest from its pixel-0 marker, as the primary's does.
        anchor = _far_corner(view, TOP_PANEL_COLUMNS)
        _place_beside(compare['label_top'],
                      (anchor[TOP_PANEL_COLUMNS[0]],
                       anchor[TOP_PANEL_COLUMNS[1]]))

        indices = np.asarray(view.volume_outline_on_detector,
                             dtype=np.float64)
        edges = _joined([indices[[first, second]]
                         for first, second in VOLUME_BOX_EDGES], 2)
        compare['edges_detector'].set_data(edges[:, 1], edges[:, 0])
        ((landing_row, landing_channel),
         (center_row, center_channel)) = \
            self.compare_scene.detector_iso_and_center_indices()
        compare['landing_detector'].set_data([float(landing_channel)],
                                             [float(landing_row)])
        compare['center_detector'].set_data([float(center_channel)],
                                            [float(center_row)])
        if self._show_trajectory:
            path = self._compare_source_trajectory()
            for axes, line in self._compare_trajectory_lines:
                if axes is self.ax_3d:
                    line.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
                elif axes is self.ax_top:
                    line.set_data(path[:, TOP_PANEL_COLUMNS[0]],
                                  path[:, TOP_PANEL_COLUMNS[1]])
                else:
                    line.set_data(path[:, SIDE_PANEL_COLUMNS[0]],
                                  path[:, SIDE_PANEL_COLUMNS[1]])

    # Panel limits

    def _object_point_groups(self, view):
        """This method returns the object-frame points that one view's drawing occupies.  The
        figure computes the panel limits from these points.
        """
        groups = [view.volume_corners, view.detector_outline,
                  view.source_draw.reshape(1, 3),
                  view.detector_origin.reshape(1, 3),
                  view.detector_center.reshape(1, 3),
                  view.corner_rays.reshape(-1, 3)]
        if view.rotation_axis is not None:
            groups.append(view.rotation_axis)
        if view.translation_path is not None:
            groups.append(view.translation_path)
        if view.rotation_direction_arc is not None:
            groups.append(view.rotation_direction_arc)
        return groups

    def _limit_sample_indices(self):
        """This method returns the views from which the figure computes the panel limits.  These
        views are the following: :data:`LIMIT_SAMPLE_VIEWS` sampled views, the view drawn, and
        earlier views that did not fit.
        """
        count = self.scene.num_views
        if count <= LIMIT_SAMPLE_VIEWS:
            sample = set(range(count))
        else:
            sample = set(int(index) for index in np.round(
                np.linspace(0, count - 1, LIMIT_SAMPLE_VIEWS)))
        sample |= self._limit_views
        sample.add(self._view_index)
        self._limit_views = sample
        return sorted(sample)

    def _compute_limits(self, view, compare_view):
        """This method computes and stores the limits of the four drawing panels."""
        groups = []
        outlines = []
        for index in self._limit_sample_indices():
            sampled = self.scene.view(index)
            groups.extend(self._object_point_groups(sampled))
            outlines.append(sampled.volume_outline_on_detector)
            if self.compare_scene is not None:
                other = self.compare_scene.view(self._compare_index(index))
                groups.extend(self._object_point_groups(other))
                outlines.append(other.volume_outline_on_detector)
        groups.extend(self._object_point_groups(view))
        outlines.append(view.volume_outline_on_detector)
        if compare_view is not None:
            groups.extend(self._object_point_groups(compare_view))
            outlines.append(compare_view.volume_outline_on_detector)
        if self._show_trajectory:
            groups.append(self._source_trajectory())
            if self.compare_scene is not None:
                groups.append(self._compare_source_trajectory())
        # The limits include the angle-0 reference even when it is hidden.  Turning
        # the reference on then cannot put it outside the limits.
        reference = self._reference_view()
        groups.extend([reference.source_draw.reshape(1, 3),
                       reference.detector_outline,
                       reference.detector_origin.reshape(1, 3)])

        points = np.concatenate([np.asarray(group, dtype=np.float64)
                                 .reshape(-1, 3) for group in groups])
        self._limits['scan'] = _cube_bounds(points)
        self._limits['volume'] = self._volume_cube()
        self._limits['top'] = _bounds(points[:, list(TOP_PANEL_COLUMNS)])
        self._limits['side'] = _bounds(points[:, list(SIDE_PANEL_COLUMNS)])

        index_points = np.concatenate(
            [np.asarray(outline, dtype=np.float64).reshape(-1, 2)
             for outline in outlines]
            + [np.array([[-0.5, -0.5],
                         [self.scene.num_det_rows - 0.5,
                          self.scene.num_det_channels - 0.5]])])
        margin = PANEL_MARGIN * max(self.scene.num_det_rows,
                                    self.scene.num_det_channels)
        row_low, row_high = (float(np.min(index_points[:, 0])) - margin,
                             float(np.max(index_points[:, 0])) + margin)
        channel_low, channel_high = (
            float(np.min(index_points[:, 1])) - margin,
            float(np.max(index_points[:, 1])) + margin)
        self._limits['detector'] = ((row_low, row_high),
                                    (channel_low, channel_high))

    def _volume_cube(self):
        """This method returns the cube of the ``'volume'`` zoom as three (low, high) pairs.  The
        cube is centered on the volume.  Its width is :data:`ZOOM_VOLUME_WIDTH_FACTOR` times the
        largest extent of the volume.
        """
        corners = self.scene.volume_corners()
        if self.compare_scene is not None:
            corners = np.concatenate([corners,
                                      self.compare_scene.volume_corners()])
        low = corners.min(axis=0)
        high = corners.max(axis=0)
        center = 0.5 * (low + high)
        extent = ZOOM_VOLUME_WIDTH_FACTOR * float(np.max(high - low))
        if extent <= 0.0:
            extent = 1.0
        return tuple((float(center[index] - 0.5 * extent),
                      float(center[index] + 0.5 * extent))
                     for index in range(3))

    def _limits_hold(self, view, compare_view):
        """This method returns True when the current view's content fits the limits already set."""
        if not self._limits:
            return False
        groups = self._object_point_groups(view)
        outlines = [view.volume_outline_on_detector]
        if compare_view is not None:
            groups.extend(self._object_point_groups(compare_view))
            outlines.append(compare_view.volume_outline_on_detector)
        points = np.concatenate([np.asarray(group, dtype=np.float64)
                                 .reshape(-1, 3) for group in groups])
        if not _within(points[:, list(TOP_PANEL_COLUMNS)],
                       self._limits['top']):
            return False
        if not _within(points[:, list(SIDE_PANEL_COLUMNS)],
                       self._limits['side']):
            return False
        if self._zoom == 'scan' and not _within(points, self._limits['scan']):
            return False
        index_points = np.concatenate(
            [np.asarray(outline, dtype=np.float64).reshape(-1, 2)
             for outline in outlines])
        return _within(index_points, self._limits['detector'])

    def _apply_limits(self):
        """This method applies the stored limits to the four drawing panels and turns
        autoscaling off.
        """
        self._apply_3d_limits()
        # Both projected panels draw y to the left.  The top view draws x
        # downward and the side view draws z downward.
        for axes, key in ((self.ax_top, 'top'), (self.ax_side, 'side')):
            horizontal, vertical = self._limits[key]
            axes.set_xlim(*_screen_pair(*horizontal))
            axes.set_ylim(*_screen_pair(*vertical))
            axes.set_autoscalex_on(False)
            axes.set_autoscaley_on(False)
        # The detector face draws the channel index increasing to the right.  It
        # inverts the row axis, which puts row 0 at the top.
        rows, channels = self._limits['detector']
        self.ax_detector.set_xlim(*channels)
        self.ax_detector.set_ylim(*_screen_pair(*rows))
        self.ax_detector.set_autoscalex_on(False)
        self.ax_detector.set_autoscaley_on(False)

    def _apply_3d_limits(self):
        """This method applies the current zoom's cube to the 3D panel, with increasing limits on
        every axis.
        """
        cube = self._limits['volume' if self._zoom == 'volume' else 'scan']
        axes = self.ax_3d
        axes.set_xlim(*cube[0])
        axes.set_ylim(*cube[1])
        axes.set_zlim(*cube[2])
        axes.set_box_aspect((1.0, 1.0, 1.0))
        axes.set_autoscalex_on(False)
        axes.set_autoscaley_on(False)
        axes.set_autoscalez_on(False)

    # Drawing, both full redraws and blitted redraws

    def _animate_moving(self):
        """This method returns True when the moving artists are marked animated, which happens
        only on a backend in ``BLIT_BACKENDS`` that supports blitting.
        """
        canvas = self.figure.canvas
        return (self.enable_blit
                and matplotlib.get_backend().lower() in BLIT_BACKENDS
                and bool(getattr(canvas, 'supports_blit', False)))

    def _blit_usable(self):
        """This method returns True when the figure can now redraw only the moving artists by
        blitting.  The figure does not blit during a save.
        """
        return self._animate_moving() and not self._suspend_blit

    def _full_redraw(self):
        """This method redraws the whole figure, places the text blocks, and stores the new
        background.
        """
        self._background = None
        canvas = self.figure.canvas
        blit_usable = self._blit_usable()
        draw = canvas.draw if blit_usable else canvas.draw_idle
        draw()
        for _ in range(3):
            # The first two passes place the blocks.  A third pass handles a block
            # that moved again.
            if not self._place_text_blocks():
                break
            if blit_usable:
                self._background = None
            draw()
        if blit_usable and self._background is None:
            # The backend did not send a draw event, so this stores the background
            # here instead of in the draw handler.
            self._capture_background()

    def _fast_redraw(self):
        """This method restores the background, redraws what moved, and blits."""
        canvas = self.figure.canvas
        if not self._blit_usable():
            canvas.draw_idle()
            return
        if self._background is None:
            self._full_redraw()
            return
        canvas.restore_region(self._background)
        self._redraw_slider_row()
        self._draw_moving_and_blit()

    def _on_draw_event(self, event):
        """This method stores a new background whenever the whole figure is redrawn."""
        if self._suspend_blit or not self._blit_usable():
            return
        if event is not None and event.canvas is not self.figure.canvas:
            return
        self._capture_background()

    def _capture_background(self):
        """This method stores the current canvas as the background and draws what moves."""
        canvas = self.figure.canvas
        try:
            self._background = canvas.copy_from_bbox(self.figure.bbox)
        except Exception:
            self._background = None
            return
        self._draw_moving_and_blit()

    def _draw_moving_and_blit(self):
        """This method draws every moving artist onto the canvas and blits the figure."""
        canvas = self.figure.canvas
        for axes, artist in self._moving + self._compare_moving:
            if artist.get_visible():
                axes.draw_artist(artist)
        if self._arrow_3d is not None and self._arrow_3d.get_visible():
            self.ax_3d.draw_artist(self._arrow_3d)
        canvas.blit(self.figure.bbox)
        try:
            canvas.flush_events()
        except NotImplementedError:
            pass

    def _redraw_slider_row(self):
        """This method redraws the slider, because the restored background holds an out-of-date
        image of it.
        """
        if self.view_slider is None or not self._slider_axes.get_visible():
            return
        canvas = self.figure.canvas
        renderer = canvas.get_renderer()
        try:
            region = self._slider_axes.get_tightbbox(renderer).padded(6)
        except Exception:
            region = self._slider_axes.bbox.padded(6)
        # The region includes the previous region, so that a shorter value text
        # does not leave part of the old text on the screen.
        if self._slider_region is not None:
            region = Bbox.union([region, self._slider_region])
        self._slider_region = region
        self._clear_rect.set_bounds(region.x0, region.y0, region.width,
                                    region.height)
        self._clear_rect.draw(renderer)
        self._slider_axes.draw(renderer)


def _finish_2d_panel(axes):
    """This function gives a 2D panel equal aspect, small tick labels, and a light grid."""
    axes.set_aspect('equal', adjustable='box')
    axes.tick_params(labelsize=TICK_FONT_SIZE)
    axes.grid(True, linewidth=0.3, alpha=0.4)


# This list keeps references to figures opened with block=False, in case the
# caller drops the return value.  The next blocking call closes them.
_NONBLOCKING_FIGURES = []


def geometry_viewer(model_or_scene, view_index=0, show_trajectory=False,
                    compare=None, show_reference=True, zoom=DEFAULT_ZOOM,
                    title=None, figsize=(15.0, 9.0),
                    elevation_deg=DEFAULT_ELEVATION_DEG,
                    azimuth_deg=DEFAULT_AZIMUTH_DEG, sinogram=None,
                    vmin=None, vmax=None, recon=None, recon_threshold=None,
                    show_sinogram=True, show_recon=True, show_compare=True,
                    block=True):
    """This function opens a window that draws a model's scan geometry, one view at a time.

    The panels show a 3D view, the top and side views, the detector face, and derived numbers
    such as the magnification.  A slider steps through the views.  A sinogram, a phantom, and a
    second geometry can be drawn with the geometry.

    Args:
        model_or_scene: A TomographyModel, or a GeometryScene.
        view_index (int, optional): The view drawn first.  Defaults to 0.
        show_trajectory (bool, optional): If True, the figure draws the source's path over
            all views.  Defaults to False.
        compare (optional): A second geometry to draw over the first.  It may be a
            model, a GeometryScene, or a dictionary of parameter overrides such as
            ``dict(det_channel_offset=12.5)``.  Defaults to None.
        show_reference (bool, optional): If True, the figure also draws the source and the
            detector at their angle-0 position.  Defaults to True.
        zoom (str, optional): The 3D panel's zoom.  'scan' fits the source, the detector, and
            the volume in the 3D panel.  'volume' fits the volume.  Defaults to 'scan'.
        title (str, optional): The window title.  None names the geometry and the shapes.
            Defaults to None.
        figsize (tuple, optional): The figure size in inches.  Defaults to (15.0, 9.0).
        elevation_deg, azimuth_deg (float, optional): The 3D camera angles in degrees.
            Default to -25 and -40.
        sinogram (array, optional): A sinogram of shape (num_views, num_det_rows,
            num_det_channels).  The figure paints the current view of the sinogram on the
            detector face.  The figure subsamples a detector wider than 128 pixels for display.
            Defaults to None.
        vmin, vmax (float, optional): The gray scale limits for the sinogram.  None uses the
            smallest and largest values of the sinogram.  Default to None.
        recon (array, optional): A reconstruction or a phantom of shape ``recon_shape``.  The
            figure draws it as a silhouette in the top and side views.  Defaults to None.
        recon_threshold (float, optional): A voxel belongs to the silhouette if its absolute
            value is above this threshold.  None uses a tenth of the largest absolute value.  Defaults
            to None.
        show_sinogram, show_recon, show_compare (bool, optional): Whether each overlay is drawn
            when the window opens.  A toggle in the window turns each on and off.  Default to True.
        block (bool, optional): If True, the function returns when the window is closed.  If
            False, the function returns at once and leaves the window open.  The window then
            becomes interactive when the next blocking call runs.  Defaults to True.

    Returns:
        GeometryFigure: The figure.  A nonblocking caller can use it to change the view or
        save an image.
    """
    figure = GeometryFigure(model_or_scene, view_index=view_index,
                            show_trajectory=show_trajectory, figsize=figsize,
                            title=title, elevation_deg=elevation_deg,
                            azimuth_deg=azimuth_deg, compare=compare,
                            zoom=zoom, show_reference=show_reference,
                            sinogram=sinogram, vmin=vmin, vmax=vmax,
                            recon=recon, recon_threshold=recon_threshold,
                            show_sinogram=show_sinogram,
                            show_recon=show_recon,
                            show_compare=show_compare)
    figure.show(block=block)
    if not block:
        _NONBLOCKING_FIGURES.append(figure)
        return figure
    # The blocking show has returned, so this also closes the earlier nonblocking
    # figures.  Each of them may have a comparison window, which is a separate figure.
    for nonblocking in _NONBLOCKING_FIGURES:
        plt.close(nonblocking.figure)
        if nonblocking.compare_figure is not None:
            plt.close(nonblocking.compare_figure)
    _NONBLOCKING_FIGURES.clear()
    # Under TkAgg, this runs garbage collection now, on the main thread.  TkAgg
    # leaves tkinter objects that a background thread cannot finalize.
    if matplotlib.get_backend() == 'TkAgg':
        import gc
        gc.collect()
    return figure


"""This module describes a scan geometry as points and lines that can be drawn.

``GeometryScene`` converts the distances, pitches, offsets, and view angles of a model into
the points and lines that ``geometry_figure`` draws.  This module imports only numpy.  All
positions are in a right-handed (x, y, z) coordinate frame attached to the object.  The z axis
of this frame is the rotation axis.  In this frame, the object stays fixed, and the source and
the detector move around it.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = ['GeometryScene', 'ViewScene', 'GEOMETRY_KINDS',
           'required_parameter_names', 'CURVED_ARC_SAMPLES', 'RIM_SAMPLES',
           'DEFAULT_DRAWING_DISTANCE_FACTOR', 'MULTIAXIS_SOURCE_ON_PLUS_Y',
           'ROTATION_ARC_SWEEP', 'ROTATION_ARC_SAMPLES',
           'CLOCKWISE_FROM_PLUS_Z', 'COUNTERCLOCKWISE_FROM_PLUS_Z',
           'DIFFERENCE_EXCLUDED_QUANTITIES', 'values_are_equal']


#: The geometry kinds that this module supports.  Each kind corresponds to an mbirtorch model
#: class.
GEOMETRY_KINDS = ('parallel', 'cone', 'multiaxis', 'translation')

#: The number of points along one curved edge of the outline of a curved detector.
CURVED_ARC_SAMPLES = 33

#: The number of points on each circular edge of the region of reconstruction.  The last
#: point repeats the first.
RIM_SAMPLES = 91

#: The drawing distance is this multiple of the largest half width of the volume.  The scene
#: uses the drawing distance when a source or a detector has no physical position.
DEFAULT_DRAWING_DISTANCE_FACTOR = 2.5

#: The half length of the drawn rotation axis is this multiple of the half height of the
#: volume along z.
DEFAULT_ROTATION_AXIS_EXTENSION = 1.25

#: The arc that shows the rotation direction covers this angle in radians.  It is drawn with
#: this many points.
ROTATION_ARC_SWEEP = 0.5
ROTATION_ARC_SAMPLES = 17

#: These are the two values of ``ViewScene.source_travel_sense``.  The direction is as seen
#: from a point on the +z axis that looks toward the origin.
CLOCKWISE_FROM_PLUS_Z = 'clockwise seen from +z'
COUNTERCLOCKWISE_FROM_PLUS_Z = 'counterclockwise seen from +z'

#: :meth:`GeometryScene.differences` does not report these entries of
#: ``derived_quantities``, because they are text.
DIFFERENCE_EXCLUDED_QUANTITIES = ('angle_note', 'drawing_note')

#: The side of the source in the multiaxis geometry.  True puts the source on the +y side, as
#: in the cone and translation geometries.  The detector indices do not depend on this choice.
MULTIAXIS_SOURCE_ON_PLUS_Y = True


# The parameter names that the scene reads from a model through ``get_params``.
_COMMON_PARAMETER_NAMES = (
    'sinogram_shape', 'recon_shape', 'delta_voxel', 'voxel_row_aspect',
    'voxel_slice_aspect', 'delta_det_channel', 'delta_det_row',
    'det_channel_offset', 'det_row_offset', 'use_ror_mask',
)

# The parallel and translation models raise an error for recon_slice_offset.  For this
# reason, the scene reads that name only for the kinds that have it.
_KIND_PARAMETER_NAMES = {
    'parallel': ('angles',),
    'cone': ('view_params_array', 'source_detector_dist', 'source_iso_dist',
             'use_curved_detector', 'recon_slice_offset'),
    'multiaxis': ('angles', 'recon_slice_offset'),
    'translation': ('translation_vectors', 'source_detector_dist',
                    'source_iso_dist'),
}


def required_parameter_names(kind):
    """This function returns the parameter names that a geometry kind needs.

    Args:
        kind (str): One of ``GEOMETRY_KINDS``.

    Returns:
        tuple of str: The names, in a fixed order.
    """
    if kind not in GEOMETRY_KINDS:
        raise ValueError(f'Unknown geometry kind {kind!r}; '
                         f'expected one of {GEOMETRY_KINDS}.')
    return _COMMON_PARAMETER_NAMES + _KIND_PARAMETER_NAMES[kind]


def _rotation_about_z(angle):
    """This function returns the 3x3 matrix that rotates a point about the z axis by
    ``angle``.  A positive angle rotates the +x axis toward the +y axis.
    """
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0],
                     [sine, cosine, 0.0],
                     [0.0, 0.0, 1.0]])


def values_are_equal(first, second, tolerance=1e-12):
    """This function returns True when two parameter values or two derived values are equal.

    A value can be a float, an integer, a bool, a string, a shape tuple, an array with one
    entry for each view, or None.  None means that the parameter does not exist for that
    geometry kind.  The function compares floats with a relative tolerance.  An infinite
    ``source_detector_dist`` equals another infinity of the same sign.

    Args:
        first, second: The two values.
        tolerance (float, optional): The relative tolerance for numbers.  Defaults to 1e-12.

    Returns:
        bool: True if the two values are equal.
    """
    if first is None or second is None:
        return first is None and second is None
    if isinstance(first, str) or isinstance(second, str):
        return str(first) == str(second)
    try:
        left = np.asarray(first, dtype=np.float64)
        right = np.asarray(second, dtype=np.float64)
    except (TypeError, ValueError):
        return bool(first == second)
    if left.shape != right.shape:
        return False
    if left.size == 0:
        return True
    return bool(np.allclose(left, right, rtol=tolerance, atol=0.0,
                            equal_nan=True))


def _merged_intervals(intervals):
    """This function merges overlapping (low, high) intervals into the fewest intervals that
    cover the same set.

    Args:
        intervals (sequence): The (low, high) pairs, in any order.

    Returns:
        list of [low, high]: The merged intervals, in increasing order.
    """
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return merged


def _unit(vector):
    """This function returns ``vector`` scaled to unit length."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError('Cannot normalize a zero vector.')
    return vector / norm


def _model_with_parameters(model, overrides):
    """This function returns a new model of the same class with some parameters replaced.

    The copy keeps the values of the model, except for the values in ``overrides``.  The
    copy is only for drawing, so its compilation is off.
    """
    required, optional, regularization = model.get_all_params()
    constructor_args = {name: value for name, value in required.items()
                        if name != 'geometry_type'}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        copy = type(model)(**constructor_args, compile_mode='off')
    copy.set_params(no_compile=True, no_warning=True, **optional)
    copy.set_params(no_compile=True, no_warning=True, **regularization)
    copy.set_params(no_compile=True, no_warning=True, **overrides)
    return copy


@dataclass
class ViewScene:
    """This class is used to hold the points and lines that draw one view.

    Every array is float64.  Every position is in ALU, in the coordinate frame of the object.
    In this frame, the object stays fixed, and the source and the detector move around it.

    Attributes:
        view_index (int): The index of the view.
        source (ndarray or None): The source position, with shape (3,).  It is None when the
            source is infinitely far away.  This is the case for the parallel and multiaxis
            geometries and for a cone geometry with an infinite source-detector distance.
        source_draw (ndarray): A finite source position for drawing, with shape (3,).  It
            equals ``source`` when ``source`` exists.  Otherwise, it is a point one drawing
            distance back along the ray direction.
        ray_direction (ndarray): The unit vector along the central ray from the source toward
            the detector, with shape (3,).
        detector_center (ndarray): The physical center of the detector grid, with shape (3,).
            Its detector coordinates are (u, v) = (-det_channel_offset, -det_row_offset).
        detector_origin (ndarray): The point where the central ray meets the detector, with
            shape (3,).  Its detector coordinates are (0, 0).
        detector_u_axis (ndarray): The unit vector along which the channel index increases,
            with shape (3,).  For a curved detector, it is the tangent at the central ray.
        detector_v_axis (ndarray): The unit vector along which the row index increases, with
            shape (3,).
        detector_normal (ndarray): The unit normal of the detector, with shape (3,).  For a
            flat detector, it equals ``ray_direction``.
        detector_outline (ndarray): A closed polyline through the physical corners of the
            detector, with shape (M, 3).  A flat detector has five points.  For a curved
            detector, the long edges are sampled along the arc.
        detector_corners (ndarray): The four physical corners of the detector, with shape
            (4, 3).  The order is (low channel, low row), (high channel, low row),
            (high channel, high row), and (low channel, high row).
        detector_pixel0 (ndarray): The center of detector pixel (row 0, channel 0), with
            shape (3,).
        volume_corners (ndarray): The eight outer corners of the volume, with shape (8, 3).
        voxel0_center (ndarray): The center of voxel (0, 0, 0), with shape (3,).
        rotation_axis (ndarray or None): A segment along the rotation axis, with shape (2, 3).
            The segment is a little longer than the volume along z.  It is None for the
            translation geometry, which does not rotate.
        translation_path (ndarray or None): For the translation geometry, the path of the
            origin of the projector frame through the object, with shape (V, 3).  It is None
            for the other geometries.
        corner_rays (ndarray): Four segments that end at the detector corners, with shape
            (4, 2, 3).  They are in the order of ``detector_corners``.  With a finite source,
            they start at the source.  For a parallel geometry, they are parallel to
            ``ray_direction`` and start in the plane through ``source_draw``.
        volume_outline_on_detector (ndarray): The detector indices of the eight volume
            corners as (row, channel), with shape (8, 2).
        ror_outline_on_detector (ndarray or None): The detector indices of the two circular
            edges of the region of reconstruction as (row, channel), with shape
            (2, RIM_SAMPLES, 2).  Entry 0 is the edge at ``z_min``, and entry 1 is the edge
            at ``z_max``.  Each entry is a closed polyline.  ``ror_outline_on_detector`` is
            None when ``ror_cylinder`` is None.
        ror_cylinder (dict or None): The region of reconstruction.  Its keys are ``center``
            (shape (3,)), ``semi_axis_x``, ``semi_axis_y``, ``radius``, ``z_min``, and
            ``z_max``.  It is None when the mask is off or is a custom array.
        rotation_direction_arc (ndarray or None): A short arc with shape (M, 3).  The arc
            starts at ``source_draw`` and covers ``ROTATION_ARC_SWEEP`` radians in the
            direction that the source moves to reach the next view.  The last two points
            give the direction of an arrowhead.  The arc is None for the translation
            geometry.  The arc is also None when the direction cannot be determined.  This happens for a
            single view, for two neighboring views at the same angle, or for a source on the
            rotation axis.
        source_travel_sense (str or None): The same direction in words, either
            ``CLOCKWISE_FROM_PLUS_Z`` or ``COUNTERCLOCKWISE_FROM_PLUS_Z``.  It is None exactly
            when ``rotation_direction_arc`` is None.
    """

    view_index: int
    source: object
    source_draw: np.ndarray
    ray_direction: np.ndarray
    detector_center: np.ndarray
    detector_origin: np.ndarray
    detector_u_axis: np.ndarray
    detector_v_axis: np.ndarray
    detector_normal: np.ndarray
    detector_outline: np.ndarray
    detector_corners: np.ndarray
    detector_pixel0: np.ndarray
    volume_corners: np.ndarray
    voxel0_center: np.ndarray
    rotation_axis: object
    translation_path: object
    corner_rays: np.ndarray
    volume_outline_on_detector: np.ndarray
    ror_outline_on_detector: object = field(default=None)
    ror_cylinder: object = field(default=None)
    rotation_direction_arc: object = field(default=None)
    source_travel_sense: object = field(default=None)


class GeometryScene:
    """This class is used to hold the geometry of one tomography model in a form that can be
    drawn.

    The scene reads the parameters of the model through ``get_params``.  It projects points
    through the ``project_points`` method of the model.  :func:`required_parameter_names`
    lists the parameter names that each geometry kind needs.

    Args:
        model: An instance of a ``TomographyModel`` subclass.
        drawing_distance_factor (float, optional): The drawing distance as a multiple of the
            largest half width of the volume.  The scene uses the drawing distance when a
            source or a detector has no physical position.  Defaults to
            ``DEFAULT_DRAWING_DISTANCE_FACTOR``.
        multiaxis_source_on_plus_y (bool, optional): The side of the source in the multiaxis
            geometry.  See ``MULTIAXIS_SOURCE_ON_PLUS_Y``.
        rotation_axis_extension (float, optional): The half length of the drawn rotation
            axis as a multiple of the half height of the volume along z.
        rotation_arc_sweep (float, optional): The angle in radians that the rotation
            direction arc covers.  See ``ROTATION_ARC_SWEEP``.

    Attributes:
        kind (str): The geometry kind.
        num_views (int): The number of views.
        num_det_rows, num_det_channels (int): The shape of the detector.
        num_rows, num_cols, num_slices (int): The shape of the volume.
        magnification (float): The ratio of the size of an image on the detector to the size
            of an object at the origin.  It is one for a parallel projection.
        is_parallel_type (bool): True when the source is infinitely far away.
        drawing_distance (float): The distance used for positions that have no physical
            value.
        row_pitch (float): The detector row pitch that the projector uses.  It is
            ``delta_det_row``, except for the parallel geometry, which uses ``delta_voxel``.
        row_offset (float): The detector row offset that the projector uses.  It is
            ``det_row_offset``, except for the parallel geometry, which uses zero.
    """

    def __init__(self, model,
                 drawing_distance_factor=DEFAULT_DRAWING_DISTANCE_FACTOR,
                 multiaxis_source_on_plus_y=MULTIAXIS_SOURCE_ON_PLUS_Y,
                 rotation_axis_extension=DEFAULT_ROTATION_AXIS_EXTENSION,
                 rotation_arc_sweep=ROTATION_ARC_SWEEP):
        self.model = model
        self.kind = self.kind_of_model(model)
        self.params = {name: model.get_params(name)
                       for name in required_parameter_names(self.kind)}
        self.drawing_distance_factor = float(drawing_distance_factor)
        self.multiaxis_source_on_plus_y = bool(multiaxis_source_on_plus_y)
        self.rotation_axis_extension = float(rotation_axis_extension)
        self.rotation_arc_sweep = float(rotation_arc_sweep)

        self._read_shapes()
        self._read_detector()
        self._read_view_parameters()
        self._read_source_and_detector_distances()
        self._set_drawing_distance()

    # Construction

    @classmethod
    def from_model(cls, model, **kwargs):
        """This method builds a scene from an mbirtorch model.  It reads the parameters only
        through ``model.get_params``.

        Args:
            model: An instance of a ``TomographyModel`` subclass.
            **kwargs: Keyword arguments for the constructor.

        Returns:
            GeometryScene
        """
        return cls(model, **kwargs)

    @staticmethod
    def kind_of_model(model):
        """This method returns the geometry kind of a model.  The kind is one of
        ``GEOMETRY_KINDS``.
        """
        try:
            geometry_type = str(model.get_params('geometry_type'))
        except Exception:
            geometry_type = ''
        text = (geometry_type + ' ' + type(model).__name__).lower()
        for keyword, kind in (('multiaxis', 'multiaxis'),
                              ('cone', 'cone'),
                              ('translation', 'translation'),
                              ('parallel', 'parallel')):
            if keyword in text:
                return kind
        raise ValueError('Geometry viewer not implemented for  '
                         f'geometry_type {geometry_type!r} and class '
                         f'{type(model).__name__}.')

    def _read_shapes(self):
        """This method reads the sinogram shape, the reconstruction shape, and the voxel
        pitches."""
        sinogram_shape = tuple(int(n) for n in self.params['sinogram_shape'])
        recon_shape = tuple(int(n) for n in self.params['recon_shape'])
        if len(sinogram_shape) != 3 or len(recon_shape) != 3:
            raise ValueError('sinogram_shape and recon_shape must each have '
                             'three entries.')
        self.sinogram_shape = sinogram_shape
        self.recon_shape = recon_shape
        self.num_views, self.num_det_rows, self.num_det_channels = sinogram_shape
        self.num_rows, self.num_cols, self.num_slices = recon_shape

        self.delta_voxel = float(self.params['delta_voxel'])
        self.voxel_row_aspect = float(self.params['voxel_row_aspect'])
        self.voxel_slice_aspect = float(self.params['voxel_slice_aspect'])
        self.delta_voxel_row = self.voxel_row_aspect * self.delta_voxel
        self.delta_voxel_slice = self.voxel_slice_aspect * self.delta_voxel
        self.recon_slice_offset = float(self.params.get('recon_slice_offset',
                                                        0.0))
        self.use_ror_mask = self.params['use_ror_mask']

    def _read_detector(self):
        """This method reads the detector pitches and offsets.  It also sets the row pitch
        and row offset that the projector uses.

        The parallel projector uses a row pitch of ``delta_voxel`` and a row offset of zero.
        """
        self.delta_det_channel = float(self.params['delta_det_channel'])
        self.delta_det_row = float(self.params['delta_det_row'])
        self.det_channel_offset = float(self.params['det_channel_offset'])
        self.det_row_offset = float(self.params['det_row_offset'])
        if self.kind == 'parallel':
            self.row_pitch = self.delta_voxel
            self.row_offset = 0.0
        else:
            self.row_pitch = self.delta_det_row
            self.row_offset = self.det_row_offset

    def _read_view_parameters(self):
        """This method reads the parameters of each view.  These are the angles, z shifts,
        elevations, or translations, depending on the geometry."""
        num_views = self.num_views
        self.angles = np.zeros(num_views)
        self.elevations = np.zeros(num_views)
        self.z_shifts = np.zeros(num_views)
        self.translation_vectors = None

        if self.kind == 'parallel':
            angles = np.asarray(self.params['angles'], dtype=np.float64)
            self._check_view_count(angles.shape[0], 'angles')
            self.angles = angles.reshape(-1)
        elif self.kind == 'cone':
            view_params = np.asarray(self.params['view_params_array'],
                                     dtype=np.float64)
            if view_params.ndim != 2 or view_params.shape[1] != 2:
                raise ValueError('view_params_array must have shape '
                                 f'(num_views, 2); got {view_params.shape}.')
            self._check_view_count(view_params.shape[0], 'view_params_array')
            self.angles = view_params[:, 0]
            self.z_shifts = view_params[:, 1]
        elif self.kind == 'multiaxis':
            angles = np.asarray(self.params['angles'], dtype=np.float64)
            if angles.ndim != 2 or angles.shape[1] != 2:
                raise ValueError('multiaxis angles must have shape '
                                 f'(num_views, 2); got {angles.shape}.')
            self._check_view_count(angles.shape[0], 'angles')
            self.angles = angles[:, 0]
            self.elevations = angles[:, 1]
        else:
            vectors = np.asarray(self.params['translation_vectors'],
                                 dtype=np.float64)
            if vectors.ndim != 2 or vectors.shape[1] != 3:
                raise ValueError('translation_vectors must have shape '
                                 f'(num_views, 3); got {vectors.shape}.')
            self._check_view_count(vectors.shape[0], 'translation_vectors')
            self.translation_vectors = vectors

    def _check_view_count(self, count, name):
        if int(count) != self.num_views:
            raise ValueError(f'{name} holds {count} views but sinogram_shape '
                             f'holds {self.num_views}.')

    def _read_source_and_detector_distances(self):
        """This method reads the source-detector and source-iso distances and computes the
        magnification.  An infinite source-detector distance gives a magnification of one.
        """
        self.source_detector_dist = None
        self.source_iso_dist = None
        self.use_curved_detector = False

        if self.kind in ('cone', 'translation'):
            self.source_detector_dist = float(
                self.params['source_detector_dist'])
            self.source_iso_dist = float(self.params['source_iso_dist'])
        if self.kind == 'cone':
            self.use_curved_detector = bool(self.params['use_curved_detector'])

        if self.kind in ('parallel', 'multiaxis'):
            self.magnification = 1.0
            self.is_parallel_type = True
        elif np.isinf(self.source_detector_dist):
            if self.kind == 'translation':
                raise ValueError('The translation geometry needs a finite '
                                 'source_detector_dist; every view would '
                                 'otherwise carry the same information.')
            if self.use_curved_detector:
                raise ValueError('A curved detector needs a finite '
                                 'source_detector_dist, which is its radius.')
            self.magnification = 1.0
            self.is_parallel_type = True
        else:
            if self.source_iso_dist == 0.0:
                raise ValueError('source_iso_dist must not be zero.')
            self.magnification = (self.source_detector_dist
                                  / self.source_iso_dist)
            self.is_parallel_type = False

    def _set_drawing_distance(self):
        """This method sets the distance used for positions that have no physical value.

        The distance is a multiple of the largest half width of the volume.  As a result,
        the drawn source and detector lie outside the volume.
        """
        half_extents = self.volume_half_extents()
        largest = float(max(half_extents))
        if largest <= 0.0:
            largest = max(self.delta_voxel, 1.0)
        self.drawing_distance = self.drawing_distance_factor * largest

    # The volume

    def volume_half_extents(self):
        """This method returns half of the physical size of the volume in x, y, and z, as a
        3-tuple.

        The z entry is measured from the origin, so it includes ``recon_slice_offset``.
        """
        half_x = 0.5 * self.num_cols * self.delta_voxel
        half_y = 0.5 * self.num_rows * self.delta_voxel_row
        half_z = (0.5 * self.num_slices * self.delta_voxel_slice
                  + abs(self.recon_slice_offset))
        return float(half_x), float(half_y), float(half_z)

    def voxel_centers(self, ijk):
        """This method returns the coordinates of the centers of the given voxels.

        Args:
            ijk (array_like): The voxel indices as (i, j, k), with shape (N, 3).  i is the row
                index, j is the column index, and k is the slice index.  Fractional indices
                are allowed.

        Returns:
            ndarray: The coordinates as (x, y, z), with shape (N, 3).
        """
        ijk = np.asarray(ijk, dtype=np.float64).reshape(-1, 3)
        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        x = self.delta_voxel * (j - (self.num_cols - 1) / 2.0)
        y = self.delta_voxel_row * (i - (self.num_rows - 1) / 2.0)
        z = (self.delta_voxel_slice * (k - (self.num_slices - 1) / 2.0)
             + self.recon_slice_offset)
        return np.stack([x, y, z], axis=1)

    def volume_corners(self):
        """This method returns the eight outer corners of the volume, with shape (8, 3).

        The box that these corners define encloses every voxel.
        """
        half_x, half_y, _ = self.volume_half_extents()
        half_z = 0.5 * self.num_slices * self.delta_voxel_slice
        z_center = self.recon_slice_offset
        corners = [(sx * half_x, sy * half_y, z_center + sz * half_z)
                   for sx in (-1.0, 1.0)
                   for sy in (-1.0, 1.0)
                   for sz in (-1.0, 1.0)]
        return np.array(corners, dtype=np.float64)

    def volume_z_range(self):
        """This method returns the range of z that the volume covers, as (z_min, z_max)."""
        half_z = 0.5 * self.num_slices * self.delta_voxel_slice
        return (self.recon_slice_offset - half_z,
                self.recon_slice_offset + half_z)

    def ror_cylinder(self):
        """This method returns the region of reconstruction as a dict.

        The region is the elliptic cylinder about the rotation axis that
        ``vcd_utils.get_2d_ror_mask`` keeps.  In x and in y, its semi-axis is (n - 1) / 2
        voxel pitches.  The ``radius`` entry is the larger of the two semi-axes.  In z, the
        cylinder covers the whole volume.  The method returns None when ``use_ror_mask`` is
        False or is a custom array.
        """
        if self.use_ror_mask is not True:
            return None
        semi_axis_x = 0.5 * (self.num_cols - 1) * self.delta_voxel
        semi_axis_y = 0.5 * (self.num_rows - 1) * self.delta_voxel_row
        z_min, z_max = self.volume_z_range()
        return dict(center=np.array([0.0, 0.0, 0.5 * (z_min + z_max)]),
                    semi_axis_x=float(semi_axis_x),
                    semi_axis_y=float(semi_axis_y),
                    radius=float(max(semi_axis_x, semi_axis_y)),
                    z_min=float(z_min), z_max=float(z_max))

    # The map from the object frame to the projector frame

    def view_action(self, view_index):
        """This method returns the affine map from the object frame to the projector frame.

        For each view, the projector moves a point p of the object to ``q = R p + d``.  The
        parallel, cone, and multiaxis geometries rotate about z.  A helical cone scan also
        shifts the object toward -z by the z shift of the view.  The translation geometry
        moves the object by minus the translation vector of the view.

        Returns:
            (ndarray, ndarray): The 3x3 rotation R and the offset d with shape (3,).
        """
        view_index = self._check_view_index(view_index)
        offset = np.zeros(3)
        if self.kind == 'translation':
            rotation = np.eye(3)
            offset = -self.translation_vectors[view_index]
        else:
            rotation = _rotation_about_z(self.angles[view_index])
            if self.kind == 'cone':
                offset = np.array([0.0, 0.0, -self.z_shifts[view_index]])
        return rotation, offset

    def to_projector_frame(self, points_xyz, view_index):
        """This method maps points with shape (N, 3) from the object frame to the projector
        frame."""
        points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        rotation, offset = self.view_action(view_index)
        return points @ rotation.T + offset

    def to_object_frame(self, points_xyz, view_index):
        """This method maps points with shape (N, 3) from the projector frame back to the
        object frame.

        It applies the inverse of the map from :meth:`view_action`.
        """
        points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        rotation, offset = self.view_action(view_index)
        return (points - offset) @ rotation

    def direction_to_object_frame(self, vectors, view_index):
        """This method maps direction vectors with shape (N, 3) from the projector frame to
        the object frame.

        It applies only the rotation, not the offset.
        """
        vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
        rotation, _ = self.view_action(view_index)
        return vectors @ rotation

    def _check_view_index(self, view_index):
        view_index = int(view_index)
        if not 0 <= view_index < self.num_views:
            raise IndexError(f'view_index {view_index} is outside '
                             f'[0, {self.num_views}).')
        return view_index

    # Projection onto the detector

    def detector_iso_and_center_indices(self):
        """This method returns the fractional (row, channel) indices of the detector iso and
        of the detector center.

        The detector iso is the point where the central ray meets the detector.  Its
        coordinates are (u, v) = (0, 0).  The detector center is the middle of the grid.
        Its index on each axis is (num - 1) / 2.
        """
        iso_row = (self.row_offset / self.row_pitch
                   + (self.num_det_rows - 1) / 2.0)
        iso_channel = (self.det_channel_offset / self.delta_det_channel
                       + (self.num_det_channels - 1) / 2.0)
        center_row = (self.num_det_rows - 1) / 2.0
        center_channel = (self.num_det_channels - 1) / 2.0
        return ((float(iso_row), float(iso_channel)),
                (float(center_row), float(center_channel)))

    def indices_to_uv(self, row, channel):
        """This method converts fractional detector indices to detector coordinates (u, v).

        This conversion places the detector grid in a drawing.  It is not a projection.
        Index (num - 1) / 2 is at the middle of the grid.  A positive ``det_channel_offset``
        moves the grid toward negative u.  ``det_row_offset`` acts the same way in v.  The
        conversion uses ``row_pitch`` and ``row_offset`` for the rows.
        """
        row = np.asarray(row, dtype=np.float64)
        channel = np.asarray(channel, dtype=np.float64)
        u = ((channel - (self.num_det_channels - 1) / 2.0)
             * self.delta_det_channel - self.det_channel_offset)
        v = ((row - (self.num_det_rows - 1) / 2.0) * self.row_pitch
             - self.row_offset)
        return u, v

    def project_points(self, points_xyz, view_index):
        """This method returns the fractional detector indices of points in the object frame.

        It calls :meth:`TomographyModel.project_points`.  The points are at their positions
        before any shift or rotation, as ``voxel_centers`` returns them.

        Args:
            points_xyz (array_like): The points in the object frame, with shape (N, 3).
            view_index (int or sequence of int): One view or several views, as the model
                method allows.

        Returns:
            (ndarray, ndarray): The fractional row and channel indices.  For one view, each
            array has shape (N,).  For a sequence of views, each array has shape
            (num_selected_views, N).
        """
        return self.model.project_points(points_xyz, view_index)

    # The source and the detector in the projector frame

    def _source_projector_frame(self):
        """This method returns the source position in the projector frame.

        It returns None when the source is infinitely far away.
        """
        if self.is_parallel_type:
            return None
        return np.array([0.0, self.source_iso_dist, 0.0])

    def _ray_direction_projector_frame(self, view_index):
        """This method returns the unit direction of the central ray in the projector frame.

        The direction points from the source toward the detector.
        """
        if self.kind != 'multiaxis':
            return np.array([0.0, -1.0, 0.0])
        elevation = self.elevations[self._check_view_index(view_index)]
        direction = np.array([0.0, -np.cos(elevation), np.sin(elevation)])
        if not self.multiaxis_source_on_plus_y:
            direction = -direction
        return direction

    def _detector_axes_projector_frame(self, view_index):
        """This method returns the u and v unit axes of the detector in the projector frame.
        """
        u_axis = np.array([1.0, 0.0, 0.0])
        if self.kind == 'multiaxis':
            elevation = self.elevations[self._check_view_index(view_index)]
            v_axis = np.array([0.0, np.sin(elevation), np.cos(elevation)])
        else:
            v_axis = np.array([0.0, 0.0, 1.0])
        return u_axis, v_axis

    def _detector_origin_projector_frame(self, view_index):
        """This method returns the point where the central ray meets the detector, in the
        projector frame.

        For a parallel projection, the drawing distance sets the position along the ray.
        """
        direction = self._ray_direction_projector_frame(view_index)
        if self.is_parallel_type:
            return self.drawing_distance * direction
        return (np.array([0.0, self.source_iso_dist, 0.0])
                + self.source_detector_dist * direction)

    def _uv_to_projector_frame(self, u, v, view_index):
        """This method returns the physical positions of detector coordinates (u, v), with
        shape (N, 3).

        On a curved detector, u is the arc length from the central ray, and v is the height
        along z.
        """
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        v = np.atleast_1d(np.asarray(v, dtype=np.float64))
        u, v = np.broadcast_arrays(u, v)
        if self.kind == 'cone' and self.use_curved_detector:
            theta = u / self.source_detector_dist
            radius = self.source_detector_dist
            x = radius * np.sin(theta)
            y = self.source_iso_dist - radius * np.cos(theta)
            return np.stack([x, y, v], axis=1)
        origin = self._detector_origin_projector_frame(view_index)
        u_axis, v_axis = self._detector_axes_projector_frame(view_index)
        return (origin[None, :] + u[:, None] * u_axis[None, :]
                + v[:, None] * v_axis[None, :])

    def detector_uv_extent(self):
        """This method returns the range of the detector in (u, v) as
        ((u_min, u_max), (v_min, v_max)).

        The range runs from index -0.5 to index num - 0.5.  These are the outer edges of
        the first and last pixels.
        """
        u_low, v_low = self.indices_to_uv(-0.5, -0.5)
        u_high, v_high = self.indices_to_uv(self.num_det_rows - 0.5,
                                            self.num_det_channels - 0.5)
        return ((float(u_low), float(u_high)), (float(v_low), float(v_high)))

    def detector_size(self):
        """This method returns the physical (width, height) of the detector in ALU.

        The height uses ``row_pitch``.  As a result, the detector of a parallel geometry is
        as tall as the volume along z.
        """
        return (float(self.num_det_channels * self.delta_det_channel),
                float(self.num_det_rows * self.row_pitch))

    # The points and lines to draw

    def _detector_outline_uv(self):
        """This method returns the detector outline as a closed polyline of (u, v) points.
        """
        (u_min, u_max), (v_min, v_max) = self.detector_uv_extent()
        if not (self.kind == 'cone' and self.use_curved_detector):
            u = np.array([u_min, u_max, u_max, u_min, u_min])
            v = np.array([v_min, v_min, v_max, v_max, v_min])
            return u, v
        arc = np.linspace(u_min, u_max, CURVED_ARC_SAMPLES)
        u = np.concatenate([arc, arc[::-1], arc[:1]])
        v = np.concatenate([np.full(CURVED_ARC_SAMPLES, v_min),
                            np.full(CURVED_ARC_SAMPLES, v_max),
                            np.array([v_min])])
        return u, v

    def _detector_corners_uv(self):
        """This method returns the four corners of the detector as (u, v), in order around
        the detector."""
        (u_min, u_max), (v_min, v_max) = self.detector_uv_extent()
        u = np.array([u_min, u_max, u_max, u_min])
        v = np.array([v_min, v_min, v_max, v_max])
        return u, v

    def view(self, view_index):
        """This method returns the points and lines that draw one view.

        The positions are in the coordinate frame of the object.

        Args:
            view_index (int): The index of the view.

        Returns:
            ViewScene: The points and lines.  The class :class:`ViewScene` describes the
            entries.
        """
        view_index = self._check_view_index(view_index)

        def to_object(points):
            return self.to_object_frame(points, view_index)

        direction = self._ray_direction_projector_frame(view_index)
        ray_direction = _unit(
            self.direction_to_object_frame(direction, view_index)[0])

        source_projector = self._source_projector_frame()
        if source_projector is None:
            source = None
        else:
            source = to_object(source_projector)[0]

        detector_origin = to_object(
            self._detector_origin_projector_frame(view_index))[0]
        if source is None:
            # The source has no finite position.  The drawn source is placed two drawing
            # distances back along the ray from the detector origin.
            source_draw = detector_origin - 2.0 * self.drawing_distance * ray_direction
        else:
            source_draw = source

        u_axis_p, v_axis_p = self._detector_axes_projector_frame(view_index)
        detector_u_axis = _unit(
            self.direction_to_object_frame(u_axis_p, view_index)[0])
        detector_v_axis = _unit(
            self.direction_to_object_frame(v_axis_p, view_index)[0])
        detector_normal = _unit(np.cross(detector_u_axis, detector_v_axis))

        detector_center = to_object(self._uv_to_projector_frame(
            -self.det_channel_offset, -self.row_offset, view_index))[0]

        outline_u, outline_v = self._detector_outline_uv()
        detector_outline = to_object(self._uv_to_projector_frame(
            outline_u, outline_v, view_index))
        corner_u, corner_v = self._detector_corners_uv()
        detector_corners = to_object(self._uv_to_projector_frame(
            corner_u, corner_v, view_index))

        pixel0_u, pixel0_v = self.indices_to_uv(0.0, 0.0)
        detector_pixel0 = to_object(self._uv_to_projector_frame(
            pixel0_u, pixel0_v, view_index))[0]

        if self.is_parallel_type:
            # The four rays stay parallel to the ray direction.
            length = float(np.linalg.norm(source_draw - detector_origin))
            starts = detector_corners - length * ray_direction[None, :]
            corner_rays = np.stack([starts, detector_corners], axis=1)
        else:
            corner_rays = np.stack([np.stack([source_draw, corner])
                                    for corner in detector_corners])

        arc, travel_sense = self._source_travel(view_index, source_draw)

        volume_corners = self.volume_corners()
        row, channel = self.project_points(volume_corners, view_index)
        volume_outline_on_detector = np.stack([row, channel], axis=1)

        cylinder = self.ror_cylinder()
        ror_outline_on_detector = None
        if cylinder is not None:
            rim_row, rim_channel = self.project_points(self.fit_points(),
                                                       view_index)
            ror_outline_on_detector = np.stack(
                [rim_row, rim_channel], axis=1).reshape(2, RIM_SAMPLES, 2)

        if self.kind == 'translation':
            rotation_axis = None
            translation_path = self.translation_vectors.copy()
        else:
            z_min, z_max = self.volume_z_range()
            z_center = 0.5 * (z_min + z_max)
            half = self.rotation_axis_extension * 0.5 * (z_max - z_min)
            if half <= 0.0:
                half = self.rotation_axis_extension * self.delta_voxel_slice
            rotation_axis = np.array([[0.0, 0.0, z_center - half],
                                      [0.0, 0.0, z_center + half]])
            translation_path = None

        return ViewScene(
            view_index=view_index,
            source=source,
            source_draw=source_draw,
            ray_direction=ray_direction,
            detector_center=detector_center,
            detector_origin=detector_origin,
            detector_u_axis=detector_u_axis,
            detector_v_axis=detector_v_axis,
            detector_normal=detector_normal,
            detector_outline=detector_outline,
            detector_corners=detector_corners,
            detector_pixel0=detector_pixel0,
            volume_corners=volume_corners,
            voxel0_center=self.voxel_centers([[0, 0, 0]])[0],
            rotation_axis=rotation_axis,
            translation_path=translation_path,
            corner_rays=corner_rays,
            volume_outline_on_detector=volume_outline_on_detector,
            ror_outline_on_detector=ror_outline_on_detector,
            ror_cylinder=cylinder,
            rotation_direction_arc=arc,
            source_travel_sense=travel_sense,
        )

    def _source_travel(self, view_index, source_draw):
        """This method returns the direction in which the source moves from this view to the
        next view.

        Args:
            view_index (int): The index of the view.
            source_draw (ndarray): The drawn source position of that view.

        Returns:
            (ndarray or None, str or None): The arc with shape (M, 3), and the direction in
            words.  Both are None when the direction cannot be determined.  This happens for
            the translation geometry, for a single view, for two neighboring views at the
            same angle, and for a source on the rotation axis.
        """
        if self.kind == 'translation' or self.num_views < 2:
            return None, None
        neighbor = (view_index + 1 if view_index + 1 < self.num_views
                    else view_index - 1)
        step = float(self.angles[neighbor] - self.angles[view_index])
        if neighbor < view_index:
            step = -step
        if step == 0.0:
            return None, None
        radius = float(np.hypot(source_draw[0], source_draw[1]))
        if radius <= 0.0:
            return None, None

        start = float(np.arctan2(source_draw[1], source_draw[0]))
        sweep = -np.sign(step) * self.rotation_arc_sweep
        azimuth = start + np.linspace(0.0, sweep, ROTATION_ARC_SAMPLES)
        arc = np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth),
                        np.full(ROTATION_ARC_SAMPLES, float(source_draw[2]))],
                       axis=1)
        sense = (CLOCKWISE_FROM_PLUS_Z if sweep < 0.0
                 else COUNTERCLOCKWISE_FROM_PLUS_Z)
        return arc, sense

    def reference_view(self):
        """This method returns the points and lines of the reference position at angle zero.

        In the reference position, the source and the detector are at angle 0.  For the
        translation geometry, they are at zero translation.  For the multiaxis geometry,
        the reference keeps the elevation of view 0.  A viewer draws the reference as a
        fixed mark against which to read the view angles.

        Returns:
            ViewScene: The points and lines, with ``view_index`` 0.
        """
        if self.kind == 'parallel':
            angles = self.angles.copy()
            angles[0] = 0.0
            overrides = dict(angles=angles)
        elif self.kind == 'cone':
            view_params = np.stack([self.angles, self.z_shifts], axis=1)
            view_params[0] = 0.0
            overrides = dict(view_params_array=view_params)
        elif self.kind == 'multiaxis':
            angles = np.stack([self.angles, self.elevations], axis=1)
            angles[0, 0] = 0.0
            overrides = dict(angles=angles)
        else:
            vectors = self.translation_vectors.copy()
            vectors[0] = 0.0
            overrides = dict(translation_vectors=vectors)
        return self.with_parameters(overrides).view(0)

    def _ray_directions_all_views(self):
        """This method returns the unit direction of the central ray in the projector frame
        for every view, with shape (V, 3).
        """
        if self.kind != 'multiaxis':
            return np.tile(np.array([0.0, -1.0, 0.0]), (self.num_views, 1))
        elevation = self.elevations
        directions = np.stack([np.zeros_like(elevation),
                               -np.cos(elevation),
                               np.sin(elevation)], axis=1)
        if not self.multiaxis_source_on_plus_y:
            directions = -directions
        return directions

    def _detector_origins_all_views(self, directions):
        """This method returns the point where the central ray meets the detector in the
        projector frame, for every view, with shape (V, 3).
        """
        if self.is_parallel_type:
            return self.drawing_distance * directions
        return (np.array([0.0, self.source_iso_dist, 0.0])
                + self.source_detector_dist * directions)

    def _detector_centers_all_views(self, directions, origins):
        """This method returns the center of the detector grid in the projector frame for
        every view, with shape (V, 3)."""
        u = -self.det_channel_offset
        v = -self.row_offset
        if self.kind == 'cone' and self.use_curved_detector:
            radius = self.source_detector_dist
            theta = u / radius
            point = np.array([radius * np.sin(theta),
                              self.source_iso_dist - radius * np.cos(theta),
                              v])
            return np.tile(point, (self.num_views, 1))
        if self.kind == 'multiaxis':
            elevation = self.elevations
            v_axis = np.stack([np.zeros_like(elevation),
                               np.sin(elevation),
                               np.cos(elevation)], axis=1)
        else:
            v_axis = np.tile(np.array([0.0, 0.0, 1.0]), (self.num_views, 1))
        u_axis = np.array([1.0, 0.0, 0.0])
        return origins + u * u_axis + v * v_axis

    def _to_object_frame_all_views(self, points):
        """This method maps one point for each view from the projector frame to the object
        frame.  The points have shape (V, 3).
        """
        points = np.asarray(points, dtype=np.float64).reshape(self.num_views, 3)
        if self.kind == 'translation':
            return points + self.translation_vectors
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        if self.kind == 'cone':
            z = z + self.z_shifts
        cosine, sine = np.cos(self.angles), np.sin(self.angles)
        return np.stack([x * cosine + y * sine,
                         -x * sine + y * cosine,
                         z], axis=1)

    def trajectory(self):
        """This method returns the paths of the source and of the detector center over all
        views.

        The method computes the paths for the whole scan at once.  It does not build a
        :class:`ViewScene` for each view.

        Returns:
            (ndarray, ndarray): The source positions and the detector centers, each with
            shape (V, 3), in the coordinate frame of the object.  When the source has no
            finite position, each source entry is the drawn point that
            :attr:`ViewScene.source_draw` uses.
        """
        directions = self._ray_directions_all_views()
        origins = self._detector_origins_all_views(directions)
        if self.is_parallel_type:
            # The drawn source follows the same rule as source_draw.
            sources = origins - 2.0 * self.drawing_distance * directions
        else:
            sources = np.tile(np.array([0.0, self.source_iso_dist, 0.0]),
                              (self.num_views, 1))
        centers = self._detector_centers_all_views(directions, origins)
        return (self._to_object_frame_all_views(sources),
                self._to_object_frame_all_views(centers))

    # Derived numbers

    def fit_shape(self):
        """This method returns the name of the shape that :meth:`fit_report` tests.

        The name is ``'cylinder'`` for the cylinder that :meth:`ror_cylinder` describes.
        The name is ``'box'`` for the volume when there is no mask.
        """
        return 'cylinder' if self.ror_cylinder() is not None else 'box'

    def fit_points(self):
        """This method returns points in the object frame whose projections bound the
        projection of the shape from :meth:`fit_shape`.

        For the box, the points are the eight :meth:`volume_corners`.  For the cylinder, the
        points lie on its two elliptical edges at ``z_min`` and at ``z_max``.  Each edge has
        :data:`RIM_SAMPLES` points, and the edge at ``z_min`` comes first.

        Returns:
            ndarray: The points.  The shape is (8, 3) for the box and
            (2 * RIM_SAMPLES, 3) for the cylinder.
        """
        cylinder = self.ror_cylinder()
        if cylinder is None:
            return self.volume_corners()
        angle = np.linspace(0.0, 2.0 * np.pi, RIM_SAMPLES)
        x = cylinder['center'][0] + cylinder['semi_axis_x'] * np.cos(angle)
        y = cylinder['center'][1] + cylinder['semi_axis_y'] * np.sin(angle)
        rims = [np.stack([x, y, np.full(RIM_SAMPLES, float(height))], axis=1)
                for height in (cylinder['z_min'], cylinder['z_max'])]
        return np.concatenate(rims)

    def fit_report(self):
        """This method reports whether the detector covers the region that the scan
        reconstructs.

        A scan that does not move along the rotation axis fits when the shape from
        :meth:`fit_shape` projects onto the detector in every view.  A helical cone scan
        fits when two conditions hold.  First, the shape stays inside the channel range of
        the detector.  Second, the detector rows of all views together cover the whole
        volume along z.  The test ignores the point spread of the projector.

        Returns:
            dict: The dict has the following entries.

            ``shape``: The tested shape, from :meth:`fit_shape`.

            ``worst_overshoot_pixels``, ``worst_channel_overshoot_pixels``,
            ``worst_row_overshoot_pixels``: The largest distance, in detector pixels, by
            which the shape reaches past a detector edge over all views.  Each value is
            zero when nothing reaches past that edge.

            ``views_leaving_detector``: The number of views in which the shape reaches past
            any detector edge.

            ``helical_rule``: True if the method used the helical test described above.

            ``swept_z_min``, ``swept_z_max``: The lowest and highest z on the line
            x = y = 0 that the detector covers in any view.  Both are nan when no view
            covers a range of z.

            ``z_extent_covered``: True if every z of :meth:`volume_z_range` lies in the
            union of the z ranges that the views cover.  A gap between two groups of views
            gives False.

            ``fits_laterally``: True if the shape stays inside the channel range of the
            detector in every view.

            ``fits_axially``: For the helical test, this value equals
            ``z_extent_covered``.  Otherwise, it is True if the shape stays inside the row
            range of the detector in every view.

            ``fits``: True when both ``fits_laterally`` and ``fits_axially`` are True.
        """
        points = self.fit_points()
        num_shape_points = int(points.shape[0])
        z_min, z_max = self.volume_z_range()
        # The two points on the axis are projected in the same call as the shape points.
        probe_points = np.concatenate([points,
                                       np.array([[0.0, 0.0, z_min],
                                                 [0.0, 0.0, z_max]])])
        row, channel = self.project_points(probe_points,
                                           np.arange(self.num_views))

        # This computes one overshoot for each view from the shape points only.
        row_over = self._overshoot(row[:, :num_shape_points],
                                   self.num_det_rows)
        channel_over = self._overshoot(channel[:, :num_shape_points],
                                       self.num_det_channels)
        worst_row = float(np.max(row_over, initial=0.0))
        worst_channel = float(np.max(channel_over, initial=0.0))
        views_leaving = int(np.count_nonzero(
            np.maximum(row_over, channel_over) > 0.0))

        # Along the line x = y = 0, the row index is an affine function of z.  For this
        # reason, the two projected rows give the z range between the outer row edges.
        # A view whose two rows are equal gives no information about z, so it is skipped.
        row_at_z_min = row[:, num_shape_points]
        row_at_z_max = row[:, num_shape_points + 1]
        if z_max == z_min:
            usable = np.zeros(row_at_z_min.shape, dtype=bool)
        else:
            usable = row_at_z_max != row_at_z_min
        if np.any(usable):
            slope = ((row_at_z_max[usable] - row_at_z_min[usable])
                     / (z_max - z_min))
            intercept = row_at_z_min[usable] - slope * z_min
            first = (-0.5 - intercept) / slope
            last = (self.num_det_rows - 0.5 - intercept) / slope
            lows = np.minimum(first, last)
            highs = np.maximum(first, last)
            intervals = list(zip(lows.tolist(), highs.tolist()))
            swept_z_min = float(np.min(lows))
            swept_z_max = float(np.max(highs))
            covered = any(low <= z_min and z_max <= high
                          for low, high in _merged_intervals(intervals))
        else:
            swept_z_min = swept_z_max = float('nan')
            covered = False

        worst = max(worst_row, worst_channel)
        helical_rule = self.kind == 'cone' and self.helical_travel() > 0.0
        # For a scan that does not move along z, the axial test checks the rows.  For a
        # helical scan, it checks the z range covered by all views together.
        fits_laterally = worst_channel <= 0.0
        fits_axially = covered if helical_rule else worst_row <= 0.0
        fits = fits_laterally and fits_axially
        return dict(
            shape=self.fit_shape(),
            fits_laterally=bool(fits_laterally),
            fits_axially=bool(fits_axially),
            worst_overshoot_pixels=float(worst),
            worst_channel_overshoot_pixels=float(worst_channel),
            worst_row_overshoot_pixels=float(worst_row),
            views_leaving_detector=int(views_leaving),
            helical_rule=bool(helical_rule),
            swept_z_min=float(swept_z_min),
            swept_z_max=float(swept_z_max),
            z_extent_covered=bool(covered),
            fits=bool(fits),
        )

    def volume_fits_detector(self):
        """This method returns two entries of :meth:`fit_report`.

        Returns:
            (bool, float): The entries ``fits`` and ``worst_overshoot_pixels`` of the report.
        """
        report = self.fit_report()
        return report['fits'], report['worst_overshoot_pixels']

    @staticmethod
    def _overshoot(indices, count):
        """This method returns how far the given indices reach past the ends of a detector
        axis.

        It takes the maximum along the last axis.  As a result, an array of shape
        (num_views, N) gives one number for each view.
        """
        indices = np.asarray(indices, dtype=np.float64)
        low = np.max(-0.5 - indices, axis=-1)
        high = np.max(indices - (count - 0.5), axis=-1)
        return np.maximum(0.0, np.maximum(low, high))

    def _fan_and_cone_angles(self):
        """This method returns the full fan angle and the full cone angle in degrees, and a
        note about them.

        Each angle is measured at the source from one edge of the detector to the other.
        Both angles are zero for a parallel projection.
        """
        if self.is_parallel_type:
            return 0.0, 0.0, ('The rays are parallel, so the source subtends '
                              'no angle and both angles are zero.')
        (u_min, u_max), (v_min, v_max) = self.detector_uv_extent()
        distance = self.source_detector_dist
        if self.use_curved_detector:
            # On a curved detector, the fan angle equals the arc length divided by the radius.
            fan = (u_max - u_min) / distance
            note = ('The fan angle is the detector arc divided by the '
                    'source-detector distance.  The cone angle uses the plane '
                    'tangent to the cylinder at the central ray, which is '
                    'where the projector places the rows.')
        else:
            fan = np.arctan(u_max / distance) - np.arctan(u_min / distance)
            note = 'Both angles are subtended at the source by the flat panel.'
        cone = np.arctan(v_max / distance) - np.arctan(v_min / distance)
        return float(np.degrees(fan)), float(np.degrees(cone)), note

    def helical_travel(self):
        """This method returns the total helical travel in ALU.

        The travel is zero for a scan that is not helical.
        """
        if self.kind != 'cone':
            return 0.0
        return float(np.ptp(self.z_shifts))

    def derived_quantities(self):
        """This method returns the numbers of the text panel as plain floats, ints, and
        strings.

        Returns:
            dict: The keys are fixed names.  The dict has the following entries.

            ``geometry_kind``, ``num_views``, ``sinogram_shape_text``, and
            ``recon_shape_text``: The geometry kind and the sizes of the problem.

            ``magnification``: The ratio of the size of an image on the detector to the
            size of an object at the origin.

            ``fan_angle_deg`` and ``cone_angle_deg``: The angles that the detector subtends
            at the source, from edge to edge.  ``angle_note`` explains them.

            ``lateral_fov_alu`` and ``axial_fov_alu``: The width and height of the detector
            divided by the magnification.  These two values give the size of the region at
            the rotation axis that the detector can see.  The axial value includes the helical travel.

            ``delta_voxel_x``, ``delta_voxel_y``, ``delta_voxel_z``: The voxel pitches along
            the three axes of the object.

            ``volume_extent_x``, ``volume_extent_y``, ``volume_extent_z``,
            ``volume_z_center``: The physical size of the volume and its center in z.

            ``detector_width``, ``detector_height``, ``detector_row_pitch``,
            ``detector_channel_pitch``: The physical size of the detector and the pitches
            used to draw it.

            ``detector_center_u`` and ``detector_center_v``: The position of the center of
            the detector grid relative to the central ray.

            ``helical_travel_alu``: The range of the z shifts of the views.

            ``volume_fits_detector``, ``helical_fit_rule``, ``fit_shape``,
            ``fits_laterally``, ``fits_axially``, ``worst_overshoot_pixels``,
            ``worst_channel_overshoot_pixels``, ``worst_row_overshoot_pixels``,
            ``views_leaving_detector``, ``swept_z_min``, ``swept_z_max``, and
            ``z_extent_covered``: The results of :meth:`fit_report`.
            ``volume_fits_detector`` is its ``fits`` entry, and ``helical_fit_rule`` is its
            ``helical_rule`` entry.

            ``drawing_distance`` and ``drawing_note``: The distance used for positions that
            have no physical value, and a note that names those positions.
        """
        fan, cone, angle_note = self._fan_and_cone_angles()
        width, height = self.detector_size()
        extent_x, extent_y, _ = self.volume_half_extents()
        z_min, z_max = self.volume_z_range()
        travel = self.helical_travel()
        fit = self.fit_report()

        quantities = dict(
            geometry_kind=self.kind,
            num_views=int(self.num_views),
            sinogram_shape_text=str(self.sinogram_shape),
            recon_shape_text=str(self.recon_shape),
            magnification=float(self.magnification),
            fan_angle_deg=fan,
            cone_angle_deg=cone,
            angle_note=angle_note,
            lateral_fov_alu=float(width / self.magnification),
            axial_fov_alu=float(height / self.magnification + travel),
            delta_voxel_x=float(self.delta_voxel),
            delta_voxel_y=float(self.delta_voxel_row),
            delta_voxel_z=float(self.delta_voxel_slice),
            volume_extent_x=float(2.0 * extent_x),
            volume_extent_y=float(2.0 * extent_y),
            volume_extent_z=float(z_max - z_min),
            volume_z_center=float(0.5 * (z_min + z_max)),
            detector_width=float(width),
            detector_height=float(height),
            detector_channel_pitch=float(self.delta_det_channel),
            detector_row_pitch=float(self.row_pitch),
            detector_center_u=float(-self.det_channel_offset),
            detector_center_v=float(-self.row_offset),
            helical_travel_alu=travel,
            volume_fits_detector=fit['fits'],
            fits_laterally=fit['fits_laterally'],
            fits_axially=fit['fits_axially'],
            fit_shape=fit['shape'],
            worst_overshoot_pixels=fit['worst_overshoot_pixels'],
            worst_channel_overshoot_pixels=fit[
                'worst_channel_overshoot_pixels'],
            worst_row_overshoot_pixels=fit['worst_row_overshoot_pixels'],
            views_leaving_detector=fit['views_leaving_detector'],
            helical_fit_rule=fit['helical_rule'],
            swept_z_min=fit['swept_z_min'],
            swept_z_max=fit['swept_z_max'],
            z_extent_covered=fit['z_extent_covered'],
            drawing_distance=float(self.drawing_distance),
            drawing_note=self.drawing_note(),
        )
        return quantities

    def drawing_note(self):
        """This method returns a short note that names each position in the drawing that
        has no physical value.  The viewer prints the note in its text panel.
        """
        if self.kind == 'parallel':
            return ('Source and detector positions are drawing choices, at '
                    'the drawing distance.  Rows are drawn at the delta_voxel '
                    'pitch; delta_det_row and det_row_offset take no part in '
                    'a parallel projection.')
        if self.kind == 'multiaxis':
            side = '+y' if self.multiaxis_source_on_plus_y else '-y'
            return ('Source and detector positions are drawing choices, at '
                    f'the drawing distance, with the source on the {side} '
                    'side by convention: a parallel projection is the same '
                    'from both ends.')
        if self.kind == 'cone' and self.is_parallel_type:
            return ('The source-detector distance is infinite: the '
                    'magnification is one, and both positions are drawing '
                    'choices at the drawing distance.')
        return 'Every position drawn is taken from the parameters.'

    # Comparing two scenes

    def drawing_options(self):
        """This method returns the drawing options of this scene as a dict.

        These options are the constructor arguments that affect only where things are
        drawn.
        """
        return dict(
            drawing_distance_factor=self.drawing_distance_factor,
            multiaxis_source_on_plus_y=self.multiaxis_source_on_plus_y,
            rotation_axis_extension=self.rotation_axis_extension,
            rotation_arc_sweep=self.rotation_arc_sweep,
        )

    def with_parameters(self, overrides):
        """This method returns a copy of this scene with some parameter values replaced.

        A viewer uses this method to draw a second geometry that differs from the first in a
        few numbers.  An example is a vendor geometry compared with the same geometry with an
        estimated offset.

        Args:
            overrides (dict): The parameter values to replace, keyed by mbirtorch parameter
                name.  Every name must be a parameter of this geometry kind.  A misspelled
                name raises ValueError.

        Returns:
            GeometryScene: A new scene of the same kind, with the same drawing options.  An
            invalid value raises the same error as a model built with that value.
        """
        allowed = required_parameter_names(self.kind)
        unknown = [name for name in overrides if name not in allowed]
        if unknown:
            raise ValueError(f'These are not parameters of a {self.kind} '
                             f'geometry: {sorted(unknown)}.  The names this '
                             f'kind uses are {list(allowed)}.')
        copy = _model_with_parameters(self.model, overrides)
        return GeometryScene(copy, **self.drawing_options())

    def differences(self, other):
        """This method lists the values that differ between this scene and another scene.

        The method compares both the parameters and the derived quantities.  It leaves out
        the two text entries named in ``DIFFERENCE_EXCLUDED_QUANTITIES``.

        Args:
            other (GeometryScene): The scene to compare with.  It can be of another geometry
                kind.

        Returns:
            list of (str, object, object): One entry for each name whose values differ.
            Each entry holds the name, the value in this scene, and the value in the other
            scene.  The parameters come first, in the order of
            :func:`required_parameter_names`.  The derived quantities come next, in the
            order of :meth:`derived_quantities`.  A value of None means that the name is not
            a parameter of the geometry kind of that scene.  An array with one entry for each
            view gives one entry in the list.  The two values of that list entry are the two
            arrays.
        """
        names = list(required_parameter_names(self.kind))
        for name in required_parameter_names(other.kind):
            if name not in names:
                names.append(name)

        rows = []
        for name in names:
            mine = self.params.get(name)
            theirs = other.params.get(name)
            if not values_are_equal(mine, theirs):
                rows.append((name, mine, theirs))

        my_quantities = self.derived_quantities()
        their_quantities = other.derived_quantities()
        for name, value in my_quantities.items():
            if name in DIFFERENCE_EXCLUDED_QUANTITIES:
                continue
            other_value = their_quantities.get(name)
            if not values_are_equal(value, other_value):
                rows.append((name, value, other_value))
        return rows

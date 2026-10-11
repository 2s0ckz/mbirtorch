"""This module estimates the scan geometry from the sinogram.

``estimate_det_channel_offset`` estimates the detector channel offset, which sets the center of
rotation, from the sinogram alone.  ``fit_det_alignment`` fits each view to the reprojection of a
first reconstruction.  It estimates the detector offsets, the detector rotation, and the shift of
each view.  ``correct_det_alignment`` applies the corrections to each view.  The steps run in
this order: stripe removal and other corrections, ``estimate_det_channel_offset``,
``fit_det_alignment``, and ``correct_det_alignment``.
"""

import concurrent.futures
import math
import os
import warnings

import numpy as np
import torch

from .. import _sharding
from ..cone_beam import ConeBeamModel
from ..multiaxis_parallel import MultiAxisParallelModel
from ..parallel_beam import ParallelBeamModel
from . import _pipeline
from .corrections import _resample_views

__all__ = ['estimate_det_channel_offset', 'fit_det_alignment', 'correct_det_alignment',
           'align_sino_views']

# The number of views read at one time when a group of detector rows is read.
_READ_VIEW_BATCH = 64

_NUM_COARSE = 11                      # Number of points on the coarse grid of a search.
_NUM_ROWS = 16                        # Largest number of detector rows compared.

# A scan covers a full rotation if every gap between neighboring view angles is at most
# _MAX_GAP_RATIO times the median gap or at most _MAX_GAP radians.
_MAX_GAP_RATIO = 3.0
_MAX_GAP = math.radians(5.0)

# The fit of a view fails if its final correlation with its reprojection is below this value.
_ECC_MIN_CORRELATION = 0.2
# The fit of a view stops after _ECC_MAX_ITERATIONS iterations, or when the correlation
# increases by less than _ECC_EPS.
_ECC_MAX_ITERATIONS = 200
_ECC_EPS = 1e-6


# ── geometry checks ───────────────────────────────────────────────────────────────────────────────

def _geometry_kind(ct_model):
    """This function classifies a model as 'parallel', 'cone', 'multiaxis', or 'other'."""
    if isinstance(ct_model, ConeBeamModel):
        return 'cone'
    if isinstance(ct_model, MultiAxisParallelModel):
        return 'multiaxis'
    if isinstance(ct_model, ParallelBeamModel):
        return 'parallel'
    return 'other'


def _is_helical(ct_model):
    """This function returns True if the model is a cone-beam model in which some view has a
    nonzero axial shift."""
    if not isinstance(ct_model, ConeBeamModel):
        return False
    z_shifts = np.asarray(ct_model.get_params('view_params_array'))[:, 1]
    return bool(np.any(z_shifts != 0))


def _view_angles(ct_model):
    """This function returns the view angles of a parallel-beam or cone-beam model, in radians,
    as a float64 array."""
    required, _, _ = ct_model.get_all_params()
    return np.asarray(required['angles'], dtype=np.float64).ravel()


def _angular_gaps(angles):
    """This function returns the gaps in radians between neighboring distinct view angles on the
    circle."""
    wrapped = np.unique(np.mod(angles, 2 * np.pi))
    if wrapped.size < 2:
        return np.array([2 * np.pi])
    return np.diff(np.append(wrapped, wrapped[0] + 2 * np.pi))


def _unsuitable_reason(ct_model):
    """This function returns the reason why the comparison of opposite views cannot be used for
    this scan, or None if it can be used.

    The comparison requires a parallel-beam or cone-beam scan over a full rotation.
    """
    kind = _geometry_kind(ct_model)
    if kind == 'other':
        return f'{type(ct_model).__name__} has no opposite views'
    if kind == 'multiaxis':
        return 'a multiaxis parallel scan is not supported yet'
    if _is_helical(ct_model):
        return 'a helical scan has no opposite view at the same axial position'
    gaps = _angular_gaps(_view_angles(ct_model))
    if gaps.max() > max(_MAX_GAP_RATIO * np.median(gaps), _MAX_GAP):
        return (f'the views cover {math.degrees(2 * np.pi - gaps.max()):.1f} degrees with a gap of '
                f'{math.degrees(gaps.max()):.1f} degrees, so not every view has an opposite view')
    return None


# ── reading a band of rows ────────────────────────────────────────────────────────────────────────

def _read_band(sino, view_indices, row_window, device):
    """This function returns the detector rows ``row_window`` of the views ``view_indices``.

    The result is a float32 array in host memory with shape ``(len(view_indices), rows, channels)``.
    """
    row_lo, row_hi = row_window
    view_indices = np.asarray(view_indices, dtype=np.int64)
    out = np.empty((view_indices.size, row_hi - row_lo, sino.shape[2]), dtype=np.float32)
    with torch.no_grad():
        for k0 in range(0, view_indices.size, _READ_VIEW_BATCH):
            k1 = min(k0 + _READ_VIEW_BATCH, view_indices.size)
            block = _pipeline._stage_batch(sino[view_indices[k0:k1], row_lo:row_hi, :], device)
            out[k0:k1] = block.to('cpu', torch.float32).numpy()
    return out


# ── the opposite-view comparison ──────────────────────────────────────────────────────────────────

class _ConjugatePairs:
    """This class is used to pair each ray with the opposite ray, for the comparison of
    opposite views.

    The class reads a group of detector rows around the central plane from every view.  For each
    view and channel, it finds the views that hold the opposite ray.  The opposite of the ray at
    view angle ``beta`` and fan angle ``gamma`` has view angle ``beta + pi - 2 * gamma`` and fan
    angle ``-gamma``.

    Args:
        ct_model: A parallel-beam or cone-beam model.
        pairing_offset (float or None): The channel offset in ALU used to compute the fan
            angles.  If None, the class uses the current value in the model.
    """

    def __init__(self, ct_model, pairing_offset=None):
        self.kind = _geometry_kind(ct_model)
        num_views, num_det_rows, num_det_channels = (int(s) for s in ct_model.get_params('sinogram_shape'))
        delta_det_channel, det_channel_offset = ct_model.get_params(['delta_det_channel', 'det_channel_offset'])
        self.delta = float(delta_det_channel)
        self.num_views = num_views
        self.num_channels = num_det_channels
        self.model_offset = float(det_channel_offset)
        self.pairing_offset = self.model_offset if pairing_offset is None else float(pairing_offset)
        self.row_window = self._band_rows(ct_model)
        self.device = ct_model.torch_device

        # target is the view angle of the opposite ray for each view and channel:
        # beta + pi - 2 gamma(u), with u = (m - c) delta + d for channel m.
        angles = _view_angles(ct_model)
        center_channel = (self.num_channels - 1) / 2.0
        u = (np.arange(self.num_channels) - center_channel) * self.delta + self.pairing_offset
        if self.kind == 'cone':
            source_detector_dist = float(ct_model.get_params('source_detector_dist'))
            if np.isinf(source_detector_dist):
                gamma = np.zeros_like(u)
            elif ct_model.get_params('use_curved_detector'):
                gamma = u / source_detector_dist
            else:
                gamma = np.arctan(u / source_detector_dist)
        else:
            gamma = np.zeros_like(u)
        target = angles[:, None] + np.pi - 2.0 * gamma[None, :]
        low, high, self.partner_weight = self._partners(angles, target)
        self.partner_indices = np.unique(np.concatenate([low.ravel(), high.ravel()]))
        self.partner_low = np.searchsorted(self.partner_indices, low)
        self.partner_high = np.searchsorted(self.partner_indices, high)

    @staticmethod
    def _band_rows(ct_model):
        """This method returns the range ``(lo, hi)`` of detector rows around the central plane
        of the scan."""
        num_det_rows = int(ct_model.get_params('sinogram_shape')[1])
        delta_det_row, det_row_offset = ct_model.get_params(['delta_det_row', 'det_row_offset'])
        num_rows = _NUM_ROWS
        if isinstance(ct_model, ConeBeamModel):
            # The rows are limited so that opposite rays through a point stay within one row.
            min_mag, _ = ct_model.pixel_magnification_bounds()
            source_detector_dist, source_iso_dist = ct_model.get_params(
                ['source_detector_dist', 'source_iso_dist'])
            if not np.isinf(source_detector_dist):
                support_radius = source_detector_dist / min_mag - source_iso_dist
                half = max(1, math.floor(source_iso_dist / (2.0 * support_radius)))
                num_rows = min(num_rows, 2 * half + 1)
        num_rows = max(1, min(num_rows, num_det_rows))
        # The rows are centered on the row where the detector height v is zero.
        central_row = (num_det_rows - 1) / 2.0 + det_row_offset / delta_det_row
        lo = int(round(central_row - (num_rows - 1) / 2.0))
        lo = max(0, min(lo, num_det_rows - num_rows))
        return lo, lo + num_rows

    @staticmethod
    def _partners(angles, target):
        """This method finds, for each target angle, the two views on either side of it on the
        circle.

        It returns three arrays with the shape of ``target``: the first view, the second view,
        and the interpolation weight of the second view.
        """
        wrapped = np.mod(angles, 2 * np.pi)
        order = np.argsort(wrapped)
        sorted_angles = wrapped[order]
        num_views = angles.size
        t = np.mod(target, 2 * np.pi)
        position = np.searchsorted(sorted_angles, t)
        high = position % num_views
        low = (position - 1) % num_views
        gap = np.mod(sorted_angles[high] - sorted_angles[low], 2 * np.pi)
        gap = np.where(gap == 0.0, 2 * np.pi, gap)
        weight = np.mod(t - sorted_angles[low], 2 * np.pi) / gap
        return order[low], order[high], weight.astype(np.float32)

    def bands(self, sino):
        """This method returns the selected detector rows of every view and of every partner view.

        A partner view is a view that holds an opposite ray.
        """
        return (_read_band(sino, np.arange(self.num_views), self.row_window, self.device),
                _read_band(sino, self.partner_indices, self.row_window, self.device))

    def pairs(self, bands):
        """This method returns the selected rows of every view and the opposite ray of each
        element, mirrored in the channel direction.

        Args:
            bands (tuple of ndarray): The rows of the views and of the partner views, from
                :meth:`bands`.

        Returns:
            tuple of ndarray: ``(views, opposites)``.  Each array is float32 with shape
            ``(num_views, num_rows, num_channels)``.  The two arrays agree up to a shift of twice
            the channel offset.
        """
        views, partners = bands
        mirrored = partners[:, :, ::-1]
        opposites = np.empty_like(views)
        columns = np.arange(self.num_channels)
        for i in range(self.num_views):
            low = mirrored[self.partner_low[i], :, columns]      # (channels, rows)
            high = mirrored[self.partner_high[i], :, columns]
            weight = self.partner_weight[i][:, None]
            opposites[i] = ((1.0 - weight) * low + weight * high).T
        return views, opposites

def _search_minimum(score_fn, bounds, tolerance):
    """This function finds the minimum of a scalar score over ``bounds``.

    The function first evaluates the score on a coarse grid.  It then refines the minimum with a
    golden-section search to ``tolerance``.

    Returns:
        tuple: ``(best, notes)``.  ``notes`` is a list of warnings.  A warning is added if the
        coarse minimum is at an edge of the bounds, or if there is more than one local minimum.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if not hi > lo:
        raise ValueError(f'bounds must satisfy lo < hi; got {bounds}.')
    coarse = np.linspace(lo, hi, _NUM_COARSE)
    evaluated = {float(x): float(score_fn(x)) for x in coarse}
    coarse_scores = np.array([evaluated[float(x)] for x in coarse])
    notes = []
    best = int(np.argmin(coarse_scores))
    interior = coarse_scores[1:-1]
    local_minima = int(np.sum((interior < coarse_scores[:-2]) & (interior <= coarse_scores[2:])))
    if local_minima > 1:
        notes.append(f'the score curve has {local_minima} local minima on the coarse grid')
    if best in (0, _NUM_COARSE - 1):
        notes.append('the coarse minimum sits at an edge of the bounds')
    a = float(coarse[max(best - 1, 0)])
    b = float(coarse[min(best + 1, _NUM_COARSE - 1)])

    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = b - ratio * (b - a)
    x2 = a + ratio * (b - a)
    f1, f2 = float(score_fn(x1)), float(score_fn(x2))
    evaluated[x1], evaluated[x2] = f1, f2
    while b - a > tolerance:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - ratio * (b - a)
            f1 = float(score_fn(x1))
            evaluated[x1] = f1
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + ratio * (b - a)
            f2 = float(score_fn(x2))
            evaluated[x2] = f2
    return min(evaluated, key=evaluated.get), notes


# ── the channel offset ────────────────────────────────────────────────────────────────────────────

_HP_SIGMA_CHANNELS = 15.0      # Width in channels of the blur removed from each profile.
_HP_SEARCH_FRACTION = 0.25  # Half-width of the shift search, as a fraction of the detector width.
_HP_MIN_PEAK = 0.2             # A correlation peak below this value causes a warning.
_HP_MAX_SPREAD = 1.0           # A median spread of the per-view peaks above this many channels causes a warning.


def _high_pass_profiles(band):
    """This function computes one high-pass filtered profile for each view.

    The profile is the average over the detector rows.  The function subtracts the mean of the
    profile and a Gaussian blur of width ``_HP_SIGMA_CHANNELS`` channels.
    """
    from scipy.ndimage import gaussian_filter1d
    profile = band.astype(np.float64).mean(axis=1)
    profile -= profile.mean(axis=1, keepdims=True)
    return profile - gaussian_filter1d(profile, _HP_SIGMA_CHANNELS, axis=1, mode='nearest')


def _overlap_correlation(views, opposites, shifts):
    """This function computes the normalized correlation between every view and its opposite
    view at each integer shift.

    The correlation uses only the channels that overlap after the shift.

    Returns:
        tuple: ``(global_curve, per_view_curves)``.  ``global_curve`` sums the products over all
        views and then normalizes the sum.  ``per_view_curves`` holds a separately normalized
        curve for each view.
    """
    num_views, num_channels = views.shape
    num = np.empty((num_views, shifts.size))
    den_v = np.empty_like(num)
    den_o = np.empty_like(num)
    for k, s in enumerate(shifts):
        lo, hi = max(0, s), min(num_channels, num_channels + s)
        v = views[:, lo:hi]
        o = opposites[:, lo - s:hi - s]
        num[:, k] = (v * o).sum(axis=1)
        den_v[:, k] = (v * v).sum(axis=1)
        den_o[:, k] = (o * o).sum(axis=1)
    global_curve = num.sum(axis=0) / np.sqrt(den_v.sum(axis=0) * den_o.sum(axis=0))
    per_view = num / np.sqrt(np.maximum(den_v * den_o, 1e-30))
    return global_curve, per_view


def _fractional_correlation(views, opposites, shift_int, fraction):
    """This function computes the normalized correlation over all views at the shift
    ``shift_int + fraction``.

    The function shifts the views and the opposite views each by half the fraction, in opposite
    directions.  In this way the interpolation blurs both arrays equally.
    """
    from scipy.ndimage import shift as ndshift
    num_channels = views.shape[1]
    lo, hi = max(0, shift_int), min(num_channels, num_channels + shift_int)
    v = ndshift(views[:, lo:hi], (0.0, -0.5 * fraction), order=3, mode='nearest')[:, 3:-3]
    o = ndshift(opposites[:, lo - shift_int:hi - shift_int], (0.0, 0.5 * fraction), order=3, mode='nearest')[:, 3:-3]
    return float((v * o).sum() / math.sqrt((v * v).sum() * (o * o).sum()))


def _estimate_offset_by_correlation(ct_model, sino, pairing_offset=None):
    """This function computes one estimate of the channel offset.

    The function correlates the high-pass filtered profile of each view with the mirrored profile
    of its opposite view at integer shifts.  It then refines the best shift to a fraction of a
    channel.

    Returns:
        dict: The dict has these entries.  ``offset`` is the offset in ALU.  ``shift`` is the
        shift in channels, which is twice the offset.  ``peak`` is the correlation at the peak.
        ``parabola`` and ``refined`` are two sub-pixel estimates of the shift.  ``spread`` is the
        median distance in channels between the peak of each view and the refined shift.
        ``curve`` and ``shifts`` give the correlation curve over all views.  ``kind`` is the
        geometry type.
    """
    problem = _ConjugatePairs(ct_model, pairing_offset=pairing_offset)
    views, opposites = problem.pairs(problem.bands(sino))
    views, opposites = _high_pass_profiles(views), _high_pass_profiles(opposites)
    delta = problem.delta
    center = int(round(2.0 * problem.model_offset / delta))
    half = int(round(_HP_SEARCH_FRACTION * problem.num_channels))
    # The shift search range is centered on the model value.  If the peak is at an edge of the
    # range, the range moves to center on the peak, at most four times.
    limit = problem.num_channels // 2
    for _ in range(4):
        shifts = np.arange(max(center - half, -limit), min(center + half, limit) + 1)
        curve, per_view = _overlap_correlation(views, opposites, shifts)
        k = int(np.argmax(curve))
        if 1 < k < shifts.size - 2 or abs(int(shifts[k])) >= limit:
            break
        center = int(shifts[k])
    best = int(shifts[k])
    if 0 < k < shifts.size - 1:
        a, b, c = curve[k - 1], curve[k], curve[k + 1]
        parabola = best + 0.5 * (a - c) / (a - 2.0 * b + c)
    else:
        parabola = float(best)
    score = lambda f: -_fractional_correlation(views, opposites, best, f)
    fraction, _ = _search_minimum(score, (-1.0, 1.0), 1.0 / 64)
    refined = best + fraction
    per_view_peaks = shifts[np.argmax(per_view, axis=1)]
    spread = float(np.median(np.abs(per_view_peaks - refined)))
    return dict(offset=refined * delta / 2.0, shift=refined, peak=float(curve[k]), parabola=parabola,
                refined=refined, spread=spread, curve=curve, shifts=shifts, kind=problem.kind)


def estimate_det_channel_offset(ct_model, sino):
    """This function estimates the detector channel offset of a scan over a full rotation.

    In a full rotation, a view and its mirrored opposite view agree up to a shift of twice the
    channel offset.  The function finds this shift by correlating the two views near the central
    plane.  If the scan is not a parallel-beam or cone-beam scan over a full rotation, the
    function issues a warning and returns the current value in the model.

    Args:
        ct_model (TomographyModel): The model of the scan.  The function does not modify it.
        sino (ndarray or tensor): The sinogram.  The function does not modify it.

    Returns:
        float: The estimate in ALU.  Set it with ``ct_model.set_params(det_channel_offset=...)``.

    Example:
        >>> det_channel_offset = mtp.estimate_det_channel_offset(ct_model, sino)
        >>> ct_model.set_params(det_channel_offset=det_channel_offset)
    """
    reason = _unsuitable_reason(ct_model)
    if reason is not None:
        warnings.warn(f'estimate_det_channel_offset: {reason}, so det_channel_offset was left at the '
                      f'model\'s value.')
        return float(ct_model.get_params('det_channel_offset'))
    _sharding.reject_shards('estimate_det_channel_offset', sino=sino)
    result = _estimate_offset_by_correlation(ct_model, sino)
    if result['kind'] == 'cone':
        result = _estimate_offset_by_correlation(ct_model, sino, pairing_offset=result['offset'])
    if result['peak'] < _HP_MIN_PEAK:
        warnings.warn(f'estimate_det_channel_offset: the correlation peak is weak ({result["peak"]:.2f}).')
    if result['spread'] > _HP_MAX_SPREAD:
        warnings.warn(f'estimate_det_channel_offset: the per-view peaks spread {result["spread"]:.1f} '
                      'channels around the estimate.')
    if abs(result['parabola'] - result['refined']) > 0.25:
        warnings.warn(f'estimate_det_channel_offset: the two sub-pixel estimates differ by '
                      f'{abs(result["parabola"] - result["refined"]):.2f} channels.')
    return float(result['offset'])


# ── view alignment ────────────────────────────────────────────────────────────────────────────────

def _sino_high_pass_filtering(sino, sigma_row=3.0, sigma_col=15.0, subtract_view_mean=True):
    """
    This function applies a high-pass filter to each view of a 3D sinogram.

    The filter subtracts a Gaussian blur from each view.  Outside the view, the blur uses the
    nearest edge value.

    Args:
        sino (numpy array or tensor): The 3D sinogram, with shape (num_views, num_det_rows,
            num_det_channels).
        sigma_row (float, optional): The Gaussian sigma along the detector rows (vertical).  It
            should be smaller than sigma_col.  Defaults to 3.0.
        sigma_col (float, optional): The Gaussian sigma along the detector channels
            (horizontal).  Defaults to 15.0.
        subtract_view_mean (bool, optional): If True, the function subtracts the mean of each
            view.  Defaults to True.

    Returns:
        filtered_sino (numpy array): The filtered sinogram, with the same shape as the input.
    """
    import cv2

    if isinstance(sino, torch.Tensor):
        sino = sino.detach().cpu().numpy()
    sino_np = np.asarray(sino)
    if sino_np.ndim != 3:
        raise ValueError(f"Expected shape (num_views, num_det_rows, num_det_channels), got {sino_np.shape}")

    num_views, num_det_rows, num_det_channels = sino_np.shape
    filtered_sino = np.empty_like(sino_np)

    for view in range(num_views):
        single_view = sino_np[view]

        if subtract_view_mean:
            single_view = single_view - single_view.mean()

        loss_pass_estimate = cv2.GaussianBlur(
            single_view,
            ksize=(0, 0),
            sigmaX=sigma_col,
            sigmaY=sigma_row,
            borderType=cv2.BORDER_REPLICATE,
        )

        filtered_sino[view] = single_view - loss_pass_estimate

    return filtered_sino


def _fit_views(ct_model, sino, recon_direct, rotation):
    """This function fits each view to its reprojection.

    It returns ``(angles, shifts, failed)``.  ``angles`` holds the rotation of each view in
    radians.  ``shifts`` holds the (row, channel) offset of each view in pixels, for a rotation
    about the detector center.  ``failed`` is a boolean mask of the views whose fit failed.
    """
    import cv2

    recon_shape = ct_model.get_params('recon_shape')
    if tuple(recon_direct.shape) != tuple(recon_shape):
        raise ValueError("Input recon shape does not match ct_model's recon shape.")

    sino_from_recon = ct_model.forward_project(recon_direct)
    filtered_sino = _sino_high_pass_filtering(sino)
    filtered_sino_from_recon = _sino_high_pass_filtering(sino_from_recon)

    num_views, num_rows, num_cols = filtered_sino.shape
    center = np.array([(num_rows - 1) / 2.0, (num_cols - 1) / 2.0])
    motion = cv2.MOTION_EUCLIDEAN if rotation else cv2.MOTION_TRANSLATION
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, _ECC_MAX_ITERATIONS, _ECC_EPS)
    angles = np.zeros(num_views)
    shifts = np.zeros((num_views, 2))
    failed = np.zeros(num_views, dtype=bool)

    def fit_view(v):
        # Each thread writes only to index v.  OpenCV releases the Python lock during the fit.
        template = np.ascontiguousarray(filtered_sino_from_recon[v], dtype=np.float32)
        view = np.ascontiguousarray(filtered_sino[v], dtype=np.float32)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            correlation, warp = cv2.findTransformECC(template, view, warp, motion, criteria)
        except cv2.error:
            failed[v] = True
            return
        if not correlation >= _ECC_MIN_CORRELATION:
            failed[v] = True
            return
        # OpenCV rotates about the image corner.  For a rotation about the detector center, the
        # translation is t + (R - I) center.
        theta = math.atan2(warp[1, 0], warp[0, 0]) if rotation else 0.0
        t = np.array([warp[1, 2], warp[0, 2]], dtype=np.float64)        # (row, channel)
        rot = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
        angles[v] = theta
        shifts[v] = t + rot @ center - center

    with concurrent.futures.ThreadPoolExecutor(min(32, os.cpu_count() or 1)) as executor:
        list(executor.map(fit_view, range(num_views)))
    return angles, shifts, failed


def fit_det_alignment(ct_model, sino, recon_direct, rotation=False):
    """This function estimates the detector offsets and rotation by fitting each view to the
    reprojection of a first reconstruction.

    The model offsets are the medians of the fitted offsets over the views.  The data corrections
    are the difference of each view from the median, and the rotation of each view.  If the fit
    of a view fails, the function names the view in a warning.  The function leaves this view
    out of the medians.  The view gets no offset correction, and its rotation is the median
    rotation.

    Args:
        ct_model (TomographyModel): The model of the scan.  The function does not change it.
        sino (numpy array or tensor): The sinogram, with shape
            (num_views, num_det_rows, num_det_channels).
        recon_direct (numpy array or tensor): A first reconstruction of ``sino``, such as the
            output of :meth:`~mbirtorch.TomographyModel.recon_direct`.
        rotation (bool, optional): If True, the function also fits a rotation about the detector
            center.  Defaults to False.

    Returns:
        tuple: ``(model_params, view_params)``.  ``model_params`` holds ``det_channel_offset``
        and ``det_row_offset`` in ALU.  Set these values with ``set_params``.  ``view_params``
        holds arrays of length ``num_views`` for :func:`correct_det_alignment`.  These arrays are
        the differences of each view from ``det_channel_offset`` and ``det_row_offset``, in ALU.
        If ``rotation`` is True, ``view_params`` also holds ``det_rotation`` in radians.

    Example:
        >>> recon_direct = ct_model.recon_direct(sino)
        >>> model_params, view_params = mtp.fit_det_alignment(ct_model, sino, recon_direct,
        ...                                                    rotation=True)
        >>> ct_model.set_params(**model_params)
        >>> sino = mtp.correct_det_alignment(ct_model, sino, view_params)
    """
    angles, shifts, failed = _fit_views(ct_model, sino, recon_direct, rotation)
    num_views = len(angles)
    if failed.all():
        raise ValueError("No view could be aligned with its reprojection.")
    if failed.any():
        warnings.warn(f"The alignment fit failed for {int(failed.sum())} of {num_views} views, "
                      f"starting with view {int(np.flatnonzero(failed)[0])}.  Each gets no per-view "
                      "correction and the median rotation.")

    delta_row, delta_channel = (float(d) for d in ct_model.get_params(['delta_det_row', 'delta_det_channel']))
    row_offset, channel_offset = (float(d) for d in ct_model.get_params(['det_row_offset', 'det_channel_offset']))
    # If the data are s pixels higher than the reprojection, the offset is s * pitch larger.
    row_values = row_offset + shifts[:, 0] * delta_row
    channel_values = channel_offset + shifts[:, 1] * delta_channel
    good = ~failed
    model_params = {'det_channel_offset': float(np.median(channel_values[good])),
                    'det_row_offset': float(np.median(row_values[good]))}
    view_params = {'det_channel_offset': np.where(good, channel_values - model_params['det_channel_offset'], 0.0),
                   'det_row_offset': np.where(good, row_values - model_params['det_row_offset'], 0.0)}
    if rotation:
        view_params['det_rotation'] = np.where(good, angles, np.median(angles[good]))
    return model_params, view_params


def correct_det_alignment(ct_model, sino, view_params, batch_size=30, devices=None):
    """This function resamples each view of a sinogram with the corrections from
    :func:`fit_det_alignment`.

    The function rotates each view about the detector center and shifts it, in one bicubic
    resampling.  A sample outside the detector takes the nearest edge value.  If a key is missing
    from ``view_params``, the function makes no correction of that kind.

    Args:
        ct_model (TomographyModel): The model of the scan.  The function uses its pixel pitches.
        sino (numpy array or tensor): The sinogram, with shape
            (num_views, num_det_rows, num_det_channels).
        view_params (dict): Arrays of length ``num_views``, as :func:`fit_det_alignment` returns
            them.  The keys can be ``det_channel_offset`` and ``det_row_offset``, in ALU, and
            ``det_rotation``, in radians.
        batch_size (int, optional): The number of views resampled at one time.  Defaults to 30.
        devices (sequence or None, optional): The function uses the first device in this
            sequence.  If None, the function uses the first CUDA device, or the default device if
            there is no CUDA device.  Defaults to None.

    Returns:
        numpy.ndarray: The corrected sinogram, with the same shape as ``sino``.

    Example:
        >>> sino = mtp.correct_det_alignment(ct_model, sino, view_params)
        >>> # Remove only the rotation of each view:
        >>> sino = mtp.correct_det_alignment(ct_model, sino,
        ...                                  {'det_rotation': view_params['det_rotation']})
    """
    _pipeline.reject_shards('correct_det_alignment', sino=sino)
    sino = torch.as_tensor(np.asarray(sino)) if not isinstance(sino, torch.Tensor) else sino
    num_views = sino.shape[0]
    delta_row, delta_channel = (float(d) for d in ct_model.get_params(['delta_det_row', 'delta_det_channel']))

    def per_view(key, scale):
        values = view_params.get(key)
        if values is None:
            return np.zeros(num_views)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape != (num_views,):
            raise ValueError(f"view_params['{key}'] must have one value per view, got shape {values.shape}.")
        return values / scale

    angles = per_view('det_rotation', 1.0)
    offsets = np.stack([per_view('det_row_offset', delta_row), per_view('det_channel_offset', delta_channel)], axis=1)

    device = torch.device(_pipeline.permitted_devices(devices)[0])
    output = np.empty(tuple(sino.shape), dtype=np.float32 if sino.dtype == torch.float32 else np.float64)
    with torch.no_grad():
        for lo in range(0, num_views, batch_size):
            hi = min(lo + batch_size, num_views)
            batch = sino[lo:hi].to(device)
            out = _resample_views(batch, torch.as_tensor(angles[lo:hi], dtype=torch.float32, device=device),
                                  torch.as_tensor(offsets[lo:hi], dtype=torch.float32, device=device))
            output[lo:hi] = out.cpu().numpy()
    return output


def _translate_views_bilinear(sino, shifts):
    """This function shifts each view by its own (dy, dx) with bilinear interpolation.

    The output at (i, j) is the input at (i - dy, j - dx).  Samples outside the view are zero.
    """
    sino = torch.as_tensor(np.asarray(sino))
    shifts = np.asarray(shifts, dtype=np.float64)
    num_views, num_rows, num_cols = sino.shape
    dtype = sino.dtype
    grid_i, grid_j = torch.meshgrid(torch.arange(num_rows, dtype=dtype),
                                    torch.arange(num_cols, dtype=dtype), indexing='ij')
    out = torch.empty_like(sino)
    for v in range(num_views):
        dy, dx = float(shifts[v, 0]), float(shifts[v, 1])
        src_row = grid_i - dy
        src_col = grid_j - dx
        lower_row = torch.floor(src_row)
        lower_col = torch.floor(src_col)
        frac_row = src_row - lower_row
        frac_col = src_col - lower_col
        r0 = torch.clamp(lower_row.to(torch.int64), 0, num_rows - 1)
        r1 = torch.clamp(torch.ceil(src_row).to(torch.int64), 0, num_rows - 1)
        c0 = torch.clamp(lower_col.to(torch.int64), 0, num_cols - 1)
        c1 = torch.clamp(torch.ceil(src_col).to(torch.int64), 0, num_cols - 1)
        view = sino[v]
        shifted = (((1.0 - frac_row) * (1.0 - frac_col)) * view[r0, c0]
                   + ((1.0 - frac_row) * frac_col) * view[r0, c1]
                   + (frac_row * (1.0 - frac_col)) * view[r1, c0]
                   + (frac_row * frac_col) * view[r1, c1])
        in_bounds = ((src_row >= 0) & (src_row <= num_rows - 1)
                     & (src_col >= 0) & (src_col <= num_cols - 1)).to(dtype)
        out[v] = shifted * in_bounds
    return out


def align_sino_views(ct_model, sino, recon_direct):
    """
    This function shifts each view of a sinogram to align it with the forward projection of a
    first reconstruction.

    This function is deprecated.  Use :func:`fit_det_alignment` and
    :func:`correct_det_alignment` instead.  The function does not change the model.

    Args:
        ct_model (TomographyModel): The model of the scan.
        sino (numpy array or tensor): The sinogram, with shape
            (num_views, num_det_rows, num_det_channels).
        recon_direct (numpy array or tensor): A first reconstruction of ``sino``, such as the
            output of :meth:`~mbirtorch.TomographyModel.recon_direct`.

    Returns:
        numpy.ndarray: The aligned sinogram, with the same shape as ``sino``.
    """
    warnings.warn("align_sino_views is deprecated; use fit_det_alignment and correct_det_alignment.",
                  DeprecationWarning, stacklevel=2)
    model_params, view_params = fit_det_alignment(ct_model, sino, recon_direct, rotation=False)
    # The model keeps its offsets, so the correction of each view also includes the change in the model offsets.
    for key in ('det_channel_offset', 'det_row_offset'):
        view_params[key] = view_params[key] + (model_params[key] - float(ct_model.get_params(key)))
    return correct_det_alignment(ct_model, sino, view_params)

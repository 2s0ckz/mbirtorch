"""This module computes a sinogram from the object, blank, and dark scans.  It also provides
functions that downsample and crop the scans and that remove zinger pixels from a sinogram."""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
import mbirtorch as mt

from . import _pipeline
from .corrections import _rotation_kernel

__all__ = ['scan_to_sino', 'crop_view_data', 'correct_zinger_pixels']


def _reference_scan_to_host(scan):
    """This function returns a blank or dark scan as a numpy array in host memory.  The callers
    average these scans with numpy."""
    return scan.detach().cpu().numpy() if torch.is_tensor(scan) else scan


def _put_in_slice(array, flat_indices, value):
    """
    This function returns a copy of ``array`` with ``array[:, flat_indices] = value``.  The
    entries of ``flat_indices`` are flat indices into ``array.shape[1:]``.

    Args:
        array (tensor): A tensor with n+1 dimensions.
        flat_indices (array of int): The flat indices that ``ravel_multi_index`` returns for
            ``array.shape[1:]``.
        value (float or tensor): The values to copy in.  The values must broadcast to
            ``array.shape[1:]``.

    Returns:
        tensor: The modified copy of ``array``.
    """
    array_shape = array.shape
    flat_indices = torch.as_tensor(np.asarray(flat_indices), dtype=torch.int64, device=array.device)
    # Out-of-range indices are clipped to the last index.
    flat_indices = torch.clamp(flat_indices, 0, int(np.prod(array_shape[1:])) - 1)
    array = array.reshape(array_shape[0], -1).clone()
    array[:, flat_indices] = value
    array = array.reshape(array_shape)
    return array


def _box3x3_sum(x):
    """This function returns the sum over each 3x3 window of (row, channel) pixels in each view.
    The sum treats pixels outside the detector edges as zero."""
    return F.avg_pool2d(x.unsqueeze(1), kernel_size=3, stride=1, padding=1,
                        count_include_pad=True).squeeze(1) * 9.0


def _interpolate_fill_pass(sino):
    """This function replaces each NaN pixel with the mean of its finite 3x3 neighbors in
    the same view.  A NaN with no finite neighbor stays NaN."""
    is_nan = torch.isnan(sino)
    valid = (~is_nan).to(sino.dtype)
    filled = torch.where(is_nan, torch.zeros((), dtype=sino.dtype, device=sino.device), sino)
    neighbor_sum = _box3x3_sum(filled)
    neighbor_count = _box3x3_sum(valid)
    neighbor_mean = neighbor_sum / torch.clamp(neighbor_count, min=1.0)  # The clamp avoids 0/0.
    return torch.where(is_nan & (neighbor_count > 0), neighbor_mean, sino)


def _fill_nan_pixels(sino, num_passes=3):
    """This function fills each NaN pixel with the mean of its finite 3x3 neighbors in the same
    view.  It repeats this fill ``num_passes`` times.  If a pixel is still NaN at the end, the
    function sets it to 0 and issues a warning."""
    for _ in range(num_passes):
        sino = _interpolate_fill_pass(sino)
    num_residual = int(torch.sum(torch.isnan(sino)).item())
    if num_residual > 0:
        warnings.warn(f"NaN-fill: {num_residual} pixel(s) left after the fill passes (defective/zinger "
                      f"clusters wider than the fill reach); setting them to 0. Increase num_passes if "
                      f"this is unexpected.")
    return torch.nan_to_num(sino, nan=0.0)


def _zinger_fill(sino, zinger_threshold, num_passes=3):
    """This function sets zinger pixels and non-finite values to NaN.  A zinger pixel is a pixel
    with ``value < zinger_threshold``.  The function then fills the NaN pixels from their finite
    3x3 neighbors in the same view."""
    nan = torch.tensor(float('nan'), dtype=sino.dtype, device=sino.device)
    sino = torch.where(torch.isfinite(sino), sino, nan)        # This sets +inf, -inf, and NaN to NaN.
    sino = torch.where(sino < zinger_threshold, nan, sino)     # This sets the zinger pixels to NaN.
    return _fill_nan_pixels(sino, num_passes)


def _transmission_kernel(obj_batch, blank_minus_dark, dark_scan_mean, flat_indices):
    """This function computes the sinogram for one batch of views.  The sinogram is
    ``-log(|obj - dark| / |blank - dark|)``.  The function sets the defective pixels to NaN in
    every view, and it then fills all NaN pixels from their neighbors.
    """
    device = obj_batch.device
    blank_minus_dark = torch.as_tensor(np.asarray(blank_minus_dark, dtype=np.float32), device=device)
    dark_scan_mean = torch.as_tensor(np.asarray(dark_scan_mean, dtype=np.float32), device=device)
    obj_batch = torch.abs(obj_batch - dark_scan_mean)
    # A non-positive ratio becomes NaN, so the fill below removes it.
    ratio = obj_batch / blank_minus_dark
    nan = torch.tensor(float('nan'), dtype=ratio.dtype, device=device)
    sino_batch = torch.where(ratio > 0, -torch.log(torch.where(ratio > 0, ratio, torch.ones_like(ratio))), nan)
    if flat_indices is not None:
        sino_batch = _put_in_slice(sino_batch, flat_indices, float('nan'))
    sino_batch = torch.where(torch.isfinite(sino_batch), sino_batch, nan)
    return _fill_nan_pixels(sino_batch)


def compute_sino_transmission(obj_scan, blank_scan, dark_scan, defective_pixel_array=(), batch_size=90,
                              devices=None):
    """
    This function computes the sinogram ``-log(|obj - dark| / |blank - dark|)`` from the object,
    blank, and dark scans.  The function fills ``inf`` values, ``NaN`` values, and the listed
    defective pixels from their neighbors.  It returns the sinogram as a numpy array in host
    memory.

    Args:
        obj_scan (numpy array or tensor):
            A 3D object scan of shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray or tensor):
            A 3D blank scan of shape (num_blank_scans, num_det_rows, num_det_channels).
            If ``num_blank_scans > 1``, the function averages the blank scans pixel by pixel.
        dark_scan (ndarray or tensor):
            A 3D dark scan of shape (num_dark_scans, num_det_rows, num_det_channels).
            If ``num_dark_scans > 1``, the function averages the dark scans pixel by pixel.
        defective_pixel_array (ndarray or tuple, optional):
            An array of defective pixel indices, with one (row_idx, channel_idx) pair per row.
            The function treats these pixels as defective in every view.  Defaults to the empty
            tuple ``()``.  The value ``None`` is not accepted.
        batch_size (int):
            The number of views to process in each GPU batch.
        devices (sequence or None):
            The devices that process the views.  If None (the default), the function uses all
            visible CUDA devices, up to ``MBIRTORCH_NUM_DEVICES`` if that variable is set.  If
            there are no CUDA devices, it uses the default device.

    Returns:
        numpy.ndarray:
            The computed sinogram, with shape (num_views, num_det_rows, num_det_channels).

    Raises:
        TypeError: If any of the scans is a ``Shards`` container.
    """
    _pipeline.reject_shards('compute_sino_transmission', obj_scan=obj_scan, blank_scan=blank_scan,
                           dark_scan=dark_scan, defective_pixel_array=defective_pixel_array)
    blank_scan = _reference_scan_to_host(blank_scan)
    dark_scan = _reference_scan_to_host(dark_scan)

    blank_scan_mean = np.mean(blank_scan, axis=0, keepdims=True)
    dark_scan_mean = np.mean(dark_scan, axis=0, keepdims=True)
    blank_minus_dark = np.abs(blank_scan_mean - dark_scan_mean)

    if len(defective_pixel_array) > 0:
        defective_pixel_array = np.asarray(defective_pixel_array)
        flat_indices = np.ravel_multi_index(defective_pixel_array.T, obj_scan.shape[1:]).astype(np.int64)
    else:
        defective_pixel_array = ()
        flat_indices = None

    sino = _pipeline.map_view_batches(
        obj_scan,
        lambda obj_batch: _transmission_kernel(obj_batch, blank_minus_dark, dark_scan_mean,
                                               flat_indices),
        batch_size, devices=_pipeline.permitted_devices(devices))
    print("Sinogram computation complete.")
    return sino


def _downsample_obj_kernel(obj_batch, flat_indices, new_size1, new_size2, block_shape):
    """This function downsamples one batch of object scan views by averaging blocks of pixels.
    It leaves the defective pixels out of the averages."""
    if flat_indices is not None:
        obj_batch = _put_in_slice(obj_batch, flat_indices, float('nan'))
    obj_batch = obj_batch[:, 0:new_size1, 0:new_size2]
    obj_batch = obj_batch.reshape((obj_batch.shape[0],) + block_shape)
    return torch.nanmean(obj_batch, dim=(2, 4))


def _downsample_blank_dark(blank_scan, dark_scan, downsample_factor, defective_pixel_array=()):
    """This function downsamples the blank and dark scans in host memory by averaging blocks of
    pixels.  It also computes the block parameters that :func:`_downsample_obj_kernel` needs.

    Returns:
        (blank_scan, dark_scan, defective_pixel_array, obj_flat_indices, new_size1, new_size2,
        block_shape)
    """
    # The defective pixels are set to NaN so that nanmean leaves them out.  The blank and dark
    # scans may have different numbers of views.
    if len(defective_pixel_array) > 0:
        flat_indices = np.ravel_multi_index(defective_pixel_array.T, blank_scan.shape[1:]).astype(np.int64)
        for i in range(blank_scan.shape[0]):
            np.put(blank_scan[i], flat_indices, np.nan)
        for i in range(dark_scan.shape[0]):
            np.put(dark_scan[i], flat_indices, np.nan)
    else:
        flat_indices = None

    # The scans are cropped to a size divisible by downsample_factor.
    new_size1 = downsample_factor[0] * (blank_scan.shape[1] // downsample_factor[0])
    new_size2 = downsample_factor[1] * (blank_scan.shape[2] // downsample_factor[1])

    blank_scan = blank_scan[:, 0:new_size1, 0:new_size2]
    dark_scan = dark_scan[:, 0:new_size1, 0:new_size2]

    block_shape = (blank_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                   blank_scan.shape[2] // downsample_factor[1], downsample_factor[1])

    blank_scan = np.stack([
        np.nanmean(scan.reshape(block_shape), axis=(1, 3))
        for scan in blank_scan
    ], axis=0)

    dark_scan = np.stack([
        np.nanmean(scan.reshape(block_shape), axis=(1, 3))
        for scan in dark_scan
    ], axis=0)

    # A downsampled pixel is defective when every pixel in its block was defective.
    nan_mask = np.isnan(blank_scan).any(axis=0)
    defective_pixel_array = np.argwhere(nan_mask)
    if len(defective_pixel_array) == 0:
        defective_pixel_array = ()

    return blank_scan, dark_scan, defective_pixel_array, flat_indices, new_size1, new_size2, block_shape


def downsample_view_data(obj_scan, blank_scan, dark_scan, downsample_factor, defective_pixel_array=(), batch_size=90,
                         devices=None):
    """
    This function downsamples the object, blank, and dark scans in the detector plane.  It also
    converts the defective pixel array to the new pixel grid.  The scans may be numpy arrays or
    tensors.  The function returns numpy arrays in host memory.

    Args:
        obj_scan (ndarray): The object scan, with shape (num_views, num_det_rows,
            num_det_channels).
        blank_scan (ndarray): The blank scans, with shape (num_blank_views, num_det_rows,
            num_det_channels).
        dark_scan (ndarray): The dark scans, with shape (num_dark_views, num_det_rows,
            num_det_channels).
        downsample_factor (tuple of int): Two integers, each at least 1.
        defective_pixel_array (ndarray): The defective pixel coordinates, with shape
            (num_defective_pixels, 2).
        batch_size (int): The number of views in one batch.  A smaller batch uses less memory.
        devices (sequence or None): The devices that process the views.  If None (the default),
            the function uses all visible CUDA devices, up to ``MBIRTORCH_NUM_DEVICES`` if that
            variable is set.  If there are no CUDA devices, it uses the default device.

    Returns:
        tuple:
        - **obj_scan** (numpy.ndarray): Shape (num_views, new_rows, new_cols).
        - **blank_scan** (numpy.ndarray): Shape (num_blank_views, new_rows, new_cols).
        - **dark_scan** (numpy.ndarray): Shape (num_dark_views, new_rows, new_cols).
        - **defective_pixel_array** (numpy.ndarray): The new coordinates, shape (N_def, 2).

    Raises:
        TypeError: If any of the scans is a ``Shards`` container.
    """
    _pipeline.reject_shards('downsample_view_data', obj_scan=obj_scan, blank_scan=blank_scan,
                           dark_scan=dark_scan, defective_pixel_array=defective_pixel_array)
    blank_scan = _reference_scan_to_host(blank_scan)
    dark_scan = _reference_scan_to_host(dark_scan)

    assert len(downsample_factor) == 2, 'factor({}) needs to be of len 2'.format(downsample_factor)
    assert (downsample_factor[0] >= 1 and downsample_factor[1] >= 1), 'factor({}) along each dimension should be greater or equal to 1'.format(downsample_factor)

    blank_scan, dark_scan, defective_pixel_array, obj_flat_indices, new_size1, new_size2, block_shape = \
        _downsample_blank_dark(blank_scan, dark_scan, downsample_factor, defective_pixel_array)

    obj_scan = _pipeline.map_view_batches(
        obj_scan,
        lambda b: _downsample_obj_kernel(b, obj_flat_indices, new_size1, new_size2, block_shape),
        batch_size, devices=_pipeline.permitted_devices(devices))

    return obj_scan, blank_scan, dark_scan, defective_pixel_array


def scan_to_sino(obj_scan, blank_scan, dark_scan, defective_pixel_array=(),
                 downsample_factor=(1, 1), det_rotation=0.0,
                 batch_size=90, devices=None):
    """
    This function computes the sinogram from object, blank, and dark scans.  It downsamples
    the views, computes ``-log(|obj - dark| / |blank - dark|)``, fills defective and
    non-positive pixels, and removes the detector rotation.  The result is a numpy array.

    Args:
        obj_scan (numpy array or tensor): The object scan, with shape (num_views, num_det_rows,
            num_det_channels).
        blank_scan (numpy array or tensor): The blank scans, with shape (num_blank,
            num_det_rows, num_det_channels).
        dark_scan (numpy array or tensor): The dark scans, with shape (num_dark, num_det_rows,
            num_det_channels).
        defective_pixel_array (numpy.ndarray or tuple, optional): The (row, col) coordinates of
            the defective pixels, shape (num_defective, 2), or () for none.  Defaults to ().
        downsample_factor (tuple[int, int], optional): The downsampling factors along the
            detector rows and channels.  Defaults to (1, 1).
        det_rotation (float, optional): The detector rotation in radians.  Defaults to 0.0.
        batch_size (int, optional): The number of views processed at a time.  Defaults to 90.
        devices (sequence or None, optional): The devices that process the views.  If None, the
            function uses all visible CUDA devices, up to ``MBIRTORCH_NUM_DEVICES`` if that
            variable is set.  If there are no CUDA devices, it uses the default device.  Defaults
            to None.

    Returns:
        numpy.ndarray: The sinogram, shape (num_views, num_det_rows, num_det_channels) after
        downsampling.
    """
    _pipeline.reject_shards('scan_to_sino', obj_scan=obj_scan, blank_scan=blank_scan,
                           dark_scan=dark_scan, defective_pixel_array=defective_pixel_array)
    blank_scan = _reference_scan_to_host(blank_scan)
    dark_scan = _reference_scan_to_host(dark_scan)

    devices = _pipeline.permitted_devices(devices)
    obj_flat_indices = new_size1 = new_size2 = block_shape = None
    do_downsample = downsample_factor[0] * downsample_factor[1] > 1
    if do_downsample:
        blank_scan, dark_scan, defective_pixel_array, obj_flat_indices, new_size1, new_size2, block_shape = \
            _downsample_blank_dark(blank_scan, dark_scan, downsample_factor, defective_pixel_array)

    blank_scan_mean = np.mean(blank_scan, axis=0, keepdims=True)
    dark_scan_mean = np.mean(dark_scan, axis=0, keepdims=True)
    blank_minus_dark = np.abs(blank_scan_mean - dark_scan_mean)

    trans_det_shape = blank_scan.shape[1:]
    if len(defective_pixel_array) > 0:
        defective_pixel_array = np.asarray(defective_pixel_array)
        trans_flat_indices = np.ravel_multi_index(defective_pixel_array.T, trans_det_shape).astype(np.int64)
    else:
        defective_pixel_array = ()
        trans_flat_indices = None

    do_rotation = det_rotation != 0.0

    # For each batch of views, one function downsamples, computes the sinogram, and rotates.
    def fused_kernel(obj_batch):
        if do_downsample:
            obj_batch = _downsample_obj_kernel(obj_batch, obj_flat_indices, new_size1, new_size2, block_shape)
        sino_batch = _transmission_kernel(obj_batch, blank_minus_dark, dark_scan_mean,
                                          trans_flat_indices)
        if do_rotation:
            sino_batch = _rotation_kernel(sino_batch, det_rotation)
        return sino_batch

    sino = _pipeline.map_view_batches(obj_scan, fused_kernel, batch_size, devices=devices)
    print("Sinogram computation complete.")
    return sino


def crop_view_data(obj_scan, blank_scan, dark_scan, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, defective_pixel_array=()):
    """
    This function crops the object, blank, and dark scans, and it updates the defective pixel
    list.  The function crops the same number of pixels from the left and from the right.

    Args:
        obj_scan (numpy.ndarray): The object scan, with shape (num_views, num_det_rows,
            num_det_channels).
        blank_scan (numpy.ndarray): The blank scans, with shape (num_blank, num_det_rows,
            num_det_channels).
        dark_scan (numpy.ndarray): The dark scans, with shape (num_dark, num_det_rows,
            num_det_channels).
        crop_pixels_sides (int, optional): The number of pixels to crop from each side.
            Defaults to 0.
        crop_pixels_top (int, optional): The number of pixels to crop from the top, which has
            the low row indices.  Defaults to 0.
        crop_pixels_bottom (int, optional): The number of pixels to crop from the bottom, which
            has the high row indices.  Defaults to 0.
        defective_pixel_array (numpy.ndarray or tuple, optional): The (row, col) coordinates of
            the defective pixels, shape (num_defective, 2), or () for none.  Defaults to ().

    Returns:
        tuple: The cropped ``(obj_scan, blank_scan, dark_scan, defective_pixel_array)``.

    Raises:
        AssertionError: If a crop is negative, if the top and bottom crops together reach the
            detector height, or if twice the side crop reaches the detector width.
    """
    assert (0 <= crop_pixels_sides < obj_scan.shape[2] // 2 and
            0 <= crop_pixels_top and 0 <= crop_pixels_bottom and crop_pixels_top + crop_pixels_bottom < obj_scan.shape[1]), \
        ('crop_pixels should be nonnegative integers so that crop_pixels_top + crop_pixels_bottom < view height and'
         ' 2*crop_pixels_sides < view width')

    Nr_lo = crop_pixels_top
    Nr_hi = obj_scan.shape[1] - crop_pixels_bottom

    Nc_lo = crop_pixels_sides
    Nc_hi = obj_scan.shape[2] - crop_pixels_sides

    obj_scan = obj_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]
    blank_scan = blank_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]
    dark_scan = dark_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]

    if len(defective_pixel_array) > 0:
        in_bounds = (defective_pixel_array[:, 0] >= Nr_lo) & (defective_pixel_array[:, 0] < Nr_hi) & \
                    (defective_pixel_array[:, 1] >= Nc_lo) & (defective_pixel_array[:, 1] < Nc_hi)
        defective_pixel_array = defective_pixel_array[in_bounds]
        defective_pixel_array -= np.array([Nr_lo, Nc_lo]).reshape(1, 2)

    return obj_scan, blank_scan, dark_scan, defective_pixel_array


def _zinger_threshold(sino, zinger_pixel_ratio, max_views_to_use=20):
    """This function returns the zinger threshold ``-zinger_pixel_ratio * RMS(sino)`` as a
    negative float.  It computes the RMS over the sinogram entries above the background, and it
    uses at most ``max_views_to_use`` views."""
    sino_sub = mt.TomographyModel.subsample_views(sino, max_views_to_use)
    sino_indicator = np.asarray(mt.TomographyModel._get_sino_indicator(sino_sub))
    typical_sino_value = float(np.average(np.asarray(sino_sub) ** 2, None, sino_indicator) ** 0.5)
    return -zinger_pixel_ratio * typical_sino_value


def correct_zinger_pixels(sino, zinger_pixel_ratio=0.1, num_passes=3, batch_size=90, devices=None,
                          max_views_to_use=20):
    """
    This function replaces the zinger pixels of a sinogram with the mean of their
    neighbors.

    A zinger is a pixel whose value is below ``-zinger_pixel_ratio`` times the RMS value of the
    sinogram.  For this reason, the background offset must be removed before this function is
    called.  The function :func:`correct_background_offset` removes the offset.  This function fills
    the zingers and the non-finite pixels from their finite 3x3 neighbors, and it repeats the fill
    ``num_passes`` times.  If a pixel is still not filled, the function sets it to 0 and issues a
    warning.

    Args:
        sino (numpy array or tensor): The sinogram with its background offset removed, with
            shape (num_views, num_det_rows, num_det_channels).
        zinger_pixel_ratio (float, optional): The ratio that sets the zinger threshold.
            Defaults to 0.1.
        num_passes (int, optional): The number of fill passes.  Defaults to 3.
        batch_size (int, optional): The number of views processed at a time.  Defaults to 90.
        devices (sequence or None, optional): The devices that process the views.  If None, the
            function uses all visible CUDA devices, up to ``MBIRTORCH_NUM_DEVICES`` if that
            variable is set.  If there are no CUDA devices, it uses the default device.  Defaults
            to None.
        max_views_to_use (int, optional): The number of views sampled to estimate the RMS
            value.  Defaults to 20.

    Returns:
        numpy.ndarray: The corrected sinogram, with the same shape as ``sino``.
    """
    _pipeline.reject_shards('correct_zinger_pixels', sino=sino)

    # The threshold is computed once, so the result does not depend on the number of devices.
    zinger_threshold = _zinger_threshold(sino, zinger_pixel_ratio, max_views_to_use)
    kernel = lambda b: _zinger_fill(b, zinger_threshold, num_passes)
    return _pipeline.map_view_batches(sino, kernel, batch_size,
                                     devices=_pipeline.permitted_devices(devices))

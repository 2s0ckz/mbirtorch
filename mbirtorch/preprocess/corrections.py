"""This module applies corrections to a sinogram before reconstruction."""

import math
import warnings

import numpy as np
import torch

from . import _pipeline

__all__ = ['correct_background_offset', 'correct_det_rotation', 'remove_all_stripe', 'remove_stripe_fw',
           'remove_sino_offset']


def _resample_views(sino_batch, angles, sampling_offsets):
    """This function resamples each view of a batch.  It rotates each view about the detector
    center by the angle of that view.  It then shifts the view by the sampling offset of that
    view.  It uses bicubic interpolation and repeats the edge values outside the detector.

    Args:
        sino_batch (tensor): The views, with shape (num_views, num_det_rows, num_det_channels).
        angles (tensor): The rotation angles in radians, with shape (num_views,).  The tensor
            must be on the device of ``sino_batch``.
        sampling_offsets (tensor): The offsets in pixels as (row, channel), with shape
            (num_views, 2).  The tensor must be on the device of ``sino_batch``.
    """
    num_views, num_rows, num_cols = sino_batch.shape
    device, dtype = sino_batch.device, sino_batch.dtype
    center_row, center_col = (num_rows - 1) / 2.0, (num_cols - 1) / 2.0
    grid_i, grid_j = torch.meshgrid(torch.arange(num_rows, dtype=dtype, device=device),
                                    torch.arange(num_cols, dtype=dtype, device=device), indexing='ij')
    cos_a = torch.cos(angles.to(dtype))[:, None, None]
    sin_a = torch.sin(angles.to(dtype))[:, None, None]
    rel_i, rel_j = grid_i[None] - center_row, grid_j[None] - center_col
    src_row = cos_a * rel_i + sin_a * rel_j + center_row + sampling_offsets[:, 0, None, None].to(dtype)
    src_col = -sin_a * rel_i + cos_a * rel_j + center_col + sampling_offsets[:, 1, None, None].to(dtype)
    # grid_sample takes (x, y) in [-1, 1].  With align_corners=True, -1 and 1 map to the edge pixels.
    grid = torch.stack([2.0 * src_col / max(num_cols - 1, 1) - 1.0,
                        2.0 * src_row / max(num_rows - 1, 1) - 1.0], dim=-1)
    out = torch.nn.functional.grid_sample(sino_batch[:, None].to(torch.float32), grid.to(torch.float32),
                                          mode='bicubic', padding_mode='border', align_corners=True)
    return out[:, 0].to(dtype)


def _rotation_kernel(sino_batch, det_rotation, center=None):
    """This function rotates each view in a batch by ``det_rotation`` radians.  It uses
    bicubic interpolation and repeats the edge values outside the detector.

    Args:
        sino_batch (tensor): The views, with shape (num_views, num_det_rows, num_det_channels).
        det_rotation (float): The rotation angle in radians.
        center (tuple of float or None): The ``(row, channel)`` point of the rotation, in the
            index coordinates of ``sino_batch``.  None means the center of the array.  Defaults
            to None.
    """
    num_views, num_rows, num_cols = sino_batch.shape
    device = sino_batch.device
    angles = torch.full((num_views,), float(det_rotation), dtype=torch.float32, device=device)
    # A rotation about ``center`` equals the rotation about the array center plus the sampling
    # offset (R - I) (array center - center).
    offsets = torch.zeros((num_views, 2), dtype=torch.float32, device=device)
    if center is not None:
        cos_a, sin_a = math.cos(det_rotation), math.sin(det_rotation)
        d_row = (num_rows - 1) / 2.0 - float(center[0])
        d_col = (num_cols - 1) / 2.0 - float(center[1])
        offsets[:, 0] = (cos_a - 1.0) * d_row + sin_a * d_col
        offsets[:, 1] = -sin_a * d_row + (cos_a - 1.0) * d_col
    return _resample_views(sino_batch, angles, offsets)


def correct_det_rotation(sino, det_rotation=0.0, batch_size=30, devices=None):
    """
    This function rotates each view of a sinogram about the center of the view to remove a
    detector rotation.  It uses bicubic interpolation and repeats the edge values outside the detector.

    Args:
        sino (numpy array or tensor): The sinogram, with shape (num_views, num_det_rows,
            num_det_channels).
        det_rotation (float, optional): The angle between the rotation axis and the detector
            columns, in radians.  Defaults to 0.0.
        batch_size (int, optional): The number of views processed at a time.  Defaults to 30.
        devices (sequence or None, optional): The devices to divide the views among.  None
            means all visible CUDA devices, up to the number set by ``MBIRTORCH_NUM_DEVICES``.
            If there is no CUDA device, None means the default device.  Defaults to None.

    Returns:
        numpy.ndarray: The corrected sinogram, with the shape of ``sino``.
    """
    _pipeline.reject_shards('correct_det_rotation', sino=sino)

    return _pipeline.map_view_batches(sino, lambda b: _rotation_kernel(b, det_rotation), batch_size,
                                     devices=_pipeline.permitted_devices(devices))


def correct_background_offset(sino, edge_width=9, option='global'):
    """
    This function estimates the background offset of a sinogram from the edges of the views.
    It then subtracts the offset from the sinogram.

    The offset of a view is the median of three values.  These values are the medians of the
    left, right, and top edge strips of the view.  With ``option='global'``, the function
    subtracts the 10th percentile of the view offsets from every view.  With
    ``option='per_view'``, the function subtracts the offset of each view from that view.

    Args:
        sino (numpy.ndarray): The sinogram, with shape (num_views, num_det_rows,
            num_det_channels).
        edge_width (int, optional): The width of the edge strips in pixels.  The function
            raises a value below 1 to 1.  Defaults to 9.
        option (str or None, optional): 'global', 'per_view', or None.  If None, the function
            returns ``sino`` unchanged.  Defaults to 'global'.

    Returns:
        numpy.ndarray: The corrected sinogram, with the shape of ``sino``.
    """

    if option is None:
        return sino

    if edge_width < 1:
        edge_width = 1
        warnings.warn("edge_width of background regions should be >= 1! Setting edge_width to 1.")

    num_views, _, num_det_channels = sino.shape

    sino_edge_left  = sino[:, :, :edge_width].reshape(num_views, -1)
    sino_edge_right = sino[:, :, num_det_channels-edge_width:].reshape(num_views, -1)
    sino_edge_top   = sino[:, :edge_width, :].reshape(num_views, -1)

    med_left  = np.median(sino_edge_left, axis=1)
    med_right = np.median(sino_edge_right, axis=1)
    med_top   = np.median(sino_edge_top, axis=1)

    edge_medians = np.stack([med_left, med_right, med_top], axis=1)
    offset = np.median(edge_medians, axis=1)   # (num_views,)

    if option == "global":
        percentile = 10
        offset = np.percentile(offset, percentile)
        sino_corrected = sino - offset

    elif option == "per_view":
        sino_corrected = sino - offset[:, None, None]

    else:
        raise ValueError("option must be None, 'global' or 'per_view'")

    return sino_corrected

def _generate_column_index_matrix(num_rows, num_cols):
    """
    This function creates the 2D array of indexes that the sorting method uses.  Each row of
    the array holds 0, 1, ..., num_cols - 1.
    The code is adapted from Tomopy (https://github.com/tomopy/tomopy.git).  It follows Vo N,
    Atwood RC, Drakopoulos M, "Superior techniques for eliminating ring artifacts in x-ray
    micro-tomography", Optics Express 26(22):28396-28412, 2018.

    Args:
        num_rows (int): The number of rows in the array.  The caller passes the number of
            detector channels.
        num_cols (int): The number of columns in the array.  The caller passes the number of
            views.

    Returns:
        index_matrix (ndarray): The 2D array of indexes, with shape (num_rows, num_cols).
    """
    list_index = np.arange(0.0, num_cols, 1.0, dtype=np.float32)
    index_matrix = np.tile(list_index, (num_rows, 1))
    return index_matrix


def _remove_small_stripes_sorting(sino, filter_size, index_matrix):
    """
    This function removes small and medium stripes, both partial and full, by the sorting method.
    The code is adapted from Tomopy (https://github.com/tomopy/tomopy.git).  It follows Vo N,
    Atwood RC, Drakopoulos M, "Superior techniques for eliminating ring artifacts in x-ray
    micro-tomography", Optics Express 26(22):28396-28412, 2018.

    Args:
        sino (ndarray): A 2D slice of the sinogram, with shape (num_views, num_det_channels).
        filter_size (int): The window size of the median filter.
        index_matrix (ndarray): The 2D array of indexes for the sorting method.

    Return:
        corrected_sino (ndarray): The corrected 2D slice of the sinogram.
    """

    from scipy.ndimage import median_filter
    sino = np.transpose(sino)
    stacked_matrix = np.stack([index_matrix, sino], axis=2)
    sorted_indices = np.argsort(stacked_matrix[:, :, 1], axis=1, kind='stable')
    sorted_indices_expanded = sorted_indices[:, :, None]
    sorted_stacked_matrix = np.take_along_axis(stacked_matrix, sorted_indices_expanded, axis=1)

    sorted_stacked_matrix[:, :, 1] = median_filter(sorted_stacked_matrix[:, :, 1], (filter_size, 1))

    sorted_indices = np.argsort(sorted_stacked_matrix[:, :, 0], axis=1, kind='stable')
    sorted_indices_expanded = sorted_indices[:, :, None]
    sort_back_matrix = np.take_along_axis(sorted_stacked_matrix, sorted_indices_expanded, axis=1)
    corrected_sino = sort_back_matrix[:, :, 1]
    corrected_sino = np.transpose(corrected_sino)

    return corrected_sino

def _detect_stripe(list_data, snr):
    """
    This function locates stripes.  It separates the values with very large positive or
    negative defects from the normal values.
    The code is adapted from Tomopy (https://github.com/tomopy/tomopy.git).  It follows Vo N,
    Atwood RC, Drakopoulos M, "Superior techniques for eliminating ring artifacts in x-ray
    micro-tomography", Optics Express 26(22):28396-28412, 2018.

    Args:
        list_data (ndarray): A normalized 1D array.
        snr (float): The ratio between the defective value and the background value.  It is
            typically 3.0 or above.

    Returns:
        list_mask (ndarray): A binary array with the shape of ``list_data``.  The value 1 marks
            a detected stripe.
    """

    num_data = list_data.shape[0]

    sorted_list = np.sort(list_data)[::-1]
    x_list = np.arange(0, num_data, 1.0, dtype=np.float32)

    # The fit ignores the top and bottom quarters of the sorted values.
    num_data_drop = np.int16(0.25 * num_data)
    (_slope, _intercept) = np.polyfit(x_list[num_data_drop:-num_data_drop - 1], sorted_list[num_data_drop:-num_data_drop - 1], 1)
    fitted_value_last_index = _intercept + _slope * x_list[-1]
    noise_level = np.abs(fitted_value_last_index - _intercept)
    noise_level = np.clip(noise_level, 1e-6, None)
    val1 = np.abs(sorted_list[0] - _intercept) / noise_level
    val2 = np.abs(sorted_list[-1] - fitted_value_last_index) / noise_level

    list_mask = np.zeros_like(list_data)
    if (val1 >= snr):
        upper_threshold = _intercept + noise_level * snr * 0.5
        list_mask = np.where(list_data > upper_threshold, np.float32(1.0), list_mask)
    if (val2 >= snr):
        lower_threshold = fitted_value_last_index - noise_level * snr * 0.5
        list_mask = np.where(list_data <= lower_threshold, np.float32(1.0), list_mask)

    return list_mask


def _remove_large_stripes_sorting(sino, snr, filter_size, index_matrix, drop_ratio=0.1):
    """
    This function removes large stripes, both partial and full, by the sorting method.
    The code is adapted from Tomopy (https://github.com/tomopy/tomopy.git).  It follows Vo N,
    Atwood RC, Drakopoulos M, "Superior techniques for eliminating ring artifacts in x-ray
    micro-tomography", Optics Express 26(22):28396-28412, 2018.

    Args:
        sino (ndarray): A 2D slice of the sinogram, with shape (num_views, num_det_channels).
        snr (float): The ratio between the defective value and the background value.
        filter_size (int): The window size of the median filter.
        index_matrix (ndarray): The 2D array of indexes for the sorting method.
        drop_ratio (float, optional): The fraction of the sorted values in each channel that
            the stripe detection ignores.  Half the fraction comes from the top and half from
            the bottom.  Defaults to 0.1.

    Returns:
        sino (ndarray): The corrected 2D slice of the sinogram.
    """

    from scipy.ndimage import median_filter, binary_dilation
    drop_ratio = np.clip(drop_ratio, 0.0, 0.8)
    (num_rows, num_cols) = sino.shape
    num_rows_drop = np.int16(0.5 * drop_ratio * num_rows)

    sorted_sino = np.sort(sino, axis=0)
    smoothed_sino = median_filter(sorted_sino, (1, filter_size))

    raw_list = np.mean(sorted_sino[num_rows_drop:num_rows - num_rows_drop], axis=0)
    smoothed_list = np.mean(smoothed_sino[num_rows_drop:num_rows - num_rows_drop], axis=0)
    normalized_list = np.where(
        smoothed_list != 0,
        np.divide(raw_list, smoothed_list, out=np.ones_like(raw_list), where=smoothed_list != 0),
        np.ones_like(raw_list)
    )

    list_mask = _detect_stripe(normalized_list, snr)
    list_mask = binary_dilation(list_mask, iterations=1).astype(list_mask.dtype)
    normalized_factor = np.tile(normalized_list, (num_rows, 1))

    sino = sino / normalized_factor

    sino_T = np.transpose(sino)
    stacked_matrix = np.stack([index_matrix, sino_T], axis=2)

    sorted_indices = np.argsort(stacked_matrix[:, :, 1], axis=1, kind='stable')
    sorted_indices_expanded = sorted_indices[:, :, None]
    sorted_matrix = np.take_along_axis(stacked_matrix, sorted_indices_expanded, axis=1)

    sorted_matrix[:, :, 1] = np.transpose(smoothed_sino)

    sorted_indices = np.argsort(sorted_matrix[:, :, 0], axis=1, kind='stable')
    sorted_indices_expanded = sorted_indices[:, :, None]
    sort_back_matrix = np.take_along_axis(sorted_matrix, sorted_indices_expanded, axis=1)

    corrected_sino = np.transpose(sort_back_matrix[:, :, 1])

    list_x_miss = np.where(list_mask > 0.0)[0]

    sino[:, list_x_miss] = corrected_sino[:, list_x_miss]
    return sino


def _remove_dead_fluctuating_stripes_interpolation(sino, snr, filter_size, index_matrix):
    """
    This function removes unresponsive and fluctuating stripes by interpolation.  The sorting
    method does not work for these stripes.
    The code is adapted from Tomopy (https://github.com/tomopy/tomopy.git).  It follows Vo N,
    Atwood RC, Drakopoulos M, "Superior techniques for eliminating ring artifacts in x-ray
    micro-tomography", Optics Express 26(22):28396-28412, 2018.

    Args:
        sino (ndarray): A 2D slice of the sinogram, with shape (num_views, num_det_channels).
        snr (float): The ratio between the defective value and the background value.
        filter_size (int): The window size of the median filter.
        index_matrix (ndarray): The 2D array of indexes for the sorting method.

    Returns:
        sino (ndarray): The corrected 2D slice of the sinogram.
    """
    from scipy import interpolate
    from scipy.ndimage import uniform_filter1d, median_filter, binary_dilation
    num_rows = sino.shape[0]

    # A large difference marks a fluctuating stripe.  A small difference marks an unresponsive stripe.
    smoothed_sino = uniform_filter1d(sino, 10, axis=0)
    difference_list = np.sum(np.abs(sino - smoothed_sino), axis=0)

    difference_list_filtered = median_filter(difference_list, size=filter_size)
    normalized_list = np.where(
        difference_list_filtered != 0,
        np.divide(difference_list, difference_list_filtered,
                  out=np.ones_like(difference_list_filtered), where=difference_list_filtered != 0),
        np.ones_like(difference_list_filtered)
    )

    list_mask = _detect_stripe(normalized_list, snr)
    list_mask = binary_dilation(list_mask, iterations=1).astype(list_mask.dtype)
    list_mask[0:2] = 0.0
    list_mask[-2:] = 0.0

    x_list = np.where(list_mask < 1.0)[0]
    y_list = np.arange(num_rows)
    z_matrix = sino[:, x_list]
    fit_function = interpolate.RectBivariateSpline(y_list, x_list, z_matrix, kx=1, ky=1)

    list_x_miss = np.where(list_mask > 0.0)[0]
    if len(list_x_miss) > 0:
        matrix_x_miss, matrix_y = np.meshgrid(list_x_miss, y_list)
        estimate_output = fit_function.ev(np.ravel(matrix_y), np.ravel(matrix_x_miss))
        sino = np.array(sino)
        sino[:, list_x_miss] = np.reshape(estimate_output, matrix_x_miss.shape).astype(sino.dtype)

    corrected_sino = _remove_large_stripes_sorting(sino, snr, filter_size, index_matrix)

    return corrected_sino

def remove_all_stripe(sino, snr=3, large_filter_size=61, small_filter_size=21):
    """
    This function removes stripes of all sizes from a sinogram.  It uses the method of Vo,
    Atwood, and Drakopoulos, Optics Express 26(22), 2018.  It matches tomopy's
    ``remove_all_stripe``.

    Args:
        sino (numpy.ndarray): The sinogram, with shape (num_views, num_det_rows,
            num_det_channels).
        snr (float, optional): The signal to noise ratio used to detect stripes.  Defaults to 3.
        large_filter_size (int, optional): The median filter window for the large stripes.
            Defaults to 61.
        small_filter_size (int, optional): The median filter window for the small stripes.
            Defaults to 21.

    Returns:
        numpy.ndarray: The corrected sinogram, with the shape of ``sino``.
    """
    from concurrent.futures import ThreadPoolExecutor
    sino = np.asarray(sino, dtype=np.float32)
    index_matrix = _generate_column_index_matrix(sino.shape[2], sino.shape[0])

    result = np.zeros_like(sino)

    def process_slice(m):
        sino_slice = sino[:, m, :]
        sino_slice = _remove_dead_fluctuating_stripes_interpolation(sino_slice, snr, large_filter_size, index_matrix)
        sino_slice = _remove_small_stripes_sorting(sino_slice, small_filter_size, index_matrix)
        return sino_slice

    with ThreadPoolExecutor() as executor:
        processed_slices = list(executor.map(process_slice, range(sino.shape[1])))

    for m, processed_slice in enumerate(processed_slices):
        result[:, m, :] = processed_slice

    return result


def remove_stripe_fw(sino, wavelet_filter_name="db5", sigma=2):
    """
    This function removes vertical stripes from a sinogram by wavelet and Fourier filtering.
    It uses the method of Münch et al., Optics Express 17(10), 2009.  It matches tomopy's
    ``remove_stripe_fw``.

    Args:
        sino (numpy.ndarray): The sinogram, with shape (num_views, num_det_rows,
            num_det_channels).
        wavelet_filter_name (str, optional): The wavelet, such as 'db5' or 'haar'.  Defaults
            to 'db5'.
        sigma (float, optional): The width of the damping in the Fourier domain.  It sets the
            strength of the stripe removal.  Defaults to 2.

    Returns:
        numpy.ndarray: The corrected sinogram, with the shape of ``sino``.
    """
    import pywt
    sino = np.array(sino, dtype=np.float32)
    level = int(np.ceil(np.log2(np.max(np.array(sino.shape)))))
    views, num_rows, num_columns = sino.shape
    padded_views = views + views // 8
    shift_val = views // 16

    for m in range(sino.shape[1]):
        sino_slice = np.zeros((padded_views, num_columns), dtype=np.float32)
        sino_slice[shift_val:views + shift_val] = sino[:, m, :]

        cH, cV, cD = {}, {}, {}
        for n in range(level):
            sino_slice, (cH[n], cV[n], cD[n]) = pywt.dwt2(sino_slice, wavelet_filter_name)

        for n in range(level):
            fcV = np.fft.fftshift(np.fft.fft(cV[n], axis=0))
            my, mx = fcV.shape

            y_hat = (np.arange(-my, my, 2, dtype='float32') + 1) / 2
            damp = -np.expm1(-np.square(y_hat) / (2 * np.square(np.float32(sigma))))
            fcV *= np.transpose(np.tile(damp, (mx, 1)))

            cV[n] = np.real(np.fft.ifft(np.fft.ifftshift(fcV), axis=0))

        for n in range(level)[::-1]:
            sino_slice = sino_slice[0:cH[n].shape[0], 0:cH[n].shape[1]]
            sino_slice = pywt.idwt2((sino_slice, (cH[n], cV[n], cD[n])), wavelet_filter_name)

        sino[:, m, :] = sino_slice[shift_val:views + shift_val, 0:num_columns]

    return sino

def remove_sino_offset(sino):
    """
    This function removes from each view the additive offset caused by material outside the
    field of view.  For each detector row, the function computes the mean over channels in
    each view.  It then shifts each view so that this mean equals the smallest mean over all
    views.

    Args:
        sino (numpy.ndarray): The sinogram, with shape (num_views, num_det_rows,
            num_det_channels).

    Returns:
        numpy.ndarray: The corrected sinogram, with the shape of ``sino``.
    """
    sino_channel_avg = np.mean(sino, axis=2)

    sino_min_channel_avg = np.min(sino_channel_avg, axis=0)

    sino_corrected = sino - sino_channel_avg[:, :, None] + sino_min_channel_avg[None, :, None]

    return sino_corrected

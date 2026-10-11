"""This module contains functions that the scanner data loaders use.

The functions read images, compute geometry vectors, crop the detector, convert units, and
build the model from the parameter dictionaries of a loader.
"""

import glob
import os

import numpy as np
import mbirtorch as mt


def _normalize_to_float32(img: np.ndarray) -> np.ndarray:
    """
    This function converts an image to float32.  If the image has an integer dtype, the function
    divides it by the largest value of that dtype.

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The float32 image.  An integer input image is scaled to [0, 1].
    """
    if np.issubdtype(img.dtype, np.integer):
        maxval = np.iinfo(img.dtype).max
        return img.astype(np.float32) / maxval
    return img.astype(np.float32)


def read_tif_img(img_path):
    """
    This function reads a TIFF image as a float32 array.  If the image has an integer dtype, the
    function divides it by the largest value of that dtype.

    Args:
        img_path (str): The path of the image file.  ``tifffile`` must be able to read the file.

    Returns:
        np.ndarray: The image as a float32 array with 2 or more dimensions.
    """
    import tifffile
    img = tifffile.imread(img_path)
    img = _normalize_to_float32(img)
    return img


def read_tif_stack_dir(scan_dir, view_ids=None):
    """This function reads a stack of TIFF scan images from a directory.

    The function reads the files whose names end in a digit followed by ``.tif`` or ``.tiff``.
    It sorts the file names and assumes that this order is the order of the views.

    Args:
        scan_dir (string): The path of the scan directory, such as
            "<absolute_path_to_dataset>/Radiographs".
        view_ids (ndarray of ints, optional, default=None): The indices of the views to read.
            If None, the function reads all views.

    Returns:
        ndarray (float): A 3D array with shape (num_views, num_det_rows, num_det_channels).
    """

    import tifffile
    img_path_list = sorted(glob.glob(os.path.join(scan_dir, '*[0-9].tif')))
    if len(img_path_list) == 0:
        img_path_list = sorted(glob.glob(os.path.join(scan_dir, '*[0-9].tiff')))

    if len(img_path_list) == 0:
        raise FileNotFoundError('No scan images found in directory: {}'.format(scan_dir))

    # The sorted file names must be in view order.
    if view_ids is None:
        view_ids = np.arange(len(img_path_list))
    else:
        max_view_id = np.amax(view_ids)
        if max_view_id >= len(img_path_list):
            raise FileNotFoundError('The max view index was given as {}, but there are only {} views in {}'.format(max_view_id, len(img_path_list), scan_dir))
    img_path_list = [img_path_list[idx] for idx in view_ids]

    output_views = tifffile.imread(img_path_list, ioworkers=48, maxworkers=8)
    output_views = _normalize_to_float32(output_views)

    return output_views


def unit_vector(v):
    """This function returns the unit vector v/||v||."""
    return v / np.linalg.norm(v)


def project_vector_to_vector(u1, u2):
    """This function returns the projection of u1 onto u2."""
    u2 = unit_vector(u2)
    u1_proj = np.dot(u1, u2)*u2
    return u1_proj


def detect_blank_margins(sino, safety_buffer=20, max_views_to_use=20):
    """
    This function counts the blank detector rows and channels around the object in a sinogram.

    The function uses a subset of the views.  The result can be passed to
    :func:`apply_detector_crop`.

    Args:
        sino (np.ndarray): The sinogram, with shape ``(num_views, num_det_rows, num_det_channels)``.
        safety_buffer (int, optional): The number of blank pixels to keep on each side of the
            object.  Defaults to 20.
        max_views_to_use (int, optional): The largest number of views that the function uses.
            Defaults to 20.

    Returns:
        tuple: ``(crop_top, crop_bottom, crop_left, crop_right)``.  The first two entries are the
        numbers of detector rows to crop from the top and bottom.  The last two entries are the
        numbers of detector channels to crop from the left and right.
    """
    sino = mt.TomographyModel.subsample_views(sino, max_views_to_use)

    sino_indicator_mask = mt.TomographyModel._get_sino_indicator(sino)

    union_mask = np.any(np.asarray(sino_indicator_mask), axis=0)

    rows = np.any(union_mask, axis=1)
    cols = np.any(union_mask, axis=0)

    # argmax on a binary array returns the index of the first 1.
    top_width = np.argmax(rows)
    bottom_width = np.argmax(rows[::-1])
    left_width = np.argmax(cols)
    right_width = np.argmax(cols[::-1])

    crop_pixels_top = max(top_width - safety_buffer, 0)
    crop_pixels_bottom = max(bottom_width - safety_buffer, 0)
    crop_pixels_left = max(left_width - safety_buffer, 0)
    crop_pixels_right = max(right_width - safety_buffer, 0)

    return crop_pixels_top, crop_pixels_bottom, crop_pixels_left, crop_pixels_right


def apply_detector_crop(required_params, optional_params, crop_top, crop_bottom, crop_left, crop_right):
    """
    This function updates the geometry parameters to match a detector crop.

    The function reduces the sinogram shape by the cropped rows and channels.  It also shifts the
    detector offsets by the distance that the crop moves the detector center.  The function does
    not change the sinogram data.  The caller must crop the sinogram array in the same way.

    Args:
        required_params (dict): The constructor parameters.  The function uses
            ``sinogram_shape``.
        optional_params (dict): The parameters applied with ``set_params``.  If
            ``det_row_offset`` or ``det_channel_offset`` is present, the function adjusts it.
            The adjustment uses ``delta_det_row`` and ``delta_det_channel``, which default to 1.0.
        crop_top (int): The number of detector rows removed from the top.
        crop_bottom (int): The number of detector rows removed from the bottom.
        crop_left (int): The number of detector channels removed from the left.
        crop_right (int): The number of detector channels removed from the right.

    Returns:
        tuple: New ``(required_params, optional_params)`` dicts.  The function does not modify
        the input dicts.

    Raises:
        AssertionError: If any crop amount is negative, or if
            ``crop_top + crop_bottom >= num_det_rows``, or if
            ``crop_left + crop_right >= num_det_channels``.
    """
    num_views, num_det_rows, num_det_channels = required_params['sinogram_shape']
    assert (crop_top >= 0 and crop_bottom >= 0 and crop_left >= 0 and crop_right >= 0 and
            crop_top + crop_bottom < num_det_rows and crop_left + crop_right < num_det_channels), \
        ('apply_detector_crop: crop amounts must be nonnegative with crop_top + crop_bottom < num_det_rows'
         ' and crop_left + crop_right < num_det_channels (got top={}, bottom={}, left={}, right={} for a'
         ' {}x{} detector).'.format(crop_top, crop_bottom, crop_left, crop_right, num_det_rows, num_det_channels))
    required_params = dict(required_params)
    optional_params = dict(optional_params)
    required_params['sinogram_shape'] = (int(num_views),
                                         int(num_det_rows - crop_top - crop_bottom),
                                         int(num_det_channels - crop_left - crop_right))

    if 'det_row_offset' in optional_params:
        delta_det_row = optional_params.get('delta_det_row', 1.0)
        optional_params['det_row_offset'] += (crop_bottom - crop_top) / 2 * delta_det_row
    if 'det_channel_offset' in optional_params:
        delta_det_channel = optional_params.get('delta_det_channel', 1.0)
        optional_params['det_channel_offset'] += (crop_right - crop_left) / 2 * delta_det_channel

    return required_params, optional_params


def _auto_crop_sino(sino, required_params, optional_params, safety_buffer=20):
    """
    This function removes the blank margins of a sinogram and updates the geometry to match.

    The function calls :func:`detect_blank_margins` and :func:`apply_detector_crop`.  It runs
    before ``build_model``.

    Args:
        sino (np.ndarray): The sinogram, with shape ``(num_views, num_det_rows, num_det_channels)``.
        required_params (dict): The constructor parameters, including ``sinogram_shape``.
        optional_params (dict): The parameters applied with ``set_params``, such as the detector
            pitches and offsets.
        safety_buffer (int, optional): The number of blank pixels to keep around the object.
            Defaults to 20.

    Returns:
        tuple: ``(sino, required_params, optional_params)``.  The sinogram is cropped, and the
        parameters are updated.
    """
    crop_top, crop_bottom, crop_left, crop_right = detect_blank_margins(sino, safety_buffer)
    sino = sino[:, crop_top:sino.shape[1] - crop_bottom, crop_left:sino.shape[2] - crop_right]
    # If the user set recon_slice_offset, this code keeps its position relative to the automatic center.
    supplied_offset = optional_params.get('recon_slice_offset')
    if supplied_offset is not None:
        _, automatic_before = mt.utilities._automatic_recon_geometry(required_params, optional_params, None)
    required_params, optional_params = apply_detector_crop(
        required_params, optional_params, crop_top, crop_bottom, crop_left, crop_right)
    if supplied_offset is not None and automatic_before is not None:
        _, automatic_after = mt.utilities._automatic_recon_geometry(required_params, optional_params, None)
        optional_params = dict(optional_params)
        optional_params['recon_slice_offset'] = float(supplied_offset) + (automatic_after - automatic_before)
    return sino, required_params, optional_params


def apply_config_crop(num_det_rows, num_det_channels, det_row_offset, det_channel_offset,
                      delta_det_row, delta_det_channel, *,
                      crop_pixels_top, crop_pixels_bottom, crop_pixels_sides):
    """
    This function applies a detector crop from a loader configuration to the scalar geometry
    values of a scanner loader.

    The function calls :func:`apply_detector_crop`.  It crops ``crop_pixels_sides`` channels from
    both the left and the right side.

    Args:
        num_det_rows (int): The number of detector rows before the crop.
        num_det_channels (int): The number of detector channels before the crop.
        det_row_offset (float): The detector row offset in ALU before the crop.
        det_channel_offset (float): The detector channel offset in ALU before the crop.
        delta_det_row (float): The detector row pitch.
        delta_det_channel (float): The detector channel pitch.
        crop_pixels_top (int): The number of detector rows cropped from the top.
        crop_pixels_bottom (int): The number of detector rows cropped from the bottom.
        crop_pixels_sides (int): The number of detector channels cropped from each side.

    Returns:
        tuple: ``(num_det_rows, num_det_channels, det_row_offset, det_channel_offset)`` after
        the crop.
    """
    required, optional = apply_detector_crop(
        {'sinogram_shape': (0, num_det_rows, num_det_channels)},  # num_views is not used, so it is 0.
        {'delta_det_row': delta_det_row, 'delta_det_channel': delta_det_channel,
         'det_row_offset': det_row_offset, 'det_channel_offset': det_channel_offset},
        crop_pixels_top, crop_pixels_bottom, crop_pixels_sides, crop_pixels_sides)
    _, num_det_rows, num_det_channels = required['sinogram_shape']
    return num_det_rows, num_det_channels, optional['det_row_offset'], optional['det_channel_offset']


def finalize_model(sino, required_params, optional_params, *, auto_crop=False, safety_buffer=20):
    """
    This function builds a model from the sinogram and parameter dictionaries of a scanner loader.

    Each loader calls this function as the last step of its ``get_sino_and_model``.

    Args:
        sino (numpy.ndarray): The sinogram.
        required_params (dict): The model constructor arguments.  The dict also has a
            ``geometry_type`` entry that names the model class.
        optional_params (dict): The arguments for ``set_params``.
        auto_crop (bool, optional): If True, the function first removes the blank margins of the
            sinogram.  Defaults to False.
        safety_buffer (int, optional): The number of blank pixels kept around the object when
            ``auto_crop`` is True.  Defaults to 20.

    Returns:
        tuple: ``(sino, model)``.  The sinogram is cropped if ``auto_crop`` is True.
    """
    if auto_crop:
        sino, required_params, optional_params = _auto_crop_sino(sino, required_params, optional_params, safety_buffer)
    model = mt.build_model(required_params, optional_params)
    return sino, model


_ALU_UNIT_CONVERSION = {'um': 1.0, 'mm': 1000.0, 'cm': 1e4, 'm': 1e6}


def to_alu(value, from_unit, alu_unit):
    """
    This function converts a physical ``value`` from ``from_unit`` to ALU.

    ALU is the arbitrary length unit of the model.  One ALU equals one ``alu_unit``.

    Args:
        value (float or array): The value or values to convert.
        from_unit (str): The unit of ``value``: ``'um'``, ``'mm'``, ``'cm'``, or ``'m'``.
        alu_unit (str): The unit that equals 1 ALU.

    Returns:
        float or array: The value in ALU.
    """
    return value * _ALU_UNIT_CONVERSION[from_unit] / _ALU_UNIT_CONVERSION[alu_unit]

import os, sys
from operator import itemgetter
import numpy as np
import warnings
import mbirtorch as mt
import mbirtorch.preprocess as mtp
from . import _loader_utils as lu
import pprint
import olefile
from . import _xradia_ole
from ._xradia_ole import (_check_read, _get_ole_data_type, _log_imported_data, _read_ole_struct,
                          _read_ole_value, _read_ole_arr, _read_ole_image, _read_ole_str,
                          get_index_in_list)
pp = pprint.PrettyPrinter(indent=4)


def get_sino_and_model(dataset_dir, *, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0,
                       alu_unit='mm', det_rotation=0.0, verbose=1):
    """
    This function loads a Zeiss translation CT scan, computes its sinogram, and returns a model
    and a weight mask.  The weight mask is 0 on the dark border of the detector and 1
    elsewhere.  The user passes the weight mask to ``model.recon`` as the weights.

    Args:
        dataset_dir (str): Path to the scan directory, with ``obj_scan``, ``blank_scan``, and
            ``dark_scan`` subfolders.
        crop_pixels_sides (int, optional): The number of pixels to crop from each side of the
            detector.  Defaults to 0.
        crop_pixels_top (int, optional): The number of pixels to crop from the top of the
            detector.  Defaults to 0.
        crop_pixels_bottom (int, optional): The number of pixels to crop from the bottom of
            the detector.  Defaults to 0.
        alu_unit (str, optional): The length unit of the model: 'um', 'mm', 'cm', or 'm'.
            Defaults to 'mm'.
        det_rotation (float, optional): The detector rotation in radians.  This function
            removes this rotation from every view.  Defaults to 0.0.
        verbose (int, optional): The value 0 prints nothing, and 1 prints progress.  Defaults
            to 1.

    Returns:
        tuple: ``(sino, model, weights)``.  The sinogram ``sino`` has shape (num_views,
        num_det_rows, num_det_channels).  The ``model`` is a TranslationModel with its
        parameters set.  The weight mask ``weights`` has the shape of the sinogram.

    Example:
        .. code-block:: python

            sino, model, weights = mbirtorch.preprocess.zeiss_tct.get_sino_and_model(dataset_dir)
            recon, recon_dict = model.recon(sino, weights=weights)
    """
    sino, required_params, optional_params, weights = _compute_sino_and_params(
        dataset_dir, crop_pixels_sides=crop_pixels_sides, crop_pixels_top=crop_pixels_top,
        crop_pixels_bottom=crop_pixels_bottom, alu_unit=alu_unit, det_rotation=det_rotation,
        verbose=verbose)
    sino, model = lu.finalize_model(sino, required_params, optional_params)
    return sino, model, weights


def _compute_sino_and_params(dataset_dir, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, alu_unit='mm', det_rotation=0.0, verbose=1):
    """
    This function loads Zeiss translation CT scans and computes the sinogram, the model
    parameters, and a weight mask for :func:`get_sino_and_model`.  The docstring of
    :func:`get_sino_and_model` describes the arguments.

    Returns:
        tuple: ``(sino, required_params, optional_params, weights)``.  The dict
        ``required_params`` holds the TranslationModel constructor arguments and
        ``geometry_type``.  The dict ``optional_params`` holds the arguments for
        ``set_params``.  The array ``weights`` has the shape of ``sino``.
    """
    if verbose > 0:
        print("\n\n########## Loading object, blank, dark scans, and geometry parameters from Zeiss dataset directory")
    obj_scan, blank_scan, dark_scan, zeiss_params = load_scans_and_params(dataset_dir, verbose=verbose)

    translation_params, optional_params = convert_zeiss_to_mbirtorch_params(zeiss_params,
                                                                          crop_pixels_sides=crop_pixels_sides,
                                                                          crop_pixels_top=crop_pixels_top,
                                                                          crop_pixels_bottom=crop_pixels_bottom,
                                                                          alu_unit=alu_unit)

    if verbose > 0:
        print("\n\n########## Cropping scans")
    obj_scan, blank_scan, dark_scan, defective_pixel_array = mtp.crop_view_data(obj_scan, blank_scan, dark_scan,
                                                                                crop_pixels_sides=crop_pixels_sides,
                                                                                crop_pixels_top=crop_pixels_top,
                                                                                crop_pixels_bottom=crop_pixels_bottom)

    weights = compute_weight(blank_scan, obj_scan)

    if verbose > 0:
        print("\n\n########## Computing sinogram from object, blank, and dark scans")
    sino = mtp.scan_to_sino(obj_scan, blank_scan, dark_scan, defective_pixel_array,
                            downsample_factor=(1, 1), det_rotation=det_rotation)

    if verbose > 0:
        print("\n\n########## Correcting sinogram data to account for background offset")
    sino = mtp.correct_background_offset(sino)

    if verbose > 0:
        print('obj_scan shape = ', obj_scan.shape)
        print('blank_scan shape = ', blank_scan.shape)
        print('dark_scan shape = ', dark_scan.shape)

    translation_params['geometry_type'] = str(mt.TranslationModel)
    return sino, translation_params, optional_params, weights


def load_scans_and_params(dataset_dir, verbose=1):
    """
    This function loads the object scan, blank scan, dark scan, and geometry from a Zeiss scan
    directory.  This function flips all scans vertically.

    Args:
        dataset_dir (str): Path to a Zeiss scan directory with ``obj_scan``, ``blank_scan``,
            and ``dark_scan`` subfolders.
        verbose (int, optional): The verbosity level.  Defaults to ``1``.

    Returns:
        tuple: ``(obj_scan, blank_scan, dark_scan, zeiss_params)``

            - ``obj_scan`` (numpy.ndarray): 3D object scan with shape
              ``(num_views, num_det_rows, num_channels)``.
            - ``blank_scan`` (numpy.ndarray): 3D blank scan with shape
              ``(num_blank_scans, num_det_rows, num_channels)``.  Each ``.xrm`` file gives
              one blank scan.
            - ``dark_scan`` (numpy.ndarray): 3D dark scan with shape
              ``(num_dark_scans, num_det_rows, num_channels)``.  When the ``dark_scan``
              subfolder is missing or empty, ``dark_scan`` is zeros with the shape of
              ``blank_scan``.
            - ``zeiss_params`` (dict): The geometry parameters read from the ``.xrm`` files.
    """
    obj_scan_dir, blank_scan_dir, dark_scan_dir = _parse_filenames_from_dataset_dir(dataset_dir)

    if verbose > 0:
        print("The following files will be used to compute the Zeiss reconstruction:\n",
              f"    - Object scan directory: {obj_scan_dir}\n",
              f"    - Blank scan directory: {blank_scan_dir}\n",
              f"    - Dark scan directory: {dark_scan_dir}\n",)

    obj_scan, zeiss_params = read_xrm_dir(obj_scan_dir)

    blank_scan, _ = read_xrm_dir(blank_scan_dir)

    # TODO: When there is no dark scan, this code uses an array of zeros.
    if os.path.isdir(dark_scan_dir) and len(os.listdir(dark_scan_dir)) > 0:
        dark_scan, _ = read_xrm_dir(dark_scan_dir)
    else:
        dark_scan = np.zeros(blank_scan.shape)

    # TODO: This code flips the scans to get the correct object orientation.
    if verbose > 0:
        print("Flipping scans vertically")
    obj_scan = np.flip(obj_scan, axis=1)
    blank_scan = np.flip(blank_scan, axis=1)
    dark_scan = np.flip(dark_scan, axis=1)

    try:
        axis_names = zeiss_params.get("axis_names")  # ['Sample X', ..., 'CCD_X', ...]
        axis_units = zeiss_params.get("axis_units")  # ['um', 'um', ...]

        source_iso_dist = zeiss_params["source_iso_dist"]
        source_iso_dist = float(np.abs(source_iso_dist))
        source_iso_dist_index = get_index_in_list(axis_names, 'Source Z')
        source_iso_dist_unit = axis_units[source_iso_dist_index] if source_iso_dist_index > -1 else 'mm'

        iso_det_dist = zeiss_params["iso_det_dist"]
        iso_det_dist = float(np.abs(iso_det_dist)) if iso_det_dist is not None else 0.0
        iso_det_dist_unit = source_iso_dist_unit

        det_pixel_pitch = zeiss_params["det_pixel_pitch"]
        det_pixel_pitch = float(np.abs(det_pixel_pitch))

        # A Zeiss detector pixel has equal width and height.
        delta_det_row = det_pixel_pitch
        delta_det_channel = det_pixel_pitch
        delta_det_index = get_index_in_list(axis_names, 'CCD_X')
        delta_det_row_unit = axis_units[delta_det_index] if delta_det_index > -1 else 'um'
        delta_det_channel_unit = delta_det_row_unit

        iso_pixel_pitch = zeiss_params["iso_pixel_pitch"]
        iso_pixel_pitch = float(np.abs(iso_pixel_pitch))
        iso_pixel_pitch_index = get_index_in_list(axis_names, 'Sample X')
        iso_pixel_pitch_unit = axis_units[iso_pixel_pitch_index] if iso_pixel_pitch_index > -1 else 'um'

        # The scanner axes map to the mbirtorch axes as follows: x -> -x, z -> y, y -> z.
        object_x_positions = -np.array(zeiss_params['x_positions'], dtype=float).ravel()
        object_y_positions = np.array(zeiss_params['z_positions'], dtype=float).ravel()
        object_z_positions = np.array(zeiss_params['y_positions'], dtype=float).ravel()

        object_x_position_index = get_index_in_list(axis_names, 'Sample X')
        object_y_position_index = get_index_in_list(axis_names, 'Sample Y')
        object_z_position_index = get_index_in_list(axis_names, 'Sample Z')

        object_x_position_unit = axis_units[object_x_position_index] if object_x_position_index > -1 else 'um'
        object_y_position_unit = axis_units[object_y_position_index] if object_y_position_index > -1 else 'um'
        object_z_position_unit = axis_units[object_z_position_index] if object_z_position_index > -1 else 'um'

        x_shifts = zeiss_params["x_shifts"]
        y_shifts = zeiss_params["y_shifts"]

    except ValueError as e:
        print("Unable to determine units for geometry parameters; cannot safely convert to mbirtorch format.")
        raise e

    opt_mag = zeiss_params["opt_mag"]
    opt_mag = 1 if opt_mag is None else opt_mag

    num_views = zeiss_params["num_views"]
    num_det_channels = zeiss_params["num_det_channels"]
    num_det_rows = zeiss_params["num_det_rows"]

    # mbirtorch uses the opposite sign convention for the channel shift.
    center_shift = zeiss_params.get("center_shift")

    if center_shift is None:
        det_channel_offset = 0.0
    else:
        det_channel_offset = -center_shift

    det_row_offset = 0.0  # The Zeiss files do not appear to give a detector row offset.

    zeiss_params = {
        'source_iso_dist': source_iso_dist,
        'iso_det_dist': iso_det_dist,
        'delta_det_channel': delta_det_channel,
        'delta_det_row': delta_det_row,
        'iso_pixel_pitch': iso_pixel_pitch,
        'opt_mag': opt_mag,
        'num_views': num_views,
        'num_det_channels': num_det_channels,
        'num_det_rows': num_det_rows,
        'det_row_offset': det_row_offset,
        'det_channel_offset': det_channel_offset,
        'object_x_positions': object_x_positions,
        'object_y_positions': object_y_positions,
        'object_z_positions': object_z_positions,
        'source_iso_dist_unit': source_iso_dist_unit,
        'iso_det_dist_unit': iso_det_dist_unit,
        'delta_det_row_unit': delta_det_row_unit,
        'delta_det_channel_unit': delta_det_channel_unit,
        'iso_pixel_pitch_unit': iso_pixel_pitch_unit,
        'object_x_position_unit': object_x_position_unit,
        'object_y_position_unit': object_y_position_unit,
        'object_z_position_unit': object_z_position_unit,
        'x_shifts': x_shifts,
        'y_shifts': y_shifts,
    }

    if verbose > 0:
        print("############ Zeiss geometry parameters ############")
        print(f"Source to iso distance: {source_iso_dist} [{source_iso_dist_unit}]")
        print(f"Iso to detector distance: {iso_det_dist} [{iso_det_dist_unit}]")
        print(f"Detector pixel pitch: {det_pixel_pitch:.3f} [{delta_det_row_unit}]")
        print(f"Pixel pitch at iso: {iso_pixel_pitch} [{iso_pixel_pitch_unit}]")
        print(f"Optical magnification: {opt_mag}")
        print(f"Number of views: {num_views}")
        print(f"Detector size: (num_det_rows, num_det_channels) = ({num_det_rows}, {num_det_channels})")
        print("############ End Zeiss geometry parameters ############")

    return obj_scan, blank_scan, dark_scan, zeiss_params


def convert_zeiss_to_mbirtorch_params(zeiss_params, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, alu_unit='mm'):
    """
    This function converts Zeiss geometry parameters into mbirtorch parameters.  The mbirtorch
    parameters account for the crop.

    Args:
        zeiss_params (dict): The Zeiss geometry parameters, as returned by
            :func:`load_scans_and_params`.
        crop_pixels_sides (int, optional): The number of pixels to crop from each side of the
            sinogram.  Defaults to 0.
        crop_pixels_top (int, optional): The number of pixels to crop from the top of the
            sinogram.  Defaults to 0.
        crop_pixels_bottom (int, optional): The number of pixels to crop from the bottom of the
            sinogram.  Defaults to 0.
        alu_unit (str, optional): The physical unit of 1 ALU: 'um', 'mm', 'cm', or 'm'.
            Defaults to 'mm'.

    Returns:
        translation_params (dict): The required parameters for the TranslationModel
        constructor.
        optional_params (dict): The other TranslationModel parameters, to be set with
        set_params().
    """
    source_iso_dist, iso_det_dist, source_iso_dist_unit, iso_det_dist_unit = itemgetter('source_iso_dist', 'iso_det_dist', 'source_iso_dist_unit', 'iso_det_dist_unit')(zeiss_params)
    delta_det_channel, delta_det_row, delta_det_channel_unit, delta_det_row_unit = itemgetter('delta_det_channel', 'delta_det_row', 'delta_det_channel_unit', 'delta_det_row_unit')(zeiss_params)
    iso_pixel_pitch, iso_pixel_pitch_unit = itemgetter('iso_pixel_pitch', 'iso_pixel_pitch_unit')(zeiss_params)
    opt_mag = itemgetter('opt_mag')(zeiss_params)
    num_det_rows, num_det_channels = itemgetter('num_det_rows', 'num_det_channels')(zeiss_params)
    object_x_positions, object_x_position_unit = itemgetter('object_x_positions', 'object_x_position_unit')(zeiss_params)
    object_y_positions, object_y_position_unit = itemgetter('object_y_positions', 'object_y_position_unit')(zeiss_params)
    object_z_positions, object_z_position_unit = itemgetter('object_z_positions', 'object_z_position_unit')(zeiss_params)
    det_row_offset, det_channel_offset = itemgetter('det_row_offset', 'det_channel_offset')(zeiss_params)

    alu_value = 1

    source_iso_dist = lu.to_alu(source_iso_dist, source_iso_dist_unit, alu_unit)
    iso_det_dist = lu.to_alu(iso_det_dist, iso_det_dist_unit, alu_unit)
    delta_det_channel = lu.to_alu(delta_det_channel, delta_det_channel_unit, alu_unit)
    delta_det_row = lu.to_alu(delta_det_row, delta_det_row_unit, alu_unit)
    iso_pixel_pitch = lu.to_alu(iso_pixel_pitch, iso_pixel_pitch_unit, alu_unit)
    object_x_positions = lu.to_alu(object_x_positions, object_x_position_unit, alu_unit)
    object_y_positions = lu.to_alu(object_y_positions, object_y_position_unit, alu_unit)
    object_z_positions = lu.to_alu(object_z_positions, object_z_position_unit, alu_unit)

    source_detector_dist = source_iso_dist + iso_det_dist

    translation_vectors = calc_translation_vec_params(object_x_positions, object_y_positions, object_z_positions)

    # The total magnification is the optical magnification times the geometric magnification.
    if opt_mag is not None:
        scintillator_mag = source_detector_dist / source_iso_dist
        magnification = opt_mag * scintillator_mag
    else:
        magnification = 1.0

    source_detector_dist = magnification * source_iso_dist
    delta_det_channel = magnification * iso_pixel_pitch
    delta_det_row = magnification * iso_pixel_pitch

    # The offsets are in pixels.  This code converts them to ALU before the crop.
    det_channel_offset *= delta_det_channel
    det_row_offset *= delta_det_row

    # This function does not downsample translation CT data.
    num_views = len(translation_vectors)
    num_det_rows, num_det_channels, det_row_offset, det_channel_offset = lu.apply_config_crop(
        num_det_rows, num_det_channels, det_row_offset, det_channel_offset, delta_det_row, delta_det_channel,
        crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom, crop_pixels_sides=crop_pixels_sides)

    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    recon_shape, delta_voxel, voxel_row_aspect = mt.utilities.calc_tct_recon_params(source_detector_dist, source_iso_dist, delta_det_row, delta_det_channel, sinogram_shape, translation_vectors)

    translation_params = dict()
    translation_params['sinogram_shape'] = sinogram_shape
    translation_params['translation_vectors'] = translation_vectors
    translation_params['source_detector_dist'] = source_detector_dist
    translation_params['source_iso_dist'] = source_iso_dist

    optional_params = dict()
    optional_params['delta_det_channel'] = delta_det_channel
    optional_params['delta_det_row'] = delta_det_row
    optional_params['delta_voxel'] = delta_voxel
    optional_params['voxel_row_aspect'] = voxel_row_aspect
    optional_params['recon_shape'] = recon_shape
    optional_params['det_row_offset'] = det_row_offset
    optional_params['det_channel_offset'] = det_channel_offset
    optional_params['alu_unit'] = alu_unit
    optional_params['alu_value'] = alu_value

    return translation_params, optional_params


######## subroutines for parsing Zeiss object scan, blank scan, and dark scan
def _parse_filenames_from_dataset_dir(dataset_dir):
    """
    This function returns the object, blank, and dark scan directories of a Zeiss scan
    directory.

    Args:
        dataset_dir (string): Path to the directory containing the Zeiss scan files.

    Returns:
        tuple:
            - obj_scan_dir (string): Path to the object scan directory
            - blank_scan_dir (string): Path to the blank scan directory
            - dark_scan_dir (string): Path to the dark scan directory
    """
    obj_scan_dir = os.path.join(dataset_dir, "obj_scan")

    blank_scan_dir = os.path.join(dataset_dir, "blank_scan")

    dark_scan_dir = os.path.join(dataset_dir, "dark_scan")

    return obj_scan_dir, blank_scan_dir, dark_scan_dir


def read_xrm(fname, normalize_to_float32=False):
    """
    This function reads a single Xradia ``.xrm`` radiograph and its metadata.  It calls
    :func:`mbirtorch.preprocess._xradia_ole.read_xrm` with this module's :func:`read_metadata`.
    """
    return _xradia_ole.read_xrm(fname, read_metadata, normalize_to_float32=normalize_to_float32)


def read_xrm_dir(dir_path):
    """
    This function reads all ``.xrm`` files in a directory and stacks them.  It calls
    :func:`mbirtorch.preprocess._xradia_ole.read_xrm_dir` with this module's
    :func:`read_metadata`.
    """
    return _xradia_ole.read_xrm_dir(dir_path, read_metadata)


def read_txrm(file_name):
    """
    This function reads the images and metadata from a .txrm file.  A .txrm file holds the
    contents of several .xrm files.  This function is adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Args:
        file_name (str): The path of the file.

    Returns:
        np.ndarray: The 3D image with shape (num_views, num_det_rows, num_det_channels).
        dict: The metadata.
    """
    file_name = _check_read(file_name)
    try:
        ole = olefile.OleFileIO(file_name)
    except IOError:
        print('No such file or directory: %s', file_name)
        return False

    metadata = read_metadata(ole)

    array_of_images = np.empty(
        (
            metadata["num_views"],
            metadata["num_det_rows"],
            metadata["num_det_channels"],
        ),
        dtype=_get_ole_data_type(metadata)
    )

    for i, idx in enumerate(range(metadata["num_views"])):
        img_string = "ImageData{}/Image{}".format(
            int(np.ceil((idx + 1) / 100.0)), int(idx + 1))
        array_of_images[i] = _read_ole_image(ole, img_string, metadata)

    _log_imported_data(file_name, array_of_images)

    ole.close()
    return array_of_images, metadata


def read_metadata(ole):
    """
    This function reads the metadata from an Xradia OLE file (.xrm, .txrm, or .txm).  Parts
    of this function are adapted from the DXchange library:
    https://github.com/data-exchange/dxchange

    Args:
        ole (OleFileIO instance): The OLE file to read from.

    Returns:
        dict: The image metadata.
    """

    number_of_images = _read_ole_value(ole, "ImageInfo/NoOfImages", "<I")

    metadata = {
        'num_det_channels': _read_ole_value(ole, 'ImageInfo/ImageWidth', '<I'),
        'num_det_rows': _read_ole_value(ole, 'ImageInfo/ImageHeight', '<I'),
        'data_type': _read_ole_value(ole, 'ImageInfo/DataType', '<1I'),
        'num_views': number_of_images,
        'iso_pixel_pitch': _read_ole_value(ole, 'ImageInfo/PixelSize', '<f'),
        'det_pixel_pitch': _read_ole_value(ole, 'ImageInfo/CamPixelSize', '<f'),
        'iso_det_dist': _read_ole_value(ole, 'ImageInfo/DtoRADistance', "<f"),
        'source_iso_dist': _read_ole_value(ole,'ImageInfo/StoRADistance', "<{0}f".format(number_of_images)),
        'thetas': _read_ole_arr(
            ole, 'ImageInfo/Angles', "<{0}f".format(number_of_images)),
        'x_positions': _read_ole_arr(
            ole, 'ImageInfo/XPosition', "<{0}f".format(number_of_images)),
        'y_positions': _read_ole_arr(
            ole, 'ImageInfo/YPosition', "<{0}f".format(number_of_images)),
        'z_positions': _read_ole_arr(
            ole, 'ImageInfo/ZPosition', "<{0}f".format(number_of_images)),
        'x_shifts':_read_ole_arr(
            ole, 'alignment/x-shifts', "<{0}f".format(number_of_images)),
        'y_shifts': _read_ole_arr(
            ole, 'alignment/y-shifts', "<{0}f".format(number_of_images)),
        'current': _read_ole_value(
            ole, "ImageInfo/XrayCurrent", "<{0}f".format(number_of_images)),
        'voltage': _read_ole_value(
            ole, "ImageInfo/XrayVoltage", "<{0}f".format(number_of_images)),
        'ExpTimes': _read_ole_value(
            ole, "ImageInfo/ExpTimes", "<{0}f".format(number_of_images)),
        'center_shift': _read_ole_value(ole, "ReconSettings/CenterShift", '<f'),
        'opt_mag': _read_ole_value(ole, 'ImageInfo/OpticalMagnification', '<f'),
        'fan_angle': _read_ole_value(ole, 'ImageInfo/FanAngle', '<f'),
        'cone_angle': _read_ole_value(ole, 'ImageInfo/ConeAngle', '<f'),
        'axis_names': _read_ole_str(ole, 'PositionInfo/AxisNames'),
        'axis_units': _read_ole_str(ole, 'PositionInfo/AxisUnits')
    }

    return metadata


######## END subroutines for parsing Zeiss object scan, blank scan, and dark scan


######## subroutines for Zeiss-MBIR parameter conversion
def calc_translation_vec_params(obj_x_positions, obj_y_positions, obj_z_positions):
    """
    This function calculates the translation vectors from the object positions.  Each
    translation vector goes from the object position of a view to the center of all object
    positions.

    Args:
        obj_x_positions (tuple): The object x position of each view, with shape (num_views,).
        obj_y_positions (tuple): The object y position of each view, with shape (num_views,).
        obj_z_positions (tuple): The object z position of each view, with shape (num_views,).

    Returns:
        translation_vectors (np.ndarray): An array of shape (num_views, 3).  Each row is a
        translation vector [dx, dy, dz].
    """
    obj_xyz_positions = np.stack([obj_x_positions, obj_y_positions, obj_z_positions], axis=1)

    max_obj_xyz_positions = np.max(obj_xyz_positions, axis=0)
    min_obj_xyz_positions = np.min(obj_xyz_positions, axis=0)

    # The center position is the midpoint of the minimum and maximum along each axis.
    center_xyz_position = (max_obj_xyz_positions + min_obj_xyz_positions) / 2

    translation_vectors = center_xyz_position - obj_xyz_positions

    return translation_vectors
######## END subroutines for Zeiss-MBIR parameter conversion


def compute_weight(blank_scan, obj_scan, dark_region_ratio=0.6, safety_buffer=20):
    """
    This function creates a 3D binary weight mask that is zero in the dark border regions of
    the detector.  This function finds the dark regions in the first blank scan.

    Args:
        blank_scan (numpy array): A 3D blank scan of shape (num_blank_scans, num_det_rows,
            num_det_channels).
        obj_scan (numpy array): A 3D object scan of shape (num_views, num_det_rows,
            num_det_channels).
        dark_region_ratio (float): A pixel gets zero weight when its blank scan value is below
            dark_region_ratio * median(blank_scan[0]).  Defaults to 0.6.
        safety_buffer (int, optional): The width in pixels of the zero-weight border that this
            function adds around each dark region.  Defaults to 20.

    Returns:
        weights: A 3D array of weights with the same shape as obj_scan.
    """
    num_views, num_rows, num_cols = obj_scan.shape

    weight_mask_2d = blank_scan[0] >= dark_region_ratio * np.median(blank_scan[0])

    if safety_buffer > 0:
        padded_mask = np.pad(weight_mask_2d, safety_buffer, mode='constant', constant_values=True)
        from scipy.ndimage import binary_erosion
        padded_mask = binary_erosion(padded_mask, iterations=safety_buffer)
        weight_mask_2d = padded_mask[safety_buffer:-safety_buffer, safety_buffer:-safety_buffer]

    weight_mask = np.broadcast_to(
        weight_mask_2d[np.newaxis, :, :],
        (num_views, num_rows, num_cols),
    ).astype(np.float32)

    return weight_mask
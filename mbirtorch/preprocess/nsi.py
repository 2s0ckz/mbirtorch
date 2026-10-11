import os, sys
import re
from operator import itemgetter
import numpy as np
import warnings
import mbirtorch
import mbirtorch.preprocess as mtp
from . import _loader_utils as lu
import glob
import pprint
pp = pprint.PrettyPrinter(indent=4)


def get_sino_and_model(dataset_dir, *, downsample_factor=(1, 1), subsample_view_factor=1,
                       crop_pixels_sides=None, crop_pixels_top=None, crop_pixels_bottom=None,
                       auto_crop=False, verbose=1, offset_correction=True):
    """
    This function loads an NSI scan, computes its sinogram, and returns a cone beam model for
    the scan.  Each crop that is None takes its value from the NSI config file.  When the top
    and bottom crops differ, this function sets both of them to the larger value.  The model
    measures all lengths in mm.

    Args:
        dataset_dir (str): Path to the NSI scan directory.
        downsample_factor (tuple[int, int], optional): The downsampling factors along the
            detector rows and channels.  Defaults to (1, 1).
        subsample_view_factor (int, optional): This function keeps every n-th view, where n
            is this factor.  Defaults to 1.
        crop_pixels_sides (int, optional): The number of pixels to crop from each side of the
            detector.  None means the value in the config file.  Defaults to None.
        crop_pixels_top (int, optional): The number of pixels to crop from the top of the
            detector.  None means the value in the config file.  Defaults to None.
        crop_pixels_bottom (int, optional): The number of pixels to crop from the bottom of
            the detector.  None means the value in the config file.  Defaults to None.
        auto_crop (bool, optional): If True, this function removes the blank margins of the
            sinogram and shrinks the reconstruction to match.  Defaults to False.
        verbose (int, optional): The value 0 prints nothing, and 1 prints progress.  Defaults
            to 1.
        offset_correction (bool, optional): If True, this function takes the detector position
            from the geometry report instead of the config file.  Defaults to True.

    Returns:
        tuple: ``(sino, model)``.  The sinogram ``sino`` has shape (num_views, num_det_rows,
        num_det_channels).  The ``model`` is a ConeBeamModel with its parameters set.  The
        function ``mbirtorch.gen_weights`` computes the weights for the reconstruction.

    Example:
        .. code-block:: python

            sino, model = mbirtorch.preprocess.nsi.get_sino_and_model(dataset_dir)
            weights = mbirtorch.gen_weights(sino, weight_type='transmission_root')
            recon, recon_dict = model.recon(sino, weights=weights)
    """
    sino, required_params, optional_params = _compute_sino_and_params(
        dataset_dir, downsample_factor=downsample_factor, subsample_view_factor=subsample_view_factor,
        crop_pixels_sides=crop_pixels_sides, crop_pixels_top=crop_pixels_top,
        crop_pixels_bottom=crop_pixels_bottom, verbose=verbose, offset_correction=offset_correction)
    return lu.finalize_model(sino, required_params, optional_params, auto_crop=auto_crop)


def _compute_sino_and_params(dataset_dir, downsample_factor=(1, 1), subsample_view_factor=1,
                             crop_pixels_sides=None, crop_pixels_top=None, crop_pixels_bottom=None, verbose=1, offset_correction=True):
    """
    This function loads NSI scans and computes the sinogram and the model parameters for
    :func:`get_sino_and_model`.  The docstring of :func:`get_sino_and_model` describes the
    arguments.

    Returns:
        tuple: ``(sino, required_params, optional_params)``.  The dict ``required_params``
        holds the ConeBeamModel constructor arguments and ``geometry_type``.  The dict
        ``optional_params`` holds the arguments for ``set_params``.
    """
    if verbose > 0:
        print("\n\n########## Loading object, blank, dark scans, and geometry parameters from NSI dataset directory")
    obj_scan, blank_scan, dark_scan, nsi_params, defective_pixel_array = \
            load_scans_and_params(dataset_dir, subsample_view_factor=subsample_view_factor, verbose=verbose, offset_correction=offset_correction)

    # TODO: Adjust the detector offsets for asymmetric crops.
    max_crop = nsi_params['max_crop']
    if crop_pixels_sides is None:
        crop_pixels_sides = max_crop
    if crop_pixels_top is None:
        crop_pixels_top = max_crop
    if crop_pixels_bottom is None:
        crop_pixels_bottom = max_crop
    if crop_pixels_bottom != crop_pixels_top:
        warnings.warn('Only symmetric cropping is allowed in this release.  Replacing top and bottom crops with their max.')
        crop_pixels_top = max(crop_pixels_top, crop_pixels_bottom)
        crop_pixels_bottom = max(crop_pixels_bottom, crop_pixels_top)

    cone_beam_params, optional_params = convert_nsi_to_mbirtorch_params(nsi_params, downsample_factor=downsample_factor,
                                                                      crop_pixels_sides=crop_pixels_sides,
                                                                      crop_pixels_top=crop_pixels_top,
                                                                      crop_pixels_bottom=crop_pixels_bottom)

    if verbose > 0:
        print("\n\n########## Cropping scans")
    obj_scan, blank_scan, dark_scan, defective_pixel_array = mtp.crop_view_data(obj_scan, blank_scan, dark_scan,
                                                                                       crop_pixels_sides=crop_pixels_sides,
                                                                                       crop_pixels_top=crop_pixels_top,
                                                                                       crop_pixels_bottom=crop_pixels_bottom,
                                                                                       defective_pixel_array=defective_pixel_array)
    obj_scan_shape = obj_scan.shape

    if verbose > 0:
        print("\n\n########## Computing sinogram (downsample -> transmission -> detector rotation, fused)")
    # scan_to_sino applies det_rotation to the sinogram, so the model does not receive it.
    det_rotation = optional_params.pop("det_rotation")
    sino = mtp.scan_to_sino(obj_scan, blank_scan, dark_scan, defective_pixel_array,
                            downsample_factor=downsample_factor, det_rotation=det_rotation)
    del obj_scan, blank_scan, dark_scan  # This frees the memory of the scan images.

    if verbose > 0:
        print("\n\n########## Correcting sinogram data to account for background offset")
    sino = mtp.correct_background_offset(sino, option='per_view')

    if verbose > 0:
        print('cropped obj_scan shape = ', obj_scan_shape)
        print('sinogram shape = ', sino.shape)

    cone_beam_params['geometry_type'] = str(mbirtorch.ConeBeamModel)
    return sino, cone_beam_params, optional_params


def load_scans_and_params(dataset_dir, view_id_start=0, view_id_end=None, subsample_view_factor=1, verbose=1, offset_correction=True):
    """
    This function loads the scans and the geometry from an NSI scan directory.  It loads the
    object, blank, and dark scans.  It also loads the defective pixels and the geometry
    parameters.

    Args:
        dataset_dir (string): Path to an NSI scan directory, which holds:

            - ``*.nsipro`` (NSI config file)
            - ``Geometry*.rtf`` (geometry report)
            - ``Radiographs*/`` (directory containing all radiograph images)
            - ``**/gain0.tif`` (blank scan image)
            - ``**/offset.tif`` (dark scan image)
            - ``**/*.defect`` (defective pixel information)

        view_id_start (int, optional): The index of the first view to load.  Defaults to 0.
        view_id_end (int, optional): The index one past the last view to load.  None means
            the number of views in the ``.nsipro`` config file.  Defaults to None.
        subsample_view_factor (int, optional): This function loads every n-th view, where n
            is this factor.  Defaults to 1.
        verbose (int, optional): The verbosity level.  Defaults to 1.
        offset_correction (bool): If True, this function takes the detector position from the
            geometry report.  Defaults to True.

    Returns:
        tuple: (obj_scan, blank_scan, dark_scan, nsi_params, defective_pixel_array)

            - obj_scan (numpy.ndarray): 3D object scan with shape (num_views, num_det_rows,
              num_det_channels).
            - blank_scan (numpy.ndarray): 3D blank scan with shape (1, num_det_rows,
              num_det_channels).
            - dark_scan (numpy.ndarray): 3D dark scan with shape (1, num_det_rows,
              num_det_channels).
            - nsi_params (dict): The geometry parameters read from the NSI config file.
            - defective_pixel_array (numpy.ndarray | tuple): An (N, 2) integer array of
              (detector_row_idx, detector_channel_idx) pairs.  The value is ``()`` when there
              is no defective pixel file.
    """
    config_file_path, geom_report_path, obj_scan_dir, blank_scan_path, dark_scan_path, defective_pixel_path = \
        _parse_filenames_from_dataset_dir(dataset_dir)

    if verbose > 0:
        print("The following files will be used to compute the NSI reconstruction:\n",
              f"    - NSI config file: {config_file_path}\n",
              f"    - Geometry report: {geom_report_path}\n",
              f"    - Radiograph directory: {obj_scan_dir}\n",
              f"    - Blank scan image: {blank_scan_path}\n",
              f"    - Dark scan image: {dark_scan_path}\n",
              f"    - Defective pixel information: {defective_pixel_path}\n")

    # Each entry is a [tag, section] pair to read from the nsipro file.
    tag_section_list = [['source', 'Result'],                           # origin to source
                        ['reference', 'Result'],                        # first detector pixel
                        ['pitch', 'Object Radiograph'],                 # pixel pitch
                        ['width pixels', 'Detector'],                   # number of detector channels
                        ['height pixels', 'Detector'],                  # number of detector rows
                        ['number', 'Object Radiograph'],                # number of views
                        ['Rotation range', 'CT Project Configuration'], # usually 360 degrees
                        ['rotate', 'Correction'],                       # radiograph rotation
                        ['flipH', 'Correction'],                        # horizontal flip
                        ['flipV', 'Correction'],                        # vertical flip
                        ['angleStep', 'Object Radiograph'],             # view angle step
                        ['clockwise', 'Processed'],                     # rotation direction
                        ['axis', 'Result'],                             # rotation axis
                        ['normal', 'Result'],                           # detector normal
                        ['horizontal', 'Result'],                       # direction of detector rows
                        ['crop', 'Radiograph']                          # pixels to crop per side
                       ]
    assert(os.path.isfile(config_file_path)), f'Error! NSI config file does not exist. Please check whether {config_file_path} is a valid file.'
    # The strings in raw_nsi_fields are in the order of tag_section_list.
    raw_nsi_fields = _read_str_from_config(config_file_path, tag_section_list)

    # r_s is the vector from the origin to the source.
    r_s = raw_nsi_fields[0].split(' ')
    r_s = np.array([np.single(elem) for elem in r_s])

    # r_r is the vector from the origin to the center of the first detector row and column.
    r_r = raw_nsi_fields[1].split(' ')
    r_r = np.array([np.single(elem) for elem in r_r])

    # The geometry report gives a more accurate position of the (0,0) detector pixel.
    if offset_correction:
        if geom_report_path is not None:
            x_r, y_r = _read_detector_location_from_geom_report(geom_report_path)
            r_r[0] = x_r
            r_r[1] = y_r
            if verbose > 0:
                print("Corrected coordinate of (0,0) detector pixel (from Geometry Report) =", r_r)
        else:
            pass
    else:
        if verbose > 0:
            print("Offset correction disabled. Using original r_r =", r_r)

    pixel_pitch_det = raw_nsi_fields[2].split(' ')
    delta_det_channel = np.single(pixel_pitch_det[0])
    delta_det_row = np.single(pixel_pitch_det[1])

    num_det_channels = int(raw_nsi_fields[3])
    num_det_rows = int(raw_nsi_fields[4])

    num_acquired_scans = int(raw_nsi_fields[5])

    # The total rotation is usually 360 degrees.  For 4D data, it is 360 times the number of turns.
    total_angles = int(raw_nsi_fields[6])

    # The radiograph rotation, in degrees.
    scan_rotate = int(raw_nsi_fields[7])
    if (scan_rotate == 180) or (scan_rotate == 0):
        if verbose > 0:
            print('Scans are in portrait mode!')
    elif (scan_rotate == 270) or (scan_rotate == 90):
        if verbose > 0:
            print('Scans are in landscape mode!')
        num_det_channels, num_det_rows = num_det_rows, num_det_channels
    else:
        warnings.warn("Picture mode unknown! Should be either portrait (0 or 180 deg rotation) or landscape (90 or 270 deg rotation). Automatically setting picture mode to portrait.")
        scan_rotate = 180

    if raw_nsi_fields[8] == "True":
        flipH = True
    else:
        flipH = False
    if raw_nsi_fields[9] == "True":
        flipV = True
    else:
        flipV = False

    # The angle step between views, in degrees.
    angle_step = np.single(raw_nsi_fields[10])

    if raw_nsi_fields[11] == "True":
        if verbose > 0:
            print("clockwise rotation.")
    else:
        if verbose > 0:
            print("counter-clockwise rotation.")
        angle_step = -angle_step

    r_a = raw_nsi_fields[12].split(' ')
    r_a = np.array([np.single(elem) for elem in r_a])
    # This code makes the rotation axis point down.
    if r_a[1] > 0:
        r_a = -r_a

    r_n = raw_nsi_fields[13].split(' ')
    r_n = np.array([np.single(elem) for elem in r_n])

    r_h = raw_nsi_fields[14].split(' ')
    r_h = np.array([np.single(elem) for elem in r_h])
    crops = raw_nsi_fields[15].split(' ')
    crops = np.array([np.int32(elem) for elem in crops])
    max_crop = np.amax(crops)

    if verbose > 0:
        print("############ NSI geometry parameters ############")
        print("vector from origin to source = ", r_s, " [mm]")
        print("vector from origin to (0,0) detector pixel = ", r_r, " [mm]")
        print("Unit vector of rotation axis = ", r_a)
        print("Unit vector of normal = ", r_n)
        print("Unit vector of horizontal = ", r_h)
        print(f"Detector pixel pitch: (delta_det_row, delta_det_channel) = ({delta_det_row:.3f},{delta_det_channel:.3f}) [mm]")
        print(f"Detector size: (num_det_rows, num_det_channels) = ({num_det_rows},{num_det_channels})")
        print(f"Pixels to crop from the border of each view = {max_crop}")
        print("############ End NSI geometry parameters ############")

    blank_scan = np.expand_dims(lu.read_tif_img(blank_scan_path), axis=0)
    if dark_scan_path is not None:
        dark_scan = np.expand_dims(lu.read_tif_img(dark_scan_path), axis=0)
    else:
        dark_scan = np.zeros(blank_scan.shape)

    if view_id_end is None:
        view_id_end = num_acquired_scans
    view_ids = np.arange(start=view_id_start, stop=view_id_end, step=subsample_view_factor, dtype=np.int32)
    if verbose > 0:
        print('Loading {} object scans from disk.'.format(len(view_ids)))
    obj_scan = lu.read_tif_stack_dir(obj_scan_dir, view_ids)
    if verbose > 0:
        print('Scans loaded.')

    if defective_pixel_path is not None:
        tag_section_list = [['Defect', 'Defective Pixels']]
        defective_loc = _read_str_from_config(defective_pixel_path, tag_section_list)
        defective_pixel_array = np.array([defective_pixel_ind.split()[1::-1] for defective_pixel_ind in defective_loc ]).astype(int)
    else:
        defective_pixel_array = ()
    num_defective_pixels = len(defective_pixel_array)

    if flipV:
        if verbose > 0:
            print("Flipping scans vertically")
        obj_scan = np.flip(obj_scan, axis=1)
        blank_scan = np.flip(blank_scan, axis=1)
        dark_scan = np.flip(dark_scan, axis=1)
        if num_defective_pixels > 0:
            defective_pixel_array[:, 0] = blank_scan.shape[1] - defective_pixel_array[:, 0] - 1

    if flipH:
        if verbose > 0:
            print("Flipping scans horizontally")
        obj_scan = np.flip(obj_scan, axis=2)
        blank_scan = np.flip(blank_scan, axis=2)
        dark_scan = np.flip(dark_scan, axis=2)
        if num_defective_pixels > 0:
            defective_pixel_array[:, 1] = blank_scan.shape[2] - defective_pixel_array[:, 1] - 1

    rot_count = scan_rotate // 90
    for n in range(rot_count):
        obj_scan = np.rot90(obj_scan, 1, axes=(2,1))
        blank_scan = np.rot90(blank_scan, 1, axes=(2,1))
        dark_scan = np.rot90(dark_scan, 1, axes=(2,1))
        if num_defective_pixels > 0:
            defective_pixel_array = np.fliplr(defective_pixel_array)
            defective_pixel_array[:, 1] = blank_scan.shape[2] - defective_pixel_array[:, 1] - 1

    angles = np.deg2rad(np.array([(view_idx*angle_step) % 360.0 for view_idx in view_ids]))

    nsi_params = {
        'r_a': r_a,
        'r_n': r_n,
        'r_h': r_h,
        'r_s': r_s,
        'r_r': r_r,
        'delta_det_channel': delta_det_channel,
        'delta_det_row': delta_det_row,
        'num_det_channels': num_det_channels,
        'num_det_rows': num_det_rows,
        'angles': angles,
        'max_crop': max_crop
    }

    return obj_scan, blank_scan, dark_scan, nsi_params, defective_pixel_array


def convert_nsi_to_mbirtorch_params(nsi_params, downsample_factor=(1, 1), crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0):
    """
    This function converts NSI geometry parameters into mbirtorch parameters.  The mbirtorch
    parameters account for the crop and the downsampling.

    Args:
        nsi_params (dict): The NSI geometry parameters.
        downsample_factor ((int, int), optional): The downsampling factors along the detector
            rows and channels.  When the number of rows or channels is not divisible by its
            factor, this function rounds the result down.  Defaults to (1, 1).
        crop_pixels_sides (int, optional): The number of pixels to crop from each side of the
            sinogram.  Defaults to 0.
        crop_pixels_top (int, optional): The number of pixels to crop from the top of the
            sinogram.  Defaults to 0.
        crop_pixels_bottom (int, optional): The number of pixels to crop from the bottom of the
            sinogram.  Defaults to 0.

    Returns:
        cone_beam_params (dict): The required parameters for the ConeBeamModel constructor.
        optional_params (dict): The other ConeBeamModel parameters, to be set with set_params().
    """
    r_a, r_n, r_h, r_s, r_r = itemgetter('r_a', 'r_n', 'r_h', 'r_s', 'r_r')(nsi_params)
    delta_det_channel, delta_det_row = itemgetter('delta_det_channel', 'delta_det_row')(nsi_params)
    num_det_channels, num_det_rows, angles = itemgetter('num_det_channels', 'num_det_rows', 'angles')(nsi_params)

    source_detector_dist, source_iso_dist, magnification, det_rotation = calc_source_detector_params(r_a, r_n, r_h, r_s, r_r)
    det_channel_offset, det_row_offset = calc_row_channel_params(r_a, r_n, r_h, r_s, r_r, delta_det_channel, delta_det_row, num_det_channels, num_det_rows, magnification)
    recon_slice_offset = - det_row_offset / magnification

    # The crop is in raw detector pixels.  The offsets are in ALU.
    num_det_rows, num_det_channels, det_row_offset, det_channel_offset = lu.apply_config_crop(
        num_det_rows, num_det_channels, det_row_offset, det_channel_offset, delta_det_row, delta_det_channel,
        crop_pixels_top=crop_pixels_top, crop_pixels_bottom=crop_pixels_bottom, crop_pixels_sides=crop_pixels_sides)

    num_det_rows = num_det_rows // downsample_factor[0]
    num_det_channels = num_det_channels // downsample_factor[1]

    delta_det_row *= downsample_factor[0]
    delta_det_channel *= downsample_factor[1]

    num_views = len(angles)
    cone_beam_params = dict()
    cone_beam_params["sinogram_shape"] = (num_views, num_det_rows, num_det_channels)
    cone_beam_params["angles"] = angles
    cone_beam_params["source_detector_dist"] = source_detector_dist
    cone_beam_params["source_iso_dist"] = source_iso_dist

    optional_params = dict()
    optional_params["delta_det_channel"] = delta_det_channel
    optional_params["delta_det_row"] = delta_det_row
    optional_params['delta_voxel'] = delta_det_channel * (source_iso_dist/source_detector_dist)
    optional_params["det_channel_offset"] = det_channel_offset
    optional_params["det_row_offset"] = det_row_offset
    optional_params['recon_slice_offset'] = recon_slice_offset
    optional_params["det_rotation"] = det_rotation # tilt angle of the rotation axis, in radians
    optional_params["alu_unit"] = 'mm' # NSI files always give lengths in mm.
    optional_params["alu_value"] = 1.0  # 1 ALU is 1 mm.

    return cone_beam_params, optional_params


######## subroutines for parsing NSI metadata
def _parse_filenames_from_dataset_dir(dataset_dir):
    """This function returns the paths of the NSI files in an NSI dataset directory.  When
    several files match one pattern, this function asks the user to choose one.

    Args:
        dataset_dir (string): Path to the directory containing the NSI scans and metadata.
    Returns:
        6-element tuple containing:
            - config_file_path (string): Path to the NSI config file (nsipro file).
            - geom_report_path (string): Path to the geometry report file (Geometry
              Report.rtf), or None when there is no geometry report.
            - obj_scan_dir (string): Path to the directory of object scan images (radiographs).
            - blank_scan_path (string): Path to the blank scan image.
            - dark_scan_path (string): Path to the dark scan image.
            - defective_pixel_path (string): Path to the defective pixel file.
    """
    config_file_path_list = glob.glob(os.path.join(dataset_dir, "*.nsipro"))
    config_file_path = _prompt_user_choice("NSI config files", config_file_path_list)

    geom_report_path_list = glob.glob(os.path.join(dataset_dir, "Geometry*.rtf"))
    if len(geom_report_path_list) == 0:
        print("No Geometry Report found. Skipping offset correction.")
        geom_report_path = None
    else:
        geom_report_path = _prompt_user_choice("Geometry Report", geom_report_path_list)

    obj_scan_dir_list = glob.glob(os.path.join(dataset_dir, "Radiographs*"))
    obj_scan_dir = _prompt_user_choice("radiograph directories", obj_scan_dir_list)

    blank_scan_path_list = glob.glob(os.path.join(dataset_dir, "**/gain0.tif"))
    blank_scan_path = _prompt_user_choice("blank scans", blank_scan_path_list)

    dark_scan_path_list = glob.glob(os.path.join(dataset_dir, "**/offset.tif"))
    dark_scan_path = _prompt_user_choice("dark scans", dark_scan_path_list)

    defective_pixel_path_list = glob.glob(os.path.join(dataset_dir, "**/*.defect"))
    defective_pixel_path = _prompt_user_choice("defective pixel files", defective_pixel_path_list)

    return config_file_path, geom_report_path, obj_scan_dir, blank_scan_path, dark_scan_path, defective_pixel_path


def _prompt_user_choice(file_description, file_path_list):
    """This function asks the user to choose one file from a list of candidate files.  When
    the list has one file, this function returns that file without asking.
    """
    assert(len(file_path_list) > 0), f"No {file_description} found!! Please make sure you provided a valid NSI scan path."

    if len(file_path_list) == 1:
        return file_path_list[0]

    choice_min = 0
    choice_max = len(file_path_list)-1
    question = f"Multiple {file_description} detected. Please select the desired one from the following candidates "
    prompt = ":\n"
    for i in range(len(file_path_list)):
        prompt += f"\n    {i}: {file_path_list[i]}"
    prompt += f"\n[{choice_min}-{choice_max}]"
    while True:
        sys.stdout.write(question + prompt)
        try:
            choice = int(input())
            if choice in range(len(file_path_list)):
                return file_path_list[choice]
            else:
                sys.stdout.write(f"Please respond with a number between {choice_min} and {choice_max}.\n")
        except:
            sys.stdout.write(f"Please respond with a number between {choice_min} and {choice_max}.\n")
    return


def _read_detector_location_from_geom_report(geom_report_path):
    """This function reads the (x, y) position in mm of the first detector row and column
    from the "Image center" line of the geometry report.  This position is more accurate than
    the <reference> field of the nsipro file.

    Returns:
        (x_r, y_r): The X and Y coordinates in mm.
    """
    rtf_file = open(geom_report_path, 'r')
    rtf_raw = rtf_file.read()
    rtf_file.close()
    start_index = rtf_raw.find('Image center')
    end_index = start_index + rtf_raw[start_index:].find('[mm]')
    line = rtf_raw[start_index:end_index+1]
    data = re.findall(r"(\d+\.*\d*, \d+\.*\d*)", line)

    data = data[0].split(",")
    x_r = float(data[0])
    y_r = float(data[1])
    return x_r, y_r


def _read_str_from_config(filepath, tags_sections):
    """This function reads the values of the requested tags from an NSI configuration file
    and returns them as strings.

    Args:
        filepath (string): Path to the NSI configuration file, with extension '.nsipro'.
        tags_sections (list[string,string]): The [tag, section] pairs to read.
    Returns:
        list[string]: The value of each tag, in the order of ``tags_sections``.
    """
    tag_strs = ['<' + tag + '>' for tag, section in tags_sections]
    section_starts = ['<' + section + '>' for tag, section in tags_sections]
    section_ends = ['</' + section + '>' for tag, section in tags_sections]
    NSI_params = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except IOError:
        print("Could not read file:", filepath)

    # TODO: Replace the nested loop with more efficient code.
    for tag_str, section_start, section_end in zip(tag_strs, section_starts, section_ends):
        section_start_inds = [ind for ind, match in enumerate(lines) if section_start in match]
        section_end_inds = [ind for ind, match in enumerate(lines) if section_end in match]
        section_start_ind = section_start_inds[0]
        section_end_ind = section_end_inds[0]

        for line_ind in range(section_start_ind + 1, section_end_ind):
            line = lines[line_ind]
            if tag_str in line:
                tag_ind = line.find(tag_str, 1) + len(tag_str)
                if tag_ind == -1:
                    NSI_params.append("")
                else:
                    NSI_params.append(line[tag_ind:].strip('\n'))

    return NSI_params
######## END subroutines for parsing NSI metadata


######## subroutines for NSI-MBIR parameter conversion
def calc_det_rotation(r_a, r_n, r_h, r_v):
    """This function calculates the tilt angle in radians between the rotation axis and the
    detector columns.  The function ``preprocess.correct_det_rotation()`` rotates the sinogram
    by this angle.

    Args:
        r_a: The 3D unit vector along the rotation axis, pointing down.
        r_n: The 3D unit vector normal to the detector plane, from source to detector.
        r_h: The 3D unit vector along the detector rows, from left to right.
        r_v: The 3D unit vector along the detector columns, pointing down.
    Returns:
        float: The angle between the rotation axis and the detector columns, in radians.
    """
    r_a_p = lu.unit_vector(r_a - lu.project_vector_to_vector(r_a, r_n))
    det_rotation = -np.arctan(np.dot(r_a_p, r_h)/np.dot(r_a_p, r_v))
    return det_rotation


def calc_source_detector_params(r_a, r_n, r_h, r_s, r_r):
    """This function calculates source_detector_dist, source_iso_dist, magnification, and
    det_rotation.

    Args:
        r_a (tuple): The 3D unit vector along the rotation axis, pointing down.
        r_n (tuple): The 3D unit vector normal to the detector plane, from source to detector.
        r_h (tuple): The 3D unit vector along the detector rows, from left to right.
        r_s (tuple): The 3D vector from the origin to the source.
        r_r (tuple): The 3D vector from the origin to the center of the first detector row and
            column.
    Returns:
        4-element tuple containing:
        - **source_detector_dist** (float): The distance from the X-ray source to the detector.
        - **source_iso_dist** (float): The distance from the X-ray source to the center of
          rotation.
        - **magnification** (float): The ratio source_detector_dist / source_iso_dist.
        - **det_rotation** (float): The angle between the rotation axis and the detector
          columns, in radians.
    """
    r_n = lu.unit_vector(r_n)
    r_v = np.cross(r_n, r_h)

    # r_s_r is the vector from the source to the center of rotation, projected onto r_n.
    r_s_r = lu.project_vector_to_vector(-r_s, r_n)

    # r_s_d is the vector from the source to the detector, projected onto r_n.
    r_s_d = lu.project_vector_to_vector(r_r-r_s, r_n)

    source_detector_dist = np.linalg.norm(r_s_d)
    source_iso_dist = np.linalg.norm(r_s_r)
    magnification = source_detector_dist/source_iso_dist
    det_rotation = calc_det_rotation(r_a, r_n, r_h, r_v)
    return source_detector_dist, source_iso_dist, magnification, det_rotation


def calc_row_channel_params(r_a, r_n, r_h, r_s, r_r, delta_det_channel, delta_det_row, num_det_channels, num_det_rows, magnification):
    """This function calculates det_channel_offset and det_row_offset.

    Args:
        r_a (tuple): The 3D unit vector along the rotation axis, pointing down.
        r_n (tuple): The 3D unit vector normal to the detector plane, from source to detector.
        r_h (tuple): The 3D unit vector along the detector rows, from left to right.
        r_s (tuple): The 3D vector from the origin to the source.
        r_r (tuple): The 3D vector from the origin to the center of the first detector row and
            column.
        delta_det_channel (float): The spacing between detector channels.
        delta_det_row (float): The spacing between detector rows.
        num_det_channels (int): The number of detector channels.
        num_det_rows (int): The number of detector rows.
        magnification (float): The magnification of the cone beam geometry.
    Returns:
        2-element tuple containing:
        - **det_channel_offset** (float): The distance from the detector center to the
          source-detector line along a row.
        - **det_row_offset** (float): The distance from the detector center to the
          source-detector line along a column.
    """
    r_n = lu.unit_vector(r_n)
    r_h = lu.unit_vector(r_h)
    r_v = np.cross(r_n, r_h)

    # c_v is the vector from the first row and column to the detector center, along the columns.
    c_v = (num_det_rows-1)/2*delta_det_row*r_v
    # c_h is the vector from the first row and column to the detector center, along the rows.
    c_h = (num_det_channels-1)/2*delta_det_channel*r_h
    # r_s_r is the vector from the source to the first detector row and column.
    r_s_r = r_r - r_s
    # r_delta is the vector from the source-detector line to the detector center.
    r_delta = r_s_r - lu.project_vector_to_vector(r_s_r, r_n) + c_v + c_h
    det_channel_offset = -np.dot(r_delta, r_h)
    det_row_offset = -np.dot(r_delta, r_v)
    r_a = lu.unit_vector(r_a)
    rotation_offset = np.dot(r_s, np.cross(r_n, r_a))
    det_channel_offset += rotation_offset*magnification
    return det_channel_offset, det_row_offset

######## END subroutines for NSI-MBIR parameter conversion

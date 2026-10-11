"""
This module contains private functions that read Xradia/Zeiss scan files in the OLE format
(``.xrm``, ``.txrm``, ``.txm``).  The modules :mod:`mbirtorch.preprocess.zeiss` and
:mod:`mbirtorch.preprocess.zeiss_tct` use these functions.  Each of those modules passes its own
``read_metadata`` function.  Portions of this module are adapted from the DXchange library:
https://github.com/data-exchange/dxchange
"""
import os
import struct
import logging
import numpy as np
import olefile
from pathlib import Path
from ._loader_utils import _normalize_to_float32

logger = logging.getLogger(__name__)


def _check_read(fname):
    """This function checks that ``fname`` is a string with a recognized extension and returns its
    absolute path.

    If the check fails, the function logs an error but does not raise one.
    """
    known_extensions = {
        '.edf', '.tiff', '.tif', '.h5', '.hdf', '.npy', '.nc',
        '.xrm', '.txrm', '.txm', '.xmt', '.nxs'
    }

    if not isinstance(fname, str):
        logger.error('File name must be a string')
    else:
        _, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in known_extensions:
            logger.error('Unknown file extension')

    return os.path.abspath(fname)


def get_index_in_list(input_list, target):
    """
    This function returns the index of ``target`` in ``input_list``, or -1 if ``target`` is not in the list.
    """
    if target in input_list:
        idx = input_list.index(target)
    else:
        idx = -1

    return idx


def _get_ole_data_type(metadata, datatype=None):
    """This function returns the numpy dtype of the image data in a Zeiss OLE file.

    The function uses ``datatype`` if it is given and ``metadata["data_type"]`` otherwise.
    """
    if datatype is None:
        datatype = metadata["data_type"]
    if datatype == 10:
        return np.dtype(np.float32)
    elif datatype == 5:
        return np.dtype(np.uint16)
    else:
        raise Exception("Unsupport XRM datatype: %s" % str(datatype))


def _read_ole_struct(ole, label, struct_fmt):
    """This function returns the struct stored under ``label`` as a tuple, or None if the label is
    absent."""
    value = None
    if ole.exists(label):
        stream = ole.openstream(label)
        data = stream.read()
        value = struct.unpack(struct_fmt, data)
    return value


def _read_ole_value(ole, label, struct_fmt):
    """This function returns the scalar stored under ``label``, or None if the label is absent."""
    value = _read_ole_struct(ole, label, struct_fmt)
    if value is not None:
        value = value[0]
    return value


def _read_ole_arr(ole, label, struct_fmt):
    """This function returns the array stored under ``label``, or None if the label is absent."""
    arr = _read_ole_struct(ole, label, struct_fmt)
    if arr is not None:
        arr = np.array(arr)
    return arr


def _read_ole_image(ole, label, metadata, datatype=None):
    """This function returns the image stored under ``label``.

    The image has shape (num_det_rows, num_det_channels).
    """
    stream = ole.openstream(label)
    data = stream.read()
    data_type = _get_ole_data_type(metadata, datatype)
    data_type = data_type.newbyteorder('<')
    image = np.reshape(
        np.frombuffer(data, data_type),
        (metadata["num_det_rows"], metadata["num_det_channels"], )
    )
    return image


def _read_ole_str(ole, label):
    """This function returns the strings stored under ``label`` as a list, or None if the label is
    absent."""
    str = None
    if ole.exists(label):
        stream = ole.openstream(label)
        data = stream.read()
        str = [name.decode('utf-8') for name in data.split(b'\x00') if name]
    return str


def _log_imported_data(fname, arr):
    """This function logs the shape and dtype of the data read from ``fname``."""
    logger.debug('Data shape & type: %s %s', arr.shape, arr.dtype)
    logger.info('Data successfully imported: %s', fname)


def read_xrm(fname, read_metadata, *, normalize_to_float32=True):
    """
    This function reads one Xradia ``.xrm`` radiograph and its metadata.  Portions of this
    function are adapted from the DXchange library.

    Args:
        fname (str): The path of the file.
        read_metadata (callable): The function that parses the metadata.  ``read_xrm`` calls
            ``read_metadata(ole)``, which returns a dict.
        normalize_to_float32 (bool, optional): If True, convert the image to float32 and
            normalize integer data to ``[0, 1]``.  If False, return the raw image.  Defaults to
            True.

    Returns:
        np.ndarray: The 2D image with shape (num_det_rows, num_det_channels).
        dict: The metadata.
    """
    fname = _check_read(fname)
    try:
        ole = olefile.OleFileIO(fname)
    except IOError:
        print('No such file or directory: %s', fname)
        return False

    metadata = read_metadata(ole)

    stream = ole.openstream("ImageData1/Image1")
    data = stream.read()

    data_type = _get_ole_data_type(metadata)
    data_type = data_type.newbyteorder('<')

    arr = np.reshape(
        np.frombuffer(data, data_type),
        (
            metadata["num_det_rows"],
            metadata["num_det_channels"]
        )
    )

    _log_imported_data(fname, arr)

    if normalize_to_float32:
        arr = _normalize_to_float32(arr)

    ole.close()
    return arr, metadata


def read_xrm_dir(dir_path, read_metadata):
    """
    This function reads all .xrm files in a directory and stacks them into one array.

    The function reads the files in the order the file system lists them.  It collects the
    positions, shifts, and angles of every file into lists in the metadata.  The other metadata
    come from the first file.  Portions of this function are adapted from the DXchange library.

    Args:
        dir_path (str): The path of the directory.
        read_metadata (callable): The function that parses the metadata (see :func:`read_xrm`).

    Returns:
        np.ndarray: The 3D array with shape (num_views, num_det_rows, num_det_channels).
        dict: The metadata.
    """
    dir_path = Path(dir_path)
    files = [p for p in dir_path.iterdir() if p.is_file()]

    # This function reads each projection without normalization and normalizes the whole stack at the end.
    proj0, md0 = read_xrm(str(files[0]), read_metadata, normalize_to_float32=False)
    num_views = len(files)
    num_det_rows, num_det_channels = proj0.shape
    arr = np.empty((num_views, num_det_rows, num_det_channels), dtype=proj0.dtype)
    arr[0] = proj0

    x0 = md0['x_positions'][0]
    y0 = md0['y_positions'][0]
    z0 = md0['z_positions'][0]

    metadata = dict(md0)
    metadata['num_views'] = num_views
    metadata['x_positions'] = [x0]
    metadata['y_positions'] = [y0]
    metadata['z_positions'] = [z0]
    metadata['x_shifts'] = [md0['x_shifts'][0]]
    metadata['y_shifts'] = [md0['y_shifts'][0]]
    # If the scan has no OLE 'Angles' stream, then 'thetas' is None.
    collect_thetas = md0.get('thetas') is not None
    metadata['thetas'] = [md0['thetas'][0]] if collect_thetas else None

    for i, p in enumerate(files[1:], start=1):
        proj, md = read_xrm(str(p), read_metadata, normalize_to_float32=False)
        arr[i] = proj
        metadata['x_positions'].append(md['x_positions'][0])
        metadata['y_positions'].append(md['y_positions'][0])
        metadata['z_positions'].append(md['z_positions'][0])
        metadata['x_shifts'].append(md['x_shifts'][0])
        metadata['y_shifts'].append(md['y_shifts'][0])
        if collect_thetas and md.get('thetas') is not None:
            metadata['thetas'].append(md['thetas'][0])

    _log_imported_data(str(dir_path), arr)

    arr = _normalize_to_float32(arr)

    return arr, metadata

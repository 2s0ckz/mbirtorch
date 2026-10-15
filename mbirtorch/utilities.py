"""This module holds utility functions for mbirtorch users.  The functions generate
phantoms and demo data, and they read and write reconstructions in HDF5 files.  Other
functions copy and build models, download data, save volumes as GIF movies, and join
arrays along an axis.
"""

import os
import shutil
import warnings
from enum import Enum

import numpy as np
import torch

from . import _sharding


def clear_cache(_root=None):
    """This function deletes the ``~/.mbirtorch`` directory and creates it again empty.
    This directory holds the files that mbirtorch keeps on disk, including the torch.compile
    cache.  If ``TORCHINDUCTOR_CACHE_DIR`` moves the cache elsewhere, the function does not
    delete that cache.

    Args:
        _root: The directory to clear in place of ``~/.mbirtorch``.  Tests use this argument.

    Returns:
        str: The path that was cleared.
    """
    root = os.path.expanduser("~/.mbirtorch") if _root is None else str(_root)
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    return root


def _as_float32(mask):
    """This function converts a numpy or torch boolean mask to float32."""
    if torch.is_tensor(mask):
        return mask.to(torch.float32)
    return mask.astype(np.float32)


def add_ellipsoid(current_volume, grids, z_locations, x0, y0, z0, a, b, c,
                  angle=0, intensity=1.0):
    """
    This function adds an ellipsoid to an existing volume.  The volume, the grids, and
    ``z_locations`` must be either all numpy arrays or all torch tensors on one device.

    Args:
        current_volume (ndarray or Tensor): The 3D volume, with axes (rows, cols, slices).
        grids (tuple): The in-plane coordinate grids (x_grid, y_grid), each with shape
            (rows, cols).
        z_locations (ndarray or Tensor): The 1D array of z coordinates of the slices.
        x0, y0, z0 (float): The center of the ellipsoid.
        a, b, c (float): The radii along x, y, and z.
        angle (float): The rotation of the ellipsoid in the xy plane around (x0, y0), in
            degrees.
        intensity (float): The constant value added inside the ellipsoid.

    Returns:
        ndarray or Tensor: The sum of current_volume and the ellipsoid, with the type of the
        input.
    """
    x_grid, y_grid = grids
    # Python floats keep the multiplications in the array library of the grids.
    cos_angle = float(np.cos(np.deg2rad(angle)))
    sin_angle = float(np.sin(np.deg2rad(angle)))
    Xr = cos_angle * (x_grid - x0) + sin_angle * (y_grid - y0)
    Yr = -sin_angle * (x_grid - x0) + cos_angle * (y_grid - y0)

    xy_norm = Xr ** 2 / a ** 2 + Yr ** 2 / b ** 2
    z_norm = (z_locations - z0) ** 2 / c ** 2
    inside = (xy_norm[:, :, None] + z_norm[None, None, :]) <= 1
    return current_volume + intensity * _as_float32(inside)


def _add_shepp_logan_ellipsoids(phantom, grids, z_locations):
    """This function adds the nine ellipsoids of the low dynamic range Shepp-Logan phantom
    to ``phantom``.  The tests compare against stored values that depend on these
    definitions."""
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0, 0, 0.69, 0.92, 0.9, intensity=1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0.0184, 0, 0.6624, 0.874, 0.88, intensity=-0.8)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0.22, 0, 0, 0.41, 0.16, 0.21, angle=108, intensity=-0.2)
    phantom = add_ellipsoid(phantom, grids, z_locations, -0.22, 0, 0, 0.31, 0.11, 0.22, angle=72, intensity=-0.2)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0.35, 0, 0.21, 0.25, 0.5, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0.1, 0, 0.046, 0.046, 0.046, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, -0.1, 0, 0.046, 0.046, 0.046, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, -0.08, -0.605, 0, 0.046, 0.023, 0.02, angle=0, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, -0.605, 0, 0.023, 0.023, 0.02, angle=0, intensity=0.1)
    return phantom


# The semi-axes (rows, cols, slices) of the largest ellipsoid.  They must match the first
# ellipsoid added above.
_MAIN_ELLIPSOID_SEMI_AXES = (0.69, 0.92, 0.9)


def _shepp_logan_attenuation_scale(phantom_shape, target_max_attenuation):
    """This function returns the intensity scale that makes the largest forward projection
    value of the phantom about ``target_max_attenuation``.  The function assumes a voxel
    spacing near 1."""
    longest_path_voxels = max(s * n for s, n in
                              zip(_MAIN_ELLIPSOID_SEMI_AXES, phantom_shape))
    interior_intensity = 0.28  # Approximate average intensity along the center.
    return (target_max_attenuation / longest_path_voxels) / interior_intensity


def _phantom_devices(devices):
    """This function returns the devices that build the phantom.  If ``devices`` is None,
    the function returns every visible CUDA device, up to the number set by
    MBIRTORCH_NUM_DEVICES.
    """
    from .preprocess import _pipeline
    return _pipeline.permitted_devices(devices)


def _phantom_block_rows(band_shape, max_block_gb):
    """This function returns the number of rows in one block when the phantom is built a
    block at a time.  The function allows sixteen bytes of temporary memory per voxel.
    """
    n_rows, n_cols, n_band = band_shape
    band_bytes = n_rows * n_cols * n_band * 4
    num_blocks = max(1, int(np.ceil(4 * band_bytes / (max_block_gb * 1024 ** 3))))
    return max(1, -(-n_rows // num_blocks))          # ceil(n_rows / num_blocks)


def _shepp_logan_band(phantom_shape, slice_range, device, max_block_gb, scale=1.0):
    """This function builds the slices ``[start, stop)`` of the phantom on one device.  The
    function builds one block of rows at a time.

    Returns:
        Tensor: A float32 tensor of shape (rows, cols, stop - start) on ``device``.
    """
    n_rows, n_cols, n_slices = phantom_shape
    start, stop = slice_range
    if stop <= start:
        # When there are more devices than slices, the last devices get no slices.
        return torch.zeros((n_rows, n_cols, 0), dtype=torch.float32, device=device)

    x_axis = torch.linspace(-1, 1, n_rows, dtype=torch.float32, device=device)
    y_axis = torch.linspace(-1, 1, n_cols, dtype=torch.float32, device=device)
    z_band = torch.linspace(-1, 1, n_slices, dtype=torch.float32,
                            device=device)[start:stop]

    # A float32 scale gives each block the same values as a build of the whole phantom.
    scale32 = float(np.float32(scale))
    band = torch.empty((n_rows, n_cols, stop - start), dtype=torch.float32,
                       device=device)
    block_rows = _phantom_block_rows((n_rows, n_cols, stop - start), max_block_gb)
    for row_start in range(0, n_rows, block_rows):
        row_stop = min(row_start + block_rows, n_rows)
        grids = torch.meshgrid(x_axis[row_start:row_stop], y_axis, indexing='ij')
        block = _add_shepp_logan_ellipsoids(
            torch.zeros((row_stop - row_start, n_cols, stop - start),
                        dtype=torch.float32, device=device), grids, z_band)
        if scale32 != 1.0:
            block = block * scale32
        band[row_start:row_stop] = block
    return band


def _gen_shepp_logan_3d_blocked(phantom_shape, device, max_block_gb, scale=1.0):
    """This function builds the phantom on one device, one block of rows at a time.  The
    function returns the phantom as a host numpy array."""
    band = _shepp_logan_band(phantom_shape, (0, phantom_shape[2]), device,
                             max_block_gb, scale)
    phantom = _to_host(band)
    del band                    # This frees the device array.
    return phantom


def _gen_shepp_logan_3d_sharded(phantom_shape, devices, max_block_gb, scale=1.0):
    """This function builds the phantom on several devices and returns it as a host numpy
    array.  Each device builds a contiguous range of slices in its own thread.
    """
    placement = _sharding.Placement(devices, axis=-1, axis_len=phantom_shape[2])
    band_ranges = [slice_range for _, slice_range in placement.shard_ranges()]
    bands = _sharding.run_per_device(
        devices,
        lambda i, device: _shepp_logan_band(phantom_shape, band_ranges[i], device,
                                            max_block_gb, scale))
    shards = _sharding.Shards(bands, placement)
    phantom = _to_host(shards)
    del bands, shards           # This frees the device arrays.
    return phantom


def gen_shepp_logan_3d(phantom_shape, devices=None,
                                              max_block_gb=4.0,
                                              target_max_attenuation=None):
    """
    This function generates a 3D Shepp-Logan phantom with the given shape.  Each device
    builds a contiguous range of slices, and the function returns the phantom as a host numpy
    array.  The result does not depend on the number of devices.

    Args:
        phantom_shape (tuple): The phantom shape (rows, columns, slices).
        devices (sequence of devices, optional): The devices that build the phantom.
            Defaults to None, which uses every visible CUDA device.  If MBIRTORCH_NUM_DEVICES
            is set, it limits the number of devices.
        max_block_gb (float, optional): The approximate largest temporary memory, in GB, that
            one block of rows uses.  Defaults to 4.0.
        target_max_attenuation (float, optional): If given, the function scales the phantom
            so that its largest line integral is about this value for any array shape.  The
            line integrals are the values of its forward projection.  Real -log attenuation
            sinograms have values from about 0 to 6 or 8.  Defaults to None, which leaves the
            phantom unscaled.

    Returns:
        numpy.ndarray: A float32 array of shape ``phantom_shape`` that holds the phantom.
    """
    scale = 1.0 if target_max_attenuation is None \
        else _shepp_logan_attenuation_scale(phantom_shape, target_max_attenuation)
    devices = _phantom_devices(devices)
    if len(devices) > 1:
        return _gen_shepp_logan_3d_sharded(phantom_shape, devices,
                                                max_block_gb, scale)
    return _gen_shepp_logan_3d_blocked(phantom_shape, devices[0],
                                            max_block_gb, scale)


def makedirs(file_path):
    """This function creates the parent directories of ``file_path`` if needed."""
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(parent_dir, exist_ok=True)


def _to_host(array):
    """This function copies a numpy array, a torch tensor on any device, or a Shards to a host
    numpy array."""
    if isinstance(array, _sharding.Shards):
        return array.gather()
    if hasattr(array, 'detach'):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _load_array_hdf5(file_path):
    """This function reads an HDF5 file that holds one dataset.  The function returns
    (array, attributes)."""
    import h5py
    with h5py.File(file_path, "r") as f:
        array_names = [key for key in f.keys()]
        if len(array_names) > 1:
            raise ValueError('More than one array found in {}. Unable to load.'.format(file_path))
        data_name = array_names[0]
        array = f[data_name][()]
        data_dict = dict()
        for name in f[data_name].attrs.keys():
            data_dict[name] = f[data_name].attrs[name]

        return array, data_dict


def load_data_hdf5(file_path):
    """This function is deprecated.  :func:`import_recon_hdf5` replaces it."""
    warnings.warn('load_data_hdf5 is deprecated; use import_recon_hdf5.', FutureWarning, stacklevel=2)
    return _load_array_hdf5(file_path)


def _shard_axis_block(shards, i0, i1):
    """This function returns a host copy of the indices [i0, i1) along the sharded axis of a
    Shards.  The copy includes all indices of the other axes.  The function copies only this
    block from the devices."""
    pl = shards.placement
    ndim = shards.tensors[0].ndim
    axis = pl.axis % ndim
    sizes = [int(t.shape[axis]) for t in shards.tensors]
    starts = np.cumsum([0] + sizes)
    axis_end = int(pl.axis_len) if pl.axis_len is not None else int(starts[-1])
    pieces = []
    for t, s0, s1 in zip(shards.tensors, starts[:-1], starts[1:]):
        lo, hi = max(i0, int(s0)), min(i1, min(int(s1), axis_end))
        if lo < hi:
            sel = [slice(None)] * ndim
            sel[axis] = slice(lo - int(s0), hi - int(s0))
            pieces.append(t[tuple(sel)].detach().cpu().numpy())
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=axis)


def _sharded_host_shape_dtype(shards):
    """This function returns the shape and the numpy dtype that a Shards would have as a host
    array.  The function does not copy the data to the host."""
    pl = shards.placement
    ndim = shards.tensors[0].ndim
    axis = pl.axis % ndim
    shape = [int(v) for v in shards.tensors[0].shape]
    total = sum(int(t.shape[axis]) for t in shards.tensors)
    shape[axis] = int(pl.axis_len) if pl.axis_len is not None else total
    np_dtype = shards.tensors[0][:0].detach().cpu().numpy().dtype
    return tuple(shape), np_dtype


# The number of bytes in one HDF5 write.  Tests reduce this value to force several writes.
_HDF5_SLAB_BYTES = 1 << 30


def _write_hdf5_streaming(file_path, array_name, out_shape, dtype, produce_slab, attributes_dict=None, chunks=None):
    """This function creates an HDF5 dataset and fills it one slab at a time along axis 0.
    A slab is a contiguous range of indices along axis 0.  ``produce_slab(i0, i1)`` returns
    the data written to dset[i0:i1].
    """
    import h5py
    from .view_utils import convert_subdicts_to_strings
    makedirs(file_path)
    with h5py.File(file_path, 'w') as f:
        dset = f.create_dataset(array_name, shape=out_shape, dtype=dtype, chunks=chunks)
        if len(out_shape) == 0:
            dset[...] = produce_slab(0, 0)
        else:
            row_bytes = np.dtype(dtype).itemsize * int(np.prod(out_shape[1:], dtype=np.int64))
            slab = max(1, _HDF5_SLAB_BYTES // max(row_bytes, 1))
            for i in range(0, out_shape[0], slab):
                dset[i:i + slab] = produce_slab(i, min(i + slab, out_shape[0]))
        if isinstance(attributes_dict, dict):
            attributes_dict = convert_subdicts_to_strings(attributes_dict)
            for key, value in attributes_dict.items():
                dset.attrs[key] = value


def _save_array_hdf5(file_path, array, array_name='array', attributes_dict=None):
    """This function writes one array to an HDF5 file as the dataset ``array_name``.  The
    function writes the entries of ``attributes_dict`` as text attributes.  It writes the
    array one slab at a time along axis 0."""
    if isinstance(array, _sharding.Shards):
        out_shape, dtype, produce_slab = _sharded_slab_source(array)
        _write_hdf5_streaming(file_path, array_name, out_shape, dtype, produce_slab, attributes_dict)
        return

    array = _to_host(array) if hasattr(array, 'detach') else array

    def produce_slab(i0, i1):
        return np.asarray(array) if array.ndim == 0 else np.ascontiguousarray(array[i0:i1])

    _write_hdf5_streaming(file_path, array_name, array.shape, array.dtype, produce_slab, attributes_dict)


def save_data_hdf5(file_path, array, array_name='array', attributes_dict=None):
    """This function is deprecated.  :func:`export_recon_hdf5` replaces it."""
    warnings.warn('save_data_hdf5 is deprecated; use export_recon_hdf5.', FutureWarning, stacklevel=2)
    _save_array_hdf5(file_path, array, array_name, attributes_dict)


def _sharded_slab_source(shards):
    """This function returns the shape, the numpy dtype, and the function ``produce_slab``
    for a Shards.  ``produce_slab(i0, i1)`` returns ``_to_host(shards)[i0:i1]``.
    ``produce_slab`` reads only the shard data that this range needs.
    """
    pl = shards.placement
    ndim = shards.tensors[0].ndim
    axis = pl.axis % ndim
    sizes = [int(t.shape[axis]) for t in shards.tensors]
    starts = np.cumsum([0] + sizes)
    axis_len = int(starts[-1])

    out_shape = list(shards.tensors[0].shape)
    out_shape[axis] = axis_len
    out_shape = tuple(out_shape)
    dtype = shards.tensors[0][:0].detach().cpu().numpy().dtype

    if axis == 0:
        def produce_slab(i0, i1):
            parts = []
            for t, s0, s1 in zip(shards.tensors, starts[:-1], starts[1:]):
                lo, hi = max(i0, int(s0)), min(i1, int(s1))
                if lo < hi:
                    parts.append(t[lo - int(s0):hi - int(s0)].detach().cpu().numpy())
            return np.ascontiguousarray(np.concatenate(parts, axis=0))
    else:
        def produce_slab(i0, i1):
            parts = [t[i0:i1].detach().cpu().numpy() for t in shards.tensors]
            return np.ascontiguousarray(np.concatenate(parts, axis=axis))

    return out_shape, dtype, produce_slab


# The reconstruction file holds the dataset 'recon' with axes (slice, col, row), or
# (time, slice, col, row) for 4D.  The attribute 'axes' states the axis order.
_RECON_FORMAT = 'mbirtorch_recon_v2'
_RECON_AXES = {3: 'slice,col,row', 4: 'time,slice,col,row'}
_RECON_TRANSPOSE = {3: (2, 1, 0), 4: (0, 3, 2, 1)}


def _json_default(value):
    """This function converts a value that json cannot encode.  The function handles arrays,
    tensors, numpy scalars, sets, and tuples.  It converts any other value to text."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {'__ndarray__': value.tolist(), 'dtype': str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _json_object_hook(obj):
    if len(obj) == 2 and '__ndarray__' in obj and 'dtype' in obj:
        return np.asarray(obj['__ndarray__'], dtype=obj['dtype'])
    return obj


def _recon_attributes(recon_dict, ndim):
    """This function returns the attributes of a reconstruction file.  The attributes are the
    format name, the axis order, the package version, and one attribute for each entry of
    recon_dict."""
    import json
    from . import __version__
    attrs = {'format': _RECON_FORMAT, 'axes': _RECON_AXES[ndim], 'mbirtorch_version': __version__}
    for key, value in (recon_dict or {}).items():
        if isinstance(value, dict):
            attrs[key] = json.dumps(value, default=_json_default)
        elif isinstance(value, str):
            attrs[key] = value
        else:
            attrs[key] = str(value)
    return attrs


def export_recon_hdf5(file_path, recon, recon_dict=None, remove_flash=False, radial_margin=10, top_margin=10, bottom_margin=10):
    """
    This function writes a reconstruction and its recon_dict to an HDF5 file.

    The file stores the volume as the dataset ``recon``.  The axes are (slice, col, row) for
    3D, or (time, slice, col, row) for 4D.  The attribute ``axes`` states the axis order.
    :func:`import_recon_hdf5` reads the file, and the slice viewers open it.

    Args:
        file_path (str): The path of the output file.  The function creates missing
            directories.
        recon (ndarray, tensor, or Shards): The volume with axes (row, col, slice), or a 4D
            volume with axes (time, row, col, slice).  If the volume is a Shards divided by
            slice, the function writes it from the devices without a full host copy.
        recon_dict (dict, optional): The dict returned by
            :meth:`~mbirtorch.TomographyModel.recon`.
        remove_flash (bool, optional): If True, the function sets to zero the voxels outside
            a cylinder and the top and bottom slices before it writes.  Defaults to False.
        radial_margin (int, optional): The number of voxels removed from the cylinder radius.
            Defaults to 10.
        top_margin (int, optional): The number of slices set to zero at the top.  Defaults
            to 10.
        bottom_margin (int, optional): The number of slices set to zero at the bottom.
            Defaults to 10.

    Example:
        >>> recon, recon_dict = ct_model.recon(sinogram)
        >>> mbirtorch.export_recon_hdf5('output/recon.h5', recon, recon_dict)
    """
    from . import preprocess

    def mask(block, s0, s1):
        # block holds slices [s0, s1) with axes (row, col, slice).
        ds = s1 - s0
        local_top = min(max(top_margin - s0, 0), ds)
        local_bottom = min(max(s1 - (num_slices - bottom_margin), 0), ds)
        return preprocess.apply_cylindrical_mask(block, radial_margin, local_top, local_bottom)

    # A 3D Shards divided by slice is written directly from the devices.  Other inputs are
    # copied to the host first.
    if (isinstance(recon, _sharding.Shards) and recon.tensors[0].ndim == 3
            and recon.placement.axis % 3 == 2):
        (num_rows, num_cols, num_slices), np_dtype = _sharded_host_shape_dtype(recon)
        ndim = 3

        def get_block(s0, s1):
            return _shard_axis_block(recon, s0, s1)
    else:
        recon = _to_host(recon)
        np_dtype = recon.dtype
        ndim = recon.ndim
        if ndim == 3:
            num_rows, num_cols, num_slices = recon.shape

            def get_block(s0, s1):
                return recon[:, :, s0:s1]
        elif ndim == 4:
            num_times, num_rows, num_cols, num_slices = recon.shape
        else:
            raise ValueError(f'recon must be a 3D or 4D volume; got shape {recon.shape}')

    if ndim == 3:
        def produce_slab(s0, s1):
            block = get_block(s0, s1)
            if remove_flash:
                block = mask(block, s0, s1)
            # (ds, cols, rows)
            return np.ascontiguousarray(np.transpose(block, _RECON_TRANSPOSE[3]))

        out_shape = (num_slices, num_cols, num_rows)
        chunks = (1, num_cols, num_rows)
    else:
        def produce_slab(t0, t1):
            block = recon[t0:t1]
            if remove_flash:
                block = np.stack([mask(frame, 0, num_slices) for frame in block])
            # (dt, slices, cols, rows)
            return np.ascontiguousarray(np.transpose(block, _RECON_TRANSPOSE[4]))

        out_shape = (num_times, num_slices, num_cols, num_rows)
        chunks = (1, 1, num_cols, num_rows)

    _write_hdf5_streaming(file_path, 'recon', out_shape, np_dtype, produce_slab,
                          _recon_attributes(recon_dict, ndim), chunks=chunks)


def _resolve_geometry_class(geometry_type):
    """This function returns the model class that a ``geometry_type`` string names."""
    import mbirtorch
    geometry_type = str(geometry_type)
    for name in ('ConeBeamModel', 'MultiAxisParallelBeamModel',
                 'MultiAxisParallelModel', 'ParallelBeamModel',
                 'TranslationModel'):
        if name in geometry_type:
            model_class = getattr(mbirtorch, name, None)
            if model_class is None:
                raise ValueError(f"The geometry class {name} is not available in mbirtorch yet.")
            return model_class
    raise ValueError(f"Cannot resolve a model class for geometry_type {geometry_type!r}.")


# The reconstruction geometry parameters that ``auto_set_recon_geometry`` sets.
_RECON_GEOMETRY_NAMES = ('recon_shape', 'delta_voxel', 'recon_slice_offset')


def _is_parallel_beam(model):
    return type(model).__name__ == 'ParallelBeamModel'


def _recon_shape_at_pitch(recon_shape, automatic_pitch, pitch, slices_are_rows=False):
    """This function rescales ``recon_shape`` from the voxel pitch ``automatic_pitch`` to the
    voxel pitch ``pitch``.  The new shape covers the same physical extent.  If
    ``slices_are_rows`` is True, the number of slices does not change."""
    if np.isclose(pitch, automatic_pitch):
        return tuple(int(n) for n in recon_shape)
    scaled = [int(np.ceil(n * automatic_pitch / pitch)) for n in recon_shape]
    if slices_are_rows:
        scaled[2] = int(recon_shape[2])
    return tuple(scaled)


def build_model(required_params, optional_params=None, regularization=None):
    """
    This function constructs a model from the parameter dicts returned by
    ``get_all_params``.  The model class comes from ``geometry_type``.  The function keeps
    the reconstruction geometry parameters that the dicts contain.
    ``auto_set_recon_geometry`` sets the other reconstruction geometry parameters.

    Args:
        required_params (dict): The arguments of the model constructor, including
            ``geometry_type``.  This dict is the first element returned by
            ``get_all_params``.
        optional_params (dict, optional): Additional parameters that the function applies
            with ``set_params``, such as detector pitches, offsets, ``delta_voxel``, and
            ``recon_shape``.  Defaults to None.
        regularization (dict, optional): Regularization parameters that the function applies
            with ``set_params``.  Defaults to None, which keeps the default regularization of
            the model.

    Returns:
        TomographyModel: The constructed model.
    """
    required_params = dict(required_params)
    model_class = _resolve_geometry_class(required_params.pop('geometry_type'))
    model = model_class(**required_params)

    optional_params = dict(optional_params) if optional_params else {}
    # The given reconstruction geometry must be applied after auto_set_recon_geometry,
    # because auto_set_recon_geometry would overwrite it.
    supplied = {name: optional_params.pop(name) for name in _RECON_GEOMETRY_NAMES if name in optional_params}
    # set_params raises an error for a misspelled optional parameter.
    if optional_params:
        model.set_params(**optional_params)
    if regularization:
        model.set_params(no_warning=True, **regularization)
    model.auto_set_recon_geometry()
    if 'delta_voxel' in supplied and 'recon_shape' not in supplied:
        supplied['recon_shape'] = _recon_shape_at_pitch(
            model.get_params('recon_shape'), float(model.get_params('delta_voxel')),
            float(supplied['delta_voxel']), slices_are_rows=_is_parallel_beam(model))
    if supplied:
        model.set_params(no_warning=True, **supplied)
    return model


def download_and_extract(download_url, save_dir):
    """
    This function downloads a file from a URL, or copies it from a local path, into
    ``save_dir``.  If the file is a tar archive, the function extracts it into ``save_dir``.
    If the file is already in ``save_dir``, the function does not download it again.

    Args:
        download_url (str): A Google Drive shared link, an HTTP or HTTPS URL, or a local path.
        save_dir (str): The directory where the function saves and extracts the file.

    Returns:
        str: For a tar file, the path of the extracted top-level directory.  For other files,
        the path of the downloaded or copied file.

    Raises:
        RuntimeError: If the file cannot be downloaded, copied, or extracted.
        ValueError: If the Google Drive URL is invalid, or if the tar file has no top-level
            directory.

    Examples:
        >>> extracted_dir = download_and_extract("https://example.com/data.tar.gz", "./data")
        >>> result = download_and_extract("/path/to/local/data.tar.gz", "./data")
    """
    import re
    import subprocess
    import tarfile
    import urllib.error
    import urllib.request
    from urllib.parse import urlparse

    def is_google_drive_url(url):
        """This function returns True if ``url`` is a Google Drive link."""
        return "drive.google.com" in url

    def is_tar_file(filename):
        """This function returns True if ``filename`` has a tar archive extension."""
        tar_extensions = ['.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz']
        return any(filename.lower().endswith(ext) for ext in tar_extensions)

    def extract_google_drive_id(url):
        """This function returns the Google Drive file ID in a URL."""
        pattern = r"(?:https?:\/\/)?(?:www\.)?drive\.google\.com\/(?:file\/d\/|open\?id=)([a-zA-Z0-9_-]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        else:
            raise ValueError("Invalid Google Drive URL format")

    parsed = urlparse(download_url)
    is_url = parsed.scheme in ('http', 'https')
    is_google_drive = is_url and is_google_drive_url(download_url)

    if is_google_drive:
        file_id = extract_google_drive_id(download_url)
        marker_file = os.path.join(save_dir, f".gdrive_{file_id}")

        if os.path.exists(marker_file):
            with open(marker_file, 'r') as f:
                actual_filename = f.read().strip()
            file_path = os.path.join(save_dir, actual_filename)
            filename = actual_filename

            if os.path.exists(file_path):
                is_download = False
            else:
                is_download = True

        else:
            filename = f"gdrive_{file_id}"
            is_download = True
    else:
        filename = os.path.basename(parsed.path if is_url else download_url)
        file_path = os.path.join(save_dir, filename)
        if os.path.exists(file_path):
            is_download = False
        else:
            is_download = True

    if is_download:
        os.makedirs(save_dir, exist_ok=True)

        if is_url:
            if is_google_drive:
                print("Downloading file from Google Drive...")
                import gdown
                try:
                    gdrive_url = f"https://drive.google.com/uc?id={file_id}"

                    downloaded_path = gdown.download(gdrive_url, output=None, quiet=False)
                    if downloaded_path and os.path.isfile(downloaded_path):
                        actual_filename = os.path.basename(downloaded_path)
                        target_path = os.path.join(save_dir, actual_filename)
                        shutil.move(downloaded_path, target_path)
                        file_path = target_path
                        filename = actual_filename

                        with open(marker_file, 'w') as f:
                            f.write(actual_filename)
                    else:
                        raise RuntimeError("Google Drive download failed or returned invalid path")

                    print(f"Download successful! File saved to {file_path}")
                except Exception as e:
                    raise RuntimeError(f"Google Drive download failed: {str(e)}")
            else:
                print("Downloading file...")
                try:
                    urllib.request.urlretrieve(download_url, file_path)
                except urllib.error.HTTPError as e:
                    if e.code == 401:
                        raise RuntimeError(f'HTTP {e.code}: authentication failed!')
                    elif e.code == 403:
                        raise RuntimeError(f'HTTP {e.code}: URL forbidden!')
                    elif e.code == 404:
                        raise RuntimeError(f'HTTP {e.code}: URL not found!')
                    else:
                        raise RuntimeError(f'HTTP {e.code}: {e.reason}')
                except urllib.error.URLError as e:
                    res = subprocess.run(
                        ["curl", "-L", "--fail", "-o", file_path, download_url],
                        capture_output=True, text=True
                    )
                    if res.returncode != 0:
                        raise RuntimeError(f"Download failed with curl: {res.stderr.strip() or res.stdout.strip()}")
                print(f"Download successful! File saved to {file_path}")
        else:
            print(f"Copying local file from {download_url} to {file_path}...")
            if not os.path.isfile(download_url):
                raise RuntimeError(f"Provided file path does not exist: {download_url}")
            shutil.copy2(download_url, file_path)
            print(f"Copy successful! File saved to {file_path}")

        if is_tar_file(filename):
            print(f"Extracting tarball file to {save_dir}...")
            try:
                with tarfile.open(file_path, 'r') as tar_file:
                    tar_file.extractall(save_dir)
                print(f"Extraction successful!")

                top_level_dir = get_top_level_tar_dir(file_path)
                extracted_path = os.path.join(save_dir, top_level_dir)
                return extracted_path
            except Exception as e:
                raise RuntimeError(f"Failed to extract tar file: {str(e)}")
        else:
            return file_path

    if is_google_drive and not is_download:
        try:
            with open(marker_file, 'r') as f:
                actual_filename = f.read().strip()
            file_path = os.path.join(save_dir, actual_filename)
            filename = actual_filename
        except:
            file_path = os.path.join(save_dir, filename)

    if is_tar_file(filename):
        top_level_dir = get_top_level_tar_dir(file_path)
        file_path = os.path.join(save_dir, top_level_dir)

    return file_path


def get_top_level_tar_dir(tar_path, max_entries=1):
    """
    This function returns the name of the top-level directory in a tar file.  The function
    reads at most ``max_entries`` members of the archive.

    Args:
        tar_path (str): The path to the tar file.
        max_entries (int): The maximum number of members to read.

    Returns:
        str: The name of the top-level directory.
    """
    import tarfile
    top_levels = set()

    with tarfile.open(tar_path, 'r') as tar:
        for i, member in enumerate(tar):
            if not member.name.strip():
                continue
            top_dir = member.name.split('/')[0]
            top_levels.add(top_dir)

            if len(top_levels) > 1 or i + 1 >= max_entries:
                break
    if len(top_levels) == 1:
        dir_name = top_levels.pop()
    else:
        raise ValueError("No top level directory found in {}".format(tar_path))
    return dir_name


def save_volume_as_gif(volume, filename, frame_axis=None, slice_axis=None, slice_index=None,
                       vmin=None, vmax=None, fps=5):
    """
    This function saves a 3D or 4D volume as an animated GIF.  Each frame of the GIF is
    one index along a chosen axis.

    Each frame shows the two remaining axes.  The axis with the lower number is vertical, as
    in :func:`mbirtorch.view_utils.slice_viewer`.  The function reduces a 4D volume to 3D by
    holding ``slice_axis`` at ``slice_index``.  With the default arguments, a 4D volume
    (num_times, nx, ny, nz) plays over time at the middle x slice.  A 3D volume plays over x.

    Args:
        volume (numpy): A 3D array (nx, ny, nz) or a 4D array (num_times, nx, ny, nz).
        filename (str): The output path of the GIF file.
        frame_axis (int, optional): The axis that the movie steps through, numbered as in
            ``volume``.  Negative values count from the end.  Defaults to None, which means
            axis 0.  If ``slice_axis`` is 0, the default is axis 1.
        slice_axis (int, optional): The axis held fixed to reduce a 4D volume to 3D.
            Negative values count from the end.  It must differ from ``frame_axis``.
            Defaults to None, which means axis 1 (x).  It is not allowed for a 3D volume.
        slice_index (int, optional): The index along ``slice_axis``.  Defaults to None, which
            means the middle of that axis.
        vmin (float, optional): The pixel value shown as black.  Defaults to None, which
            means the minimum over the frames shown.  The whole movie uses the same display
            range.
        vmax (float, optional): The pixel value shown as white.  Defaults to None, which
            means the maximum over the frames shown.
        fps (float, optional): The number of frames per second in the GIF.  Defaults to 5.
            A GIF stores frame times in hundredths of a second, so the function rounds the
            frame time.

    Raises:
        ValueError: If ``volume`` is not 3D or 4D, if ``frame_axis`` and ``slice_axis`` are
            the same axis, if ``slice_axis`` or ``slice_index`` is given for a 3D volume, or
            if ``fps`` is not positive.
        IndexError: If an axis or an index is out of range for ``volume``.

    Example:
        >>> # A 3D reconstruction, scaled to its own data range.
        >>> mbirtorch.save_volume_as_gif(recon, 'recon.gif')
        >>> # A 4D reconstruction: the middle x slice, playing over time.
        >>> mbirtorch.save_volume_as_gif(recon_4d, 'recon_4d.gif', vmax=0.06)
        >>> # A 4D reconstruction: an XY plane at the middle z, playing over time.
        >>> mbirtorch.save_volume_as_gif(recon_4d, 'recon_4d_xy.gif', slice_axis=3)
        >>> # A single time frame, stepping through z.
        >>> mbirtorch.save_volume_as_gif(recon_4d, 'frame0_z.gif', frame_axis=3, slice_axis=0,
        ...                              slice_index=0)
    """

    def _save_frames_as_gif(frames, filename, titles, vmin, vmax, fps):
        """This function writes a stack of 2D frames as an animated GIF.  Axis 0 indexes the
        frames.  Each frame gets its label from ``titles``."""
        if vmin is None or vmax is None:
            # The display range is set from the frames shown.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)  # A frame of all NaN gives a warning.
                data_min, data_max = float(np.nanmin(frames)), float(np.nanmax(frames))
            if not (np.isfinite(data_min) and np.isfinite(data_max)):
                data_min, data_max = 0.0, 1.0   # The frames have no finite values.
            vmin = data_min if vmin is None else vmin
            vmax = data_max if vmax is None else vmax
        if vmin == vmax:
            # A constant volume would give imshow a display range of zero width.
            scale = max(1e-6 * abs(vmax), 1e-6)
            vmin, vmax = vmin - scale, vmax + scale

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from PIL import Image

        # The whole movie uses one figure, because a GIF needs frames of equal size.
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        image_artist = ax.imshow(frames[0], cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        title_artist = ax.set_title('')
        images = []
        for i, frame in enumerate(frames):
            image_artist.set_data(frame)
            title_artist.set_text(titles[i])
            canvas.draw()
            # The next draw reuses the RGBA buffer, so the code copies each frame.
            buf = canvas.get_renderer().buffer_rgba()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(
                canvas.get_width_height()[::-1] + (4,))
            images.append(Image.fromarray(image[..., :3].copy(), mode='RGB'))
        plt.close(fig)

        makedirs(filename)
        images[0].save(filename, save_all=True, append_images=images[1:],
                       duration=round(1000 / fps), loop=0)

    volume = np.asarray(volume)
    if volume.ndim not in (3, 4):
        raise ValueError('volume must be 3D (nx, ny, nz) or 4D (num_times, nx, ny, nz); '
                         'got shape {}.'.format(volume.shape))
    if fps <= 0:
        raise ValueError('fps must be positive; got {}.'.format(fps))

    # The code converts only a negative axis that is in range.  numpy then raises an error
    # for an axis that is out of range.
    if frame_axis is not None and -volume.ndim <= frame_axis < 0:
        frame_axis += volume.ndim
    if slice_axis is not None and -volume.ndim <= slice_axis < 0:
        slice_axis += volume.ndim

    axis_names = ('x', 'y', 'z') if volume.ndim == 3 else ('t', 'x', 'y', 'z')

    if volume.ndim == 3:
        if slice_axis is not None or slice_index is not None:
            raise ValueError('slice_axis and slice_index apply only to a 4D volume; fixing '
                             'an axis of a 3D volume would leave a single image, not a movie.')
        if frame_axis is None:
            frame_axis = 0
        frames = np.moveaxis(volume, frame_axis, 0)
        titles = ['{} = {}'.format(axis_names[frame_axis], i) for i in range(len(frames))]
    else:
        if slice_axis is None:
            slice_axis = 1
        if frame_axis is None:
            frame_axis = 0 if slice_axis != 0 else 1
        if frame_axis == slice_axis:
            raise ValueError('frame_axis and slice_axis must differ; both are {} ({}).'
                             .format(frame_axis, axis_names[frame_axis]))
        num_slices = volume.shape[slice_axis]
        if slice_index is None:
            slice_index = num_slices // 2

        # Removing slice_axis lowers the number of each axis above it by one.
        index = [slice(None)] * 4
        index[slice_axis] = slice_index
        frames = np.moveaxis(volume[tuple(index)], frame_axis - (frame_axis > slice_axis), 0)
        titles = ['{} slice = {}, {} = {}'.format(axis_names[slice_axis], slice_index,
                                                  axis_names[frame_axis], i)
                  for i in range(len(frames))]

    _save_frames_as_gif(frames, filename, titles, vmin, vmax, fps)


def stitch_arrays(array_list, overlap, axis=2, ramp_overlap=None):
    """
    This function concatenates arrays along one axis.  Adjacent arrays overlap by a fixed
    number of elements, and the function blends the overlap with linear weights.

    The inputs may be numpy arrays, torch tensors on one device, or a mix of the two.  The
    function does not accept tensors on several devices or ``Shards`` containers.  All
    dimensions other than ``axis`` must match.

    Args:
        array_list (list of ndarray or tensor): The two or more arrays to join.
        overlap (int): The number of elements that adjacent arrays share.  It must not exceed
            the length of any input along `axis`.
        axis (int, optional): The axis along which to join.  Defaults to 2.
        ramp_overlap (int, optional): The target number of blended elements, whose weights
            are strictly between 0 and 1.  Defaults to None.

    Returns:
        ndarray or tensor: The joined array.  It is a numpy array for numpy input.
        Otherwise it is a tensor on the device of the input tensors.  Its length along `axis`
        is ``sum(len_k) - (len(array_list) - 1) * overlap``.

    Raises:
        ValueError: If fewer than two arrays are given, if the dimensions other than `axis`
            differ, if any array is shorter than `overlap` along `axis`, or if the tensors
            are not all on one device.
        TypeError: If any element is a ``Shards`` container.

    Example:
        >>> import numpy as np
        >>> a0 = np.arange(2*2*5.).reshape(2, 2, 5)
        >>> a1 = np.arange(2*2*6.).reshape(2, 2, 6)
        >>> out = stitch_arrays([a0, a1], overlap=3, axis=2)
        >>> out.shape
        (2, 2, 8)
    """
    import torch

    if not isinstance(array_list, list) or len(array_list) < 2:
        raise ValueError('array_list must be a list of 2 or more arrays.')

    for array in array_list:
        if isinstance(array, _sharding.Shards):
            raise TypeError(
                'stitch_arrays does not accept an array in the divided device form.  Gather the '
                'shards to the host first with shards.gather(), then stitch the host arrays.')

    # The caller must choose the device, because a whole volume may not fit on one GPU.
    tensor_devices = {array.device for array in array_list if isinstance(array, torch.Tensor)}
    if len(tensor_devices) > 1:
        found = ', '.join(sorted(str(device) for device in tensor_devices))
        raise ValueError(
            'stitch_arrays requires all tensors in array_list to be on one device, but found '
            'tensors on {}.  Move them to a common device before calling, for example with '
            'array.to(device).'.format(found))

    for dim in range(array_list[0].ndim):
        lengths = [array.shape[dim] for array in array_list]
        if dim != axis:
            if np.amax(lengths) != np.amin(lengths):
                raise ValueError('The shapes of the arrays in array_list must be the same except in the dimension specified by axis.')
        if dim == axis:
            if np.amin(lengths) < overlap:
                raise ValueError('Each array must have length at least overlap in the dimension specified by axis.')

    # ramp_overlap is the target number of elements with a weight strictly between 0 and 1.
    # The code gives it the parity of overlap, so the two flat parts have equal length.
    if ramp_overlap is None:
        ramp_overlap = overlap // 2
    ramp_overlap = min(ramp_overlap, overlap)
    ramp_overlap -= (overlap - ramp_overlap) % 2
    ramp_overlap = max(ramp_overlap, overlap % 2)
    flat_pad = (overlap - ramp_overlap) // 2

    # The weights are float32 in the array library of the inputs, so a float32 input stays
    # float32.
    is_torch = any(isinstance(a, torch.Tensor) for a in array_list)
    if is_torch:
        device = next(iter(tensor_devices))  # The check above ensures exactly one device.
        ramp = (torch.arange(ramp_overlap, dtype=torch.float32, device=device) + 1) / (ramp_overlap + 1)
        weights = torch.cat([torch.zeros(flat_pad, dtype=torch.float32, device=device), ramp,
                             torch.ones(flat_pad, dtype=torch.float32, device=device)])
        swap, cat = torch.swapaxes, torch.cat
        array_list = [torch.as_tensor(a, device=device) for a in array_list]
    else:
        ramp = (np.arange(ramp_overlap, dtype=np.float32) + 1) / (ramp_overlap + 1)
        weights = np.concatenate([np.zeros(flat_pad, dtype=np.float32), ramp,
                                  np.ones(flat_pad, dtype=np.float32)])
        swap, cat = np.swapaxes, np.concatenate

    weights_shape = [1] * array_list[0].ndim
    weights_shape[0] = len(weights)
    weights = weights.reshape(weights_shape)

    stitched = swap(array_list[0], 0, axis)

    for next_array in array_list[1:]:
        overlap_current = stitched[-overlap:]
        next_array = swap(next_array, 0, axis)
        overlap_next = next_array[:overlap]

        weighted_overlap = (1 - weights) * overlap_current + weights * overlap_next

        stitched = cat([stitched[:-overlap], weighted_overlap], 0)

        stitched = cat([stitched, next_array[overlap:]], 0)

    return swap(stitched, 0, axis)


def _automatic_recon_geometry(required, optional, regularization):
    """This function returns the (recon_shape, recon_slice_offset) that
    ``auto_set_recon_geometry`` sets for a model built from these parameter dicts.  The
    offset is None for a geometry that has no ``recon_slice_offset``."""
    optional = {name: value for name, value in optional.items() if name not in _RECON_GEOMETRY_NAMES}
    optional['verbose'] = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reference = build_model(dict(required), optional, regularization)
    recon_shape = tuple(int(n) for n in reference.get_params('recon_shape'))
    has_offset = 'recon_slice_offset' in reference.get_all_params()[1]
    offset = float(reference.get_params('recon_slice_offset')) if has_offset else None
    return recon_shape, offset


def copy_ct_model(ct_model, new_angles=None, new_helical_z_shifts=None, new_num_det_rows=None, new_num_det_cols=None,
                  new_translation_vectors=None, no_warning=False):
    """
    This function creates a model with the same type and parameters as ct_model.  The new
    model can have new view parameters and a matching sinogram shape.  The function supports
    ParallelBeamModel, ConeBeamModel, MultiAxisParallelModel, and TranslationModel.

    The copy keeps the voxel pitch and the aspect ratios of ct_model.  Along each axis whose
    inputs did not change, the copy also keeps ``recon_shape`` and ``recon_slice_offset``.
    Along each axis whose detector size or helical travel changed, ``auto_set_recon_geometry``
    sets these values again.  If ct_model has devices set with configure_devices, the copy
    uses the same devices.  The copy also keeps ``compile_mode`` and ``view_batch_size``.

    Args:
        ct_model (TomographyModel): The model to copy.
        new_angles (ndarray of float, optional): The projection angles in radians.  For
            ParallelBeamModel and ConeBeamModel, this is a 1D vector.  For
            MultiAxisParallelModel, this is a (num_views, 2) array of (azimuth, elevation)
            pairs.  It is not allowed for TranslationModel.  If None, the copy uses the angles
            of ct_model.  Defaults to None.
        new_helical_z_shifts (ndarray of float, optional): A 1D vector of the axial shift of
            each view in ALU, for ConeBeamModel.  The function ignores it for
            ParallelBeamModel and MultiAxisParallelModel.  It is not allowed for
            TranslationModel.  Defaults to None.
        new_num_det_rows (int, optional): The number of detector rows in the new model.  If
            None, the copy uses the num_det_rows of ct_model.  Defaults to None.
        new_num_det_cols (int, optional): The number of detector columns in the new model.
            If None, the copy uses the num_det_cols of ct_model.  Defaults to None.
        new_translation_vectors (ndarray of float, optional): A (num_views, 3) array of object
            translations (x, y, z) in ALU.  It applies only to TranslationModel.  If None, the
            copy uses the translation_vectors of ct_model.  Defaults to None.
        no_warning (bool, optional): If True, the function gives no warning when the copy
            replaces reconstruction geometry that was set by hand.  Defaults to False.

    Returns:
        An instance of the same model class as ct_model.

    Raises:
        ValueError: If a view parameter is given that the geometry of the model does not use.
    """
    model_name = str(type(ct_model))
    is_cone = model_name.find('ConeBeamModel') > 0
    is_parallel = model_name.find('ParallelBeamModel') > 0
    is_translation = model_name.find('TranslationModel') > 0
    # 'ParallelBeamModel' is not a substring of 'MultiAxisParallelModel'.
    if not (is_cone or is_translation or model_name.find('ParallelBeamModel') > 0
            or model_name.find('MultiAxisParallelModel') > 0):
        raise TypeError('copy_ct_model() supports ConeBeamModel, ParallelBeamModel, MultiAxisParallelModel and '
                        f'TranslationModel; got {type(ct_model).__name__}.  Construct the new model directly.')

    required, optional, regularization = ct_model.get_all_params()
    parent_required = dict(required)   # The constructor arguments of ct_model.

    if is_translation:
        view_key, new_view_params = 'translation_vectors', new_translation_vectors
        if new_angles is not None or new_helical_z_shifts is not None:
            raise ValueError('copy_ct_model: a TranslationModel has per-view translations rather than angles; '
                             'pass new_translation_vectors.')
    else:
        view_key, new_view_params = 'angles', new_angles
        if new_translation_vectors is not None:
            raise ValueError('copy_ct_model: new_translation_vectors applies to a TranslationModel only; '
                             f'got {type(ct_model).__name__}, so pass new_angles.')

    old_view_params = required[view_key]
    new_shape = list(required['sinogram_shape'])
    old_num_det_rows, old_num_det_cols = new_shape[1], new_shape[2]

    if is_cone:
        old_helical_z_shifts = required['helical_z_shifts']
        if new_angles is None and new_helical_z_shifts is None:
            new_helical_z_shifts = old_helical_z_shifts
        elif new_angles is not None and new_helical_z_shifts is None:
            if np.any(np.asarray(old_helical_z_shifts) != 0):
                raise ValueError('copy_ct_model: new_helical_z_shifts must be specified when changing angles for a helical scan.')
            new_helical_z_shifts = np.zeros_like(new_angles)
        elif new_angles is not None and new_helical_z_shifts is not None:
            if len(new_angles) != len(new_helical_z_shifts):
                raise ValueError('copy_ct_model: new_angles and new_helical_z_shifts must have the same length.')
        elif new_angles is None and new_helical_z_shifts is not None:
            if len(old_helical_z_shifts) != len(new_helical_z_shifts):
                raise ValueError('copy_ct_model: new_helical_z_shifts must have the same length as the existing angles.')
        required['helical_z_shifts'] = new_helical_z_shifts

    # auto_set_recon_geometry uses only the range of the helical shifts.
    travel_changed = False
    if is_cone:
        old_z, new_z = np.asarray(old_helical_z_shifts, dtype=float), np.asarray(new_helical_z_shifts, dtype=float)
        travel_changed = not np.allclose([old_z.min(), old_z.max()], [new_z.min(), new_z.max()])

    if new_view_params is None:
        new_view_params = old_view_params
    new_shape[0] = len(new_view_params)
    if new_num_det_rows is not None:
        new_shape[1] = new_num_det_rows
    if new_num_det_cols is not None:
        new_shape[2] = new_num_det_cols
    required[view_key] = new_view_params
    required['sinogram_shape'] = tuple(new_shape)

    # The geometry of ct_model is applied after auto_set_recon_geometry, one axis at a time.
    parent_geometry = {name: optional.pop(name) for name in _RECON_GEOMETRY_NAMES if name in optional}
    new_model = build_model(required, optional, regularization)

    # Axes whose inputs did not change keep the geometry of ct_model.  A new count is
    # rescaled to the pitch of ct_model, except the slice count in parallel beam, which has
    # one slice per detector row.
    rows_changed = new_shape[1] != old_num_det_rows
    cols_changed = new_shape[2] != old_num_det_cols
    if is_translation:
        redo_in_plane = redo_slices = rows_changed or cols_changed
    else:
        redo_in_plane = cols_changed
        redo_slices = rows_changed or travel_changed
    automatic_pitch = float(new_model.get_params('delta_voxel'))
    parent_pitch = (automatic_pitch if parent_geometry.get('delta_voxel') is None
                    else float(parent_geometry['delta_voxel']))
    automatic_shape = tuple(int(n) for n in new_model.get_params('recon_shape'))
    parent_shape = tuple(int(n) for n in ct_model.get_params('recon_shape'))

    def at_parent_pitch(count):
        if np.isclose(parent_pitch, automatic_pitch):
            return count
        return int(np.ceil(count * automatic_pitch / parent_pitch))

    in_plane = tuple(at_parent_pitch(n) for n in automatic_shape[:2]) if redo_in_plane else parent_shape[:2]
    if not redo_slices:
        slices = parent_shape[2]
    elif is_parallel:
        slices = automatic_shape[2]
    else:
        slices = at_parent_pitch(automatic_shape[2])
    recon_shape = tuple(in_plane) + (slices,)
    has_offset = 'recon_slice_offset' in parent_geometry
    geometry = dict(delta_voxel=parent_pitch, recon_shape=recon_shape)
    if has_offset and not redo_slices:
        geometry['recon_slice_offset'] = parent_geometry['recon_slice_offset']
    new_model.set_params(no_warning=True, **geometry)

    if not no_warning and (redo_in_plane or redo_slices):
        # The warning names each value of ct_model that was set by hand and is replaced.
        automatic_parent_shape, automatic_parent_offset = _automatic_recon_geometry(
            parent_required, optional, regularization)
        lost = []
        if redo_in_plane and parent_shape[:2] != automatic_parent_shape[:2] and recon_shape[:2] != parent_shape[:2]:
            lost.append('the in-plane recon shape')
        if redo_slices and parent_shape[2] != automatic_parent_shape[2] and recon_shape[2] != parent_shape[2]:
            lost.append('the slice count')
        if redo_slices and has_offset:
            parent_offset = float(parent_geometry['recon_slice_offset'])
            if not np.isclose(parent_offset, automatic_parent_offset) \
                    and not np.isclose(float(new_model.get_params('recon_slice_offset')), parent_offset):
                lost.append('recon_slice_offset')
        if lost:
            changed = [name for flag, name in ((rows_changed, 'detector row count'),
                                               (cols_changed, 'detector channel count'),
                                               (travel_changed, 'helical travel')) if flag]
            warnings.warn(f"copy_ct_model: the {' and '.join(changed)} changed, so the automatic pass re-derived "
                          f"{' and '.join(lost)}, which the parent had set by hand.  Set them on the copy to keep "
                          "the parent's values, or pass no_warning=True.")
    # compile_mode and view_batch_size are constructor arguments but not parameters, so
    # get_all_params does not carry them.  The copy runs the way ct_model runs.
    new_model.compile_mode = ct_model.compile_mode
    new_model.view_batch_size = ct_model.view_batch_size
    if not ct_model.device_layout_is_automatic:
        new_model.configure_devices(devices=list(ct_model.sino_placement.devices))
    return new_model


def construct_time_frame_models(model, frames_per_rotation=6, frame_overlap_factor=2.0):
    """
    This function splits a scan into overlapping time frames and builds one model for each
    frame.

    The function assumes that the views are in time order and that the angle changes at a
    constant rate.  Each frame is a range of consecutive views.  A new frame starts every
    360 / ``frames_per_rotation`` degrees.  Each frame spans ``frame_overlap_factor`` of these
    steps.  The function discards the last views if they cannot fill a whole frame.  Each
    frame model is a :func:`copy_ct_model` of ``model`` with the angles of the frame.

    Args:
        model (TomographyModel): The model of the full scan, with one angle per view.  The
            model is a ConeBeamModel or a ParallelBeamModel.
        frames_per_rotation (int, optional): The number of frames in a full rotation.
            Defaults to 6.
        frame_overlap_factor (float, optional): The span of a frame in units of the step
            between frames.  Defaults to 2.0.

    Returns:
        (model_list, view_slices): One model and one slice for each frame.
            - model_list (list of TomographyModel): The model of each frame.
            - view_slices (list of slice): The views of the full sinogram in each frame.
              ``sinogram[view_slices[k]]`` is the sinogram of frame ``k``.

    Raises:
        ValueError: If the model has no 1D angle vector, if the angles have zero spacing, if
            the frame span or the stride is smaller than one view, or if a frame would be
            longer than the scan.

    Example:
        >>> frames, view_slices = mbirtorch.utilities.construct_time_frame_models(ct_model)
        >>> sinogram_of_frame_1 = sinogram[view_slices[1]]
    """
    angle_stride = 2.0 * np.pi / frames_per_rotation
    angle_span_per_frame = frame_overlap_factor * angle_stride

    required_params, _, _ = model.get_all_params()
    angles = required_params.get('angles')
    if angles is None or np.asarray(angles).ndim != 1:
        raise ValueError('construct_time_frame_models needs a model with one angle per view, such '
                         f'as a ConeBeamModel or a ParallelBeamModel; got {type(model).__name__}.')
    angles = np.asarray(angles)
    num_views = len(angles)

    angle_step = float(np.median(np.abs(np.diff(angles)))) if num_views > 1 else 0.0
    if not angle_step > 0:
        raise ValueError('The model angles must have nonzero spacing.')
    views_per_frame = int(round(angle_span_per_frame / angle_step))
    stride = int(round(angle_stride / angle_step))

    if views_per_frame <= 0:
        raise ValueError('frame_overlap_factor gives a frame span smaller than one view.')
    if stride <= 0:
        raise ValueError('frames_per_rotation gives a stride smaller than one view.')
    if views_per_frame > num_views:
        raise ValueError('The frame span cannot exceed the full scan.')

    model_list = []
    view_slices = []
    for start in range(0, num_views - views_per_frame + 1, stride):
        view_slice = slice(start, start + views_per_frame)
        view_slices.append(view_slice)
        model_list.append(copy_ct_model(model, new_angles=angles[view_slice]))
    return model_list, view_slices


def calc_tct_recon_params(source_det_dist, source_iso_dist, delta_det_row, delta_det_channel, sinogram_shape, translation_vectors, voxel_row_aspect=1.0, voxel_slice_aspect=1.0):
    """
    This function computes the reconstruction parameters recon_shape, delta_voxel, and
    voxel_row_aspect for the translation geometry.

    Args:
        source_det_dist (float): The distance from the X-ray source to the detector, in ALU.
        source_iso_dist (float): The distance from the X-ray source to the isocenter, in ALU.
        delta_det_row (float): The spacing between detector rows, in ALU.
        delta_det_channel (float): The spacing between detector channels, in ALU.
        sinogram_shape (tuple): The sinogram shape (num_views, num_det_rows,
            num_det_channels).
        translation_vectors (numpy array): A (num_views, 3) array of translations (x, y, z)
            in ALU.
        voxel_row_aspect (float): The ratio of delta_voxel_row to delta_voxel.  Defaults
            to 1.0.
        voxel_slice_aspect (float): The ratio of delta_voxel_slice to delta_voxel.
            Defaults to 1.0.

    Returns:
        recon_shape (tuple): (num_recon_rows, num_recon_cols, num_recon_slices).
        delta_voxel (float): The voxel pitch at the isocenter, in ALU.
        voxel_row_aspect (float): The ratio of delta_voxel_row to delta_voxel.
    """
    num_views, num_det_rows, num_det_channels = sinogram_shape

    magnification = source_det_dist / source_iso_dist

    # The detector width and height in ALU.
    detect_box = np.array([delta_det_channel * num_det_channels, delta_det_row * num_det_rows])

    # The average slope of a ray at the isocenter.  It is the tangent of half the cone angle.
    avg_view_slope = (detect_box / 4) / source_det_dist

    # The detector pixel pitch at the isocenter.
    det_pixel_pitch_iso_vec = np.array([delta_det_row, delta_det_channel]) / magnification
    det_pixel_pitch_iso = np.max(det_pixel_pitch_iso_vec)

    delta_voxel = float(det_pixel_pitch_iso)

    delta_voxel_slice = voxel_slice_aspect * delta_voxel

    # This heuristic gives isotropic voxels when the average view slope is above 76 degrees.
    # The row resolution is never finer than the detector resolution.
    nominal_row_pitch = 4.0 * det_pixel_pitch_iso_vec / avg_view_slope
    nominal_row_pitch = np.max(nominal_row_pitch)
    delta_recon_row = np.maximum(nominal_row_pitch, det_pixel_pitch_iso)
    delta_recon_row = float(delta_recon_row)

    # A voxel_row_aspect of 1.0 means that the caller chose no aspect ratio.  Then the
    # heuristic sets delta_recon_row.
    if voxel_row_aspect == 1.0:
        voxel_row_aspect = delta_recon_row / delta_voxel
    else:
        delta_recon_row = voxel_row_aspect * delta_voxel

    # cube holds the width, depth, and height of the scanned region in ALU.
    max_translation = np.amax(translation_vectors, axis=0)
    min_translation = np.amin(translation_vectors, axis=0)
    cube = max_translation - min_translation

    recon_box = np.ceil(np.array([cube[0], cube[2]]) / np.array([delta_voxel, delta_voxel_slice]))

    # The number of rows makes the number of unknowns twice the number of measurements.
    num_pixels_per_view = ((recon_box[0] + num_det_rows) * (recon_box[1] + num_det_channels)) / num_views
    num_measurements_per_view = num_det_channels * num_det_rows
    num_recon_rows = 2 * np.ceil(num_measurements_per_view / num_pixels_per_view)

    # The object must extend no further than halfway to the source.
    max_recon_rows = np.floor((source_iso_dist - cube[1]) / delta_recon_row)
    if max_recon_rows < 1:
        print(f"[Error] Computed max_recon_rows = {max_recon_rows} < 1. This suggests the object extends beyond the source.")
    num_recon_rows = np.minimum(num_recon_rows, max_recon_rows)

    num_recon_cols, num_recon_slices = recon_box
    num_recon_cols = int(num_recon_cols)
    num_recon_rows = int(num_recon_rows)
    num_recon_slices = int(num_recon_slices)
    recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)

    return recon_shape, delta_voxel, voxel_row_aspect


def import_recon_hdf5(file_path):
    """
    This function reads a reconstruction and its recon_dict from a file written by
    :func:`export_recon_hdf5`.  The returned volume has axes (row, col, slice), or
    (time, row, col, slice) for a 4D volume.  If an older file records no axis order, the
    function reads it and gives a warning.

    Args:
        file_path (str): The path of the file.

    Returns:
        (recon, recon_dict)

    Example:
        >>> recon, recon_dict = mbirtorch.import_recon_hdf5('output/recon.h5')
    """
    import json
    import h5py
    with h5py.File(file_path, 'r') as f:
        names = list(f.keys())
        if 'recon' in names:
            name = 'recon'
        elif len(names) == 1:
            name = names[0]
        else:
            raise ValueError(f'{file_path} holds {len(names)} datasets and none is named recon')
        recon = f[name][()]
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f[name].attrs.items()}

    if attrs.pop('format', None) == _RECON_FORMAT:
        axes = attrs.pop('axes')
        attrs.pop('mbirtorch_version', None)
        recon = np.transpose(recon, _RECON_TRANSPOSE[len(axes.split(','))])
        recon_dict = {}
        for key, value in attrs.items():
            if isinstance(value, str) and value[:1] == '{':
                try:
                    value = json.loads(value, object_hook=_json_object_hook)
                except ValueError:
                    pass
            recon_dict[key] = value
        return recon, recon_dict

    # This is an older file that records no axis order.
    if name == 'recon':
        warnings.warn(f'{file_path} is in the old layout; it is read as (slice, col, row), so a file '
                      f'written by save_recon_hdf5 comes back transposed.')
        if recon.ndim == 3:
            recon = np.transpose(recon, (2, 1, 0))
    else:
        warnings.warn(f'{file_path} is in the old layout; it is read as stored.')
    return recon, attrs


def merge_log_files(merged_path, labeled_paths):
    """This function merges temporary log files into one file and deletes them.  Each file
    appears under a section header.  The function skips missing files.  If no file exists,
    the function writes nothing.

    Args:
        merged_path (str): The path of the merged output file.
        labeled_paths (iterable): The (label, path) pairs, in the order they appear in the
            merged file.
    """
    labeled_paths = [(label, path) for label, path in labeled_paths
                     if path is not None and os.path.exists(path)]
    if not labeled_paths:
        return
    with open(merged_path, 'w') as merged:
        for label, path in labeled_paths:
            merged.write('======== {} ========\n'.format(label))
            with open(path, 'r') as f:
                merged.write(f.read())
            os.remove(path)
        merged.write('Merged logs written to {}\n'.format(
            os.path.abspath(merged_path)))


def get_ct_model(geometry_type, sinogram_shape, angles=None, source_detector_dist=None, source_iso_dist=None,
                 helical_z_shifts=None, translation_vectors=None):
    """
    This function creates an instance of TomographyModel with the given parameters.

    Args:
        geometry_type (str): 'parallel', 'cone', 'multiaxis', or 'translation'.
        sinogram_shape (tuple of int): (num_views, num_rows, num_channels).
        angles (ndarray of float, optional): The projection angles in radians.  For
            'parallel' and 'cone', this is a 1D vector.  For 'multiaxis', this is a
            (num_views, 2) array of (azimuth, elevation) pairs.  'translation' does not use
            it.  Defaults to None.
        source_detector_dist (float or None, optional): The distance from the source to the
            detector, in ALU.  Defaults to None for geometries that do not use it.
        source_iso_dist (float or None, optional): The distance from the source to the
            isocenter, in ALU.  Defaults to None for geometries that do not use it.
        helical_z_shifts (ndarray, optional): The axial shift of each view in ALU, with the
            same length as angles.  Only 'cone' uses it, and then the scan is helical.  For
            'parallel' and 'multiaxis', the function ignores it and gives a warning.  Defaults
            to None.
        translation_vectors (ndarray of float, optional): A (num_views, 3) array of object
            translations (x, y, z) in ALU.  'translation' requires it, and the other
            geometries do not use it.  Defaults to None.

    Returns:
        An instance of ConeBeamModel, ParallelBeamModel, MultiAxisParallelModel, or
        TranslationModel.
    """
    import mbirtorch

    if geometry_type == 'cone':
        model = mbirtorch.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                        source_iso_dist=source_iso_dist, helical_z_shifts=helical_z_shifts)
    elif geometry_type == 'parallel':
        if helical_z_shifts is not None:
            warnings.warn("Helical mode (helical_z_shifts) is only supported for geometry_type='cone'; ignoring z_shifts.", UserWarning)
        model = mbirtorch.ParallelBeamModel(sinogram_shape, angles)
    elif geometry_type == 'multiaxis':
        if helical_z_shifts is not None:
            warnings.warn("Helical mode (helical_z_shifts) is only supported for geometry_type='cone'; ignoring z_shifts.", UserWarning)
        model = mbirtorch.MultiAxisParallelModel(sinogram_shape, angles)
    elif geometry_type == 'translation':
        if translation_vectors is None:
            raise ValueError("get_ct_model() with geometry_type 'translation' needs translation_vectors, a "
                             "(num_views, 3) array of object translations in ALU; a translation geometry has no "
                             "angles.")
        model = mbirtorch.TranslationModel(sinogram_shape, translation_vectors,
                                           source_detector_dist=source_detector_dist,
                                           source_iso_dist=source_iso_dist)
    else:
        raise ValueError("get_ct_model() supports geometry_type 'cone', 'parallel', 'multiaxis' and 'translation'; "
                         f"got {geometry_type!r}.")

    return model


def _gen_ellipsoid(x_grid, y_grid, z_grid, x0, y0, z0, a, b, c, gray_level, alpha=0, beta=0, gamma=0):
    """
    This function returns an image of a 3D ellipsoid with center [x0, y0, z0].

    Args:
        x_grid(ndarray): The 3D grid of x coordinates.
        y_grid(ndarray): The 3D grid of y coordinates.
        z_grid(ndarray): The 3D grid of z coordinates.
        x0(float): The x coordinate of the center.
        y0(float): The y coordinate of the center.
        z0(float): The z coordinate of the center.
        a(float): The radius along the x axis.
        b(float): The radius along the y axis.
        c(float): The radius along the z axis.
        gray_level(float): The value inside the ellipsoid.
        alpha(float): The counter-clockwise rotation about the x axis, in radians.  Defaults
            to 0.
        beta(float): The counter-clockwise rotation about the y axis, in radians.  Defaults
            to 0.
        gamma(float): The counter-clockwise rotation about the z axis, in radians.  Defaults
            to 0.

    Return:
        ndarray: A 3D array with the same shape as x_grid, y_grid, and z_grid.

    """
    rx = np.array([[1, 0, 0], [0, np.cos(-alpha), -np.sin(-alpha)], [0, np.sin(-alpha), np.cos(-alpha)]])
    ry = np.array([[np.cos(-beta), 0, np.sin(-beta)], [0, 1, 0], [-np.sin(-beta), 0, np.cos(-beta)]])
    rz = np.array([[np.cos(-gamma), -np.sin(-gamma), 0], [np.sin(-gamma), np.cos(-gamma), 0], [0, 0, 1]])
    r = np.dot(rx, np.dot(ry, rz))

    cor = np.array([x_grid.flatten() - x0, y_grid.flatten() - y0, z_grid.flatten() - z0])

    image = ((np.dot(r[0], cor)) ** 2 / a ** 2 + (np.dot(r[1], cor)) ** 2 / b ** 2 + (
        np.dot(r[2], cor)) ** 2 / c ** 2 <= 1.0) * gray_level

    return image.reshape(x_grid.shape)


def gen_shepp_logan_3d_reference(phantom_shape):
    """
    This function generates the 3D Shepp-Logan phantom defined in Kak AC, Slaney M,
    Principles of computerized tomographic imaging, page 102, IEEE Press, New York, 1988.
    https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf

    Args:
        phantom_shape (tuple or list of ints): (num_rows, num_cols, num_slices).

    Return:
        out_image: A 3D numpy array with shape (num_rows, num_cols, num_slices).

    Note:
        This function holds 6 temporary arrays of shape phantom_shape, so a large shape
        needs a lot of memory.
    """

    # The phantom is the sum of ten ellipsoids inside a 2 by 2 by 2 cube.
    sl3d_paras = [
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.69, 'b': 0.92, 'c': 0.9, 'gamma': 0, 'gray_level': 2.0},
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.6624, 'b': 0.874, 'c': 0.88, 'gamma': 0, 'gray_level': -0.98},
        {'x0': -0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.41, 'b': 0.16, 'c': 0.21, 'gamma': 108, 'gray_level': -0.02},
        {'x0': 0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.31, 'b': 0.11, 'c': 0.22, 'gamma': 72, 'gray_level': -0.02},
        {'x0': 0.0, 'y0': 0.35, 'z0': -0.25, 'a': 0.21, 'b': 0.25, 'c': 0.5, 'gamma': 0, 'gray_level': 0.02},
        {'x0': 0.0, 'y0': 0.1, 'z0': -0.25, 'a': 0.046, 'b': 0.046, 'c': 0.046, 'gamma': 0, 'gray_level': 0.02},
        {'x0': -0.08, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 0, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 90, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.105, 'z0': 0.625, 'a': 0.056, 'b': 0.04, 'c': 0.1, 'gamma': 90, 'gray_level': 0.02},
        {'x0': 0.0, 'y0': 0.1, 'z0': 0.625, 'a': 0.056, 'b': 0.056, 'c': 0.1, 'gamma': 0, 'gray_level': -0.02}
    ]

    num_rows, num_cols, num_slices = phantom_shape
    axis_x = np.linspace(-1.0, 1.0, num_cols)
    axis_y = np.linspace(1.0, -1.0, num_rows)
    axis_z = np.linspace(-1.0, 1.0, num_slices)

    x_grid, y_grid, z_grid = np.meshgrid(axis_x, axis_y, axis_z)
    image = x_grid * 0.0

    for el_paras in sl3d_paras:
        image += _gen_ellipsoid(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, x0=el_paras['x0'], y0=el_paras['y0'],
                               z0=el_paras['z0'],
                               a=el_paras['a'], b=el_paras['b'], c=el_paras['c'],
                               gamma=el_paras['gamma'] / 180.0 * np.pi,
                               gray_level=el_paras['gray_level'])

    # meshgrid puts y on axis 0, so the image already has axes (rows, cols, slices).
    return image


def _gen_translation_phantom(recon_shape, option, text, fill_rate=0.05, font_size=20, text_row_indices=None,
                            horizontal_offset=0, vertical_offset=0, voxel_slice_aspect=1.0):
    """
    This function generates a synthetic phantom of dots or text.

    Args:
        recon_shape (tuple[int, int, int]): The shape of the reconstruction volume.
        option (str): The phantom type, either 'dots' or 'text'.
        text (list[str]): The ASCII strings to draw.  Only 'text' uses this argument.
        fill_rate (float, optional): The fraction of voxels set to 1.  Only 'dots' uses this
            argument.  Defaults to 0.05.
        font_size (int, optional): The font size of the text.  Only 'text' uses this
            argument.  Defaults to 20.
        text_row_indices (list[int], optional): The row index of each string in 'text'.
            Only 'text' uses this argument.  Defaults to None, which spaces the strings evenly
            along the first axis.
        horizontal_offset (int, optional): The horizontal offset of the text.  A positive
            value shifts the text right.  Only 'text' uses this argument.  Defaults to 0.
        vertical_offset (int, optional): The vertical offset of the text.  A positive value
            shifts the text up.  Only 'text' uses this argument.  Defaults to 0.
        voxel_slice_aspect (float, optional): The ratio of the slice spacing to the column
            spacing.  Only 'text' uses this argument.  Defaults to 1.0.

    Returns:
        np.ndarray: The phantom volume.
    """
    if option == 'dots':
        return gen_dot_phantom(recon_shape, fill_rate)
    elif option == 'text':
        return gen_text_phantom(recon_shape, text, font_size, text_row_indices, horizontal_offset, vertical_offset,
                                voxel_slice_aspect=voxel_slice_aspect)
    else:
        raise ValueError(f"Unsupported phantom option: {option}")


def gen_dot_phantom(recon_shape, fill_rate):
    """
    This function generates a phantom of randomly placed dots.  In each row of the central
    two thirds of the rows, the function sets a fraction ``fill_rate`` of the voxels to 1.

    Args:
        recon_shape (tuple[int, int, int]): The shape of the reconstruction volume.
        fill_rate (float): The fraction of voxels set to 1 in each filled row.

    Returns:
        np.ndarray: The phantom, with values 0 and 1.
    """
    np.random.seed(42)
    gt_recon = np.zeros(recon_shape, dtype=np.float32)

    y_pad = recon_shape[0] // 6
    central_start = y_pad
    central_end = recon_shape[0] - y_pad

    row_size = recon_shape[1] * recon_shape[2]
    num_ones_per_row = int(row_size * fill_rate)

    for row_idx in range(central_start, central_end):
        flat_row = gt_recon[row_idx].flatten()
        positions_ones = np.random.choice(row_size, num_ones_per_row, replace=False)
        flat_row[positions_ones] = 1.0
        gt_recon[row_idx] = flat_row.reshape(recon_shape[1:])

    return gt_recon


def gen_text_phantom(recon_shape, words, font_size, row_indices=None, horizontal_offset=0,
                     vertical_offset=0, voxel_slice_aspect=1.0, font_path="DejaVuSans.ttf"):
    """
    This function generates a 3D text phantom.  Each word is drawn with value 1 in one row
    of the volume.

    Args:
        recon_shape (tuple[int, int, int]): The shape of the phantom (num_rows, num_cols,
            num_slices).
        words (list[str]): The ASCII words to draw.
        font_size (int): The font size of the words.
        row_indices (list[int], optional): The row index of each word.  Defaults to None,
            which spaces the words evenly along the first axis.
        horizontal_offset (int, optional): The horizontal offset of the text.  A positive
            value shifts the text right.  Defaults to 0.
        vertical_offset (int, optional): The vertical offset of the text.  A positive value
            shifts the text up.  Defaults to 0.
        voxel_slice_aspect (float, optional): The ratio of the slice spacing to the column
            spacing.  The text keeps its physical aspect ratio.  Defaults to 1.0.
        font_path (str, optional): The path to the TrueType font file.  Defaults to
            "DejaVuSans.ttf".

    Returns:
        np.ndarray: A 3D numpy array of shape `recon_shape` that holds the text phantom.
    """
    # Imported here so that importing mbirtorch does not require Pillow.
    from PIL import Image, ImageDraw, ImageFont

    if voxel_slice_aspect <= 0:
        raise ValueError(f"voxel_slice_aspect must be positive. Got {voxel_slice_aspect}.")

    if row_indices is not None:
        if len(row_indices) != len(words):
            raise ValueError(
                f"Length of row_indices ({len(row_indices)}) must match length of words ({len(words)})")

        for idx in row_indices:
            if not (0 <= idx < recon_shape[0]):
                raise ValueError(f"Row index {idx} is out of bounds for first dimension of size {recon_shape[0]}")

        positions = []
        for row_idx in row_indices:
            col_pos = recon_shape[1] // 2 + horizontal_offset
            slice_pos = recon_shape[2] // 2 - vertical_offset
            positions.append((row_idx, col_pos, slice_pos))
    else:
        positions = []
        row_positions = np.linspace(0, recon_shape[0] - 1, len(words) + 2)[1:-1]
        for r in row_positions:
            col_pos = recon_shape[1] // 2 + horizontal_offset
            slice_pos = recon_shape[2] // 2 - vertical_offset
            positions.append((int(round(r)), col_pos, slice_pos))

    array_size = int(np.minimum(recon_shape[1], recon_shape[2]))
    array_num_cols = array_size
    array_num_slices = int(round(array_size / voxel_slice_aspect))
    array_num_slices = min(max(array_num_slices, 1), recon_shape[2])

    phantom = np.zeros(recon_shape, dtype=np.float32)
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except OSError:
        from pathlib import Path
        fallback_paths = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS.
            "/Library/Fonts/Arial.ttf",  # macOS.
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux.
        ]
        for fallback in fallback_paths:
            if Path(fallback).exists():
                font = ImageFont.truetype(fallback, size=font_size)
                break
        else:
            raise FileNotFoundError(
                f"Could not find a usable font. Tried the following paths:\n"
                + "\n".join(fallback_paths)
                + "\nPlease install one of these fonts or specify a valid font_path."
            )

    for word, (r, c, s) in zip(words, positions):
        img = Image.new('L', (array_size, array_size), 0)
        draw = ImageDraw.Draw(img)

        text_box = draw.textbbox((0, 0), word, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        x = (array_size - text_width) // 2 - text_box[0]
        y = (array_size - text_height) // 2 - text_box[1]
        draw.text((x, y), word, fill=1, font=font)

        word_array = np.array(img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT))
        if array_num_slices != array_size:
            word_img = Image.fromarray(word_array)
            nearest_resampling = getattr(getattr(Image, 'Resampling', Image), 'NEAREST')
            word_img = word_img.resize((array_num_slices, array_num_cols), resample=nearest_resampling)
            word_array = np.array(word_img)
        word_array = (word_array > 0).astype(np.float32)

        r_start, r_end = r, r + 1
        c_start = c - array_num_cols // 2
        c_end = c_start + array_num_cols
        s_start = s - array_num_slices // 2
        s_end = s_start + array_num_slices

        c_start_valid = max(c_start, 0)
        c_end_valid = min(c_end, recon_shape[1])
        s_start_valid = max(s_start, 0)
        s_end_valid = min(s_end, recon_shape[2])

        word_c_start = c_start_valid - c_start
        word_c_end = word_c_start + (c_end_valid - c_start_valid)
        word_s_start = s_start_valid - s_start
        word_s_end = word_s_start + (s_end_valid - s_start_valid)

        word_crop = word_array[word_c_start:word_c_end, word_s_start:word_s_end]
        phantom[r_start:r_end, c_start_valid:c_end_valid, s_start_valid:s_end_valid] = word_crop

    return phantom


def gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing):
    """
    This function generates a grid of translation vectors in the lateral (x) and axial (z)
    directions.  The grid is centered at zero.

    Args:
        num_x_translations (int): The number of translations in x.
        num_z_translations (int): The number of translations in z.
        x_spacing (float): The spacing between x translations, in ALU.
        z_spacing (float): The spacing between z translations, in ALU.

    Returns:
        np.ndarray: An array of shape (num_views, 3) of translation vectors [dx, dy, dz].
    """
    num_views = num_x_translations * num_z_translations
    translation_vectors = np.zeros((num_views, 3))

    x_center = (num_x_translations - 1) / 2
    z_center = (num_z_translations - 1) / 2

    idx = 0
    for row in range(num_z_translations):
        for col in range(num_x_translations):
            dx = (col - x_center) * x_spacing
            dz = (row - z_center) * z_spacing
            dy = 0
            translation_vectors[idx] = [dx, dy, dz]
            idx += 1

    return translation_vectors


def _gen_polygon_phantom(recon_shape):
    """
    This function returns a phantom of one asymmetric polygon with five sides.  Every slice
    is the same.  The phantom is used to test view selection.

    Args:
        recon_shape (tuple of int): (num_rows, num_cols, num_slices).

    Returns:
        numpy.ndarray: The float32 phantom.  It is 1.0 inside the polygon and 0 elsewhere.
    """
    from matplotlib.path import Path

    num_rows, num_cols, num_slices = recon_shape
    # The vertices are (row, col) points on a 512 by 512 grid.  The code scales them to
    # this grid.
    vertices = np.array([[100, 440], [80, 260], [280, 60], [420, 280], [360, 460]], dtype=np.float64)
    vertices *= np.array([num_rows, num_cols]) / 512.0
    rows, cols = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
    inside = Path(vertices).contains_points(np.stack([rows.ravel(), cols.ravel()], axis=-1))
    image = inside.reshape(num_rows, num_cols).astype(np.float32)
    return np.repeat(image[:, :, None], num_slices, axis=2)


def gen_cube_phantom(recon_shape, device=None):
    """This function generates a phantom of a rectangular bar that shifts across the columns
    from slice to slice.  The function returns it as a torch tensor on ``device``."""
    import torch

    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    phantom_rows = num_recon_rows // 4  # The phantom height.
    phantom_cols = num_recon_cols // 4  # The phantom width.

    # The phantom is float32, because the mps backend does not support float64.
    phantom = np.zeros((num_recon_rows, num_recon_cols, num_recon_slices),
                       dtype=np.float32)

    start_rows = (num_recon_rows - phantom_rows) // 2
    stop_rows = (num_recon_rows + phantom_rows) // 2
    start_cols = (num_recon_cols - phantom_cols) // 2
    stop_cols = (num_recon_cols + phantom_cols) // 2
    for slice_index in np.arange(num_recon_slices):
        shift_cols = int(slice_index * phantom_cols / num_recon_slices)
        phantom[start_rows:stop_rows, (shift_cols + start_cols):(shift_cols + stop_cols), slice_index] = 1.0 / max(
            phantom_rows, phantom_cols)

    return torch.as_tensor(phantom, device=device)


def get_helical_half_rotation_slice_range(
    ct_model,
    helical_pitch,
    helical_z_shifts,
):
    """
    This function returns the contiguous range of slices that the helical scan sees for at
    least half a rotation.  The function assumes that helical_z_shifts are monotone and
    evenly spaced.

    Returns:
        start_slice, stop_slice.  The range excludes stop_slice.
    """
    recon_shape = ct_model.get_params('recon_shape')
    delta_voxel, voxel_slice_aspect, recon_slice_offset = ct_model.get_params(
        ['delta_voxel', 'voxel_slice_aspect', 'recon_slice_offset']
    )
    sinogram_shape = ct_model.get_params('sinogram_shape')
    delta_det_row = ct_model.get_params('delta_det_row')
    magnification = ct_model.get_magnification()

    num_slices = recon_shape[2]
    num_det_rows = sinogram_shape[1]

    delta_voxel_slice = voxel_slice_aspect * delta_voxel

    # These are the z locations of the slice centers, in reconstruction coordinates.
    k = np.arange(num_slices)
    z_k = delta_voxel_slice * (k - (num_slices - 1) / 2.0) + recon_slice_offset

    det_height_iso = num_det_rows * delta_det_row / magnification

    z_shift_min = np.min(helical_z_shifts)
    z_shift_max = np.max(helical_z_shifts)

    # The range is trimmed only when the pitch is above 1.
    trim = 0.5 * det_height_iso * np.maximum(float(helical_pitch) - 1.0, 0.0)

    z_min = z_shift_min + trim
    z_max = z_shift_max - trim

    valid_slice_mask = (z_k >= z_min) & (z_k <= z_max)

    valid_indices = np.where(np.asarray(valid_slice_mask))[0]
    if len(valid_indices) == 0:
        raise ValueError(
            "No slices are visible for at least half a rotation. "
            "Check helical_pitch, helical_z_range, num_views, and detector height."
        )

    start_slice = int(valid_indices[0])
    stop_slice = int(valid_indices[-1] + 1)

    return start_slice, stop_slice


class ObjectType(str, Enum):
    SHEPP_LOGAN = 'shepp-logan'
    CUBE = 'cube'
    POLYGON = 'polygon'


class ModelType(str, Enum):
    PARALLEL = 'parallel'
    CONE = 'cone'
    TRANSLATION = 'translation'
    MULTIAXIS = 'multiaxis'


def gen_demo_data(
    object_type='shepp-logan',
    model_type='cone',
    num_views=64,
    num_det_rows=96,
    delta_det_row=1,
    num_det_channels=128,
    num_x_translations=7,
    num_z_translations=7,
    x_spacing=22,
    z_spacing=22,
    use_helical=False,
    helical_pitch=None,
    helical_z_range=None,
    helical_z_center=0.0,
    use_curved_detector=False,
    elevation_degrees=0.0,
    voxel_row_aspect=1.0,
    voxel_slice_aspect=1.0,
    target_max_attenuation=None,
    devices=None,
):
    """
    This function creates a simple object and a simulated sinogram for demonstrations.

    The object is a simplified Shepp-Logan phantom, a cube, or an asymmetric polygon.  The
    polygon is used to test view selection.  The 'translation' model type always uses a
    sparse pattern of dots instead.  Each 2D array sinogram[view_index] is a simulated
    detector image.

    Args:
        object_type (str, optional): One of 'shepp-logan', 'cube', or 'polygon'.  Defaults to
            'shepp-logan'.  The function ignores it when model_type is 'translation'.
        model_type (str, optional): One of 'parallel', 'cone', 'translation', or 'multiaxis'.
            Defaults to 'cone'.
        num_views (int, optional): The number of views in the output sinogram.  Defaults to
            64.  The function ignores it when model_type is 'translation'.
        num_det_rows (int, optional): The number of detector rows (vertical) in the output
            sinogram.  Defaults to 96.
        delta_det_row (float, optional): The detector row pitch in ALU.  It has an effect only
            for a helical cone beam scan.  Defaults to 1.
        num_det_channels (int, optional): The number of detector channels (horizontal) in the
            output sinogram.  Defaults to 128.
        num_x_translations (int, optional): The number of horizontal translations for the
            translation model.  Defaults to 7.
        num_z_translations (int, optional): The number of vertical translations for the
            translation model.  Defaults to 7.
        x_spacing (float, optional): The horizontal spacing between translations, in ALU.
            Defaults to 22.
        z_spacing (float, optional): The vertical spacing between translations, in ALU.
            Defaults to 22.
        use_helical (bool, optional): If True and model_type is 'cone', the scan is helical.
            Defaults to False.
        helical_pitch (float, optional): The helical pitch.  It equals the table travel per
            rotation divided by the detector height at the isocenter.
        helical_z_range (float, optional): The total axial travel of a helical scan, in ALU.
        helical_z_center (float, optional): The midpoint of the axial travel of a helical
            scan, in ALU.
        use_curved_detector (bool, optional): If True, the cone beam detector is curved.
        elevation_degrees (float, optional): The constant tilt of every view out of the
            horizontal plane, in degrees.  Only 'multiaxis' uses it.  Defaults to 0.0.
        voxel_row_aspect (float, optional): The ratio of the row spacing to the column
            spacing of the reconstruction.  Defaults to 1.0.
        voxel_slice_aspect (float, optional): The ratio of the slice spacing to the in-plane
            voxel spacing delta_voxel.  Defaults to 1.0.
        target_max_attenuation (float, optional): The target maximum sinogram attenuation for
            the Shepp-Logan phantom.  Defaults to None, which keeps each voxel in [0, 1].  The
            scaling may not be accurate if any detector or voxel spacing is not 1.
        devices (sequence of devices, optional): The devices that generate the data.  Defaults
            to None, which uses the automatic choice of the model.  The result does not depend
            on the devices.

    Returns:
        tuple: (object, sinogram, params)
            - object: The phantom, a host numpy float32 array of shape recon_shape.
            - sinogram: A host numpy array of shape (num_views, num_det_rows,
              num_det_channels).
            - params (dict): The geometry parameters of the simulation.  For all model types
              except 'translation', it holds 'angles' and the voxel aspect ratios.  For 'cone',
              it also holds 'source_detector_dist', 'source_iso_dist', and
              'use_curved_detector'.  For a helical scan, it also holds 'helical_z_shifts'.
              For 'multiaxis', it also holds 'elevation_degrees'.  For 'translation', it holds
              only 'translation_vectors'.
    """
    import mbirtorch

    object_type = ObjectType(object_type)
    model_type = ModelType(model_type)

    start_angle = -np.pi
    end_angle = np.pi

    if model_type == ModelType.MULTIAXIS:
        azimuths = np.linspace(0, np.pi, num_views, endpoint=False)
        elevations = np.deg2rad(elevation_degrees) * np.ones(num_views)
        angles = np.column_stack([azimuths, elevations]).astype(np.float32)
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        ct_model_for_generation = mbirtorch.MultiAxisParallelModel(sinogram_shape, angles)
        ct_model_for_generation.set_params(voxel_row_aspect=voxel_row_aspect)
        ct_model_for_generation.set_params(voxel_slice_aspect=voxel_slice_aspect)
        ct_model_for_generation.auto_set_recon_geometry()
        params = {'angles': angles, 'elevation_degrees': elevation_degrees,
                  'voxel_row_aspect': voxel_row_aspect, 'voxel_slice_aspect': voxel_slice_aspect}
    elif model_type == ModelType.PARALLEL:
        start_angle = 0
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        angles = np.linspace(start_angle, end_angle, num_views, endpoint=False)
        ct_model_for_generation = mbirtorch.ParallelBeamModel(sinogram_shape, angles)
        ct_model_for_generation.set_params(voxel_row_aspect=voxel_row_aspect)
        ct_model_for_generation.set_params(voxel_slice_aspect=voxel_slice_aspect)
        ct_model_for_generation.auto_set_recon_geometry()
        params = {'angles': angles, 'voxel_row_aspect': voxel_row_aspect, 'voxel_slice_aspect': voxel_slice_aspect}
    elif model_type == ModelType.CONE:
        # An infinite source distance is allowed and gives parallel beam.
        source_detector_dist = 4 * num_det_channels
        source_iso_dist = source_detector_dist/2
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        if not use_helical:
            angles = np.linspace(start_angle, end_angle, num_views, endpoint=False)
            ct_model_for_generation = mbirtorch.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                                              source_iso_dist=source_iso_dist, use_curved_detector=use_curved_detector)
            ct_model_for_generation.set_params(voxel_row_aspect=voxel_row_aspect)
            ct_model_for_generation.set_params(voxel_slice_aspect=voxel_slice_aspect)
            ct_model_for_generation.auto_set_recon_geometry()
            params = {'angles': angles, 'source_detector_dist': source_detector_dist, 'source_iso_dist': source_iso_dist,
                      'use_curved_detector': use_curved_detector, 'voxel_row_aspect': voxel_row_aspect, 'voxel_slice_aspect': voxel_slice_aspect}
        else:
            if helical_pitch is None or helical_z_range is None:
                raise ValueError("Helical trajectory requires both helical_pitch and helical_z_range.")

            if np.isinf(source_detector_dist):
                magnification = 1
            else:
                magnification = source_detector_dist / source_iso_dist

            # The detector height at the isocenter, in ALU.
            det_height_iso = float(num_det_rows) * (delta_det_row / magnification)

            # The table travel per rotation, in ALU.
            z_per_rot = float(helical_pitch) * det_height_iso
            if z_per_rot <= 0:
                raise ValueError(f"helical_pitch must be > 0 (got {helical_pitch}).")
            if float(helical_z_range) < 0:
                raise ValueError(f"helical_z_range must be >= 0 (got {helical_z_range}).")

            if float(helical_z_range) == 0.0:  # The scan is circular.
                num_rotations = 1.0
                views_per_rotation = float(num_views)
            else:
                num_rotations = float(helical_z_range) / z_per_rot
                if num_rotations <= 0:
                    raise ValueError("Derived num_rotations <= 0; check pitch/z_range.")
                views_per_rotation = float(num_views) / num_rotations

            angle_step = (2.0 * np.pi) / views_per_rotation
            angles = start_angle + angle_step * np.arange(num_views)

            z0 = float(helical_z_center) - 0.5 * float(helical_z_range)
            z1 = float(helical_z_center) + 0.5 * float(helical_z_range)
            helical_z_shifts = np.linspace(z0, z1, num_views, endpoint=True)

            ct_model_for_generation = mbirtorch.ConeBeamModel(
                sinogram_shape,
                angles,
                source_detector_dist=source_detector_dist,
                source_iso_dist=source_iso_dist,
                helical_z_shifts=helical_z_shifts,
                use_curved_detector=use_curved_detector
            )
            ct_model_for_generation.set_params(voxel_row_aspect=voxel_row_aspect)
            ct_model_for_generation.set_params(voxel_slice_aspect=voxel_slice_aspect)
            ct_model_for_generation.auto_set_recon_geometry()

            params = {
                'angles': angles,
                'source_detector_dist': source_detector_dist,
                'source_iso_dist': source_iso_dist,
                'helical_z_shifts': helical_z_shifts,
                'use_curved_detector': use_curved_detector,
                'voxel_row_aspect': voxel_row_aspect,
                'voxel_slice_aspect': voxel_slice_aspect
            }
    elif model_type == ModelType.TRANSLATION:
        source_iso_dist = min(num_det_rows, num_det_channels) / 2
        source_detector_dist = source_iso_dist
        translation_vectors = gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing)
        num_views = translation_vectors.shape[0]
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        ct_model_for_generation = mbirtorch.TranslationModel(sinogram_shape, translation_vectors, source_detector_dist=source_detector_dist,
                                                             source_iso_dist=source_iso_dist)
        params = {'translation_vectors': translation_vectors}
    else:
        raise ValueError(f'Invalid model type. Expected one of {[m.value for m in ModelType]}, got {model_type}')

    if devices is not None:
        ct_model_for_generation.configure_devices(devices=list(devices))
    # This model skips the memory check, because it does only one forward projection.
    ct_model_for_generation.skip_memory_check = True
    ct_model_for_generation._apply_device_policy()
    device_provenance = 'requested' if devices is not None else 'default'

    print('Creating phantom')
    recon_shape = ct_model_for_generation.get_params('recon_shape')
    phantom_shape = recon_shape
    embed_slice_start = 0
    embed_slice_stop = recon_shape[2]
    if model_type == ModelType.CONE and use_helical:
        embed_slice_start, embed_slice_stop = get_helical_half_rotation_slice_range(
            ct_model_for_generation,
            helical_pitch,
            helical_z_shifts
        )
        phantom_shape = (
            recon_shape[0],
            recon_shape[1],
            embed_slice_stop - embed_slice_start,
        )
    if model_type == ModelType.TRANSLATION:
        # The other phantoms are empty in the thin translation volume, so the code uses dots.
        phantom_core = _gen_translation_phantom(phantom_shape, option='dots', text=None)
    elif object_type == ObjectType.SHEPP_LOGAN:
        phantom_core = gen_shepp_logan_3d(
            phantom_shape, target_max_attenuation=target_max_attenuation)
    elif object_type == ObjectType.CUBE:
        phantom_core = gen_cube_phantom(phantom_shape).cpu().numpy()
    elif object_type == ObjectType.POLYGON:
        phantom_core = _gen_polygon_phantom(phantom_shape)
    else:
        raise ValueError(f'Invalid object type. Expected one of {[o.value for o in ObjectType]}, got {object_type}')
    if model_type == ModelType.CONE and use_helical:
        # In a helical scan, only the slices seen for at least half a rotation hold the
        # phantom.
        phantom = np.zeros(recon_shape, dtype=np.float32)
        phantom[:, :, embed_slice_start:embed_slice_stop] = phantom_core
    else:
        phantom = phantom_core

    print('Creating sinogram on the {} devices'.format(device_provenance))
    sinogram_sharded = ct_model_for_generation._forward_project(phantom, output_sharded=True)
    sinogram = ct_model_for_generation._gather_sinogram(sinogram_sharded)

    del ct_model_for_generation
    return phantom, sinogram, params
def gen_demo_data_4d(
    object_type='rack-and-pinion',
    model_type='parallel',
    num_views=240,
    num_rotations=2,
    num_det_rows=64,
    num_det_channels=64,
    num_steps=24,
    devices=None,
    **object_options,
):
    """
    This function creates a moving object and the sinogram of one continuous scan of the
    object.  The 4D demos use this data.

    The function samples the object at ``num_steps`` evenly spaced times.  The views are in
    time order, and the function divides them evenly among the time steps.  The views of
    step k see the object at step k.  The only object is 'rack-and-pinion'.  It is a toothed
    bar along the rotation axis with a toothed wheel beside it.

    Args:
        object_type (str, optional): The moving object.  Defaults to 'rack-and-pinion'.
        model_type (str, optional): 'parallel' or 'cone'.  Defaults to 'parallel'.
        num_views (int, optional): The number of views in the scan.  Defaults to 240.
        num_rotations (float, optional): The number of rotations in the scan.  Defaults to 2.
        num_det_rows (int, optional): The number of detector rows.  Defaults to 64.
        num_det_channels (int, optional): The number of detector channels.  Defaults to 64.
        num_steps (int, optional): The number of times at which the function samples the
            object.  Defaults to 24.
        devices (sequence of devices, optional): The devices that compute the projections.
            Defaults to None, which uses the automatic choice of the model.
        **object_options: Options passed to the phantom function, such as
            ``rotation_degrees`` for the rack and pinion.

    Returns:
        tuple: (phantom_4d, sinogram, params)
            - phantom_4d: The float32 object at each step, with shape (num_steps, num_rows,
              num_cols, num_slices).  The volume shape is the reconstruction shape of the
              model.
            - sinogram: A host numpy array with shape (num_views, num_det_rows,
              num_det_channels).
            - params (dict): 'angles' holds the angle of each view.  'step_of_view' holds the
              time step of each view.  The dict also holds 'num_rotations' and 'num_steps'.
              For 'cone', it also holds the two source distances.
    """
    import mbirtorch
    from ._phantoms_4d import _gen_moving_phantom

    model_type = ModelType(model_type)
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = 2.0 * np.pi * num_rotations * np.arange(num_views) / num_views
    params = {'angles': angles, 'num_rotations': num_rotations, 'num_steps': num_steps}
    if model_type == ModelType.PARALLEL:
        ct_model = mbirtorch.ParallelBeamModel(sinogram_shape, angles)
    elif model_type == ModelType.CONE:
        source_detector_dist = 4 * num_det_channels
        source_iso_dist = source_detector_dist / 2
        ct_model = mbirtorch.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                           source_iso_dist=source_iso_dist)
        params.update(source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
    else:
        raise ValueError(f"model_type must be 'parallel' or 'cone' for 4D data; got {model_type.value!r}.")
    ct_model.set_params(verbose=0)
    if devices is not None:
        ct_model.configure_devices(devices)

    phantom_shape = tuple(int(n) for n in ct_model.get_params('recon_shape'))
    phantom_4d = _gen_moving_phantom(object_type, phantom_shape, num_steps, **object_options)

    step_of_view = (np.arange(num_views) * num_steps) // num_views
    params['step_of_view'] = step_of_view
    sinogram = np.zeros(sinogram_shape, dtype=np.float32)
    for step in range(num_steps):
        views = step_of_view == step
        sinogram[views] = ct_model._forward_project(phantom_4d[step])[views]
    return phantom_4d, sinogram, params



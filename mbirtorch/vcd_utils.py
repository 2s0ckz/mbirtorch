"""This module builds pixel partitions, masks, and sinogram weights.

The partition functions draw random numbers from the global np.random state.  If the caller
passes a ``numpy.random.Generator`` as ``rng``, they draw from that generator instead.  A run
with a fixed seed must draw the same subsets in the same order.  For this reason, the order of
the random calls must not change.
"""

import warnings

import numpy as np
import torch

from . import _sharding


def named_rng(seed, name):
    """
    This function returns a random generator for one named use within a run with a fixed seed.
    The generator depends only on the seed and the name.  As a result, it does not depend on
    which thread uses it.

    Args:
        seed (int or None): The seed of the run.  If the seed is None, the function returns
            None.  Every ``rng`` argument in mbirtorch treats None as the global np.random state.
        name (int or str): A label for the use, such as an iteration number or a short word.

    Returns:
        numpy.random.Generator or None
    """
    if seed is None:
        return None
    if isinstance(name, str):
        # A seed sequence takes only integers, so the name is converted to an integer.
        name = int.from_bytes(name.encode('utf-8'), 'little')
    return np.random.default_rng([int(seed), int(name)])


def named_seed(seed, name):
    """
    This function returns a seed for one named part of a run with a fixed seed.  That part then
    uses the returned seed as its own seed.  If ``seed`` is None, the function returns None.
    """
    generator = named_rng(seed, name)
    return None if generator is None else int(generator.integers(1 << 31))


def get_2d_ror_mask(recon_shape, *, use_ror_mask=True, crop_radius_pixels=0,
                    crop_radius_fraction=0.0):
    """
    This function returns a binary mask for the region of reconstruction.  By default, the mask
    is an ellipse inscribed in the rectangle of size recon_shape[0:2].  The function shrinks
    the axes of the ellipse by ``crop_radius_pixels`` or by ``crop_radius_fraction``.  At most
    one of these two arguments can be nonzero.

    Args:
        recon_shape (tuple): The shape of the recon as (rows, columns, slices) or as
            (rows, columns).
        use_ror_mask (bool or ndarray, optional): The type of mask.  Defaults to True.
            False:
                The mask is all True.
            True:
                The mask is an ellipse inscribed in the reconstruction volume.
            2D array:
                The function uses this binary mask.  It must have shape recon_shape[:2].
        crop_radius_pixels (int): The number of pixels to subtract from each radius of the
            ellipse.  Negative values are treated as 0.
        crop_radius_fraction (float): The fraction to subtract from each radius of the ellipse.

    Returns:
        np.ndarray: A 2D boolean mask.
    """
    if use_ror_mask is False:
        if crop_radius_pixels != 0 and crop_radius_fraction != 0.0:
            raise ValueError('crop_radius_pixels and crop_radius_fraction must be zero '
                             'if use_ror_mask is set to False.')
        return np.ones(recon_shape[:2], dtype=bool)

    elif use_ror_mask is True:
        if crop_radius_pixels != 0 and crop_radius_fraction != 0.0:
            raise ValueError('Only one of crop_radius_pixels and crop_radius_fraction '
                             'can be nonzero.')
        num_recon_rows, num_recon_cols = recon_shape[:2]
        row_center = (num_recon_rows - 1) / 2
        col_center = (num_recon_cols - 1) / 2

        row_radius = row_center - max(int(row_center * crop_radius_fraction), crop_radius_pixels, 0)
        col_radius = col_center - max(int(col_center * crop_radius_fraction), crop_radius_pixels, 0)

        col_coords = np.arange(num_recon_cols) - col_center
        row_coords = np.arange(num_recon_rows) - row_center
        coords = np.meshgrid(col_coords, row_coords)

        mask = (coords[0] / col_radius) ** 2 + (coords[1] / row_radius) ** 2 <= 1.0
        return mask

    else:  # user-provided mask
        if crop_radius_pixels != 0 and crop_radius_fraction != 0.0:
            raise ValueError('crop_radius_pixels and crop_radius_fraction must be zero '
                             'if use_ror_mask is a custom array.')
        mask = np.asarray(use_ror_mask)
        if mask.shape != tuple(recon_shape[:2]):
            raise ValueError(f"Custom use_ror_mask must have shape recon_shape[:2]. "
                             f"Got mask.shape={mask.shape}, expected {tuple(recon_shape[:2])}.")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Custom use_ror_mask must contain only 0s and 1s.")
        return mask


def get_support_radius(recon_shape, delta_voxel_row, delta_voxel_col, use_ror_mask=True):
    """This function returns the largest distance, in ALU, from the rotation axis to the outer
    edge of any voxel that the projectors can update.

    With the default mask, this distance is half of the larger physical width of the volume.
    Otherwise, it is half of the diagonal.
    """
    num_rows, num_cols = recon_shape[:2]
    half_width_row = 0.5 * num_rows * delta_voxel_row
    half_width_col = 0.5 * num_cols * delta_voxel_col
    if use_ror_mask is True:
        return float(max(half_width_row, half_width_col))
    return float(np.hypot(half_width_row, half_width_col))


def gen_pixel_partition(recon_shape, num_subsets, use_ror_mask=True, rng=None):
    """
    This function divides the pixel indices into ``num_subsets`` random subsets of equal size
    for VCD reconstruction.  A run with a fixed seed must give the same subsets every time.  For
    this reason, the order of the random calls must not change.

    Args:
        recon_shape (tuple): The shape of the recon as (rows, columns, slices).
        num_subsets (int): The number of subsets.
        use_ror_mask (bool or ndarray, optional): The mask, as in :func:`get_2d_ror_mask`.
            Defaults to True.
        rng (numpy.random.Generator, optional): The generator that draws the subsets.  None
            means the global np.random state.  Defaults to None.

    Returns:
        np.ndarray: An array in which each row is one subset of pixel indices.  The indices in
        each row are sorted.
    """
    draw = np.random if rng is None else rng
    num_recon_rows, num_recon_cols = recon_shape[:2]
    max_index_val = num_recon_rows * num_recon_cols
    indices = np.arange(max_index_val, dtype=np.int32)

    if use_ror_mask is not False:
        mask = get_2d_ror_mask(recon_shape, use_ror_mask=use_ror_mask)
        indices = indices[mask.flatten() == 1]
    if num_subsets > len(indices):
        num_subsets = len(indices)
        warnings.warn('\nThe number of partition subsets is greater than the number of '
                      'pixels in the region of reconstruction.  \nReducing the number '
                      'of subsets to equal the number of indices.')

    # The function does not permute a single subset, so it does not advance the generator.
    if num_subsets == 1:
        return np.sort(indices).reshape(1, -1)

    num_indices_per_subset = int(np.ceil(len(indices) / num_subsets))
    array_size = num_subsets * num_indices_per_subset
    num_extra_indices = array_size - len(indices)
    indices = draw.permutation(indices)

    # The extra indices that pad the subsets to equal size come from outside the final subset.
    num_non_final_indices = (num_subsets - 1) * num_indices_per_subset
    extra_indices = draw.choice(indices[:num_non_final_indices],
                                size=num_extra_indices, replace=False)
    indices = np.concatenate((indices, extra_indices))

    indices = indices.reshape(num_subsets, indices.size // num_subsets)
    return np.sort(indices, axis=1)


def gen_set_of_pixel_partitions(recon_shape, granularity, device=None, use_ror_mask=True,
                                rng=None):
    """
    This function generates one random pixel partition for each entry of ``granularity``.

    Args:
        recon_shape (tuple): The shape of the recon as (rows, columns, slices).
        granularity (list or tuple): The number of subsets in each partition.
        device (torch.device, optional): The device that holds each partition tensor.
            Defaults to None.
        use_ror_mask (bool or ndarray, optional): The mask, as in :func:`get_2d_ror_mask`.
            Defaults to True.
        rng (numpy.random.Generator, optional): The generator that draws the partitions.  None
            means the global np.random state.  Defaults to None.

    Returns:
        list of torch.Tensor: One int64 tensor for each partition.  Each tensor divides the
        pixels into the given number of subsets.
    """
    partitions = []
    for num_subsets in granularity:
        partition = gen_pixel_partition(recon_shape, num_subsets,
                                        use_ror_mask=use_ror_mask, rng=rng)
        partitions.append(torch.as_tensor(np.ascontiguousarray(partition),
                                          dtype=torch.int64, device=device))
    return partitions


def gen_partition_sequence(partition_sequence, max_iterations):
    """
    This function returns a partition sequence of length ``max_iterations``.  If the input
    sequence is too long, the function truncates it.  If the input is too short, the function
    extends it by repeating its last element.
    """
    partition_sequence = np.array(partition_sequence)
    current_length = partition_sequence.size
    if max_iterations > current_length:
        extension = np.full(max_iterations - current_length, partition_sequence[-1])
        return np.concatenate((partition_sequence, extension))
    return partition_sequence[:max_iterations]


def gen_full_indices(recon_shape, use_ror_mask=True):
    """
    This function returns the indices of every pixel in the region of reconstruction.
    """
    partition = gen_pixel_partition(recon_shape, num_subsets=1, use_ror_mask=use_ror_mask)
    return partition[0]


def gen_weights(sinogram, weight_type):
    """
    This function computes weights for MBIR reconstruction from a noise model.  The weights are
    proportional to the inverse noise variance of each sinogram entry.  A numpy sinogram gives
    numpy weights.  A tensor sinogram gives a tensor on the same device.

    Args:
        sinogram (ndarray or tensor): A 3D array of shape
            (num_views, num_det_rows, num_det_channels).  The function does not accept a
            sinogram that is already divided among the GPUs.  The weights are computed from the
            sinogram in CPU memory.  Then ``prepare_sino_for_devices`` copies both arrays to the GPUs.
        weight_type (str): The noise model.  It is one of the following:
            - 'unweighted': All weights are one.
            - 'transmission': The weights are `exp(-sinogram)`.
            - 'transmission_root': The weights are `exp(-sinogram / 2)`.
            - 'emission': The weights are `1 / (abs(sinogram) + 0.1)`.

    Returns:
        ndarray or tensor: The weights.  They have the same shape, type, and device as the
        sinogram.

    Raises:
        ValueError: If ``sinogram`` is already divided among the GPUs.
        Exception: If ``weight_type`` is not one of the supported options.

    Note:
        For the transmission noise models, a sinogram value above about 5 means that the
        transmission is near zero.  Such values are not physically meaningful in typical
        X-ray imaging.
    """
    # Without this check, a divided sinogram would give a wrong answer with no error below.
    if isinstance(sinogram, _sharding.Shards):
        raise ValueError(
            'gen_weights does not accept a sinogram that has been placed on '
            'the devices.  Compute the weights from the host sinogram first, '
            'then place the sinogram and the weights together with '
            'ct_model.prepare_sino_for_devices(sinogram, weights).')
    xp = torch if isinstance(sinogram, torch.Tensor) else np
    if weight_type == 'unweighted':
        weights = xp.ones_like(sinogram)
    elif weight_type == 'transmission':
        weights = xp.exp(-sinogram)
    elif weight_type == 'transmission_root':
        weights = xp.exp(-sinogram / 2)
    elif weight_type == 'emission':
        weights = 1.0 / (xp.abs(sinogram) + 0.1)
    else:
        raise Exception("gen_weights: undefined weight_type {}".format(weight_type))
    return weights


def estimate_background_cluster_boundaries(sinogram):
    """
    This function estimates the left and right boundaries of the background cluster in the
    sinogram histogram.  The function assumes that the background values are near zero.

    Args:
        sinogram (ndarray): A 3D array with shape (num_views, num_det_rows, num_det_channels).

    Returns:
        left_boundary (float): The left boundary of the background cluster.
        right_boundary (float): The right boundary of the background cluster.
    """
    hist, edges = np.histogram(np.asarray(sinogram).ravel(), bins=400)
    centers = 0.5 * (edges[:-1] + edges[1:])

    peak_indices = []
    if len(hist) > 1 and hist[0] > hist[1]:
        peak_indices.append(0)
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]:
            peak_indices.append(i)

    # The background peak is the histogram peak closest to intensity 0.
    if len(peak_indices) == 0:
        peak_idx = int(np.argmin(np.abs(centers - 0.0)))
    else:
        peak_idx = min(peak_indices, key=lambda i: abs(centers[i] - 0.0))

    # The background cluster ends where the histogram falls to one tenth of the peak height.
    cutoff = 0.1 * hist[peak_idx]

    left_boundary_idx = peak_idx
    while left_boundary_idx > 0 and hist[left_boundary_idx] > cutoff:
        left_boundary_idx -= 1
    right_boundary_idx = peak_idx
    while right_boundary_idx < len(hist) - 1 and hist[right_boundary_idx] > cutoff:
        right_boundary_idx += 1

    return centers[left_boundary_idx], centers[right_boundary_idx]


def gen_weights_mar(ct_model, sinogram, init_recon=None, metal_threshold=None, beta=1.0, gamma=3.0):
    """
    This function computes sinogram weights that reduce metal artifacts:

        weights = exp( -(sinogram/beta) * ( 1 + gamma * delta(metal) ) )

    where delta(metal) is 1 for sinogram entries that contain projections of metal.  An
    ``init_recon`` gives better metal artifact reduction.  Without ``init_recon``, the function
    finds the metal directly in the sinogram.

    Args:
        ct_model (TomographyModel): The model that forward projects the metal mask.  The
            function uses it only when ``init_recon`` is given.
        sinogram (ndarray): A 3D sinogram with shape (num_views, num_det_rows, num_det_channels).
        init_recon (ndarray, optional): An initial reconstruction used to find the metal voxels.
            If it is not given, Otsu's method segments the sinogram directly.
        metal_threshold (float, optional): Values in ``init_recon`` above ``metal_threshold``
            are metal.  If it is not given, Otsu's method segments ``init_recon``.
        beta (float, optional): A scalar greater than 0.  A larger ``beta`` makes the noise
            more uniform.  Too large a value can increase the overall noise level.
        gamma (float, optional): A scalar greater than or equal to 0.  A larger ``gamma``
            reduces the weight of sinogram entries with metal.  Too large a value can reduce
            image quality inside the metal regions.

    Returns:
        (ndarray): The weights, with the same shape as ``sinogram``.

    Raises:
        TypeError: If ``sinogram`` or ``init_recon`` is already divided among the GPUs.

    Note:
        Suppose that ``init_recon`` is given and that the user has not chosen the GPUs of the
        model.  Then this call chooses the GPUs, and later reconstructions with the model use
        them.  The call can also raise ``MemoryCheckError`` if the problem is too large for
        the available GPUs.
    """
    _sharding.reject_shards('gen_weights_mar', sinogram=sinogram,
                            init_recon=init_recon)

    import mbirtorch.preprocess as mtp

    # Without init_recon, Otsu's method splits the sinogram into three classes: metal,
    # non-metal, and background.
    if init_recon is None:
        print("init_recon is not provided. Automatically determine distorted sinogram entries with Otsu's method.")
        [bk_thresh_sino, metal_thresh_sino] = mtp.multi_threshold_otsu(sinogram, classes=3)
        print("Distorted sinogram threshold = ", metal_thresh_sino)
        delta_metal = (np.asarray(sinogram) > metal_thresh_sino).astype(np.float32)

    # With init_recon, the function forward projects the metal mask.
    else:
        if metal_threshold is None:
            print("Metal_threshold calculated with Otsu's method.")
            [bk_threshold, metal_threshold] = mtp.multi_threshold_otsu(init_recon, classes=3)

        print("metal_threshold = ", metal_threshold)
        metal_mask = (np.asarray(init_recon) > metal_threshold).astype(np.float32)
        # This call chooses the GPUs before the projection copies anything to them.
        ct_model._apply_device_policy()
        metal_mask_projected = ct_model.forward_project(metal_mask)

        delta_metal = (np.asarray(metal_mask_projected) > 0.0).astype(np.float32)

    weights = np.exp(-np.asarray(sinogram) * (1 + gamma * delta_metal) / beta)

    return weights

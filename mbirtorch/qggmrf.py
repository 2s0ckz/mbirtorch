"""This module computes the gradient and the Hessian diagonal of the qGGMRF prior at selected
pixels.  It also computes the exact quadratic term of the prior that the line search uses.  All
of the computation runs on one device.  The formulas come from FCI Figure 8.5 and Table 8.1.
In the docstrings, b~ is the surrogate coefficient of :func:`b_tilde_by_definition`, and b is a
neighbor weight.
"""

import numpy as np
import torch

from . import _sharding
from ._memory_estimate import (PRIOR_CHUNK_BYTES, PRIOR_PASS_CYLINDERS_EAGER,
                               QGGMRF_CYLINDERS_EAGER)

_F32_EPS = torch.finfo(torch.float32).eps


def get_b_from_nbr_wts(qggmrf_nbr_wts):
    """This function converts the 3 neighbor weights to 6 weights, one for each direction.  The
    order is [row+1, row-1, col+1, col-1, slice+1, slice-1].  The 6 weights sum to 1.
    """
    import numpy as np
    b = np.array([qggmrf_nbr_wts, qggmrf_nbr_wts]).T.flatten()
    b = b / np.sum(b)
    return tuple(b)


def b_tilde_by_definition(delta, sigma_x, p, q, T):
    """This function computes ``b_tilde = rho'(delta) / (2 delta)`` from Table 8.1 in FCI.
    Before the computation, it clips ``|delta|`` below at ``T * sigma_x * eps_f32`` to avoid
    dividing 0 by 0.
    """
    a_min = T * sigma_x * _F32_EPS
    abs_delta = torch.clamp(torch.abs(delta), min=a_min)
    delta_scale = abs_delta / (T * sigma_x)  # delta_scale has a min of eps

    ds_q_minus_p = delta_scale ** (q - p)
    ds_p_minus_2 = abs_delta ** (p - 2)

    numerator = ds_p_minus_2 / (2 * sigma_x ** p)
    numerator = numerator * ds_q_minus_p * ((q / p) + ds_q_minus_p)
    return numerator / (1 + ds_q_minus_p) ** 2


def get_2_b_tilde(delta, b_for_delta, qggmrf_params):
    """This function computes ``b_for_delta * rho'(delta) / delta``.  This value is the
    coefficient of the symmetric bound surrogate in FCI Eq. 8.16 and 8.17.

    Args:
        delta (tensor): The differences between the center pixel values and the neighbor values.
        b_for_delta (float): The neighbor weight b for this delta.
        qggmrf_params (tuple): The parameters in the form (b, sigma_x, p, q, T).

    Returns:
        tensor: ``b_for_delta * rho'(delta) / delta``.
    """
    b, sigma_x, p, q, T = qggmrf_params
    return b_for_delta * 2.0 * b_tilde_by_definition(delta, sigma_x, p, q, T)


def qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices,
                                           qggmrf_params, left_halo=None,
                                           right_halo=None):
    """
    This function computes the gradient and the Hessian diagonal of the qGGMRF surrogate at
    selected pixels of a reconstruction.  The formulas come from FCI Figure 8.5.  The arguments
    and the first two outputs are those of :func:`qggmrf_terms_at_indices`.

    Returns:
        tuple of two tensors (first_derivative, second_derivative), each of shape
        (N_indices, num_slices).
    """
    return qggmrf_terms_at_indices(flat_recon, recon_shape, pixel_indices, qggmrf_params,
                                   left_halo=left_halo, right_halo=right_halo)[:2]


def qggmrf_terms_at_indices(flat_recon, recon_shape, pixel_indices, qggmrf_params,
                            left_halo=None, right_halo=None):
    """
    This function computes the gradient and the Hessian diagonal of the qGGMRF surrogate at
    selected pixels of a reconstruction.  It also returns the surrogate coefficients of the
    pairs of adjacent slices in each cylinder, which the exact line search uses.  The formulas
    come from FCI Figure 8.5.

    Args:
        flat_recon (tensor): A 2D reconstruction array with shape
            (num_recon_rows x num_recon_cols, num_slices).  The array may hold only a
            contiguous block of the slices of the full reconstruction.
        recon_shape (tuple of ints): The shape (num_recon_rows, num_recon_cols,
            num_recon_slices) of the full reconstruction.  This function does not use
            the number of slices.
        pixel_indices (int tensor): A 1D array of shape (N_indices,).  It holds the indices
            of the voxel cylinders to update.  The indices refer to the flattened
            (num_recon_rows x num_recon_cols) grid.
        qggmrf_params (tuple): The parameters (b, sigma_x, p, q, T).  b is the tuple of 6
            direction weights from :func:`get_b_from_nbr_wts`.
        left_halo (tensor or None): A 1D array of shape (num_rows x num_cols,) that holds the
            slice just before the slices in ``flat_recon``.  None gives a reflected boundary.
        right_halo (tensor or None): A 1D array like ``left_halo`` that holds the slice just
            after the slices in ``flat_recon``.  None gives a reflected boundary.

    Returns:
        tuple of three tensors (first_derivative, second_derivative, slice_b_tilde_2).  The
        first two have shape (N_indices, num_slices).  ``slice_b_tilde_2`` has shape
        (N_indices, num_slices - 1).  Entry k holds 2 b~ for the pair of slices k and k + 1,
        without the neighbor weight b.

    Raises:
        TypeError: If ``flat_recon``, ``left_halo``, or ``right_halo`` is a ``Shards`` object,
            which holds an array split across GPUs.
    """
    _sharding.reject_shards('qggmrf_terms_at_indices',
                            flat_recon=flat_recon, left_halo=left_halo,
                            right_halo=right_halo)
    # Neighbor weight order is [row+1, row-1, col+1, col-1, slice+1, slice-1].
    b, sigma_x, p, q, T = qggmrf_params
    num_rows, num_cols = recon_shape[0], recon_shape[1]

    # Slice axis term.  With no halo, the boundary delta is zero, which gives a reflected boundary.
    cylinders = flat_recon[pixel_indices]                     # (N, S)
    left_val = (left_halo[pixel_indices][:, None] if left_halo is not None
                else cylinders[:, :1])
    right_val = (right_halo[pixel_indices][:, None] if right_halo is not None
                 else cylinders[:, -1:])
    delta = torch.cat((cylinders[:, :1] - left_val,
                       cylinders[:, 1:] - cylinders[:, :-1],
                       right_val - cylinders[:, -1:]), dim=1)  # (N, S+1)

    # b_for_delta is 1 here, and the slice-direction b is applied below.
    b_tilde_2 = get_2_b_tilde(delta, 1.0, qggmrf_params)
    b_tilde_2_delta = b_tilde_2 * delta

    # delta is xs - xr, where xs is the pixel being updated.  So the slice+1 neighbor uses
    # -delta[1:].
    b_slice_plus, b_slice_minus = b[4], b[5]
    gradient = -b_slice_plus * b_tilde_2_delta[:, 1:] + b_slice_minus * b_tilde_2_delta[:, :-1]
    hessian = b_slice_plus * b_tilde_2[:, 1:] + b_slice_minus * b_tilde_2[:, :-1]
    slice_b_tilde_2 = b_tilde_2[:, 1:-1]

    # In-slice term.  Each neighbor index is clamped to the grid, so a neighbor outside the
    # grid gives a delta of zero.
    row_index = pixel_indices // num_cols
    col_index = pixel_indices % num_cols

    xs0 = cylinders                                            # (N, S)

    offsets_and_b = [((1, 0), b[0]), ((-1, 0), b[1]),
                     ((0, 1), b[2]), ((0, -1), b[3])]
    for (dr, dc), b_value in offsets_and_b:
        r = (row_index + dr).clamp(0, num_rows - 1)
        c = (col_index + dc).clamp(0, num_cols - 1)
        neighbor = flat_recon[r * num_cols + c]                # (N, S)
        delta = xs0 - neighbor

        b_tilde_2 = get_2_b_tilde(delta, b_value, qggmrf_params)
        gradient = gradient + b_tilde_2 * delta
        hessian = hessian + b_tilde_2

    return gradient, hessian, slice_b_tilde_2


def _accumulation_dtype(device):
    """This function returns the data type for the sums of the exact line search.  It returns
    float64, except on MPS, which has no float64."""
    return torch.float32 if torch.device(device).type == 'mps' else torch.float64


def _sum_last_two_axes(values):
    """This function sums an array over its last two axes.  It sums the last axis in the data
    type of ``values`` and the other axis in the data type of :func:`_accumulation_dtype`."""
    partial = torch.sum(values, dim=-1)
    return torch.sum(partial.to(_accumulation_dtype(values.device)), dim=-1)


def qggmrf_prior_pass_at_indices(flat_recon, recon_shape, pixel_indices, qggmrf_params,
                                 forward_grad, forward_hess, valid, left_halo=None,
                                 right_halo=None):
    """
    This function computes the gradient and the Hessian diagonal of the qGGMRF surrogate at
    selected pixels, as :func:`qggmrf_terms_at_indices` does.  It also computes four sums for
    each pair of adjacent slices.  :func:`qggmrf_slice_pair_sum` combines these sums into the
    slice term of the exact line search.

    For the pair of slices k and k + 1, let w = b 2 b~ u_k u_(k+1), where u = 1 / (forward_hess
    + prior Hessian) and b is the sum of the two slice neighbor weights.  Let g = forward_grad +
    prior gradient.  The four sums over the pixels are sum(w g_k g_(k+1)), sum(w g_k),
    sum(w g_(k+1)), and sum(w).

    Args:
        flat_recon, recon_shape, pixel_indices, qggmrf_params, left_halo, right_halo: The
            same as for :func:`qggmrf_terms_at_indices`.
        forward_grad (tensor): The gradient of the forward model term at the pixels, with
            shape (N_indices, num_slices).
        forward_hess (tensor): The Hessian diagonal of the forward model term at the pixels,
            with shape (N_indices, num_slices).
        valid (tensor): An array of shape (N_indices, 1) that holds 1 for each pixel that
            counts in the sums and 0 for each pixel that does not.

    Returns:
        tuple (first_derivative, second_derivative, slice_sums).  The first two have shape
        (N_indices, num_slices).  ``slice_sums`` has shape (4, num_slices - 1).
    """
    b = qggmrf_params[0]
    gradient, hessian, slice_b_tilde_2 = qggmrf_terms_at_indices(
        flat_recon, recon_shape, pixel_indices, qggmrf_params, left_halo=left_halo,
        right_halo=right_halo)
    g = forward_grad + gradient
    u = 1.0 / (forward_hess + hessian)
    w = (b[4] + b[5]) * slice_b_tilde_2 * u[:, :-1] * u[:, 1:] * valid
    dtype = _accumulation_dtype(flat_recon.device)
    slice_sums = torch.stack((torch.sum(w * g[:, :-1] * g[:, 1:], dim=0, dtype=dtype),
                              torch.sum(w * g[:, :-1], dim=0, dtype=dtype),
                              torch.sum(w * g[:, 1:], dim=0, dtype=dtype),
                              torch.sum(w, dim=0, dtype=dtype)))
    return gradient, hessian, slice_sums


def qggmrf_prior_pass_in_chunks(flat_recon, recon_shape, pixel_indices, qggmrf_params,
                                forward_grad, forward_hess, left_halo=None, right_halo=None,
                                chunk_function=None, max_chunk_bytes=PRIOR_CHUNK_BYTES):
    """
    This function computes the same results as :func:`qggmrf_prior_pass_at_indices`, but it
    works on one chunk of pixels at a time.  The temporary arrays of one chunk use at most
    ``max_chunk_bytes`` bytes.

    Every chunk has the same number of pixels, so a compiled ``chunk_function`` compiles once.
    The last chunk is padded by repeating the last pixel.  The padded results are discarded,
    and the padded pixels do not count in the slice sums.

    Args:
        flat_recon, recon_shape, pixel_indices, qggmrf_params, forward_grad, forward_hess,
            left_halo, right_halo: The same as for :func:`qggmrf_prior_pass_at_indices`.
        chunk_function (callable or None): The function that computes one chunk.  It has the
            signature of :func:`qggmrf_prior_pass_at_indices`, and it may be a compiled form
            of that function.  None uses that function itself.
        max_chunk_bytes (int): The most bytes that the temporary arrays of one chunk use.

    Returns:
        tuple (first_derivative, second_derivative, slice_sums), as for
        :func:`qggmrf_prior_pass_at_indices`.
    """
    if chunk_function is None:
        chunk_function = qggmrf_prior_pass_at_indices
    num_indices = pixel_indices.shape[0]
    num_slices = flat_recon.shape[1]
    # A device can hold no slices.
    bytes_per_pixel = max(1, PRIOR_PASS_CYLINDERS_EAGER * num_slices
                          * flat_recon.element_size())
    chunk_size = max(1, int(max_chunk_bytes) // bytes_per_pixel)
    valid = torch.ones((min(chunk_size, num_indices), 1), dtype=flat_recon.dtype,
                       device=flat_recon.device)
    if num_indices <= chunk_size:
        return chunk_function(flat_recon, recon_shape, pixel_indices, qggmrf_params,
                              forward_grad, forward_hess, valid, left_halo=left_halo,
                              right_halo=right_halo)

    gradient = torch.empty((num_indices, num_slices), dtype=flat_recon.dtype,
                           device=flat_recon.device)
    hessian = torch.empty_like(gradient)
    slice_sums = 0
    for start in range(0, num_indices, chunk_size):
        stop = min(start + chunk_size, num_indices)
        chunk = [pixel_indices[start:stop], forward_grad[start:stop],
                 forward_hess[start:stop]]
        chunk_valid = valid
        num_padded = chunk_size - (stop - start)
        if num_padded > 0:
            chunk = [torch.cat((a, a[-1:].expand((num_padded,) + a.shape[1:])))
                     for a in chunk]
            chunk_valid = torch.cat((valid[:stop - start],
                                     torch.zeros_like(valid[:num_padded])))
        chunk_gradient, chunk_hessian, chunk_sums = chunk_function(
            flat_recon, recon_shape, chunk[0], qggmrf_params, chunk[1], chunk[2],
            chunk_valid, left_halo=left_halo, right_halo=right_halo)
        gradient[start:stop] = chunk_gradient[:stop - start]
        hessian[start:stop] = chunk_hessian[:stop - start]
        slice_sums = slice_sums + chunk_sums
    return gradient, hessian, slice_sums


def qggmrf_slice_pair_sum(slice_sums, slice_offsets=None):
    """
    This function returns the sum over the pairs of adjacent slices of b 2 b~ d_k d_(k+1).
    The direction is d = -(g - c) u, where c holds one value for each slice.  The sum is
    computed from the slice sums of :func:`qggmrf_prior_pass_at_indices`.

    Args:
        slice_sums (tensor): The slice sums, with shape (4, num_slices - 1).
        slice_offsets (tensor or None): The values c, with shape (num_slices,).  None means
            that c is zero.

    Returns:
        tensor: A scalar.
    """
    if slice_offsets is None:
        return torch.sum(slice_sums[0])
    c = slice_offsets.to(slice_sums.dtype)
    return torch.sum(slice_sums[0] - c[1:] * slice_sums[1] - c[:-1] * slice_sums[2]
                     + c[:-1] * c[1:] * slice_sums[3])


def qggmrf_slice_pair_sum_of_direction(slice_b_tilde_2, delta, qggmrf_params):
    """
    This function returns the sum over the pairs of adjacent slices of b 2 b~ d_k d_(k+1),
    computed directly from the direction d.  The first axes of ``delta`` may index volumes.

    Args:
        slice_b_tilde_2 (tensor): The coefficients from :func:`qggmrf_terms_at_indices`, with
            shape (..., N_indices, num_slices - 1).
        delta (tensor): The direction d, with shape (..., N_indices, num_slices).
        qggmrf_params (tuple): The parameters (b, sigma_x, p, q, T).

    Returns:
        tensor: The sum for each volume, with shape (...).
    """
    b = qggmrf_params[0]
    return _sum_last_two_axes((b[4] + b[5]) * slice_b_tilde_2
                              * delta[..., :-1] * delta[..., 1:])


def qggmrf_edge_term(delta, pixel_indices, recon_shape, qggmrf_params, first_slice_is_edge,
                     last_slice_is_edge):
    """
    This function returns the part of the approximate quadratic term sum(hessian d^2) that
    comes from missing neighbors.  The prior Hessian uses the pixel itself as a missing
    neighbor at the edges of the grid and at the end slices of the volume.  Each such neighbor
    adds b 2 b~(0) d^2.  The first axes of ``delta`` may index volumes.

    Args:
        delta (tensor): The direction d, with shape (..., N_indices, num_slices).
        pixel_indices (int tensor): The pixel indices of the subset, with shape (N_indices,).
        recon_shape (tuple of ints): The shape (num_rows, num_cols, num_slices).
        qggmrf_params (tuple): The parameters (b, sigma_x, p, q, T).
        first_slice_is_edge (bool): True if the first slice of ``delta`` has no neighbor
            before it.
        last_slice_is_edge (bool): True if the last slice of ``delta`` has no neighbor after
            it.

    Returns:
        tensor: The sum for each volume, with shape (...).
    """
    b = qggmrf_params[0]
    num_rows, num_cols = recon_shape[0], recon_shape[1]
    row_index = pixel_indices // num_cols
    col_index = pixel_indices % num_cols
    # Neighbor weight order is [row+1, row-1, col+1, col-1, slice+1, slice-1].
    edge_b = (b[0] * (row_index == num_rows - 1) + b[1] * (row_index == 0)
              + b[2] * (col_index == num_cols - 1) + b[3] * (col_index == 0))
    # The weights of the missing slice neighbors are added in the first and last slices.
    num_slices = delta.shape[-1]
    slice_index = torch.arange(num_slices, device=delta.device)
    slice_b = ((b[5] if first_slice_is_edge else 0.0) * (slice_index == 0)
               + (b[4] if last_slice_is_edge else 0.0) * (slice_index == num_slices - 1))
    weight = edge_b.to(delta.dtype)[:, None] + slice_b.to(delta.dtype)[None, :]
    total = _sum_last_two_axes(weight * delta * delta)
    zero = torch.zeros((), dtype=delta.dtype, device=delta.device)
    return get_2_b_tilde(zero, 1.0, qggmrf_params).to(total.dtype) * total


def qggmrf_pair_sum(flat_recon, pixel_indices, delta, pairs, valid, b_value, qggmrf_params):
    """
    This function returns the sum over a list of pairs of neighboring pixels of
    b 2 b~ d_s d_r over all slices.  Both pixels of each pair are in the subset.  The first
    axes of ``flat_recon`` and ``delta`` may index volumes.

    Args:
        flat_recon (tensor): The reconstruction, with shape (..., num_pixels, num_slices).
        pixel_indices (int tensor): The pixel indices of the subset, with shape (N_indices,).
        delta (tensor): The direction d, with shape (..., N_indices, num_slices).
        pairs (int tensor): The pairs, with shape (M, 2).  Each entry is a row of
            ``pixel_indices`` and ``delta``.
        valid (tensor): An array of shape (M, 1) that holds 1 for each pair that counts and 0
            for each pair that does not.
        b_value (float): The sum of the two neighbor weights of the direction of the pairs.
        qggmrf_params (tuple): The parameters (b, sigma_x, p, q, T).

    Returns:
        tensor: The sum for each volume, with shape (...).
    """
    first = pairs[:, 0].long()
    second = pairs[:, 1].long()
    x_difference = (flat_recon[..., pixel_indices[first], :]
                    - flat_recon[..., pixel_indices[second], :])
    b_tilde_2 = get_2_b_tilde(x_difference, b_value, qggmrf_params)
    return _sum_last_two_axes(b_tilde_2 * delta[..., first, :] * delta[..., second, :] * valid)


def qggmrf_pair_sum_in_chunks(flat_recon, pixel_indices, delta, pairs, num_pairs, b_value,
                              qggmrf_params, chunk_function=None,
                              max_chunk_bytes=PRIOR_CHUNK_BYTES):
    """
    This function computes the same sum as :func:`qggmrf_pair_sum` for the first
    ``num_pairs`` pairs of ``pairs``, one chunk of pairs at a time.  The temporary arrays of
    one chunk use at most ``max_chunk_bytes`` bytes.  Every chunk has the same number of pairs,
    so a compiled ``chunk_function`` compiles once.

    Args:
        flat_recon, pixel_indices, delta, b_value, qggmrf_params: The same as for
            :func:`qggmrf_pair_sum`, without the axis of volumes.
        pairs (int tensor): The pairs, with shape (M, 2).  The rows after the first
            ``num_pairs`` rows are not used.
        num_pairs (int): The number of pairs that count.
        chunk_function (callable or None): The function that computes one chunk.  It has the
            signature of :func:`qggmrf_pair_sum`, and it may be a compiled form of that
            function.  None uses that function itself.
        max_chunk_bytes (int): The most bytes that the temporary arrays of one chunk use.

    Returns:
        tensor: A scalar.
    """
    if chunk_function is None:
        chunk_function = qggmrf_pair_sum
    num_rows = pairs.shape[0]
    num_slices = flat_recon.shape[1]
    total = torch.zeros((), dtype=_accumulation_dtype(flat_recon.device),
                        device=flat_recon.device)
    if num_pairs == 0 or num_slices == 0:
        return total
    bytes_per_pair = QGGMRF_CYLINDERS_EAGER * num_slices * flat_recon.element_size()
    chunk_size = min(num_rows, max(1, int(max_chunk_bytes) // bytes_per_pair))
    row_numbers = torch.arange(chunk_size, device=flat_recon.device)
    for start in range(0, num_pairs, chunk_size):
        chunk_pairs = pairs[start:start + chunk_size]
        if chunk_pairs.shape[0] < chunk_size:
            chunk_pairs = torch.cat((chunk_pairs, chunk_pairs[-1:].expand(
                chunk_size - chunk_pairs.shape[0], 2)))
        valid = (row_numbers < num_pairs - start).to(flat_recon.dtype)[:, None]
        total = total + chunk_function(flat_recon, pixel_indices, delta, chunk_pairs, valid,
                                       b_value, qggmrf_params)
    return total


def qggmrf_boundary_pair_sum(last_slice, next_slice, delta_last, delta_next, qggmrf_params):
    """
    This function returns the sum of b 2 b~ d_s d_r over the pairs between the last slice of
    one block of slices and the first slice of the next block.  Each input holds the values at
    the pixels of the subset, with shape (N_indices,).

    Args:
        last_slice (tensor): The reconstruction in the last slice of the first block.
        next_slice (tensor): The reconstruction in the first slice of the next block.
        delta_last (tensor): The direction in the last slice of the first block.
        delta_next (tensor): The direction in the first slice of the next block.
        qggmrf_params (tuple): The parameters (b, sigma_x, p, q, T).

    Returns:
        tensor: A scalar.
    """
    b = qggmrf_params[0]
    b_tilde_2 = get_2_b_tilde(last_slice - next_slice, b[4] + b[5], qggmrf_params)
    terms = b_tilde_2 * delta_last * delta_next
    return torch.sum(terms.to(_accumulation_dtype(terms.device)))


def qggmrf_exact_quadratic(prior_hess, slice_b_tilde_2, delta, flat_recon, pixel_indices,
                           recon_shape, qggmrf_params, pair_lists, first_slice_is_edge=True,
                           last_slice_is_edge=True):
    """
    This function returns the exact quadratic term of the qGGMRF surrogate along the direction
    d, FCI Eq. 8.24, for a subset held in one array.  The term is computed as
    sum(prior_hess d^2) minus the edge term and minus the pair terms of the slice pairs and of
    the pairs in each slice.  The first axes of ``delta`` and ``flat_recon`` may index
    volumes.

    Args:
        prior_hess (tensor): The Hessian diagonal of the prior, with the shape of ``delta``.
        slice_b_tilde_2 (tensor): The slice coefficients from :func:`qggmrf_terms_at_indices`.
        delta (tensor): The direction d, with shape (..., N_indices, num_slices).
        flat_recon (tensor): The reconstruction, with shape (..., num_pixels, num_slices).
        pixel_indices (int tensor): The pixel indices of the subset, with shape (N_indices,).
        recon_shape (tuple of ints): The shape (num_rows, num_cols, num_slices).
        qggmrf_params (tuple): The parameters (b, sigma_x, p, q, T).
        pair_lists (tuple of tensors): The tuple (row_pairs, row_valid, col_pairs, col_valid)
            of the subset, in the form of the arguments of :func:`qggmrf_pair_sum`.
        first_slice_is_edge, last_slice_is_edge (bool): The same as for
            :func:`qggmrf_edge_term`.

    Returns:
        tensor: The term for each volume, with shape (...).
    """
    b = qggmrf_params[0]
    row_pairs, row_valid, col_pairs, col_valid = pair_lists
    return (_sum_last_two_axes(prior_hess * delta * delta)
            - qggmrf_edge_term(delta, pixel_indices, recon_shape, qggmrf_params,
                               first_slice_is_edge, last_slice_is_edge)
            - qggmrf_slice_pair_sum_of_direction(slice_b_tilde_2, delta, qggmrf_params)
            - qggmrf_pair_sum(flat_recon, pixel_indices, delta, row_pairs, row_valid,
                              b[0] + b[1], qggmrf_params)
            - qggmrf_pair_sum(flat_recon, pixel_indices, delta, col_pairs, col_valid,
                              b[2] + b[3], qggmrf_params))


def qggmrf_terms_batched(flat_stack, recon_shape, pixel_indices, qggmrf_params):
    """
    This function does the computation of :func:`qggmrf_terms_at_indices` for a stack of
    volumes on one device.  The first axis indexes the volumes.  Each volume is
    computed independently of the others.

    Args:
        flat_stack (tensor): A 3D array with shape
            (num_volumes, num_recon_rows x num_recon_cols, num_recon_slices).
        recon_shape (tuple of ints): The shape (num_recon_rows, num_recon_cols,
            num_recon_slices) of one volume.  This function does not use
            the number of slices.
        pixel_indices (int tensor): A 1D array of shape (N_indices,).  It holds the indices
            of the voxel cylinders to update in every volume.  The indices refer to the
            flattened (num_recon_rows x num_recon_cols) grid.
        qggmrf_params (tuple): The parameters (b, sigma_x, p, q, T).  b is the tuple of 6
            direction weights from :func:`get_b_from_nbr_wts`.

    Returns:
        tuple of three tensors (first_derivative, second_derivative, slice_b_tilde_2).  The
        first two have shape (num_volumes, N_indices, num_recon_slices).  ``slice_b_tilde_2``
        has shape (num_volumes, N_indices, num_recon_slices - 1), as in
        :func:`qggmrf_terms_at_indices`.

    Raises:
        TypeError: If ``flat_stack`` is a ``Shards`` object, which holds an array split
            across GPUs.
    """
    _sharding.reject_shards('qggmrf_terms_batched',
                            flat_stack=flat_stack)
    # Neighbor weight order is [row+1, row-1, col+1, col-1, slice+1, slice-1].
    b, sigma_x, p, q, T = qggmrf_params
    num_rows, num_cols = recon_shape[0], recon_shape[1]

    # Slice axis term.  The boundary delta is zero, which gives a reflected boundary.
    cylinders = flat_stack[:, pixel_indices]                   # (B, N, S)
    left_val = cylinders[..., :1]
    right_val = cylinders[..., -1:]
    delta = torch.cat((cylinders[..., :1] - left_val,
                       cylinders[..., 1:] - cylinders[..., :-1],
                       right_val - cylinders[..., -1:]), dim=-1)  # (B, N, S+1)

    # b_for_delta is 1 here, and the slice-direction b is applied below.
    b_tilde_2 = get_2_b_tilde(delta, 1.0, qggmrf_params)
    b_tilde_2_delta = b_tilde_2 * delta

    # The slice+1 neighbor uses -delta[1:] and the slice-1 neighbor uses delta[:-1].
    b_slice_plus, b_slice_minus = b[4], b[5]
    gradient = (-b_slice_plus * b_tilde_2_delta[..., 1:]
                + b_slice_minus * b_tilde_2_delta[..., :-1])
    hessian = (b_slice_plus * b_tilde_2[..., 1:]
               + b_slice_minus * b_tilde_2[..., :-1])
    slice_b_tilde_2 = b_tilde_2[..., 1:-1]

    # In-slice term.  Each neighbor index is clamped to the grid.
    row_index = pixel_indices // num_cols
    col_index = pixel_indices % num_cols
    xs0 = cylinders                                            # (B, N, S)

    offsets_and_b = [((1, 0), b[0]), ((-1, 0), b[1]),
                     ((0, 1), b[2]), ((0, -1), b[3])]
    for (dr, dc), b_value in offsets_and_b:
        r = (row_index + dr).clamp(0, num_rows - 1)
        c = (col_index + dc).clamp(0, num_cols - 1)
        neighbor = flat_stack[:, r * num_cols + c]             # (B, N, S)
        delta = xs0 - neighbor

        b_tilde_2 = get_2_b_tilde(delta, b_value, qggmrf_params)
        gradient = gradient + b_tilde_2 * delta
        hessian = hessian + b_tilde_2

    return gradient, hessian, slice_b_tilde_2


def prox_gradient_at_indices(recon, prox_input, pixel_indices, sigma_prox):
    """
    This function computes the gradient of the proximal map prior at selected pixels of a
    reconstruction.

    Args:
        recon (tensor): A 2D reconstruction array with shape
            (num_recon_rows x num_recon_cols, num_recon_slices).
        prox_input (tensor): A 2D array with the same shape as ``recon``.
        pixel_indices (int tensor): A 1D array of shape (N_indices,).  It holds the indices
            of the voxel cylinders to update.  The indices refer to the flattened
            (num_recon_rows x num_recon_cols) grid.
        sigma_prox (float): The standard deviation parameter of the proximal map.

    Returns:
        tensor of shape (N_indices, num_recon_slices): The gradient of the proximal map term
        at the selected pixels.
    """
    cur_diff = recon[pixel_indices] - prox_input[pixel_indices]
    return (1.0 / (sigma_prox ** 2.0)) * cur_diff


def qggmrf_loss(full_recon, qggmrf_params):
    """
    This function computes the qGGMRF prior loss of a reconstruction on the host.  It is
    intended for debugging with small reconstructions.

    Args:
        full_recon (ndarray): A 3D volume with axes (rows, cols, slices).
        qggmrf_params (tuple): The parameters (b, sigma_x, p, q, T).

    Returns:
        float: The prior loss.

    Raises:
        TypeError: If ``full_recon`` is a ``Shards`` object, which holds an array split
            across GPUs.
    """
    # np.asarray turns a ``Shards`` object into a 0-d object array without an error.
    _sharding.reject_shards('qggmrf_loss', full_recon=full_recon)
    b, sigma_x, p, q, T = qggmrf_params
    full_recon = np.asarray(full_recon)

    b_per_axis = [(b[j] + b[j + 1]) / (2 * sum(b)) for j in [0, 2, 4]]

    def rho_ref(delta):
        # rho from Table 8.1 in FCI.
        a_min = T * sigma_x * np.finfo(np.float32).eps
        abs_delta = np.clip(np.abs(delta), a_min, None)
        delta_scale = abs_delta / (T * sigma_x)  # delta_scale has a min of eps
        ds_q_minus_p = delta_scale ** (q - p)
        numerator = (abs_delta ** p) / (p * sigma_x ** p)
        numerator *= ds_q_minus_p
        return numerator / (1 + ds_q_minus_p)

    loss = 0.0
    for axis in [0, 1, 2]:
        cur_delta = np.diff(full_recon, axis=axis)
        loss += float(np.sum(b_per_axis[axis] * rho_ref(cur_delta)))

    return loss

"""This module computes the gradient and the Hessian diagonal of the qGGMRF prior at selected
pixels.  All of the computation runs on one device.  The formulas come from FCI Figure 8.5 and
Table 8.1.
"""

import numpy as np
import torch

from . import _sharding

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
    selected pixels of a reconstruction.  The formulas come from FCI Figure 8.5.

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
        tuple of two tensors (first_derivative, second_derivative), each of shape
        (N_indices, num_slices).

    Raises:
        TypeError: If ``flat_recon``, ``left_halo``, or ``right_halo`` is a ``Shards`` object,
            which holds an array split across GPUs.
    """
    _sharding.reject_shards('qggmrf_gradient_and_hessian_at_indices',
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

    return gradient, hessian


def qggmrf_gradient_and_hessian_batched(flat_stack, recon_shape, pixel_indices,
                                        qggmrf_params):
    """
    This function does the computation of :func:`qggmrf_gradient_and_hessian_at_indices` for
    a stack of volumes on one device.  The first axis indexes the volumes.  Each volume is
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
        tuple of two tensors (first_derivative, second_derivative), each of
        shape (num_volumes, N_indices, num_recon_slices).

    Raises:
        TypeError: If ``flat_stack`` is a ``Shards`` object, which holds an array split
            across GPUs.
    """
    _sharding.reject_shards('qggmrf_gradient_and_hessian_batched',
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

    return gradient, hessian


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

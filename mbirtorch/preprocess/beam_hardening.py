import warnings

import numpy as np
import scipy
import torch
from mbirtorch import _sharding

from . import _pipeline
from .recon_utils import segment_plastic_metal, compute_scaling_factor

__all__ = ['BH_correction']


# The helpers below accept a sinogram as one tensor or as a Shards container with one
# tensor per device.

def _ps_map(fn, *xs):
    """This function applies fn to the inputs, one device tensor at a time, and returns the
    result in the same form as the inputs."""
    if isinstance(xs[0], _sharding.Shards):
        parts = [fn(*[x.tensors[i] for x in xs])
                 for i in range(xs[0].placement.n_devices)]
        return _sharding.Shards(parts, xs[0].placement)
    return fn(*xs)


def _ps_sum(fn, *xs):
    """This function applies the scalar reduction fn to each device tensor and returns the sum
    as a float."""
    if isinstance(xs[0], _sharding.Shards):
        return sum(float(fn(*[x.tensors[i] for x in xs]))
                   for i in range(xs[0].placement.n_devices))
    return float(fn(*xs))


def _ps_max(fn, x):
    """This function applies the scalar reduction fn to each nonempty device tensor and
    returns the maximum as a float."""
    if isinstance(x, _sharding.Shards):
        return max(float(fn(t)) for t in x.tensors if t.numel() > 0)
    return float(fn(x))


def _ps_numel(x):
    """This function returns the total number of elements in all device tensors, as an int."""
    if isinstance(x, _sharding.Shards):
        return sum(t.numel() for t in x.tensors)
    return x.numel()


def _ps_item(x, idx):
    """This function returns the value at a (view, row, col) index of the whole sinogram, as
    a float."""
    if isinstance(x, _sharding.Shards):
        pl = x.placement
        for t, (_d, (v0, v1)) in zip(x.tensors, pl.shard_ranges()):
            if v0 <= idx[0] < v1:
                return float(t[idx[0] - v0, idx[1], idx[2]])
        raise IndexError(f'view {idx[0]} outside the sharded axis')
    return float(x[idx])


def _ps_argmin3d(x):
    """This function returns the (view, row, col) index of the minimum in the whole sinogram,
    and the minimum value.  If several views share the minimum, it returns the first view."""
    if not isinstance(x, _sharding.Shards):
        return _argmin_3d(x)
    pl = x.placement
    best_idx, best_val = None, None
    for t, (_d, (v0, _v1)) in zip(x.tensors, pl.shard_ranges()):
        if t.numel() == 0:
            continue
        (v, r, c), val = _argmin_3d(t)
        if best_val is None or float(val) < best_val:
            best_idx, best_val = (v + v0, r, c), float(val)
    return best_idx, best_val


def BH_correction(sino, alpha, batch_size=64, devices=None):
    """
    This function applies a polynomial beam hardening correction to a sinogram.  It replaces
    each value ``p`` by ``alpha[0] p + alpha[1] p**2 + alpha[2] p**3 + ...``.

    Args:
        sino (numpy array or tensor): The sinogram, with shape (num_views, num_det_rows,
            num_det_channels).
        alpha (sequence of float): The polynomial coefficients.  The coefficient ``alpha[k]``
            multiplies ``p**(k+1)``.
        batch_size (int, optional): The number of views processed at a time.  Defaults to 64.
        devices (sequence or None, optional): The devices to divide the views among.  None
            means all visible CUDA devices, up to the number set by ``MBIRTORCH_NUM_DEVICES``.
            If there is no CUDA device, None means the default device.  Defaults to None.

    Returns:
        numpy.ndarray: The corrected sinogram, with the shape of ``sino``.

    Example:
        >>> sino = BH_correction(sino, alpha=[1.0, 0.2, 0.1])
    """
    _pipeline.reject_shards('BH_correction', sino=sino)

    alpha = np.asarray(alpha)

    def kernel(sino_batch):
        corrected = torch.zeros_like(sino_batch)
        for k in range(len(alpha)):
            corrected = corrected + float(alpha[k]) * torch.pow(sino_batch, k + 1)
        return corrected

    return _pipeline.map_view_batches(sino, kernel, batch_size,
                                     devices=_pipeline.permitted_devices(devices))


def _generate_metal_exponent_list(num_metal, max_order):
    """
    This function returns every exponent tuple with total degree from 1 to max_order.  It
    sorts the tuples by total degree.

    Args:
        num_metal (int): The number of metals, which is the length of each tuple.
        max_order (int): The maximum total degree of the polynomial.

    Returns:
        list[tuple[int]]: The exponent tuples.  Each tuple gives the exponents of one term.
    """
    combinations = []

    def generate_recursive(current_combination, remaining_terms):
        if remaining_terms == 0:
            total_degree = sum(current_combination)
            if 0 < total_degree <= max_order:
                combinations.append(tuple(current_combination))
            return

        for power in range(max_order + 1):
            generate_recursive(current_combination + [power], remaining_terms - 1)

    generate_recursive([], num_metal)

    combinations.sort(key=lambda x: sum(x))
    return combinations


def _est_plastic_metal_sinos_from_recon(recon, num_metal, ct_model,
                                        radial_margin=None, top_margin=None, bottom_margin=None):
    """
    This function segments the plastic and metal regions of a reconstruction and forward
    projects them.  It returns the unnormalized sinograms p, m0, m1, ... that the beam
    hardening model uses.

    Args:
        recon (ndarray or tensor): The reconstruction.
        num_metal (int): The number of metal types to segment.
        ct_model: The model used for the forward projection.
        radial_margin, top_margin, bottom_margin (int or None, optional): The margins of the
            segmentation mask.  None means a margin proportional to the volume size, as in
            ``segment_plastic_metal``.  Defaults to None.

    Returns:
        plastic_sino_est (tensor): The unnormalized estimate of the plastic sinogram.
        metal_sino_est (list of tensor): The unnormalized estimates of the metal sinograms.
    """
    recon = ct_model._shard_recon(recon)

    plastic_mask, metal_masks, plastic_scale, metal_scales = segment_plastic_metal(
        recon, num_metal=num_metal, radial_margin=radial_margin, top_margin=top_margin,
        bottom_margin=bottom_margin)

    plastic_sino_est = _ps_map(lambda t: plastic_scale * t,
                               ct_model._forward_project(plastic_mask, output_sharded=True))

    metal_sino_est = []
    for mask in metal_masks:
        masked = (_sharding.Shards([mk * t for mk, t in zip(mask.tensors, recon.tensors)],
                                   recon.placement)
                  if isinstance(recon, _sharding.Shards) else mask * recon)
        m = ct_model._forward_project(masked, output_sharded=True)
        metal_sino_est.append(m)

    return plastic_sino_est, metal_sino_est


def _get_column_H(col_index, plastic_sino_est, metal_sino_est, H_exponent_list):
    """
    This function computes column ``col_index`` of the matrix H.  The column is the monomial
    ``p^e0 * m_0^e1 * ... * m_{n-1}^en``, where the exponents are
    ``H_exponent_list[col_index]``.

    Args:
        col_index (int): The index of the column to compute.
        plastic_sino_est (tensor): The normalized estimate of the plastic sinogram.
        metal_sino_est (list of tensor): The normalized estimates of the metal sinograms
            [m_0, m_1, ..., m_{n-1}].
        H_exponent_list (list of tuple): The exponent tuple of each column of H.

    Returns:
        tensor: The column of H, with the same shape as p and m_i.
    """
    exponents = H_exponent_list[col_index]
    assert len(exponents) == 1 + len(metal_sino_est), "Mismatch between exponent tuple and number of sinograms."

    col = None
    for arr, exp in zip([plastic_sino_est] + list(metal_sino_est), exponents):
        if exp == 0:
            continue
        term = arr if exp == 1 else arr ** exp
        col = term if col is None else col * term
    if col is None:
        col = torch.ones_like(plastic_sino_est)
    return col

def _get_row_H(pixel_index, plastic_sino_est, metal_sino_est, H_exponent_list):
    """
    This function computes the row of the matrix H for one sinogram pixel.

    Args:
        pixel_index (tuple of int): The (view, row, col) index of the pixel.
        plastic_sino_est (tensor): The normalized estimate of the plastic sinogram.
        metal_sino_est (list of tensor): The normalized estimates of the metal sinograms
            [m_0, m_1, ..., m_{n-1}].
        H_exponent_list (list of tuple): The exponent tuple of each column of H.

    Returns:
        ndarray: The row of H.
    """
    pi = _ps_item(plastic_sino_est, pixel_index)
    mi = [_ps_item(m, pixel_index) for m in metal_sino_est]
    row_vals = []
    for exps in H_exponent_list:
        val = (pi ** exps[0])
        for mk, ek in zip(mi, exps[1:]):
            val = val * (mk ** ek)
        row_vals.append(val)
    return np.asarray(row_vals, dtype=np.float32)


def _argmin_3d(x):
    """This function returns the (view, row, col) index of the minimum of a 3D array, as
    Python ints, and the minimum value.  If several entries share the minimum, it returns the
    first one."""
    num_views, num_rows, num_channels = x.shape
    per_view = x.reshape(num_views, -1)              # (V, R*C)
    per_view_min, plane_argmin = torch.min(per_view, dim=1)
    view = int(torch.argmin(per_view_min))
    row, col = divmod(int(plane_argmin[view]), num_channels)
    return (view, row, col), per_view_min[view]


# The constraint y - H_m θ_m >= 0 is applied only at pixels where some metal estimate exceeds
# this value.  The metal estimates are normalized to a maximum of 1.
_METAL_SUPPORT_FLOOR = 1e-3


def _find_most_violated_constraints(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms):
    """
    This function finds the most violated constraint of each of the two types in the beam
    hardening model.  The constraints are ``H_p[i,:] θ_p ≥ 0`` and ``y[i] − H_m[i,:] θ_m ≥ 0``.
    The function checks the second constraint only where some metal estimate exceeds
    ``_METAL_SUPPORT_FLOOR``.

    Returns:
        idx_min_Sp (tuple of int): The (view, row, col) index of the smallest entry of Sp.
        v_min_Sp (scalar): The value of Sp at that index.
        idx_min_residual (tuple of int): The (view, row, col) index of the smallest entry of
            (y − Sm).
        v_min_residual (scalar): The value of (y − Sm) at that index.
    """
    num_cols = len(H_exponent_list)
    # Setting the p exponent to zero gives the coefficient of p in each column.
    p_coeff_exponents = [(0,) + exps[1:] for exps in H_exponent_list]

    def build_sp(p, *ms):
        sp = torch.zeros_like(p)
        for i in range(0, 1 + num_cross_terms):
            sp = sp + float(theta[i]) * _get_column_H(i, p, list(ms), p_coeff_exponents)
        return sp

    def build_y_minus_sm(y, p, *ms):
        out = y
        for j in range(1 + num_cross_terms, num_cols):
            out = out - float(theta[j]) * _get_column_H(j, p, list(ms), H_exponent_list)
        return out

    Sp = _ps_map(build_sp, plastic_sino_est, *metal_sino_est)
    y_minus_Sm = _ps_map(build_y_minus_sm, measured_sino, plastic_sino_est, *metal_sino_est)

    # theta cannot change y - Sm at a pixel with no metal, so such a pixel is never a constraint.
    def mask_residual(ym, *ms):
        support = torch.zeros_like(ym, dtype=torch.bool)
        for metal in ms:
            support = support | (metal > _METAL_SUPPORT_FLOOR)
        inf = torch.tensor(float('inf'), dtype=ym.dtype, device=ym.device)
        return torch.where(support, ym, inf)

    ymSm_masked = _ps_map(mask_residual, y_minus_Sm, *metal_sino_est)
    idx_min_Sp, v_min_Sp = _ps_argmin3d(Sp)
    idx_min_residual, v_min_residual = _ps_argmin3d(ymSm_masked)

    return idx_min_Sp, v_min_Sp, idx_min_residual, v_min_residual



def _estimate_BH_model_params_using_OSQP(P, q, A, u):
    """
    This function solves ``minimize 0.5 θᵀ P θ + qᵀ θ subject to A θ ≤ u`` with OSQP.  If
    ``A`` or ``u`` is None, it solves the unconstrained problem with a linear solve.

    Args:
        P (ndarray): The matrix of the quadratic term.
        q (ndarray): The vector of the linear term.
        A (ndarray): The inequality constraint matrix.
        u (ndarray): The right side of the inequality constraints.

    Returns:
        ndarray or None: The solution θ, or None if the constrained solve fails.
    """
    P_numpy = np.asarray(P, dtype=np.float64)
    q_numpy = np.asarray(q, dtype=np.float64)

    if A is None or u is None:
        theta = np.linalg.solve(P_numpy, -q_numpy)
        return np.asarray(theta, dtype=np.float32)

    # The import is local because most callers never fit a beam hardening model.
    from scipy.sparse import csc_matrix
    import osqp
    A_numpy = np.asarray(A, dtype=np.float64)
    u_numpy = np.asarray(u, dtype=np.float64)

    P_sparse = csc_matrix(P_numpy)
    A_sparse = csc_matrix(A_numpy)

    solver = osqp.OSQP()
    solver.setup(P=P_sparse, q=q_numpy, A=A_sparse, l=None, u=u_numpy, alpha=1.0, verbose=0)
    result = solver.solve()

    # On failure, OSQP sets result.info.status and fills result.x with 2143289344.0.  So the
    # code accepts only a solved status with finite values.
    status = str(result.info.status).strip().lower()
    theta = np.asarray(result.x, dtype=np.float64)
    if not status.startswith('solved') or not np.all(np.isfinite(theta)):
        return None

    return np.asarray(theta, dtype=np.float32)

def _compute_entry_for_OSQP(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta):
    """This function computes the matrix P and the vector q of the quadratic program that
    OSQP solves."""
    num_cols = len(H_exponent_list)

    HtH = np.zeros((num_cols, num_cols), dtype=np.float64)
    Hty = np.zeros(num_cols, dtype=np.float64)

    # The code computes the upper triangle of HtH and copies it to the lower triangle.
    def column(i):
        return _ps_map(lambda p, *ms: _get_column_H(i, p, list(ms), H_exponent_list),
                       plastic_sino_est, *metal_sino_est)

    for i in range(num_cols):
        h_i = column(i)
        Hty[i] = _ps_sum(lambda a, b: torch.sum(a * b), h_i, measured_sino)
        for j in range(i, num_cols):
            h_j = column(j)
            dot_ij = _ps_sum(lambda a, b: torch.sum(a * b), h_i, h_j)
            HtH[i, j] = dot_ij
            if i != j:
                HtH[j, i] = dot_ij

    cross_degree = [sum(exponent) for exponent in H_exponent_list[0:1+num_cross_terms]]
    metal_degree = [sum(exponent) for exponent in H_exponent_list[1+num_cross_terms:]]

    # When alpha > 0, a term of higher degree has a larger penalty.
    weights = np.asarray(cross_degree + metal_degree, dtype=np.float64)
    weight_matrix = np.diag(1 + weights ** alpha)

    scaling_const = np.trace(HtH) / np.trace(weight_matrix)
    lambda_reg = beta * scaling_const

    P = HtH + lambda_reg * weight_matrix
    q = -Hty

    return P, q

def _estimate_BH_model_params(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta, num_constraint_update_iter=10, tolerance=-1e-5):
    """
    This function estimates the parameters of the polynomial beam hardening model.  It solves

        minimize_θ   0.5‖Hθ − y‖² + 0.5λ‖θ‖²_Λ
        subject to   H_p[i,:] θ_p ≥ 0 and y[i] − H_m[i,:] θ_m ≥ 0.

    In each iteration, the function adds the most violated constraint of each type and
    solves again.  H_p holds the plastic column and the cross term columns.  H_m holds the
    metal-only columns.

    Args:
        plastic_sino_est (tensor): The normalized estimate of the plastic sinogram.
        metal_sino_est (list of tensor): The normalized estimates of the metal sinograms.
        measured_sino (tensor): The measured sinogram.
        H_exponent_list (list of tuple[int]): The exponent tuple of each column of H.
        num_cross_terms (int): The number of cross terms (plastic × metal).  The remaining
            terms after the plastic and cross terms are metal-only.
        alpha (float): The regularization exponent.  A higher alpha penalizes terms of higher
            degree more.
        beta (float): The scale factor of the regularization strength.
        num_constraint_update_iter (int): The maximum number of constraint update iterations.
        tolerance (float): The iterations stop when no constraint is below this value.

    Returns:
        theta (ndarray): The model parameters, one for each column of H.
    """
    num_cols = len(H_exponent_list)
    dp = 1 + num_cross_terms

    C_p = []
    C_m = []

    P, q = _compute_entry_for_OSQP(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta)
    A = np.zeros((0, num_cols))  # No constraints are active yet.
    u = np.zeros((0,))

    theta = _estimate_BH_model_params_using_OSQP(P, q, A=None, u=None)

    for iter in range(num_constraint_update_iter):
        idx_min_Sp, v_min_Sp, idx_min_residual, v_min_residual = _find_most_violated_constraints(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms)

        # (1) Hp θp ≥ 0  ->  (-Hp) θ ≤ 0
        if v_min_Sp < tolerance and (idx_min_Sp not in C_p):
            p_coeff_exponents = [(0,) + exps[1:] for exps in H_exponent_list]
            row_p = _get_row_H(idx_min_Sp, plastic_sino_est, metal_sino_est, p_coeff_exponents)
            # row_p[:dp] is negated so that the constraint reads (-Hp) θ <= 0.
            A_p = np.concatenate([-row_p[:dp], np.zeros((num_cols - dp,))])
            u_p = np.array([0.0])
            A = np.vstack([A, A_p[None, :]])
            u = np.concatenate([u, u_p])
            C_p.append(idx_min_Sp)

        # (2) y − Hm θm ≥ 0  ->  (Hm) θ ≤ y
        if v_min_residual < tolerance and (idx_min_residual not in C_m):
            row_m = _get_row_H(idx_min_residual, plastic_sino_est, metal_sino_est, H_exponent_list)
            # row_m[dp:] keeps its sign so that the constraint reads Hm θ <= y.
            A_m = np.concatenate([np.zeros(dp), row_m[dp:]])
            # The bound is clamped at 0, because a negative measurement would force large
            # negative metal coefficients.
            u_m = np.array([max(_ps_item(measured_sino, idx_min_residual), 0.0)])
            A = np.vstack([A, A_m[None, :]])
            u = np.concatenate([u, u_m])
            C_m.append(idx_min_residual)

        if (v_min_Sp >= tolerance) and (v_min_residual >= tolerance):
            break
        theta_new = _estimate_BH_model_params_using_OSQP(P, q, A, u)
        if theta_new is None:
            # The function keeps the last theta that OSQP solved.
            warnings.warn("OSQP failed to solve the constrained beam-hardening fit; keeping the "
                          "parameters from the previous constraint iteration.", RuntimeWarning)
            break
        theta = theta_new
    return theta


def _correct_plastic_sinogram(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms, num_metal_terms, p_normalization, gamma):
    """
    This function returns the part of the measured sinogram that comes from the plastic.
    The result is
    ``p_normalization * max(y - H_metal θ_m, 0) / max(H_plastic θ_p, γ mean(H_plastic θ_p))``.
    The columns of H are p, the cross terms p*m, p*m^2, ..., and the metal terms m, m^2, ....

    Args:
        measured_sino (tensor): The measured sinogram.
        plastic_sino_est (tensor): The normalized estimate of the plastic sinogram.
        metal_sino_est (list of tensor): The normalized estimates of the metal sinograms.
        theta (ndarray): The coefficients of the polynomial terms in H.
        H_exponent_list (list of tuple): The exponent tuple of each column of H.
        num_cross_terms (int): The number of cross terms that contain both p and a metal.
        num_metal_terms (int): The number of metal-only terms in H.
        p_normalization (float): The normalization factor of p.
        gamma (float): The stabilization factor γ.  It sets the smallest denominator as a
            fraction of the mean of H_plastic θ_p.

    Returns:
        corrected_plastic_sino (tensor): The plastic sinogram with beam hardening corrected.
    """

    # The denominator uses the first 1 + num_cross_terms columns of H.
    p_coeff_exponents = [(0,) + exps[1:] for exps in H_exponent_list]

    def build_sp(p, *ms):
        sp = torch.zeros_like(p)
        for i in range(0, 1 + num_cross_terms):
            sp = sp + float(theta[i]) * _get_column_H(i, p, list(ms), p_coeff_exponents)
        return sp

    def build_y_minus_sm(y, p, *ms):
        out = y
        for j in range(1 + num_cross_terms, 1 + num_cross_terms + num_metal_terms):
            out = out - float(theta[j]) * _get_column_H(j, p, list(ms), H_exponent_list)
        return torch.clamp(out, min=0)

    Sp = _ps_map(build_sp, plastic_sino_est, *metal_sino_est)
    y_minus_Sm = _ps_map(build_y_minus_sm, measured_sino, plastic_sino_est, *metal_sino_est)

    if not isinstance(Sp, _sharding.Shards):
        mean_plastic_coef = torch.mean(Sp)
    else:
        mean_plastic_coef = _ps_sum(torch.sum, Sp) / _ps_numel(Sp)
    Sp_floor = gamma * mean_plastic_coef

    if float(mean_plastic_coef) <= 0:
        warnings.warn("Mean of Sp is negative", RuntimeWarning)

    def clamp_and_divide(sp, ym):
        floor = (Sp_floor if torch.is_tensor(Sp_floor)
                 else torch.as_tensor(Sp_floor, dtype=sp.dtype, device=sp.device))
        return p_normalization * ym / torch.maximum(sp, floor)

    corrected_plastic_sino = _ps_map(clamp_and_divide, Sp, y_minus_Sm)

    return corrected_plastic_sino

def _estimate_plastic_scaling(plastic_sino_est, metal_sino_est, measured_sino, plastic_sino_corrected):
    # This function returns a least squares scale fit over the pixels with plastic and no metal.
    def keep_plastic_only(x, p, *ms):
        condition = (p != 0)
        for metal in ms:
            condition = condition & (metal == 0)
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        return torch.where(condition, x, zero)

    plastic_sino_scale = compute_scaling_factor(
        _ps_map(keep_plastic_only, measured_sino, plastic_sino_est, *metal_sino_est),
        _ps_map(keep_plastic_only, plastic_sino_corrected, plastic_sino_est, *metal_sino_est))
    return plastic_sino_scale

def correct_sino_plastic_metal(ct_model, measured_sino, recon, num_metal=1, order=3, alpha=1, beta=0.002, gamma=0.1, num_constraint_update_iter=10,
                               radial_margin=None, top_margin=None, bottom_margin=None):
    """
    This function corrects beam hardening in the sinogram of an object with plastic and metal
    parts.  It fits a beam hardening model to the sinogram.  The result is the corrected plastic
    sinogram plus the forward projections of the metal regions.

    Args:
        ct_model: The CT model of the scan.
        measured_sino (ndarray): The measured sinogram.
        recon (ndarray or tensor): A reconstruction of the 3D volume.  The function segments it
            into plastic and metal regions.
        num_metal (int, optional): The number of metal materials to segment and correct.
            Defaults to 1.
        order (int, optional): The maximum total degree of the correction polynomial.
            Defaults to 3.
        alpha (float, optional): The exponent of the degree in the regularization weights.
            Higher values penalize terms of higher degree more.  Defaults to 1.
        beta (float, optional): The strength of the ridge regularization.  Defaults to 0.002.
        gamma (float, optional): The stabilization factor.  It sets the smallest denominator
            in the plastic correction as a fraction of the mean of the denominator.  Defaults
            to 0.1.
        num_constraint_update_iter (int, optional): The maximum number of constraint update
            iterations.  Defaults to 10.
        radial_margin, top_margin, bottom_margin (int or None, optional): The margins of the
            segmentation mask.  None means a margin proportional to the volume size, as in
            ``segment_plastic_metal``.  Defaults to None.

    Returns:
        ndarray: The corrected sinogram, with the shape of ``measured_sino``.
    """
    metal_exponent_list = _generate_metal_exponent_list(num_metal, order)
    cross_exponent_list = _generate_metal_exponent_list(num_metal, order - 1)
    num_metal_terms = len(metal_exponent_list)
    num_cross_terms = len(cross_exponent_list)

    # Each tuple holds the exponents of (p, m_0, ..., m_{num_metal-1}).  The order is the
    # linear plastic term, then the cross terms, then the metal-only terms.
    H_exponent_list = (
            [(1,) + (0,) * num_metal] +
            [(1, *t) for t in cross_exponent_list] +
            [(0, *t) for t in metal_exponent_list])

    measured_sino = ct_model.prepare_sino_for_devices(measured_sino)

    plastic_sino_est, metal_sino_est = _est_plastic_metal_sinos_from_recon(
        recon, num_metal, ct_model, radial_margin=radial_margin, top_margin=top_margin,
        bottom_margin=bottom_margin)
    plastic_sino_scale = _ps_max(lambda t: torch.max(torch.abs(t)), plastic_sino_est)
    metal_sino_scale = [_ps_max(lambda t: torch.max(torch.abs(t)), arr) for arr in metal_sino_est]
    # An empty estimate would give NaNs.  The test ``not > 0`` also catches a NaN scale.
    if not float(plastic_sino_scale) > 0:
        raise ValueError(
            "The estimated plastic sinogram is empty (the plastic segmentation class contains no "
            "voxels).  Check the input reconstruction, num_metal, and the cylindrical-mask margins.")
    for metal_index, scale in enumerate(metal_sino_scale):
        if not float(scale) > 0:
            raise ValueError(
                f"The estimated sinogram for metal {metal_index} is empty (its segmentation class "
                f"contains no voxels).  num_metal={num_metal} may be too large for this object.")
    plastic_sino_est = _ps_map(lambda t: t / plastic_sino_scale, plastic_sino_est)
    metal_sino_est = [_ps_map(lambda t, n=norm: t / n, arr)
                      for arr, norm in zip(metal_sino_est, metal_sino_scale)]

    theta = _estimate_BH_model_params(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta, num_constraint_update_iter)

    plastic_sino_corrected = _correct_plastic_sinogram(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list,
                                                       num_cross_terms, num_metal_terms, float(plastic_sino_scale), gamma)

    plastic_sino_corrected_scale = _estimate_plastic_scaling(plastic_sino_est, metal_sino_est, measured_sino, plastic_sino_corrected)

    def combine(corrected, *ms):
        out = plastic_sino_corrected_scale * corrected
        for arr, norm in zip(ms, metal_sino_scale):
            out = out + arr * norm
        return out

    corrected_sino = _ps_map(combine, plastic_sino_corrected, *metal_sino_est)
    return ct_model._gather_sinogram(corrected_sino)


def fit_beam_hardening_curve(linear_projection, target_projection, num_parameters=5, zero_offset_normalized=True):
    """
    This function fits a beam hardening curve to paired samples.  The curve is
    ``f(p) = -log( sum_{i=1..N} exp(theta_i - i * theta_0 * p) )``, with
    ``N = num_parameters - 1``.  If ``zero_offset_normalized`` is True, the function shifts
    the curve so that ``f(0) = 0``.

    Args:
        linear_projection (np.ndarray): The samples of the ideal linear projection or the path
            length.
        target_projection (np.ndarray): The samples of the beam hardened projection, paired
            with ``linear_projection``.
        num_parameters (int, optional): The total number of fitted parameters.
            Defaults to 5.
        zero_offset_normalized (bool, optional): If True, the function uses the shifted
            curve with ``f(0) = 0``.  Defaults to True.

    Returns:
        ndarray: The fitted parameters
            ``[theta_0, theta_1, ..., theta_{num_parameters-1}]``.

    Example:
        >>> fitted_params = fit_beam_hardening_curve(
        ...     sinogram.ravel(), sinogram_nonlinear.ravel(), num_parameters=5)
        >>> y_pred = apply_beam_hardening_curve(sinogram_test, fitted_params)
    """
    num_parameters = int(num_parameters)
    if num_parameters < 2:
        raise ValueError(
            'fit_beam_hardening_curve: num_parameters must be at least 2.')

    linear_projection = np.asarray(linear_projection, dtype=np.float64)
    target_projection = np.asarray(target_projection, dtype=np.float64)

    if linear_projection.size != target_projection.size:
        raise ValueError(
            'fit_beam_hardening_curve: Input and target projection arrays must contain the same number of samples.')

    linear_projection = linear_projection.ravel()
    target_projection = target_projection.ravel()

    valid_mask = (
        np.isfinite(linear_projection) & np.isfinite(target_projection)
        & (linear_projection > 1e-6) & (target_projection > 1e-6)
    )
    linear_projection = linear_projection[valid_mask]
    target_projection = target_projection[valid_mask]

    if linear_projection.size == 0:
        raise ValueError(
            'fit_beam_hardening_curve: No valid training samples remain.')

    initial_params = np.zeros(num_parameters, dtype=np.float64)
    initial_params[0] = 1.0

    max_nfev = 20 * num_parameters
    solver_options = dict(
        loss="linear",
        method="trf",
        max_nfev=max_nfev,
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
        verbose=1,
    )

    optimization_result = scipy.optimize.least_squares(
        _beam_hardening_curve_residuals,
        initial_params,
        args=(linear_projection, target_projection, zero_offset_normalized),
        **solver_options,
    )

    if not optimization_result.success:
        warnings.warn(
            f'fit_beam_hardening_curve: Beam-hardening curve fit did not converge: {optimization_result.message}',
            RuntimeWarning)

    return optimization_result.x


def apply_beam_hardening_curve(linear_projection, params, zero_offset_normalized=True):
    """
    This function applies a fitted beam hardening curve to linear projection values.

    If ``zero_offset_normalized`` is True, the function uses

        f(p) = log(sum_i exp(theta_i))
               - log(sum_i exp(theta_i - i * theta_0 * p)),

    which gives ``f(0) = 0``.  If False, the function uses

        f(p) = -log(sum_i exp(theta_i - i * theta_0 * p)).

    Args:
        linear_projection (np.ndarray): The linear projection values.
        params (np.ndarray): The parameters ``[theta_0, theta_1, ..., theta_N]``.
        zero_offset_normalized (bool, optional): If True, the function uses the curve with
            ``f(0) = 0``.  Defaults to True.

    Returns:
        ndarray: The beam hardened projection values, with the shape of
            ``linear_projection``.
    """
    linear_projection = np.asarray(linear_projection, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64).reshape(-1)

    if params.size < 2:
        raise ValueError(
            'Expected at least 2 parameters: theta_0 and one log-weight.')

    theta_0 = params[0]
    theta_rest = params[1:]

    log_sum_exp_p = np.full_like(linear_projection, -np.inf, dtype=np.float64)
    for i, theta_i in enumerate(theta_rest, start=1):
        exponent = theta_i - i * theta_0 * linear_projection
        log_sum_exp_p = np.logaddexp(log_sum_exp_p, exponent)

    if not zero_offset_normalized:
        return -log_sum_exp_p

    log_sum_exp_0 = -np.inf
    for theta_i in theta_rest:
        log_sum_exp_0 = np.logaddexp(log_sum_exp_0, theta_i)

    return log_sum_exp_0 - log_sum_exp_p


def _beam_hardening_curve_residuals(params, linear_projection, target_projection, zero_offset_normalized):
    """
    This function returns the fitted values minus the target values, for
    :func:`scipy.optimize.least_squares`.
    """
    fitted_projection = apply_beam_hardening_curve(
        linear_projection, params,
        zero_offset_normalized=zero_offset_normalized)

    return fitted_projection.ravel() - target_projection.ravel()


def fit_inverse_beam_hardening_curve(forward_params, vmin=0.0, vmax=5.0, degree=10, num_samples=2000, zero_offset_normalized=True):
    """
    This function fits a Chebyshev polynomial that inverts a beam hardening curve.  The
    polynomial maps beam hardened projections to linear projections.

    Args:
        forward_params (np.ndarray): The beam hardening curve parameters from
            :func:`fit_beam_hardening_curve`.
        vmin (float, optional): The smallest projection value to correct.  Defaults to 0.0.
        vmax (float, optional): The largest projection value to correct.  Defaults to 5.0.
        degree (int, optional): The degree of the Chebyshev polynomial.  Defaults to 10.
        num_samples (int, optional): The number of samples used in the fit.  Defaults to
            2000.
        zero_offset_normalized (bool, optional): The value of this argument used to fit
            ``forward_params``.  Defaults to True.

    Returns:
        tuple: ``(cheb_coeffs, y_domain)``.  ``cheb_coeffs`` is an ndarray of length
            ``degree + 1``.  ``y_domain`` is ``(vmin, vmax)``, which
            :func:`apply_inverse_beam_hardening_curve` needs.

    Example:
        >>> forward_params = fit_beam_hardening_curve(
        ...     sinogram.ravel(), sinogram_nonlinear.ravel(),
        ...     num_parameters=5)
        >>> cheb_coeffs, y_domain = fit_inverse_beam_hardening_curve(
        ...     forward_params,
        ...     vmin=0.0,
        ...     vmax=float(sinogram_nonlinear.max()),
        ...     degree=10)
        >>> sinogram_linearized = apply_inverse_beam_hardening_curve(
        ...     sinogram_nonlinear, cheb_coeffs, y_domain)
    """
    forward_params = np.asarray(forward_params, dtype=np.float64).reshape(-1)
    vmin = float(vmin)
    vmax = float(vmax)
    degree = int(degree)
    num_samples = int(num_samples)

    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: require finite vmin < vmax.')
    if degree < 1:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: degree must be at least 1.')
    if num_samples < degree + 1:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: num_samples must be at least '
            'degree + 1.')

    # The effective attenuation is the slope h'(0) of the forward curve.
    epsilon = 1e-6
    forward_at_zero = apply_beam_hardening_curve(
        0.0, forward_params,
        zero_offset_normalized=zero_offset_normalized)

    forward_at_epsilon = apply_beam_hardening_curve(
        epsilon, forward_params,
        zero_offset_normalized=zero_offset_normalized)

    effective_attenuation = float(
        (forward_at_epsilon - forward_at_zero) / epsilon)

    path_min = 0.0
    path_max = max(path_min + 1.0, abs(vmax) + 1.0)
    # The loop doubles path_max at most this many times.
    max_expand_iterations = 64
    for _ in range(max_expand_iterations):
        y_at_path_max = apply_beam_hardening_curve(
            path_max, forward_params,
            zero_offset_normalized=zero_offset_normalized)
        if np.isfinite(y_at_path_max) and y_at_path_max >= vmax:
            break
        path_max = path_min + 2.0 * (path_max - path_min)
    else:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: could not expand the path '
            'length grid enough to cover vmax.')

    p_grid = np.linspace(
        path_min, path_max, 4 * num_samples, dtype=np.float64)
    y_grid = apply_beam_hardening_curve(
        p_grid, forward_params,
        zero_offset_normalized=zero_offset_normalized)

    valid_mask = np.isfinite(p_grid) & np.isfinite(y_grid)
    p_grid = p_grid[valid_mask]
    y_grid = y_grid[valid_mask]
    if p_grid.size < degree + 1:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: too few finite forward '
            'samples.')

    sort_idx = np.argsort(y_grid)
    y_sorted = y_grid[sort_idx]
    p_sorted = p_grid[sort_idx]
    y_unique, unique_idx = np.unique(y_sorted, return_index=True)
    p_unique = p_sorted[unique_idx]

    if y_unique.size < degree + 1 or y_unique[-1] <= y_unique[0]:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: forward model is not '
            'invertible on the sampled grid.')
    if vmax > y_unique[-1]:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: sampled forward '
            f'model only reaches {y_unique[-1]:.6g}, below vmax={vmax:.6g}.')
    if vmin < y_unique[0]:
        warnings.warn(
            'fit_inverse_beam_hardening_curve: vmin is below the forward '
            'value at zero path length; low-end inverse samples will be '
            'clamped to zero.',
            RuntimeWarning)

    y_samples = np.linspace(vmin, vmax, num_samples, dtype=np.float64)
    p_samples = np.interp(
        y_samples, y_unique, p_unique,
        left=p_unique[0], right=p_unique[-1])
    linearized_projection_samples = p_samples * effective_attenuation

    y_scaled = 2.0 * (y_samples - vmin) / (vmax - vmin) - 1.0
    cheb_coeffs = np.polynomial.chebyshev.chebfit(
        y_scaled, linearized_projection_samples, deg=degree)

    return cheb_coeffs, (vmin, vmax)


def apply_inverse_beam_hardening_curve(beam_hardened_projection, cheb_coeffs, y_domain, clip=False):
    """
    This function applies a fitted Chebyshev inverse to beam hardened projections.  The
    result is the linearized projections.

    Args:
        beam_hardened_projection (np.ndarray): The beam hardened projection values.  The
            array may have any shape.
        cheb_coeffs (np.ndarray): The coefficients returned by
            :func:`fit_inverse_beam_hardening_curve`.
        y_domain (tuple): The projection range ``(vmin, vmax)`` used in the fit.
        clip (bool, optional): If True, the function clips the input values to ``y_domain``.
            If False, the function warns when it extrapolates.  Defaults to False.

    Returns:
        ndarray: The linearized projection values, with the shape of
            ``beam_hardened_projection``.
    """
    beam_hardened_projection = np.asarray(
        beam_hardened_projection, dtype=np.float64)
    cheb_coeffs = np.asarray(cheb_coeffs, dtype=np.float64).reshape(-1)
    y_min, y_max = float(y_domain[0]), float(y_domain[1])

    if y_max <= y_min:
        raise ValueError(
            'apply_inverse_beam_hardening_curve: y_domain must '
            'satisfy y_max > y_min.')

    if clip:
        y_eval = np.clip(beam_hardened_projection, y_min, y_max)
    else:
        if (np.any(beam_hardened_projection < y_min)
                or np.any(beam_hardened_projection > y_max)):
            warnings.warn(
                'apply_inverse_beam_hardening_curve: inputs lie '
                'outside the fitted y_domain; extrapolated values may be '
                'unreliable.',
                RuntimeWarning)
        y_eval = beam_hardened_projection

    y_scaled = 2.0 * (y_eval - y_min) / (y_max - y_min) - 1.0
    return np.polynomial.chebyshev.chebval(y_scaled, cheb_coeffs)

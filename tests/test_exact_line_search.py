"""The exact line search.  The subset update computes the exact prior term of the line search
from the approximate term sum(prior_hess * d^2) and from correction sums.  These tests compare
the result with a direct computation in float64 on the full volume."""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import qggmrf


def _direct_prior_terms(flat_recon, recon_shape, pixel_indices, d, qggmrf_params):
    # This function returns the linear and the quadratic prior terms of the line search, FCI
    # Eq. 8.24.  Each term is a sum over each pixel s of the subset and each of its 6 neighbors
    # r of b 2 b~ d_s (x_s - x_r) or of b 2 b~ d_s (d_s - d_r), where d_r is 0 outside the
    # subset.  A missing neighbor at an edge of the volume is the pixel itself, so it adds 0.
    b = qggmrf_params[0]
    num_rows, num_cols = recon_shape[0], recon_shape[1]
    cylinders = flat_recon[pixel_indices]
    delta = torch.cat((torch.zeros_like(cylinders[:, :1]), cylinders[:, 1:] - cylinders[:, :-1],
                       torch.zeros_like(cylinders[:, :1])), dim=1)
    b_tilde_2 = qggmrf.get_2_b_tilde(delta, 1.0, qggmrf_params)
    d_ext = torch.cat((d[:, :1], d, d[:, -1:]), dim=1)
    linear = torch.sum(b[4] * b_tilde_2[:, 1:] * d * -delta[:, 1:])
    linear = linear + torch.sum(b[5] * b_tilde_2[:, :-1] * d * delta[:, :-1])
    quadratic = torch.sum(b[4] * b_tilde_2[:, 1:] * d * (d - d_ext[:, 2:]))
    quadratic = quadratic + torch.sum(b[5] * b_tilde_2[:, :-1] * d * (d - d_ext[:, :-2]))
    position = torch.full((num_rows * num_cols,), -1, dtype=torch.int64)
    position[pixel_indices] = torch.arange(pixel_indices.numel())
    row_index = pixel_indices // num_cols
    col_index = pixel_indices % num_cols
    for (dr, dc), b_value in [((1, 0), b[0]), ((-1, 0), b[1]), ((0, 1), b[2]), ((0, -1), b[3])]:
        neighbor = ((row_index + dr).clamp(0, num_rows - 1) * num_cols
                    + (col_index + dc).clamp(0, num_cols - 1))
        difference = cylinders - flat_recon[neighbor]
        b_tilde_2 = qggmrf.get_2_b_tilde(difference, b_value, qggmrf_params)
        neighbor_position = position[neighbor]
        d_neighbor = (d[neighbor_position.clamp(min=0)]
                      * (neighbor_position >= 0).to(d.dtype)[:, None])
        linear = linear + torch.sum(b_tilde_2 * d * difference)
        quadratic = quadratic + torch.sum(b_tilde_2 * d * (d - d_neighbor))
    return float(linear), float(quadratic)


@pytest.mark.parametrize('geometry', ['parallel', 'cone'])
def test_exact_term_matches_the_direct_sum(geometry, monkeypatch):
    # The test records the inputs of each subset update and the step size alpha that the update
    # returns.  It then computes alpha = -theta1 / theta2 directly in float64.  Two CPU devices
    # cover the pairs that cross a boundary between devices.  The cone-beam model uses slice
    # damping.  The grid has 512 pixels, so the 4 subsets do not overlap.
    np.random.seed(0)
    _, sinogram, params = mbirtorch.gen_demo_data(model_type=geometry, object_type='cube',
                                                  num_views=16, num_det_rows=6,
                                                  num_det_channels=40)
    if geometry == 'parallel':
        model = mbirtorch.ParallelBeamModel(sinogram.shape, params['angles'])
    else:
        model = mbirtorch.ConeBeamModel(sinogram.shape, params['angles'],
                                        source_detector_dist=params['source_detector_dist'],
                                        source_iso_dist=params['source_iso_dist'])
    recon_shape = (32, 16, 6)
    model.set_params(recon_shape=recon_shape, verbose=0, granularity=[4],
                     partition_sequence=[0], use_ror_mask=False, positivity_flag=False)
    model.configure_devices(devices=['cpu', 'cpu'])

    def as_float64(shards):
        return [tensor.double() for tensor in shards.tensors]

    # Each subset update records the recon, the error sinogram, the direction d, the projection
    # A d, and the returned alpha.
    updates = []
    sino_weights = []
    create = model.create_vcd_subset_updater

    def recording_create(fm_hessian, weights, prox_input=None):
        sino_weights[:] = [weights]
        updater = create(fm_hessian, weights, prox_input=prox_input)

        def recording_updater(flat_recon, error_sinogram, pixel_indices, subset_index=0):
            updates.append(dict(recon=torch.cat(flat_recon.tensors, dim=1).double(),
                                error=as_float64(error_sinogram),
                                indices=torch.as_tensor(pixel_indices).long()))
            result = updater(flat_recon, error_sinogram, pixel_indices, subset_index=subset_index)
            updates[-1]['alpha'] = float(result[3])
            return result
        recording_updater.stage_halos = updater.stage_halos
        recording_updater.stage_pair_lists = updater.stage_pair_lists
        return recording_updater
    forward_project = model.sparse_forward_project

    def recording_forward_project(delta, pixel_indices, *args, **kwargs):
        delta_sinogram = forward_project(delta, pixel_indices, *args, **kwargs)
        if updates and 'd' not in updates[-1]:
            updates[-1]['d'] = torch.cat(delta.tensors, dim=1).double()
            updates[-1]['A_d'] = as_float64(delta_sinogram)
        return delta_sinogram
    monkeypatch.setattr(model, 'create_vcd_subset_updater', recording_create)
    monkeypatch.setattr(model, 'sparse_forward_project', recording_forward_project)
    model.recon(sinogram, max_iterations=2, stop_threshold_change_pct=0, print_logs=False)

    qggmrf_nbr_wts, sigma_x, p, q, T, sigma_y, max_alpha = model.get_params(
        ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T', 'sigma_y', 'max_alpha'])
    qggmrf_params = (qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts), sigma_x, p, q, T)
    fm_constant = 1.0 / sigma_y ** 2
    if isinstance(sino_weights[0], (int, float)):
        weight_shards = [1.0] * len(updates[0]['error'])
    else:
        weight_shards = as_float64(model._as_shards(sino_weights[0], model.sino_placement))
    assert len(updates) == 8
    for update in updates:
        prior_linear, prior_quadratic = _direct_prior_terms(
            update['recon'], recon_shape, update['indices'], update['d'], qggmrf_params)
        theta1 = prior_linear - fm_constant * sum(
            float(torch.sum(w * e * a)) for w, e, a in zip(weight_shards, update['error'],
                                                          update['A_d']))
        theta2 = prior_quadratic + fm_constant * sum(
            float(torch.sum(w * a * a)) for w, a in zip(weight_shards, update['A_d']))
        expected_alpha = -theta1 / theta2
        assert 1e-6 < expected_alpha < max_alpha
        assert abs(update['alpha'] - expected_alpha) < 1e-5 * expected_alpha


def test_denoiser_exact_term_matches_the_direct_sum(monkeypatch):
    # The denoiser computes the exact term directly on one device.
    results = []
    exact_quadratic = qggmrf.qggmrf_exact_quadratic

    def recording(prior_hess, slice_b_tilde_2, delta, flat_recon, pixel_indices, recon_shape,
                  qggmrf_params, pair_lists, *args):
        value = exact_quadratic(prior_hess, slice_b_tilde_2, delta, flat_recon, pixel_indices,
                                recon_shape, qggmrf_params, pair_lists, *args)
        _, direct = _direct_prior_terms(flat_recon.double(), recon_shape, pixel_indices.long(),
                                        delta.double(), qggmrf_params)
        results.append((float(value), direct))
        return value
    monkeypatch.setattr(qggmrf, 'qggmrf_exact_quadratic', recording)
    rng = np.random.default_rng(1)
    shape = (16, 12, 5)
    image = np.zeros(shape, np.float32)
    image[4:12, 3:9, 1:4] = 1.0
    image += 0.1 * rng.standard_normal(shape).astype(np.float32)
    denoiser = mbirtorch.QGGMRFDenoiser(shape, compile_mode='off')
    denoiser.set_params(verbose=0, granularity=[4], use_ror_mask=False)
    denoiser.configure_devices(devices=['cpu'])
    denoiser.denoise(image, sigma_noise=0.1, max_iterations=2)
    assert len(results) == 8
    for value, direct in results:
        assert abs(value - direct) < 1e-5 * abs(direct)

"""The exact prior term of the line search.  The subset update computes the term from the
approximate term sum(prior_hess * d^2) and from correction sums.  These tests compare the result
with a direct computation of the exact term, FCI Eq. 8.24, on the full volume in float64."""

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import qggmrf
from mbirtorch import tomography_model


def _direct_exact_term(flat_recon, recon_shape, pixel_indices, d, qggmrf_params):
    # This function sums b 2 b~ d_s (d_s - d_r) over each pixel s of the subset and each of its
    # 6 neighbors r, where d_r is 0 outside the subset.  A missing neighbor at an edge of the
    # volume is the pixel itself, so it adds 0.
    b = qggmrf_params[0]
    num_rows, num_cols = recon_shape[0], recon_shape[1]
    cylinders = flat_recon[pixel_indices]
    delta = torch.cat((torch.zeros_like(cylinders[:, :1]), cylinders[:, 1:] - cylinders[:, :-1],
                       torch.zeros_like(cylinders[:, :1])), dim=1)
    b_tilde_2 = qggmrf.get_2_b_tilde(delta, 1.0, qggmrf_params)
    d_ext = torch.cat((d[:, :1], d, d[:, -1:]), dim=1)
    total = torch.sum(b[4] * b_tilde_2[:, 1:] * d * (d - d_ext[:, 2:]))
    total = total + torch.sum(b[5] * b_tilde_2[:, :-1] * d * (d - d_ext[:, :-2]))
    position = torch.full((num_rows * num_cols,), -1, dtype=torch.int64)
    position[pixel_indices] = torch.arange(pixel_indices.numel())
    row_index = pixel_indices // num_cols
    col_index = pixel_indices % num_cols
    for (dr, dc), b_value in [((1, 0), b[0]), ((-1, 0), b[1]), ((0, 1), b[2]), ((0, -1), b[3])]:
        neighbor = ((row_index + dr).clamp(0, num_rows - 1) * num_cols
                    + (col_index + dc).clamp(0, num_cols - 1))
        b_tilde_2 = qggmrf.get_2_b_tilde(cylinders - flat_recon[neighbor], b_value,
                                         qggmrf_params)
        neighbor_position = position[neighbor]
        d_neighbor = (d[neighbor_position.clamp(min=0)]
                      * (neighbor_position >= 0).to(d.dtype)[:, None])
        total = total + torch.sum(b_tilde_2 * d * (d - d_neighbor))
    return float(total)


@pytest.mark.parametrize('geometry', ['parallel', 'cone'])
def test_exact_term_matches_the_direct_sum(geometry, monkeypatch):
    # Two CPU devices cover the pairs that cross a boundary between devices.  The cone-beam
    # model uses slice damping.  The grid has 512 pixels, so the 4 subsets do not overlap.
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

    # Each subset update records the recon before the update and the direction d.
    states = []
    create = model.create_vcd_subset_updater

    def recording_create(*args, **kwargs):
        updater = create(*args, **kwargs)

        def recording_updater(flat_recon, error_sinogram, pixel_indices, subset_index=0):
            states.append([torch.cat(flat_recon.tensors, dim=1).double(),
                           torch.as_tensor(pixel_indices).long()])
            return updater(flat_recon, error_sinogram, pixel_indices, subset_index=subset_index)
        recording_updater.stage_halos = updater.stage_halos
        recording_updater.stage_pair_lists = updater.stage_pair_lists
        return recording_updater
    forward_project = model.sparse_forward_project

    def recording_forward_project(delta, pixel_indices, *args, **kwargs):
        if states and len(states[-1]) == 2:
            states[-1].append(torch.cat(delta.tensors, dim=1).double())
        return forward_project(delta, pixel_indices, *args, **kwargs)
    monkeypatch.setattr(model, 'create_vcd_subset_updater', recording_create)
    monkeypatch.setattr(model, 'sparse_forward_project', recording_forward_project)
    monkeypatch.setattr(tomography_model, '_LINE_SEARCH_LOG', [])
    model.recon(sinogram, max_iterations=2, stop_threshold_change_pct=0, print_logs=False)

    qggmrf_nbr_wts, sigma_x, p, q, T = model.get_params(['qggmrf_nbr_wts', 'sigma_x', 'p', 'q',
                                                         'T'])
    qggmrf_params = (qggmrf.get_b_from_nbr_wts(qggmrf_nbr_wts), sigma_x, p, q, T)
    log = tomography_model._LINE_SEARCH_LOG
    assert len(log) == len(states) == 8
    for entry, (recon, indices, d) in zip(log, states):
        direct = _direct_exact_term(recon, recon_shape, indices, d, qggmrf_params)
        assert abs(entry['prior_quadratic'] - direct) < 1e-5 * abs(direct)
        assert entry['prior_quadratic'] < entry['prior_quadratic_approx']


def test_denoiser_exact_term_matches_the_direct_sum(monkeypatch):
    # The denoiser computes the exact term directly on one device.
    results = []
    exact_quadratic = qggmrf.qggmrf_exact_quadratic

    def recording(prior_hess, slice_b_tilde_2, delta, flat_recon, pixel_indices, recon_shape,
                  qggmrf_params, pair_lists, *args):
        value = exact_quadratic(prior_hess, slice_b_tilde_2, delta, flat_recon, pixel_indices,
                                recon_shape, qggmrf_params, pair_lists, *args)
        direct = _direct_exact_term(flat_recon.double(), recon_shape, pixel_indices.long(),
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

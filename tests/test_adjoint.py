"""Adjointness gate: <Ax, y> == <x, A'y> for each geometry on every device."""

import numpy as np
import pytest
import torch

import mbirtorch

GEOMETRIES = ['parallel', 'cone', 'multiaxis', 'translation']


def small_model(geometry, device):
    """Return a small model of the given geometry on one device."""
    if geometry == 'parallel':
        shape = (40, 32, 32)
        angles = np.linspace(0, np.pi, shape[0], endpoint=False)
        model = mbirtorch.ParallelBeamModel(shape, angles)
    elif geometry == 'cone':
        shape = (24, 16, 16)
        angles = np.linspace(0, 2 * np.pi, shape[0], endpoint=False)
        model = mbirtorch.ConeBeamModel(shape, angles, source_detector_dist=4 * shape[2],
                                        source_iso_dist=2 * shape[2])
    elif geometry == 'multiaxis':
        n_views = 16
        az = np.linspace(0, np.pi, n_views, endpoint=False)
        el = np.linspace(-0.5, 0.5, n_views)
        model = mbirtorch.MultiAxisParallelModel((n_views, 24, 20), np.stack([az, el], axis=1))
    else:
        tvecs = mbirtorch.gen_translation_vectors(4, 4, x_spacing=3.0, z_spacing=2.0)
        model = mbirtorch.TranslationModel((tvecs.shape[0], 40, 32), tvecs,
                                           source_detector_dist=128.0, source_iso_dist=32.0)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    return model


@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_projector_adjointness(geometry, any_device):
    torch.manual_seed(0)
    m = small_model(geometry, any_device)
    rs = m.get_params('recon_shape')
    idx = torch.as_tensor(mbirtorch.gen_full_indices(
        rs, use_ror_mask=m.get_params('use_ror_mask')), dtype=torch.int64,
        device=m.torch_device)
    x = torch.rand((idx.shape[0], rs[2]), device=m.torch_device)
    y = torch.rand(tuple(m.get_params('sinogram_shape')), device=m.torch_device)
    lhs = float(torch.sum(m.sparse_forward_project(x, idx) * y))
    rhs = float(torch.sum(x * m.sparse_back_project(y, idx)))
    # f32 sums over many terms: run-to-run atomics noise is ~1e-6 relative.
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30) < 1e-4, (lhs, rhs)

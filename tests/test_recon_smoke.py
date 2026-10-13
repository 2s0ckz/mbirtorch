"""Recon smoke gate: the VCD loop runs for each geometry on every device and
reduces the forward loss on a simple synthetic object."""

import numpy as np
import pytest

import mbirtorch
from test_adjoint import GEOMETRIES, small_model


@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_recon_reduces_loss(geometry, any_device):
    m = small_model(geometry, any_device)
    rs = tuple(m.get_params('recon_shape'))
    if geometry == 'parallel':
        # A centered box phantom, checked against the recon below.
        phantom = np.zeros(rs, dtype=np.float32)
        r0, c0, s0 = [n // 4 for n in rs]
        phantom[r0:-r0, c0:-c0, s0:-s0] = 1.0
    elif geometry == 'cone':
        phantom = mbirtorch.gen_shepp_logan_3d(rs)
    else:
        phantom = mbirtorch.utilities._gen_translation_phantom(rs, 'dots', None, fill_rate=0.05)
    sino = m.forward_project(phantom)
    np.random.seed(0)
    recon, rd = m.recon(sino, max_iterations=4 if geometry == 'parallel' else 3,
                        stop_threshold_change_pct=0.0, logfile_path=None)
    fm = rd['recon_params']['fm_rmse']
    assert fm[-1] < fm[0], fm
    assert recon.shape == rs
    if geometry == 'parallel':
        nrmse = float(np.linalg.norm(recon - phantom) / np.linalg.norm(phantom))
        assert nrmse < 0.5, nrmse

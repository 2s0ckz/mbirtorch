"""Multiaxis-parallel gate: agreement with the parallel-beam model at zero elevation."""

import numpy as np

import mbirtorch


def _rel_max(out, ref):
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


def test_multiaxis_zero_elevation_matches_parallel(device):
    """At zero elevation the geometry is parallel beam; the forward
    projections must agree at float level."""
    n_views = 12
    az = np.linspace(0, np.pi, n_views, endpoint=False)
    shape = (n_views, 16, 16)
    pm = mbirtorch.ParallelBeamModel(shape, az)
    pm.configure_devices(devices=[device])
    pm.set_params(no_warning=True, verbose=0)
    rs = tuple(pm.get_params('recon_shape'))
    mm = mbirtorch.MultiAxisParallelModel(shape, np.stack([az, np.zeros(n_views)], axis=1))
    mm.configure_devices(devices=[device])
    mm.set_params(no_warning=True, verbose=0, recon_shape=rs)
    vol = np.random.RandomState(0).rand(*rs).astype(np.float32)
    sino_p = np.asarray(pm.forward_project(vol))
    sino_m = np.asarray(mm.forward_project(vol))
    rel = _rel_max(sino_m, sino_p)
    print(f"multiaxis vs parallel at zero elevation rel_max = {rel:.2e}")
    assert rel < 1e-5

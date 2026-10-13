"""Crash guard for mbirtorch.estimate_resources: it reports every step and leaves the model unchanged."""
import numpy as np
import pytest

import mbirtorch
from mbirtorch import projectors


@pytest.fixture(autouse=True)
def _no_recorded_compile_errors(monkeypatch):
    """This function clears the compile errors that earlier tests recorded.  Those errors change the memory count."""
    monkeypatch.setattr(projectors, '_COMPILE_ERRORS', {})


def _small_model():
    """Return a small cone model on the CPU."""
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    model = mbirtorch.ConeBeamModel((24, 40, 32), angles, source_detector_dist=128.0,
                                    source_iso_dist=64.0, compile_mode='off')
    model.configure_devices(devices=['cpu'])
    model.set_params(no_warning=True, verbose=0)
    return model


def test_estimate_reports_and_leaves_the_model_unchanged():
    model = _small_model()
    before = model.get_params('recon_shape')
    estimate = mbirtorch.estimate_resources(model, gpu_model='H100', num_gpus=2)
    assert model.get_params('recon_shape') == before
    for step in (estimate.direct, estimate.recon, estimate.split):
        assert step.gpu_memory_gib > 0
        assert step.fits is True
    assert 'full reconstruction' in str(estimate)
    for step in (estimate.forward, estimate.back):
        assert step.gpu_memory_gib > 0
        assert step.time_minutes > 0
    assert 'forward projection' in str(estimate) and 'back projection' in str(estimate)
    assert mbirtorch.estimate_resources(model, gpu_memory_gb=2.0001).recon.fits is False

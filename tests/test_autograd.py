"""Check the gradients of the differentiable projectors in mbirtorch.autograd."""

import numpy as np
import torch

import mbirtorch


SINO_SHAPE = (16, 32, 32)


def _max_rel_err(tensor, reference):
    return float((tensor - reference).abs().max() / reference.abs().max())


def test_projector_gradients(device):
    """d/dx sum(A x * y) equals A' y, and d/dy sum(A' y * x) equals A x."""
    torch.manual_seed(0)
    angles = np.linspace(0, np.pi, SINO_SHAPE[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(SINO_SHAPE, angles)
    model.configure_devices(devices=[device])
    recon_shape = tuple(model.get_params('recon_shape'))

    x = torch.rand(recon_shape, device=model.torch_device, requires_grad=True)
    y = torch.rand(SINO_SHAPE, device=model.torch_device, requires_grad=True)

    torch.sum(mbirtorch.forward_project_differentiable(model, x) * y.detach()).backward()
    error = _max_rel_err(x.grad, mbirtorch.back_project_differentiable(model, y.detach()))
    assert error < 1e-5, error

    torch.sum(mbirtorch.back_project_differentiable(model, y) * x.detach()).backward()
    error = _max_rel_err(y.grad, mbirtorch.forward_project_differentiable(model, x.detach()))
    assert error < 1e-5, error

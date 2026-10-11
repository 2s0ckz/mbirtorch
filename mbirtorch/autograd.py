"""This module provides differentiable projectors for use in deep-learning pipelines.

The forward projector and the back projector are exact adjoints of each other.  For this
reason, the autograd backward pass of each projector is the other projector.  The function
``forward_project_differentiable`` lets gradients flow back to the volume.  The class
``TorchProjector`` wraps this function as an ``nn.Module``.
"""

import torch

from . import _sharding



class _ForwardProjectFunction(torch.autograd.Function):
    """This class is used to apply the forward projector A to the voxel values at fixed pixel
    indices.  Its backward pass applies the back projector A'."""

    @staticmethod
    def forward(ctx, voxel_values, model, pixel_indices):
        ctx.model = model
        ctx.pixel_indices = pixel_indices
        return model.sparse_forward_project(voxel_values, pixel_indices)

    @staticmethod
    def backward(ctx, grad_sinogram):
        grad_values = ctx.model.sparse_back_project(
            grad_sinogram.contiguous(), ctx.pixel_indices)
        return grad_values, None, None


class _BackProjectFunction(torch.autograd.Function):
    """This class is used to apply the back projector A' to a sinogram.  It returns the voxel
    values at fixed pixel indices.  Its backward pass applies the forward projector A."""

    @staticmethod
    def forward(ctx, sinogram, model, pixel_indices):
        ctx.model = model
        ctx.pixel_indices = pixel_indices
        return model.sparse_back_project(sinogram, pixel_indices)

    @staticmethod
    def backward(ctx, grad_values):
        grad_sinogram = ctx.model.sparse_forward_project(
            grad_values.contiguous(), ctx.pixel_indices)
        return grad_sinogram, None, None


def _require_single_device(model):
    if not model.sino_placement.is_trivial:
        raise NotImplementedError(
            'The differentiable wrappers support single-device models only: '
            'under a multi-device configuration the projector outputs are '
            'shard containers, which would silently detach gradients.  '
            'configure_devices(1) for training.')


def forward_project_differentiable(model, volume):
    """This function computes a differentiable forward projection of a full volume.

    The volume may have any device and any dtype.  The function copies the volume to the
    model device as float32.  The gradient with respect to the volume has the device and
    dtype of the input volume.  The function works only when the model uses one device.

    Args:
        model: (TomographyModel) The model, for example a ParallelBeamModel.
        volume: (tensor) The volume, with shape (num_rows, num_cols, num_slices).  The volume
            may require grad.

    Returns:
        A tensor with shape (num_views, num_det_rows, num_det_channels).  Gradients flow
        from this tensor back to ``volume``.
    """
    _require_single_device(model)
    # The check above does not detect a sharded array that another model built.
    _sharding.reject_shards('forward_project_differentiable', volume=volume)
    volume = volume.to(device=model.torch_device, dtype=torch.float32)
    indices = model.full_indices_device()
    voxel_values = volume.reshape(-1, volume.shape[-1])[indices]
    return _ForwardProjectFunction.apply(voxel_values, model, indices)


def back_project_differentiable(model, sinogram):
    """This function computes a differentiable back projection that returns a full volume.
    It is the adjoint of :func:`forward_project_differentiable`, and it handles device and
    dtype in the same way."""
    _require_single_device(model)
    _sharding.reject_shards('back_project_differentiable', sinogram=sinogram)
    recon_shape = model.get_params('recon_shape')
    sinogram = sinogram.to(device=model.torch_device, dtype=torch.float32)
    indices = model.full_indices_device()
    cylinders = _BackProjectFunction.apply(sinogram, model, indices)
    volume = torch.zeros(recon_shape[0] * recon_shape[1], cylinders.shape[-1],
                         dtype=cylinders.dtype, device=cylinders.device)
    volume = volume.index_put((indices,), cylinders)
    return volume.reshape(tuple(recon_shape[:2]) + (cylinders.shape[-1],))


class TorchProjector(torch.nn.Module):
    """This class is used to wrap the forward projector as an ``nn.Module``.  The method
    ``forward(volume)`` returns the sinogram, and the sinogram is differentiable with respect
    to ``volume``.  The method ``adjoint(sinogram)`` returns the back projection.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, volume):
        return forward_project_differentiable(self.model, volume)

    def adjoint(self, sinogram):
        return back_project_differentiable(self.model, sinogram)

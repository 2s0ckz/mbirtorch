import torch


def _default_device(device=None):
    """This function returns the name of the device to compute on.  If the caller gives a
    device, the function returns that device.  Otherwise it returns the automatic choice, with
    MPS replaced by the CPU because MPS does not support float64."""
    if device is not None and device != 'auto':
        return str(torch.device(device))
    from ..tomography_model import _resolve_device
    resolved = _resolve_device('auto')
    return 'cpu' if resolved.type == 'mps' else str(resolved)

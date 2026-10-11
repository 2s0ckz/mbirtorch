"""This module runs the per-view steps that convert scan data into a sinogram.

The module moves the views to the devices in batches.  It applies a kernel from
:mod:`mbirtorch.preprocess.sinogram` to each batch.  It writes each result into one host array
that it allocates once.  With several devices, each device processes one contiguous block of
views in its own thread.
"""
import numpy as np
import torch

from .. import _memory_estimate
from .. import _sharding
from ..tomography_model import _resolve_device


def permitted_devices(devices=None):
    """This function returns the devices that a preprocessing function divides its views
    among.  By default it returns every visible CUDA device, up to the number set by
    ``MBIRTORCH_NUM_DEVICES``.  If there is no CUDA device, it returns the default device.

    Args:
        devices (sequence or None): A list of devices, or None for the default.  The function
            returns a given list unchanged.

    Returns:
        list: The devices to pass to :func:`map_view_batches`.
    """
    if devices is not None:
        return list(devices)
    num_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    env_count = _memory_estimate.env_device_count()
    if env_count is not None:
        num_visible = min(num_visible, env_count)
    if num_visible == 0:
        return [_resolve_device('auto')]
    return [f'cuda:{i}' for i in range(num_visible)]


# The preprocessing functions accept CPU arrays only, not arrays split across devices.
reject_shards = _sharding.reject_shards


def _stage_batch(batch, device):
    """This function copies one batch of views to ``device`` as a contiguous floating point
    tensor.  It converts integer data to float32.
    """
    if torch.is_tensor(batch):
        batch = batch.detach()
        if not torch.is_floating_point(batch):
            batch = batch.to(torch.float32)
        return batch.to(device).contiguous()
    batch = np.ascontiguousarray(batch)
    if not np.issubdtype(batch.dtype, np.floating):
        batch = batch.astype(np.float32)
    return torch.as_tensor(batch, device=device)


def _fill_view_batches(array, kernel, output, batch_size, device, lo, hi, desc=None):
    """This function applies ``kernel`` to views ``[lo, hi)`` of ``array`` on ``device``,
    ``batch_size`` views at a time.  It writes each result into the CPU array ``output``.
    """
    import tqdm
    steps = range(lo, hi, batch_size)
    if desc is not None:
        steps = tqdm.tqdm(steps, desc=desc)
    with torch.no_grad():
        for j in steps:
            end = min(j + batch_size, hi)
            batch = _stage_batch(array[j:end], device)
            output[j:end] = kernel(batch).cpu().numpy()


def map_view_batches(array, kernel, batch_size, desc=None, devices=None):
    """This function applies a kernel to the views in batches on the devices and returns the
    results as one CPU array.  With several devices, each device processes one contiguous block
    of views in its own thread.  The kernel must move any constants it uses to the device of
    each batch.

    Args:
        array (numpy array or tensor): The data, with views along axis 0.
        kernel (callable): A function that maps a batch on a device to a batch on the same
            device.  It must act on each view separately and must not copy data to the CPU.
        batch_size (int): The number of views in each batch.
        desc (str or None, optional): The tqdm progress bar label.  The function shows the
            progress bar only when it uses one device.
        devices (sequence or None): The devices to divide the views among.  None means the
            default device.

    Returns:
        numpy.ndarray: The kernel outputs, stacked along axis 0 in view order.
    """
    if devices is None or len(list(devices)) == 0:
        devices = [_resolve_device('auto')]
    else:
        devices = [torch.device(d) for d in devices]
    num_views = array.shape[0]

    # The first batch determines the output shape and dtype.
    probe_hi = min(batch_size, num_views)
    with torch.no_grad():
        probe = kernel(_stage_batch(array[0:probe_hi], devices[0])).cpu().numpy()
    output = np.empty((num_views,) + probe.shape[1:], dtype=probe.dtype)
    output[0:probe_hi] = probe
    del probe

    if len(devices) <= 1:
        _fill_view_batches(array, kernel, output, batch_size, devices[0], probe_hi, num_views,
                           desc=desc)
        return output

    view_ranges = np.array_split(np.arange(num_views), len(devices))

    def worker(i, device):
        rng = view_ranges[i]
        if len(rng) == 0:
            return
        lo = max(int(rng[0]), probe_hi)  # The first probe_hi views are already filled.
        hi = int(rng[-1]) + 1
        if lo >= hi:
            return
        _fill_view_batches(array, kernel, output, batch_size, device, lo, hi)

    _sharding.run_per_device(devices, worker)
    return output

"""This module estimates the peak memory that a reconstruction uses on each GPU.

The estimate is computed from the array sizes, before the reconstruction starts.  The code
divides the reconstruction into steps, which it calls phases, and sums the bytes of each step.
mbirtorch uses the estimate to choose how many GPUs to use.  If the reconstruction does not
fit on the available GPUs, mbirtorch raises an error.  mbirtorch does not check memory when it
uses one GPU, when the user sets the GPUs with ``configure_devices``, or when the devices are
not CUDA devices.  In this module, a cylinder array has shape (pixels, slices).
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from . import _sharding, tomography_utils
from ._utils import padded_kernel_width

_F32_BYTES = 4
_INT64_BYTES = 8

# The qGGMRF prior code holds this many cylinder arrays at once.  qggmrf_cylinder_count
# chooses between the compiled count and the eager count.
QGGMRF_CYLINDERS_COMPILED = 9
QGGMRF_CYLINDERS_EAGER = 16
# The proximal map prior acts on each voxel separately.  It holds four cylinder arrays: the
# recon, the prox input, their difference, and the scaled result.
PROX_CYLINDERS = 4
# These are the numbers of cylinder arrays that direction_worker and apply_worker hold at once.
DIRECTION_CYLINDERS = 7
APPLY_CYLINDERS = 2
# A forward projection on several GPUs holds this many copied batches of cylinder arrays at
# once.  The copy of the next batch overlaps the projection of the current batch.
CYLINDER_TRANSFER_COPIES = 3

# These are the numbers of cylinder arrays that one QGGMRFDenoiser subset update holds at
# once.  The first is for computing the update direction.  The second is for applying it.
DENOISE_DIRECTION_CYLINDERS = 8
DENOISE_APPLY_CYLINDERS = 3
# Each GPU holds this many boundary slices while the denoiser on several GPUs exchanges
# boundary slices.  The first and last GPUs hold fewer, but the estimate counts four for
# them too.
DENOISE_HALO_COLUMNS = 4
# The L1 norms and the weighted dot products are summed one chunk at a time.  These two
# values set the chunk size for all of them.
L1_CHUNK_BYTES = 16 * 2 ** 20
L1_MAX_CHUNKS = 1024


def _chunk_count(n_bytes):
    """This function returns the number of chunks used to sum an array of ``n_bytes`` bytes."""
    return min(L1_MAX_CHUNKS, max(1, round(int(n_bytes) / L1_CHUNK_BYTES)))


def image_l1_norm(flat_image):
    """This function returns the L1 norm of an array with the shape of the recon.

    The function sums the array one chunk at a time, so the temporary array holds only one
    chunk.  An array smaller than one chunk is summed in one step.
    """
    n_chunks = _chunk_count(flat_image.numel() * flat_image.element_size())
    if n_chunks == 1:
        return torch.sum(torch.abs(flat_image))
    return torch.stack([torch.sum(torch.abs(chunk)) for chunk
                        in torch.chunk(flat_image, n_chunks, dim=0)]).sum()


def stack_l1_norm(flat_stack):
    """This function returns the L1 norm of each volume in a stack of volumes.

    The function does not allocate a temporary array with the shape of the stack.
    ``flat_stack`` has shape ``(num_volumes, num_pixels, num_slices)``.  The result has shape
    ``(num_volumes,)``.  The function chooses the number of chunks as :func:`image_l1_norm` does
    for the whole stack.  It cuts the chunks along the pixel axis.  A stack smaller than one
    chunk is summed in one step.
    """
    n_bytes = flat_stack.numel() * flat_stack.element_size()
    n_chunks = min(L1_MAX_CHUNKS, max(1, round(n_bytes / L1_CHUNK_BYTES)))
    if n_chunks == 1:
        return torch.sum(torch.abs(flat_stack), dim=(1, 2))
    return torch.stack([torch.sum(torch.abs(chunk), dim=(1, 2)) for chunk
                        in torch.chunk(flat_stack, n_chunks, dim=1)]).sum(dim=0)


def _block_dot(a_block, b_block, weights):
    """This function returns the weighted dot product of one block."""
    if weights is None:
        return torch.sum(a_block * b_block)
    return torch.sum(a_block * b_block * weights)


def _paired_blocks(operand, reference, n_chunks, n_blocks):
    """This function splits ``operand`` into blocks that match the blocks of ``reference``.

    If ``operand`` does not have the view axis of ``reference``, the function returns
    ``operand`` itself once for each block.
    """
    if (torch.is_tensor(operand) and operand.ndim == reference.ndim
            and operand.shape[0] == reference.shape[0]):
        return torch.chunk(operand, n_chunks, dim=0)
    return [operand] * n_blocks


def weighted_dot(a, b, weights=None):
    """This function returns the weighted dot product of two sinograms.

    The function does not allocate a temporary array with the shape of the sinogram.
    ``b`` has the shape of ``a``.  ``weights`` is None, a scalar, or an array with the shape
    of ``a``.  None gives the unweighted dot product.  The function sums one block of views at
    a time, so each temporary array holds one block.  A sinogram smaller than one chunk is
    summed in one step.
    """
    n_chunks = _chunk_count(a.numel() * a.element_size())
    if n_chunks == 1:
        return _block_dot(a, b, weights)
    blocks = torch.chunk(a, n_chunks, dim=0)
    b_blocks = _paired_blocks(b, a, n_chunks, len(blocks))
    weight_blocks = _paired_blocks(weights, a, n_chunks, len(blocks))
    totals = [_block_dot(block, b_block, block_weights)
              for block, b_block, block_weights
              in zip(blocks, b_blocks, weight_blocks)]
    return torch.stack(totals).sum()


def weighted_square_sum(error_sinogram, weights=None):
    """This function returns the weighted sum of squares of a sinogram.

    The function calls :func:`weighted_dot` with the sinogram as both operands.
    """
    return weighted_dot(error_sinogram, error_sinogram, weights)


def reduction_chunk_bytes(array_bytes):
    """This function returns (chunk_bytes, n_chunks) for a sum over an array of
    ``array_bytes`` bytes.

    The L1 norms and the weighted dot products use this rule.  A step that holds more than
    one array for each chunk multiplies chunk_bytes by the number of arrays.  An array smaller
    than one chunk is summed in one step, so its chunk is the whole array.
    """
    array_bytes = int(array_bytes)
    n_chunks = _chunk_count(array_bytes)
    if n_chunks == 1:
        return array_bytes, 1
    return math.ceil(array_bytes / n_chunks), n_chunks

# The workspace of the torch libraries, which is not part of any array.  The cuBLAS workspace
# alone is 32 MiB per stream.
FIXED_DEVICE_OVERHEAD_BYTES = 64 * 2 ** 20

CALIBRATION_ENV_VAR = 'MBIRTORCH_MEMORY_CALIBRATION'
# This environment variable sets the device count for the whole process.  mbirtorch does not
# reduce this count, but it still checks that the reconstruction fits.
DEVICE_COUNT_ENV_VAR = 'MBIRTORCH_NUM_DEVICES'
# One view batch of a torch projection function holds this many arrays for each view.  A
# torch projection function is written in torch code and has no ``_view_batch_cost``.
TORCH_PROJECTOR_VIEW_SLABS = 14

# The ratio of the estimated peak to the measured peak must fall in this range.
CALIBRATION_BAND = (1.00, 1.30)
# This wider range applies to torch projection functions.  These functions hold a different
# number of arrays in each geometry.
TORCH_PROJECTOR_CALIBRATION_BAND = (1.00, 5.80)


class MemoryCheckError(RuntimeError):
    """This class is used to signal that no number of GPUs can hold the reconstruction."""


# MemoryPreflightError is the deprecated name of MemoryCheckError.
MemoryPreflightError = MemoryCheckError


@dataclass
class StepMemory:
    """This class is used to hold the bytes of one step on each device and the terms that make
    up those bytes."""
    name: str
    per_device: list
    terms: list = field(default_factory=list)   # [(term_name, [bytes on each device])]

    def dominant_terms(self, dev_index, count=3):
        """This method returns the largest terms on one device, in decreasing order."""
        ranked = sorted(self.terms, key=lambda t: -t[1][dev_index])
        return [(name, vals[dev_index]) for name, vals in ranked[:count]
                if vals[dev_index] > 0]


@dataclass
class MemoryEstimate:
    """This class is used to hold the memory estimate of every step on every device.

    The peak on a device is the maximum over the steps.
    """
    devices: list
    phases: list
    # The number of pixels inside the ROR mask, which the subset steps use.
    num_pixels_full: int = 0

    def peak_bytes(self, dev_index):
        return max(p.per_device[dev_index] for p in self.phases)

    def per_device_peaks(self):
        return [self.peak_bytes(i) for i in range(len(self.devices))]

    def dominant_phase(self, dev_index):
        return max(self.phases, key=lambda p: p.per_device[dev_index])

    def format_table(self):
        """This method formats a table of bytes with one row for each step and one column for
        each device.  The verbose printout and the error message use this table."""
        width = max(28, max(len(p.name) for p in self.phases) + 2)
        head = 'phase'.ljust(width) + ''.join(
            f'{str(d):>14}' for d in self.devices)
        lines = [head, '-' * len(head)]
        for phase in self.phases:
            lines.append(phase.name.ljust(width) + ''.join(
                f'{_gb(b):>14}' for b in phase.per_device))
        lines.append('-' * len(head))
        lines.append('PEAK'.ljust(width) + ''.join(
            f'{_gb(b):>14}' for b in self.per_device_peaks()))
        return '\n'.join(lines)


def _gb(num_bytes):
    return f'{num_bytes / 2 ** 30:.2f} GB'


@dataclass
class MemoryEstimateInputs:
    """This class is used to hold the inputs of the memory estimate.

    :func:`plan_from_model` builds it from a model, and the tests build it by hand.  No field
    requires CUDA, so the estimate can run on the CPU.
    """
    # ── shapes and how they are divided among devices ───────────────────────
    sinogram_shape: tuple                 # (V, R, C), the full sinogram shape.
    recon_shape: tuple                    # (Rr, Rc, S), the full recon shape.
    devices: list                         # One entry for each device, in order.
    view_blocks: list                     # The number of views on each device.
    slice_blocks: list                    # The number of slices on each device.
    sino_rows: int                        # The number of detector rows.
    rows_track_slices: bool
    # ── pixel counts ─────────────────────────────────────────────────────────
    num_pixels_full: int                  # The number of pixels inside the ROR mask.
    num_pixels_grid: int                  # The number of pixels in the full grid.
    granularities: tuple                  # The subset counts that the sequence uses.
    partition_granularities: tuple        # Every subset count, built at the start.
    # ── what this call runs and what the caller supplies ────────────────────
    # The workload names the call that the plan describes.  'recon' is a full
    # reconstruction.  'direct' is a direct reconstruction alone.  'denoise' is one
    # call of QGGMRFDenoiser.  'forward' and 'back' are one call of forward_project and
    # back_project.
    workload: str = 'recon'
    weights_supplied: bool = False
    fm_hessian_supplied: bool = False
    init_recon_supplied: bool = False
    resume: bool = False
    prox: bool = False
    positivity: bool = False
    helical: bool = False
    # True when the Hessian back projection uses only the pixels inside the ROR mask.
    # _vcd_recon does this, and a direct call of the public function does not.
    hessian_masked: bool = False
    # ── settings and model choices ───────────────────────────────────────────
    back_band: int = None
    # The number of pixels in one copied batch of the forward projection.  A denoise plan
    # has no forward projection, so it leaves this value None.
    pixel_batch: int = None
    qggmrf_cylinders: int = QGGMRF_CYLINDERS_COMPILED
    # A function of (direction, num_pixels, band_cols) that returns
    # (view_batch, bytes_per_view).  The direction is 'forward' or 'back'.  None means that
    # the view batches add no bytes.
    view_batch_bytes: object = None
    # The directions, 'forward' or 'back', that use a torch projection function.  The
    # estimate counts the bytes of those view batches itself.  See TORCH_PROJECTOR_VIEW_SLABS.
    torch_projector_directions: tuple = ()
    # True when the back projection function adds each view batch into the running sum.
    back_adds_into_output: bool = False

    @property
    def n_devices(self):
        return len(self.devices)

    def batch_bytes(self, direction, num_pixels, band_cols):
        if self.view_batch_bytes is None:
            return 0
        view_batch, bytes_per_view = self.view_batch_bytes(
            direction, int(num_pixels), int(band_cols))
        return int(view_batch) * int(bytes_per_view)

    def band_length(self, dev_index):
        """This method returns the number of slices that one device computes at a time in the
        back projection.

        The result matches ``TomographyModel._slice_band_length``.  By default, it is all the
        slices on the device.  The forward projection does not divide the slices.
        """
        local_slices = self.slice_blocks[dev_index]
        fixed = self.back_band
        return min(int(fixed), local_slices) if fixed else local_slices


def estimate_peak_device_bytes(plan):
    """This function estimates the peak memory on each device for one reconstruction.

    The function does not query any device and does not allocate anything.  The result
    depends only on ``plan``.  ``plan.workload`` selects the steps.  The default is a full
    reconstruction.  ``'direct'`` gives the filter and one back projection.  ``'denoise'``
    gives one call of the denoiser.  ``'forward'`` and ``'back'`` give one call of
    ``forward_project`` and ``back_project``.

    Returns:
        MemoryEstimate: The steps and the bytes of each step on each device.
    """
    n = plan.n_devices
    num_rows_dev, num_channels = plan.sino_rows, int(plan.sinogram_shape[2])
    rows_recon, cols_recon = int(plan.recon_shape[0]), int(plan.recon_shape[1])

    def sino_dev(i):
        return plan.view_blocks[i] * num_rows_dev * num_channels * _F32_BYTES

    def recon_dev(i):
        return rows_recon * cols_recon * plan.slice_blocks[i] * _F32_BYTES

    def cyl(i, num_pixels):
        return int(num_pixels) * plan.slice_blocks[i] * _F32_BYTES

    def sino_reduction_block(i):
        """This function returns the bytes in one block of a sum over the sinogram.

        Each block holds a whole number of views.
        """
        chunk, n_chunks = reduction_chunk_bytes(sino_dev(i))
        views = int(plan.view_blocks[i])
        if n_chunks == 1 or views <= 0:
            return chunk
        return (math.ceil(views / n_chunks) * num_rows_dev * num_channels
                * _F32_BYTES)

    def back_block(i, num_pixels):
        """This function returns the bytes of one partial result of the back projection.

        The partial result has shape (pixels, slices).  The kernel allocates the partial
        result with the number of slices rounded up to a multiple of 16.
        """
        slices = max(int(plan.slice_blocks[i]),
                     padded_kernel_width(plan.band_length(i)))
        return int(num_pixels) * slices * _F32_BYTES

    def is_view_owner(i):
        return plan.view_blocks[i] > 0

    def is_slice_owner(i):
        return plan.slice_blocks[i] > 0

    def per_dev(fn):
        return [int(fn(i)) for i in range(n)]

    # ── the projection terms that several steps share ────────────────────────
    def back_cols(i):
        """This function returns ``band_cols`` for one back projection call.

        This value is the number of rows in the sinogram on the device.
        """
        if n == 1:
            return int(plan.sinogram_shape[1])
        return (plan.band_length(i) if plan.rows_track_slices
                else num_rows_dev)

    def whole_slice_extent():
        """This function returns the total number of slices over all devices.

        One forward projection call receives all of these slices.
        """
        return sum(int(block) for block in plan.slice_blocks)

    def forward_call_pixels(num_pixels):
        """This function returns the number of pixels that one forward projection call
        receives.

        If the plan sets ``pixel_batch``, the number is the smaller of ``pixel_batch`` and
        ``num_pixels``.  Otherwise, the number is ``num_pixels``.
        """
        if plan.pixel_batch:
            return min(int(num_pixels), int(plan.pixel_batch))
        return int(num_pixels)

    def forward_cols(i):
        """This function returns the number of slices that one forward projection receives.

        The forward projection always receives all of the slices.
        """
        if n == 1:
            return int(plan.recon_shape[2])
        return whole_slice_extent()

    def band_slices(i, direction):
        """This function returns the number of slices that one projection call receives.

        On several devices, a back projection call receives the slice range of its device.
        In all other cases, the call receives all of the slices.
        """
        if n == 1:
            return int(plan.recon_shape[2])
        if direction == 'forward':
            return whole_slice_extent()
        return plan.band_length(i)

    def torch_projector_batch(i, direction, num_pixels):
        """This function returns the bytes that one view batch of a torch projection function
        holds.

        The function counts ``TORCH_PROJECTOR_VIEW_SLABS`` arrays.  Each array has shape (views,
        pixels, width).  The width is the larger of the number of rows and the number of
        slices.
        """
        if plan.view_batch_bytes is None:
            return 0
        cols = back_cols(i) if direction == 'back' else forward_cols(i)
        view_batch = int(plan.view_batch_bytes(direction, int(num_pixels), cols)[0])
        width = max(int(plan.sino_rows), int(band_slices(i, direction)))
        return (TORCH_PROJECTOR_VIEW_SLABS * view_batch * int(num_pixels)
                * width * _F32_BYTES)

    def back_batch(i, num_pixels):
        if not is_view_owner(i):
            return 0
        if 'back' in plan.torch_projector_directions:
            return torch_projector_batch(i, 'back', num_pixels)
        return plan.batch_bytes('back', num_pixels, back_cols(i))

    def forward_batch(i, num_pixels):
        if not is_view_owner(i):
            return 0
        call_pixels = forward_call_pixels(num_pixels)
        if 'forward' in plan.torch_projector_directions:
            return torch_projector_batch(i, 'forward', call_pixels)
        return plan.batch_bytes('forward', call_pixels, forward_cols(i))

    def band_reduce(i, num_pixels):
        """This function returns the bytes that the sum of the back projection results holds
        on one device that holds slices.

        The sum combines the partial results from all devices.  The count includes the
        running total, the partial result of the device itself, and one slab of rows for each
        partial result that arrives from another device.
        """
        if n == 1 or not is_slice_owner(i):
            return 0
        band = plan.band_length(i)
        shard = plan.slice_blocks[i]
        own_partial = padded_kernel_width(band)
        row_bytes = int(band) * _F32_BYTES
        slab_rows = _sharding.reduce_slab_rows(int(num_pixels), row_bytes)
        return (int(num_pixels) * (int(shard) + own_partial) * _F32_BYTES
                + (n - 1) * slab_rows * row_bytes)

    def back_view_batches(i, num_pixels):
        """This function returns the number of view batches in the back projection on one
        device.

        The function returns None when the plan has no ``view_batch_bytes``.
        """
        local_views = plan.view_blocks[i]
        if local_views <= 0 or plan.view_batch_bytes is None:
            return None
        view_batch = int(plan.view_batch_bytes(
            'back', int(num_pixels), back_cols(i))[0])
        return max(1, -(-int(local_views) // max(1, view_batch)))

    def back_fixed(i, num_pixels):
        """This function returns the bytes of the cylinder arrays that the back projection
        loop holds.

        The loop holds two such arrays on several devices and three on a single device.  A
        back projection function that adds into the running sum holds one array fewer.
        """
        fewer = 1 if plan.back_adds_into_output else 0
        if n == 1:
            return ((3 - fewer) * back_block(i, num_pixels) if is_slice_owner(i) else 0)
        if not is_view_owner(i):
            return 0
        batches = back_view_batches(i, num_pixels)
        live = 2 if batches is None else min(2, batches)
        return max(1, live - fewer) * back_block(i, num_pixels)

    def back_own_band(i, num_pixels):
        """This function returns the bytes of the slices that this device has finished.

        The device keeps these slices for the rest of the loop.
        """
        if n == 1 or not is_slice_owner(i):
            return 0
        return cyl(i, num_pixels)

    # The forward terms have no safety margin.  If one of them is made smaller, a run can
    # start and then run out of memory.
    def forward_fixed(i):
        """This function returns the bytes of the output of the forward projection.

        On several devices, the function counts the output twice.
        """
        if not is_view_owner(i):
            return 0
        return sino_dev(i) if n == 1 else 2 * sino_dev(i)

    def forward_transferred_cylinders(i, num_pixels):
        """This function returns the bytes of the copied batches of cylinder arrays that a
        forward projection on several devices holds.

        See ``CYLINDER_TRANSFER_COPIES``.
        """
        if n == 1 or not is_view_owner(i) or not plan.pixel_batch:
            return 0
        return (CYLINDER_TRANSFER_COPIES * forward_call_pixels(num_pixels)
                * whole_slice_extent() * _F32_BYTES)

    def forward_view_batches(i, num_pixels):
        """This function returns the number of view batches in the forward projection on one
        device.

        The function returns None when the plan has no ``view_batch_bytes``.
        """
        local_views = plan.view_blocks[i]
        if local_views <= 0 or plan.view_batch_bytes is None:
            return None
        view_batch = int(plan.view_batch_bytes(
            'forward', forward_call_pixels(num_pixels), forward_cols(i))[0])
        return max(1, -(-int(local_views) // max(1, view_batch)))

    def forward_block_rows(i):
        """This function returns the number of detector rows in one view block of the forward
        projection.

        For a hand-written kernel, the number is rounded up to a multiple of 16.
        """
        rows = forward_cols(i) if plan.rows_track_slices else plan.sino_rows
        if 'forward' in plan.torch_projector_directions:
            return int(rows)
        return padded_kernel_width(rows)

    def forward_block(i, num_pixels):
        """This function returns the bytes of the extra view blocks that the forward
        projection loop holds.

        The loop holds one extra block for a hand-written kernel and two for a torch
        projection function.  A subset step passes its own pixel count.
        """
        if not is_view_owner(i):
            return 0
        batches = forward_view_batches(i, num_pixels)
        live = 2 if batches is None else min(2, batches)
        already_paid = 0 if 'forward' in plan.torch_projector_directions else 1
        view_batch = 1
        if plan.view_batch_bytes is not None:
            view_batch = plan.view_batch_bytes('forward',
                                          forward_call_pixels(num_pixels),
                                          forward_cols(i))[0]
        return ((live - already_paid) * int(view_batch)
                * forward_block_rows(i) * num_channels * _F32_BYTES)

    # Only the 'direct' plan counts the filter terms.  In a full reconstruction, the
    # filter runs between steps that hold more memory than the filter does.
    def filter_row_weights(i):
        """This function returns the bytes of the FDK cosine weight on one device.

        The weight has the size of one detector image.  The FBP filters have no such weight,
        so this term overestimates their memory.
        """
        return int(plan.sino_rows) * num_channels * _F32_BYTES

    def filter_row_batch(i):
        """This function returns the bytes that one batch of ``ROW_FILTER_BATCH`` detector rows
        holds in the row filter.

        This value does not decrease as the device count increases.
        """
        rows_in_shard = plan.view_blocks[i] * int(plan.sino_rows)
        batch = min(tomography_utils.ROW_FILTER_BATCH, rows_in_shard)
        # This is the FFT length of the linear convolution in apply_row_filter.  It equals
        # the channel count plus the number of filter taps minus one.
        padded = 3 * num_channels - 2
        per_row = (num_channels * _F32_BYTES
                   + 2 * (padded // 2 + 1) * (2 * _F32_BYTES)
                   + padded * _F32_BYTES)
        return batch * per_row

    # There is at most one weights array with the shape of the sinogram.  A supplied array is
    # held from the direct reconstruction onward.  An unweighted run builds an array of ones
    # in the Hessian step and holds no weights before that step.
    weights_held = plan.weights_supplied or not plan.fm_hessian_supplied

    def weights_term(i):
        """This function returns the bytes of the weights from the Hessian step onward."""
        return sino_dev(i) if weights_held else 0

    def supplied_weights_term(i):
        """This function returns the bytes of the weights in the steps before the Hessian
        step builds its own array."""
        return sino_dev(i) if plan.weights_supplied else 0

    persistent = {
        'error sinogram': per_dev(sino_dev),
        'weights': per_dev(weights_term),
        'flat recon': per_dev(recon_dev),
        'hessian diagonal': per_dev(recon_dev),
    }
    if plan.prox:
        persistent['prox input'] = per_dev(recon_dev)
    # The partitions and the cached full index set are on the first device.
    partition_bytes = sum(
        g * math.ceil(plan.num_pixels_full / g) * _INT64_BYTES
        for g in plan.partition_granularities)
    partition_bytes += plan.num_pixels_full * _INT64_BYTES
    persistent['partitions (first device)'] = [
        partition_bytes if i == 0 else 0 for i in range(n)]

    # The partitions and the index cache stay on the first device for the whole
    # reconstruction.  For this reason, every step includes them.
    workspace_term = ('library workspace', [FIXED_DEVICE_OVERHEAD_BYTES] * n)
    constant_terms = [
        ('partitions (first device)',
         persistent.pop('partitions (first device)')),
        workspace_term,
    ]
    constant_base = [sum(vals[i] for _name, vals in constant_terms)
                     for i in range(n)]
    persistent_total = [constant_base[i]
                        + sum(vals[i] for vals in persistent.values())
                        for i in range(n)]

    phases = []

    def back_phases(name, held_terms, num_pixels, base, base_terms,
                    back_output=None):
        """This function returns the steps of one back projection.

        On several devices, the back projection has two steps.  The first step is the
        projection, and the second step sums the results across devices.  The step names
        start with ``name``.  On a single device, the function returns one step named
        ``name``.  The optional function ``back_output(i)`` replaces ``back_fixed`` for the
        bytes of the output arrays.
        """
        if back_output is None:
            def back_output(i):
                return back_fixed(i, num_pixels)
        worker_terms = list(held_terms) + [
            ('back output', per_dev(back_output)),
            ('finished own band',
             per_dev(lambda i: back_own_band(i, num_pixels))),
            ('back batch', per_dev(lambda i: back_batch(i, num_pixels))),
        ]
        reduce_term = ('band reduce',
                       per_dev(lambda i: band_reduce(i, num_pixels)))
        if n == 1:
            return [_phase(name, worker_terms + [reduce_term], n,
                           base=base, base_terms=base_terms)]
        reduce_terms = list(held_terms) + [reduce_term]
        return [_phase(f'{name} [back workers]', worker_terms, n,
                       base=base, base_terms=base_terms),
                _phase(f'{name} [band reduce]', reduce_terms, n,
                       base=base, base_terms=base_terms)]

    # A direct reconstruction is the filter and one back projection.  It builds no prior,
    # no Hessian diagonal, no partition sequence, and no reconstruction loop.  For this
    # reason, the library workspace is the only term that all of its steps share.
    if plan.workload == 'direct':
        p_full = plan.num_pixels_full
        base_terms = [workspace_term]
        base = list(workspace_term[1])
        # The filter copies the sinogram to the devices and writes a filtered array of the same
        # shape.  The later steps hold only the filtered sinogram.  A sinogram that the caller
        # placed on the devices is already excluded from the available memory.
        held_arrays = [
            ('filtered sinogram', per_dev(sino_dev)),
        ]
        filter_terms = [('sinogram', per_dev(sino_dev))] + held_arrays + [
            ('filter row weights', per_dev(filter_row_weights)),
            ('filter row batch', per_dev(filter_row_batch)),
        ]
        scatter_terms = held_arrays + [
            ('back cylinders', per_dev(lambda i: cyl(i, p_full))),
            ('scatter buffer', per_dev(recon_dev)),
        ]
        if plan.helical:
            scatter_terms.append(('helical z-weight', per_dev(recon_dev)))
        phases.append(_phase('direct recon (filter)', filter_terms, n,
                             base=base, base_terms=base_terms))
        phases.extend(back_phases('direct recon (back loop)', held_arrays,
                                  p_full, base, base_terms))
        phases.append(_phase('direct recon (scatter)', scatter_terms, n,
                             base=base, base_terms=base_terms))
        return MemoryEstimate(devices=list(plan.devices), phases=phases,
                      num_pixels_full=int(plan.num_pixels_full))

    # The denoise plan describes one call of QGGMRFDenoiser.  Every term has the shape of
    # the image and is divided among devices by slice_blocks.
    if plan.workload == 'denoise':
        base_terms = [workspace_term]
        base = list(workspace_term[1])

        def halo_columns(i):
            """This function returns the bytes of the boundary slices that one device holds
            during a pass of the denoiser.

            The value is zero on a single device.
            """
            if n == 1:
                return 0
            return (DENOISE_HALO_COLUMNS * int(plan.num_pixels_grid)
                    * _F32_BYTES)

        def partition_indices(i):
            """This function returns the bytes of the subset partition.

            Every device holds the whole partition.
            """
            return sum(
                int(g) * math.ceil(plan.num_pixels_full / max(1, int(g)))
                * _INT64_BYTES for g in plan.partition_granularities)

        # These arrays exist from the start of the denoiser call until it returns.  An
        # initial image supplied by the caller is a fourth array.  By default, the initial
        # image is the input image, so the initial image adds no memory.
        held_arrays = [
            ('input image', per_dev(recon_dev)),
            ('init image', per_dev(
                lambda i: recon_dev(i) if plan.init_recon_supplied else 0)),
            ('working image', per_dev(recon_dev)),
            ('residual', per_dev(recon_dev)),
            ('subset indices', per_dev(partition_indices)),
            ('qggmrf halos', per_dev(halo_columns)),
        ]
        phases.append(_phase('denoise state placement', held_arrays, n,
                             base=base, base_terms=base_terms))
        for granularity in plan.granularities:
            p_sub = math.ceil(plan.num_pixels_full / max(1, int(granularity)))
            # The prior and the update direction run one after the other.  The maximum over
            # steps on each device therefore chooses the larger of the two.
            sub_phases = (
                ('prior', [('prior cylinders', per_dev(
                    lambda i: plan.qggmrf_cylinders * cyl(i, p_sub)))]),
                ('update direction', [('direction cylinders', per_dev(
                    lambda i: DENOISE_DIRECTION_CYLINDERS * cyl(i, p_sub)))]),
                ('state application', [
                    ('direction and scaled direction', per_dev(
                        lambda i: DENOISE_APPLY_CYLINDERS * cyl(i, p_sub)))]),
            )
            for name, terms in sub_phases:
                phases.append(_phase(
                    f'denoise subset {name} (granularity {granularity})',
                    held_arrays + terms, n, base=base, base_terms=base_terms))
        # image_l1_norm sums the image one chunk at a time, so its array of absolute values
        # has the size of one chunk.
        def l1_chunk(i):
            return reduction_chunk_bytes(recon_dev(i))[0]

        phases.append(_phase(
            'denoise per-pass statistics',
            held_arrays + [('L1 norm chunk', per_dev(l1_chunk))], n,
            base=base, base_terms=base_terms))
        return MemoryEstimate(devices=list(plan.devices), phases=phases,
                      num_pixels_full=int(plan.num_pixels_full))

    # A call of forward_project or back_project holds only the arrays of that one projection.
    # It holds no weights, no filtered sinogram, and no reconstruction state.  Both calls hold
    # the pixel indices on every device.
    if plan.workload in ('forward', 'back'):
        p_full = plan.num_pixels_full
        base_terms = [workspace_term]
        base = list(workspace_term[1])
        index_term = ('pixel indices', per_dev(lambda i: p_full * _INT64_BYTES))
        if plan.workload == 'forward':
            # The gather copies the voxels inside the ROR mask from the volume.  The projection
            # step holds this copy but not the volume.  A volume that the caller placed on the
            # devices is already excluded from the available memory.
            gather_terms = [
                index_term,
                ('recon', per_dev(recon_dev)),
                ('voxel gather', per_dev(lambda i: cyl(i, p_full))),
            ]
            phases.append(_phase('forward projection (gather)', gather_terms, n,
                                 base=base, base_terms=base_terms))
            # forward_fixed counts the output twice on several devices.  This call holds one
            # output, so the output is counted here with sino_dev.
            forward_terms = [
                index_term,
                ('voxel gather', per_dev(lambda i: cyl(i, p_full))),
                ('transferred cylinders', per_dev(
                    lambda i: forward_transferred_cylinders(i, p_full))),
                ('forward output', per_dev(
                    lambda i: sino_dev(i) if is_view_owner(i) else 0)),
                ('forward block', per_dev(lambda i: forward_block(i, p_full))),
                ('forward batch', per_dev(lambda i: forward_batch(i, p_full))),
            ]
            phases.append(_phase('forward projection', forward_terms, n,
                                 base=base, base_terms=base_terms))
        else:
            held_arrays = [index_term, ('sinogram', per_dev(sino_dev))]

            # On a single device, this call holds one output array fewer than back_fixed counts.
            def back_output(i):
                if n == 1:
                    return (back_fixed(i, p_full) - back_block(i, p_full)
                            if is_slice_owner(i) else 0)
                return back_fixed(i, p_full)

            phases.extend(back_phases('back projection (loop)', held_arrays,
                                      p_full, base, base_terms,
                                      back_output=back_output))
            scatter_terms = held_arrays + [
                ('back cylinders', per_dev(lambda i: cyl(i, p_full))),
                ('scatter volume', per_dev(recon_dev)),
            ]
            phases.append(_phase('back projection (scatter)', scatter_terms, n,
                                 base=base, base_terms=base_terms))
        return MemoryEstimate(devices=list(plan.devices), phases=phases,
                              num_pixels_full=int(plan.num_pixels_full))

    # This step is the direct reconstruction.  It runs only when the caller supplies no
    # initial reconstruction.  Its back projection uses all pixels, so it is the largest
    # projection of the run.
    if not plan.init_recon_supplied and not plan.resume:
        p_full = plan.num_pixels_full
        # The back projection loop and the scatter run one after the other.  The output of
        # the back projection becomes the input of the scatter.
        loop_held_arrays = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('filtered sinogram', per_dev(sino_dev)),
        ]
        scatter_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('filtered sinogram', per_dev(sino_dev)),
            ('back cylinders', per_dev(lambda i: cyl(i, p_full))),
            ('scatter buffer', per_dev(recon_dev)),
        ]
        if plan.helical:
            scatter_terms.append(('helical z-weight', per_dev(recon_dev)))
        # Both steps are added.  The maximum on each device chooses the larger one.
        phases.extend(back_phases('direct recon (back loop)', loop_held_arrays,
                                  p_full, constant_base, constant_terms))
        phases.append(_phase('direct recon (scatter)', scatter_terms, n,
                             base=constant_base, base_terms=constant_terms))

    # These steps compute the initial error sinogram.  The forward projection and the
    # formation of the error sinogram hold different arrays, so they are separate steps.
    if not plan.resume:
        p_full = plan.num_pixels_full
        forward_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('init recon', per_dev(recon_dev)),
            ('voxel gather', per_dev(lambda i: cyl(i, p_full))),
            ('transferred cylinders', per_dev(
                lambda i: forward_transferred_cylinders(i, p_full))),
            ('forward output', per_dev(forward_fixed)),
            ('forward block', per_dev(lambda i: forward_block(i, p_full))),
            ('forward batch', per_dev(lambda i: forward_batch(i, p_full))),
        ]
        phases.append(_phase('initial forward projection', forward_terms,
                             n, base=constant_base,
                             base_terms=constant_terms))
        # The error sinogram is formed in the array of the projection, so the two are one
        # array.  The dot products that set the scale are summed one block of views at a
        # time.  A weighted block holds the products and the weighted products, so the
        # estimate counts two blocks.
        dot_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('forward projection', per_dev(sino_dev)),
            ('dot product blocks', per_dev(
                lambda i: 2 * sino_reduction_block(i))),
            ('init recon', per_dev(recon_dev)),
        ]
        error_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            # The projection is scaled by -alpha in place, and the sinogram is added to it.
            # The estimate therefore counts one array for the pair.
            ('error sinogram', per_dev(sino_dev)),
            ('init recon', per_dev(recon_dev)),
        ]
        scale_terms = [
            ('sinogram', per_dev(sino_dev)),
            ('weights', per_dev(supplied_weights_term)),
            ('error sinogram', per_dev(sino_dev)),
            ('init recon (x2, scaling)', per_dev(lambda i: 2 * recon_dev(i))),
        ]
        phases.append(_phase('initial dot products', dot_terms, n,
                             base=constant_base, base_terms=constant_terms))
        phases.append(_phase('error sinogram formation', error_terms, n,
                             base=constant_base, base_terms=constant_terms))
        phases.append(_phase('init recon scaling', scale_terms, n,
                             base=constant_base, base_terms=constant_terms))

    # This step computes the Hessian diagonal.  It is the only step that can use all
    # pixels of the grid rather than the pixels inside the ROR mask.
    if not plan.fm_hessian_supplied:
        # With the mask, the code back projects the pixels inside the mask and scatters
        # them into a volume of zeros.  Without the mask, it back projects the whole grid.
        p_hess = (plan.num_pixels_full if plan.hessian_masked
                  else plan.num_pixels_grid)
        hessian_held_arrays = [
            ('error sinogram', per_dev(sino_dev)),
            ('hessian weights', per_dev(weights_term)),
            ('init recon', per_dev(recon_dev)),
        ]
        phases.extend(back_phases('hessian diagonal', hessian_held_arrays,
                                  p_hess, constant_base, constant_terms))
        if plan.hessian_masked:
            # The scatter holds the cylinder arrays and the volume of zeros at once.  For
            # this reason, it is a separate step from the back projection loop.
            scatter_terms = [
                ('error sinogram', per_dev(sino_dev)),
                ('hessian weights', per_dev(weights_term)),
                ('init recon', per_dev(recon_dev)),
                ('hessian cylinders', per_dev(lambda i: cyl(i, p_hess))),
                ('hessian scatter volume', per_dev(recon_dev)),
            ]
            phases.append(_phase('hessian scatter', scatter_terms, n,
                                 base=constant_base,
                                 base_terms=constant_terms))

    # These steps compute the statistics of each iteration.  weighted_square_sum sums one
    # block of views at a time, so the squared error holds two blocks.  The L1 norm of the
    # recon runs after those blocks are freed, so it is a separate step.
    stats_sub_phases = (
        ('squared error',
         ('squared-error products',
          per_dev(lambda i: 2 * sino_reduction_block(i)))),
        ('recon L1 norm',
         ('recon L1 norm chunk',
          per_dev(lambda i: reduction_chunk_bytes(recon_dev(i))[0]))),
    )
    stats_base_terms = constant_terms + list(persistent.items())
    for name, term in stats_sub_phases:
        phases.append(_phase(f'per-iteration statistics ({name})', [term], n,
                             base=persistent_total,
                             base_terms=stats_base_terms))

    # ── the subset update, once for each granularity in the sequence ─────────
    prior_cylinders = PROX_CYLINDERS if plan.prox else plan.qggmrf_cylinders
    for granularity in plan.granularities:
        p_sub = math.ceil(plan.num_pixels_full / max(1, int(granularity)))
        index_bytes = per_dev(lambda i: p_sub * _INT64_BYTES)
        sub_phases = {
            'prior': [
                ('prior cylinders', per_dev(
                    lambda i: prior_cylinders * cyl(i, p_sub))),
            ],
            # This entry lists only the arrays that the back projection holds throughout.
            # back_phases below builds the two steps of the back projection and their terms.
            'back projection': [
                ('prior gradient and hessian',
                 per_dev(lambda i: 2 * cyl(i, p_sub))),
                ('weighted error sinogram', per_dev(
                    lambda i: sino_dev(i) if plan.weights_supplied else 0)),
            ],
            'update direction': [
                ('direction cylinders', per_dev(
                    lambda i: DIRECTION_CYLINDERS * cyl(i, p_sub))),
            ],
            'delta forward projection': [
                ('update direction', per_dev(
                    lambda i: (2 if plan.positivity else 1) * cyl(i, p_sub))),
                ('delta sinogram', per_dev(sino_dev)),
                # This is the second output block of a forward projection on several
                # devices, as described in forward_fixed.  With positivity, the update
                # projects a second delta while it holds the first.
                ('forward assembly', per_dev(
                    lambda i: sino_dev(i) if n > 1 and is_view_owner(i) else 0)),
                ('transferred cylinders', per_dev(
                    lambda i: forward_transferred_cylinders(i, p_sub))),
                ('forward block', per_dev(lambda i: forward_block(i, p_sub))),
                ('forward batch', per_dev(lambda i: forward_batch(i, p_sub))),
            ],
            'state application': [
                ('direction and scaled direction',
                 per_dev(lambda i: APPLY_CYLINDERS * cyl(i, p_sub))),
                ('delta sinogram', per_dev(sino_dev)),
            ],
        }
        loop_base_terms = constant_terms + list(persistent.items())
        for name, terms in sub_phases.items():
            all_terms = terms + [('subset indices', index_bytes)]
            phase_name = f'subset {name} (granularity {granularity})'
            if name == 'back projection':
                phases.extend(back_phases(phase_name, all_terms, p_sub,
                                          persistent_total, loop_base_terms))
                continue
            phases.append(_phase(phase_name, all_terms, n,
                                 base=persistent_total,
                                 base_terms=loop_base_terms))

    return MemoryEstimate(devices=list(plan.devices), phases=phases,
                  num_pixels_full=int(plan.num_pixels_full))


def _phase(name, terms, n, base=None, base_terms=None):
    totals = [sum(vals[i] for _, vals in terms) for i in range(n)]
    if base is not None:
        totals = [totals[i] + base[i] for i in range(n)]
        terms = list(base_terms or []) + list(terms)
    return StepMemory(name=name, per_device=totals, terms=list(terms))


# ── building a plan from a model ─────────────────────────────────────────────
def qggmrf_cylinder_count(model):
    """This function returns the number of cylinder arrays that the qGGMRF prior holds.

    The function returns the compiled count while compilation works.  After a qGGMRF compile
    fails and the code runs in eager mode, the function returns the eager count.  Eager mode holds more
    temporary arrays.
    """
    from . import projectors
    if not model.compile_enabled:
        return QGGMRF_CYLINDERS_EAGER
    for key in projectors._COMPILE_ERRORS:
        if 'qggmrf' in key:
            return QGGMRF_CYLINDERS_EAGER
    return QGGMRF_CYLINDERS_COMPILED


def plan_from_model(model, devices, workload='recon', partition_sequence=None,
                    weights=None, init_recon=None, fm_hessian=None,
                    prox_input=None, init_error_sinogram=None):
    """This function builds a :class:`MemoryEstimateInputs` for ``model`` on a given list of devices.

    The device list is an argument, so the caller can test device counts that differ from
    the current count of the model.  ``workload`` names the call that the plan describes.
    ``'recon'`` is a full reconstruction, and it is the default.  ``'direct'`` is a direct
    reconstruction, which is the filter and one back projection.  ``'denoise'`` is one call
    of QGGMRFDenoiser.  ``'forward'`` and ``'back'`` are one call of ``forward_project`` and
    ``back_project``.  For a denoiser, the function checks that the sinogram shape equals
    the image shape.
    """
    sinogram_shape = tuple(int(s) for s in model.get_params('sinogram_shape'))
    recon_shape = tuple(int(s) for s in model.get_params('recon_shape'))
    devices = [torch.device(d) for d in devices]
    denoising = workload == 'denoise'
    if denoising and sinogram_shape != recon_shape:
        raise ValueError(
            "the 'denoise' workload prices a QGGMRFDenoiser, whose "
            'sinogram_shape is its image shape.  This model has '
            f'sinogram_shape {sinogram_shape} and recon_shape {recon_shape}.')

    sino_placement = _sharding.Placement(devices, axis=0,
                                         axis_len=sinogram_shape[0])
    recon_placement = _sharding.Placement(devices, axis=-1,
                                          axis_len=recon_shape[2])
    view_blocks = [end - start for _d, (start, end)
                   in sino_placement.shard_ranges()]
    slice_blocks = [end - start for _d, (start, end)
                    in recon_placement.shard_ranges()]
    rows_track_slices = bool(getattr(model, 'rows_track_slices', False))
    sino_rows = sinogram_shape[1]

    granularity = list(model.get_params('granularity'))
    if partition_sequence is None:
        partition_sequence = list(model.get_params('partition_sequence'))
    if denoising:
        # A denoiser call builds and uses only the partition that the first entry of the
        # sequence names.  A reconstruction uses the whole sequence.
        index = int(partition_sequence[0]) if len(partition_sequence) else 0
        visited = [granularity[index]] if index < len(granularity) else []
        built = list(visited)
    else:
        visited = sorted({granularity[int(k)] for k in partition_sequence
                          if int(k) < len(granularity)})
        built = list(granularity)

    num_pixels_full = int(model.full_index_count())
    num_pixels_grid = recon_shape[0] * recon_shape[1]

    view_batch_bytes = None if denoising else _model_view_batch_bytes(model, len(devices))

    return MemoryEstimateInputs(
        sinogram_shape=sinogram_shape,
        recon_shape=recon_shape,
        devices=devices,
        workload=workload,
        view_blocks=view_blocks,
        slice_blocks=slice_blocks,
        sino_rows=int(sino_rows),
        rows_track_slices=rows_track_slices,
        num_pixels_full=num_pixels_full,
        num_pixels_grid=num_pixels_grid,
        granularities=tuple(visited) or (granularity[0],),
        partition_granularities=tuple(built) or (granularity[0],),
        weights_supplied=weights is not None,
        fm_hessian_supplied=fm_hessian is not None,
        init_recon_supplied=init_recon is not None,
        resume=init_error_sinogram is not None,
        prox=prox_input is not None,
        positivity=bool(model.get_params('positivity_flag')),
        helical=_is_helical(model),
        hessian_masked=model.get_params('use_ror_mask') is not False,
        back_band=getattr(model, 'back_project_slice_band', None),
        # The model method supplies the pixel batch, so the count always matches the
        # projection loop.  A denoiser has no forward projection.
        pixel_batch=(None if denoising
                            else model._forward_pixel_batch()),
        qggmrf_cylinders=qggmrf_cylinder_count(model),
        view_batch_bytes=view_batch_bytes,
        torch_projector_directions=(() if denoising
                               else torch_projector_directions(model)),
        back_adds_into_output=(not denoising and bool(getattr(
            model._view_batch_bodies()[1], '_adds_into_output', False))),
    )


def workload_covers(checked, incoming):
    """This function returns True when a device count that passed the check for the workload
    ``checked`` needs no new check before the workload ``incoming`` runs.

    A device count that passed a ``'recon'`` check also passes a ``'direct'``, ``'forward'``,
    or ``'back'`` check.  A device count that passed a ``'direct'`` check also passes a
    ``'back'`` check.  Every workload covers itself.  ``'denoise'`` covers no other workload.
    """
    return checked == incoming or (checked, incoming) in (
        ('recon', 'direct'), ('recon', 'forward'), ('recon', 'back'),
        ('direct', 'back'))


def torch_projector_directions(model):
    """This function returns the projection directions that this model runs with a torch
    projection function.

    A torch projection function has no ``_view_batch_cost`` attribute.  The function checks
    the forward and back directions separately.
    """
    fwd_body, back_body = model._view_batch_bodies()
    return tuple(name for name, body in (('forward', fwd_body),
                                         ('back', back_body))
                 if getattr(body, '_view_batch_cost', None) is None)


def _model_view_batch_bytes(model, n_devices):
    """This function returns a function of (direction, num_pixels, band_cols).

    The returned function uses the projection functions of the model and returns
    (batch, bytes_per_view).
    """
    projector_functions = model.projector_functions
    fwd_body, back_body = model._view_batch_bodies()
    args = model._view_batch_args()

    def view_batch_bytes(direction, num_pixels, band_cols):
        body = fwd_body if direction == 'forward' else back_body
        view_batch, bytes_per_view = projector_functions.view_batch_bytes(
            body, num_pixels, band_cols, args, n_devices=n_devices)
        # A kernel with a ``_view_peak_bytes`` attribute states its peak memory per view
        # separately from the cost that sets the view batch size.
        peak_bytes = getattr(body, '_view_peak_bytes', None)
        if peak_bytes is not None:
            bytes_per_view = peak_bytes(num_pixels, band_cols, args)
        return view_batch, bytes_per_view
    return view_batch_bytes


def _is_helical(model):
    params = model.params
    if 'view_params_array' not in params:
        return False
    try:
        import numpy as np
        shifts = np.asarray(model.get_params('view_params_array'))
        if shifts.ndim != 2 or shifts.shape[1] < 2:
            return False
        return float(np.max(shifts[:, 1]) - np.min(shifts[:, 1])) > 0
    except (TypeError, ValueError, IndexError):
        return False


# ── the available memory and the check that the estimate fits ────────────────
def device_budget_bytes(device):
    """This function returns the bytes that a new allocation can still obtain on one GPU.

    The result is the free memory reported by the driver plus the memory that torch holds
    but does not use.  Memory held by other processes counts as unavailable.  The value is
    read at one moment and does not account for fragmentation.  The safety margin in
    :func:`layout_fits` covers both effects.
    """
    device = torch.device(device)
    if device.type != 'cuda':
        return None
    free, _total = torch.cuda.mem_get_info(device)
    reclaimable = (torch.cuda.memory_reserved(device)
                   - torch.cuda.memory_allocated(device))
    return int(free) + int(reclaimable)


def bytes_already_on_devices(devices, arrays):
    """This function returns the bytes of ``arrays`` that are already on each device.

    The available memory already excludes these bytes.  For this reason, the required
    memory leaves them out.
    """
    totals = [0] * len(devices)
    index = {}
    for position, device in enumerate(devices):
        index.setdefault(str(torch.device(device)), position)
    for array in arrays:
        for tensor in _iter_tensors(array):
            if tensor.device.type != 'cuda':
                continue
            position = index.get(str(tensor.device))
            if position is None:
                continue
            totals[position] += tensor.numel() * tensor.element_size()
    return totals


def _iter_tensors(array):
    if array is None:
        return
    if isinstance(array, _sharding.Shards):
        for tensor in array.tensors:
            yield tensor
    elif torch.is_tensor(array):
        yield array
    elif isinstance(array, (list, tuple)):
        for item in array:
            for tensor in _iter_tensors(item):
                yield tensor


def layout_fits(estimate, budgets, bytes_already_on_devices=None, margin=0.15):
    """This function checks whether every device can hold its estimated peak memory.

    The required memory on a device is the estimated peak minus
    ``bytes_already_on_devices``, times ``1 + margin``.

    Returns:
        (fits, rows): fits is True when every device has enough memory.  rows holds one
        (device, demand, budget) tuple for each device, for reporting.
    """
    peaks = estimate.per_device_peaks()
    bytes_already_on_devices = bytes_already_on_devices or [0] * len(peaks)
    rows, fits = [], True
    for i, device in enumerate(estimate.devices):
        demand = int((1.0 + margin) * max(0, peaks[i] - bytes_already_on_devices[i]))
        budget = budgets[i]
        ok = budget is None or demand <= budget
        fits = fits and ok
        rows.append((device, demand, budget))
    return fits, rows


# QGGMRFDenoiser.denoise_stack denoises several volumes of the same shape at once on one
# device.  As a result, every term with the shape of the image is held once for each volume.
# The terms named here are held only once for the whole batch.
DENOISE_BATCH_FIXED_TERMS = frozenset({'library workspace', 'subset indices',
                                       'qggmrf halos'})


def _denoise_batch_phase_split(phase):
    """This function splits the bytes of one denoiser step into fixed bytes and bytes per
    volume.

    The denoiser step runs on a single device.
    """
    fixed = sum(vals[0] for name, vals in phase.terms
                if name in DENOISE_BATCH_FIXED_TERMS)
    per_volume = sum(vals[0] for name, vals in phase.terms
                     if name not in DENOISE_BATCH_FIXED_TERMS)
    return int(fixed), int(per_volume)


def denoise_batch_peak_bytes(estimate, batch_size):
    """This function returns the estimated peak memory of one denoiser call on a batch of
    ``batch_size`` volumes.

    ``estimate`` is the estimate for one volume on one device.  The bytes of each step are
    its fixed terms plus ``batch_size`` times its terms per volume.  The peak is the largest
    step.  The result can be larger than the memory that the call holds, but it is never
    smaller.
    """
    peaks = []
    for phase in estimate.phases:
        fixed, per_volume = _denoise_batch_phase_split(phase)
        peaks.append(fixed + int(batch_size) * per_volume)
    return max(peaks)


def largest_denoise_batch(estimate, budget, margin=0.15):
    """This function returns the largest number of volumes that one denoiser call can
    process within ``budget`` bytes.

    The function applies the same test as :func:`layout_fits`.  The estimated peak times
    ``1 + margin`` must not exceed the budget.  The function returns 0 when one volume does
    not fit.
    """
    allowed = float(budget) / (1.0 + margin)
    limit = None
    for phase in estimate.phases:
        fixed, per_volume = _denoise_batch_phase_split(phase)
        if per_volume <= 0:
            room = 0 if fixed > allowed else None
        else:
            room = math.floor((allowed - fixed) / per_volume)
        if room is not None:
            limit = room if limit is None else min(limit, room)
    return max(0, limit if limit is not None else 0)


def format_shortfall(estimate, rows, num_devices_tried, closest_count=None,
                     remedies=None):
    """This function writes the error message for a reconstruction that does not fit.

    The message says which device did not fit, which step uses the most memory, and which
    settings reduce that step.  ``rows`` describes the device count that came closest to
    fitting, not the last count tried.
    """
    lines = ['this reconstruction needs more memory than the available '
             'CUDA devices have free.', '']
    header = f'{"device":>10}{"modeled need":>16}{"available":>14}{"shortfall":>14}'
    lines.append(header)
    worst_index, worst_gap = 0, -1
    for i, (device, demand, budget) in enumerate(rows):
        gap = 0 if budget is None else demand - budget
        if gap > worst_gap:
            worst_index, worst_gap = i, gap
        shortfall = '-' if gap <= 0 else _gb(gap)
        available = '-' if budget is None else _gb(budget)
        lines.append(f'{str(device):>10}{_gb(demand):>16}'
                     f'{available:>14}{shortfall:>14}')

    phase = estimate.dominant_phase(worst_index)
    peak = estimate.peak_bytes(worst_index)
    lines += ['', f'The dominant phase on {estimate.devices[worst_index]} is '
                  f'"{phase.name}" at {_gb(phase.per_device[worst_index])} '
                  f'of the {_gb(peak)} peak.']
    top = phase.dominant_terms(worst_index)
    if top:
        named = ', '.join(f'{name} {_gb(value)}' for name, value in top)
        lines.append(f'Its largest terms are: {named}.')
    tried = (', '.join(str(c) for c in num_devices_tried)
             if isinstance(num_devices_tried, (list, tuple))
             else str(num_devices_tried))
    closest = ('' if closest_count is None
               else f'  The closest was {closest_count}, shown above.')
    lines += [
        '', f'Device counts tried, largest first: {tried}.  None fit.'
            + closest,
        '', 'Remedies, most effective first:',
        '  model.back_project_slice_band = <slices>   '
        '# shrinks every back projection transient',
        '                                             '
        '# that is sized by a band, on top of what',
        '                                             '
        '# more devices already save',
        '  model.view_batch_size = <views>            '
        '# caps the projector batch transient',
        '  model.set_params(granularity=[...])        '
        '# a finer coarsest granularity shrinks the',
        '                                             '
        '# per-subset transients',
        '  CUDA_VISIBLE_DEVICES=...                   '
        '# exclude a device another process is using',
    ]
    lines += list(remedies or [])
    lines += ['', 'To run anyway: model.skip_memory_check = True']
    return '\n'.join(lines)


# ── calibration ──────────────────────────────────────────────────────────────
def env_device_count():
    """This function returns the device count set by the environment variable
    ``MBIRTORCH_NUM_DEVICES``, or None if the variable is not set.

    The function raises ValueError if the value is not a positive integer.
    """
    raw = os.environ.get(DEVICE_COUNT_ENV_VAR, '').strip()
    if not raw:
        return None
    try:
        count = int(raw)
    except ValueError:
        count = 0
    if count < 1:
        raise ValueError(
            f'{DEVICE_COUNT_ENV_VAR}={raw!r} is not a positive integer.  '
            'Unset it for the automatic device count, or set it to the '
            'number of devices to use.')
    return count


def calibration_enabled():
    return os.environ.get(CALIBRATION_ENV_VAR, '') not in ('', '0', 'false')


def calibration_start(devices):
    """This function resets the peak memory counters of torch on each CUDA device.

    The reset changes the counters for other code too.  For this reason, calibration runs
    only when its environment variable is set.
    """
    for device in devices:
        device = torch.device(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)


def calibration_report(estimate, devices):
    """This function returns the estimated and measured peaks on each CUDA device.

    Each row is a tuple (device, modeled, measured, ratio).
    """
    rows = []
    for i, device in enumerate(devices):
        device = torch.device(device)
        if device.type != 'cuda':
            continue
        measured = int(torch.cuda.max_memory_allocated(device))
        modeled = estimate.peak_bytes(i)
        ratio = (modeled / measured) if measured else float('inf')
        rows.append((device, modeled, measured, ratio))
    return rows


def format_calibration(rows, band=None):
    """This function formats the calibration table.

    ``band`` defaults to ``CALIBRATION_BAND``.  A reconstruction that uses torch projection
    functions is judged against ``TORCH_PROJECTOR_CALIBRATION_BAND`` instead.
    """
    low, high = band or CALIBRATION_BAND
    lines = ['memory estimate calibration (this mode owns '
             'torch.cuda.max_memory_allocated)',
             f'{"device":>10}{"modeled":>14}{"measured":>14}'
             f'{"ratio":>10}{"verdict":>12}']
    for device, modeled, measured, ratio in rows:
        if ratio < low:
            verdict = 'UNDER'
        elif ratio > high:
            verdict = 'over'
        else:
            verdict = 'ok'
        lines.append(f'{str(device):>10}{_gb(modeled):>14}{_gb(measured):>14}'
                     f'{ratio:>10.3f}{verdict:>12}')
    return '\n'.join(lines)

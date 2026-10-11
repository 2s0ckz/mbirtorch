"""This module divides arrays among several devices and moves data between them.

One process controls every device, with one Python thread per device.  The module splits an
array along one axis into contiguous blocks, one block (shard) per device.  A :class:`Shards`
container holds the shards, and a :class:`Placement` records which block each device holds.
The sinogram stays on its devices, and only voxel data moves between devices.  The function
``sum_band_to_owner`` sums the partial back projections from the devices.  The function
``transfer_cylinder_batch`` collects the voxel data that a forward projection needs.  Here a
cylinder is the set of voxels at one (row, col) pixel over all slices.
"""

import contextlib
import math
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


class Placement:
    """This class is used to record how an array is divided among devices.  It
    gives the axis that is split and the list of devices.  Each device holds one
    contiguous block of the array along that axis.

    The block lengths differ by at most one, with the longer blocks first.  If
    there are more devices than entries on the axis, the last devices hold
    nothing.

    Args:
        devices (sequence of torch.device or str): The devices that hold the
            array.  A single device means that the array is not divided.
        axis (int): The axis that is divided among the devices.  It may be
            negative.  A reconstruction is divided along the slice axis (-1),
            and a sinogram along the view axis (0).
        axis_len (int or None): The length of the divided axis, such as
            num_views for a sinogram.  If given, :meth:`shard_ranges` splits
            this length by default.
    """

    def __init__(self, devices, axis, axis_len=None):
        self.devices = [torch.device(d) for d in devices]
        if len(self.devices) < 1:
            raise ValueError("Placement requires at least one device.")
        self.axis = axis
        self.axis_len = int(axis_len) if axis_len is not None else None

    def __eq__(self, other):
        """This method returns True when both placements have the same devices
        and split the same axis of the same length.  It does not compare the
        other axes of the array.  It treats ``torch.device('cuda')`` and
        ``torch.device('cuda:0')`` as different devices.
        """
        if not isinstance(other, Placement):
            return NotImplemented
        return (self.devices == other.devices
                and self.axis == other.axis
                and self.axis_len == other.axis_len)

    def __hash__(self):
        """This method returns a hash of the three fields that :meth:`__eq__` compares."""
        return hash((tuple(self.devices), self.axis, self.axis_len))

    def __repr__(self):
        return 'Placement({}, axis={}, axis_len={})'.format(
            ','.join(str(d) for d in self.devices), self.axis, self.axis_len)

    @property
    def n_devices(self):
        return len(self.devices)

    @property
    def is_trivial(self):
        """This property is True when the placement has a single device."""
        return len(self.devices) == 1

    def shard_ranges(self, axis_len=None):
        """This method splits an axis of length ``axis_len`` into contiguous
        blocks, one per device.  It returns the half-open range of each block.
        The blocks are the same as those of ``numpy.array_split``.

        Args:
            axis_len (int, optional): The length of the axis to split.
                Defaults to the ``axis_len`` of this placement.

        Returns:
            list of (device, (start, end)): The half-open range held by each
            device, in device order.
        """
        if axis_len is None:
            axis_len = self.axis_len
        if axis_len is None:
            raise ValueError(
                'shard_ranges needs an axis length.  This placement was built '
                'without axis_len, so pass the length explicitly, as in '
                'shard_ranges(num_slices).')
        blocks = np.array_split(np.arange(int(axis_len)), len(self.devices))
        bounds = np.cumsum([0] + [len(b) for b in blocks])
        return [(dev, (int(bounds[i]), int(bounds[i + 1])))
                for i, dev in enumerate(self.devices)]


#: The number of bytes that ``Shards.gather`` copies to the CPU in one step.  Each CPU
#: thread copies through two pinned buffers of this size.
GATHER_SLOT_BYTES = 16 * 2 ** 20

#: The maximum number of CPU threads that ``Shards.gather`` uses per shard.
GATHER_THREADS_PER_SHARD = 16


def _gather_threads_per_shard(n_shards):
    """This function returns the number of CPU threads used to copy each shard.
    The number is the count of available CPUs divided by the number of shards
    and by torch's intra-op thread count.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    busy = max(1, n_shards) * max(1, torch.get_num_threads())
    return max(1, min(GATHER_THREADS_PER_SHARD, cpus // busy))


def _split_grid(rows, width, parts):
    """This function splits a (rows, width) grid into at most ``parts`` pieces,
    each given as (row0, row1, col0, col1).  The pieces are groups of whole rows.
    If the grid has one row, the pieces are groups of columns."""
    if rows == 1 and parts > 1:
        bounds = np.linspace(0, width, min(parts, width) + 1).astype(int)
        return [(0, 1, int(a), int(b))
                for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    bounds = np.linspace(0, rows, min(parts, rows) + 1).astype(int)
    return [(int(a), int(b), 0, width)
            for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _slab_pieces(rows, width, slot_elems):
    """This function yields the pieces (row0, row1, col0, col1) of a (rows,
    width) grid.  Each piece has at most ``slot_elems`` elements."""
    rows_per_slab = slot_elems // width
    if rows_per_slab >= 1:
        for r0 in range(0, rows, rows_per_slab):
            yield r0, min(r0 + rows_per_slab, rows), 0, width
    else:
        for r in range(rows):
            for c0 in range(0, width, slot_elems):
                yield r, r + 1, c0, min(c0 + slot_elems, width)


def _copy_through_slots(source, dest, slots):
    """This function copies a contiguous (rows, width) part of a shard into the
    CPU array ``dest``.  It copies one piece at a time and alternates between the
    two pinned buffers in ``slots``.
    """
    rows, width = source.shape
    if rows == 0 or width == 0:
        return
    pieces = list(_slab_pieces(rows, width, slots[0].numel()))
    on_cuda = source.device.type == 'cuda'
    stream = torch.cuda.current_stream(source.device) if on_cuda else None

    def issue(k):
        r0, r1, c0, c1 = pieces[k]
        piece = source[r0:r1, c0:c1]
        staged = slots[k % 2][:piece.numel()].view(r1 - r0, c1 - c0)
        staged.copy_(piece, non_blocking=on_cuda)
        landed = None
        if on_cuda:
            landed = torch.cuda.Event()
            landed.record(stream)
        return staged, landed

    pending = issue(0)
    for k in range(len(pieces)):
        staged, landed = pending
        if k + 1 < len(pieces):
            # This copy goes into the other buffer, whose last piece has already been written to dest.
            pending = issue(k + 1)
        if landed is not None:
            landed.synchronize()
        r0, r1, c0, c1 = pieces[k]
        dest[r0:r1, c0:c1].copy_(staged)


def _fill_from_shards_in_slabs(sources, out, axis):
    """This function copies every shard into its block of the CPU array ``out``.
    It uses several CPU threads per shard and copies :data:`GATHER_SLOT_BYTES` at a
    time.
    """
    shape = tuple(out.shape)
    rows = math.prod(shape[:axis])
    inner = math.prod(shape[axis + 1:])
    grid = out.view(rows, shape[axis] * inner)
    threads = _gather_threads_per_shard(len(sources))
    tasks = []
    start = 0
    for source in sources:
        length = int(source.shape[axis])
        if source.numel():
            source2d = source.reshape(rows, length * inner)
            dest2d = grid[:, start * inner:(start + length) * inner]
            for r0, r1, c0, c1 in _split_grid(rows, length * inner, threads):
                tasks.append((source2d[r0:r1, c0:c1], dest2d[r0:r1, c0:c1]))
        start += length

    def run(task):
        source2d, dest2d = task
        slot_elems = max(1, min(GATHER_SLOT_BYTES // out.element_size(),
                                source2d.numel()))
        pinned = source2d.device.type == 'cuda'
        try:
            slots = [torch.empty(slot_elems, dtype=out.dtype, pin_memory=pinned)
                     for _ in range(2)]
        except RuntimeError:
            # Pinned memory is unavailable.  Ordinary CPU memory gives the same
            # values, but the copies are slower.
            slots = [torch.empty(slot_elems, dtype=out.dtype) for _ in range(2)]
        _copy_through_slots(source2d, dest2d, slots)

    if len(tasks) == 1:
        run(tasks[0])
    elif tasks:
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            list(pool.map(run, tasks))


class Shards:
    """This class is used to hold an array that is divided among devices.  It
    holds one tensor per device and a :class:`Placement`.  The placement records
    which block of the whole array each tensor holds.

    This class does no arithmetic.  The projectors and the VCD loop work on the
    tensors directly.  :meth:`gather` returns the whole array as a numpy array.
    """

    def __init__(self, tensors, placement):
        if len(tensors) != placement.n_devices:
            raise ValueError(
                f"{len(tensors)} shard tensors for {placement.n_devices} devices.")
        self.tensors = list(tensors)
        self.placement = placement
        # Each shard must be on its placement device.  A device without an index,
        # such as 'cuda', matches any index of its type.
        for t, d in zip(self.tensors, placement.devices):
            if t.device.type != d.type or (
                    t.device.index is not None and d.index is not None
                    and t.device.index != d.index):
                raise ValueError(
                    f'Shard on {t.device} does not match its placement '
                    f'device {d}.')

    @property
    def dtype(self):
        """This property returns the dtype of the first shard."""
        return self.tensors[0].dtype

    def gather(self):
        """This method returns the whole array as a numpy array.  It places the
        shards end to end along the divided axis.  The result is a new
        C-contiguous array that shares no memory with the shards.

        Returns:
            numpy.ndarray: The whole array.
        """
        first = self.tensors[0]
        # The divided axis may be negative.
        axis = self.placement.axis % first.ndim
        # A device that holds nothing contributes length zero.
        lengths = [int(t.shape[axis]) for t in self.tensors]
        shape = [int(n) for n in first.shape]
        shape[axis] = sum(lengths)
        axis_len = self.placement.axis_len
        if axis_len is not None and axis_len != shape[axis]:
            raise ValueError(
                f'The shards cover {shape[axis]} entries of axis {axis}, but '
                f'their placement says that axis is {axis_len} long.')
        # The numpy array returned at the end shares memory with this CPU tensor.
        out = torch.empty(shape, dtype=first.dtype, device='cpu')
        # Each shard is paired with a view of its block in ``out``.
        pairs = []
        start = 0
        for tensor, length in zip(self.tensors, lengths):
            block = out.narrow(axis, start, length)
            start += length
            if tuple(tensor.shape) != tuple(block.shape):
                # Without this check, the copy below would broadcast a shard with a
                # wrong shape.
                raise ValueError(
                    f'A shard of shape {tuple(tensor.shape)} does not fill the '
                    f'{tuple(block.shape)} block of the whole array it covers.')
            pairs.append((tensor.detach(), block))
        if all(source.device.type == 'cuda' for source, _ in pairs):
            _fill_from_shards_in_slabs([source for source, _ in pairs], out,
                                       axis)
        else:
            for source, block in pairs:
                if block.numel() == 0:
                    continue            # This device holds nothing.
                try:
                    block.copy_(source)
                except RuntimeError:
                    # Metal copies only at 4-byte boundaries.  A 1-byte or 2-byte dtype
                    # must be copied to the CPU first.
                    block.copy_(source.cpu())
        # The result shares memory with ``out``, so it is C-contiguous.
        return out.numpy()


def reject_shards(function_name, **arrays):
    """This function raises an error if any argument is a :class:`Shards`
    container.

    A ``Shards`` container has no shape and does not support indexing,
    reshaping, or arithmetic.  The error message names the function, the
    argument, and the fix.

    Args:
        function_name (str): The name of the calling function, for the message.
        **arrays: The array arguments of the calling function, given by name.
            The function ignores any value that is not a ``Shards`` container.

    Raises:
        TypeError: If any of the given arguments is a ``Shards`` container.
    """
    divided = [name for name, value in arrays.items() if isinstance(value, Shards)]
    if divided:
        raise TypeError(
            '{} does not accept an array in the divided device form.  Passed in that '
            'form: {}.  Gather the shards to the host first with shards.gather(), then '
            'pass the host array.'.format(function_name, ', '.join(divided)))


# This flag records whether move_shard has already warned that it copies through CPU memory.
_warned_host_bounce = False


def is_dev2dev_safe(devices) -> bool:
    """This function tests whether a direct copy from devices[0] to devices[1]
    gives correct values.  It returns True for a single device.
    """
    devices = [torch.device(d) for d in devices]
    if len(devices) < 2:
        return True
    probe = torch.tensor([1.0, 2.0, 3.0, 4.0])
    src = probe.to(devices[0])
    dst = src.to(devices[1])
    return bool(torch.equal(dst.cpu(), probe))


def move_shard(x, target, dev2dev_safe=True):
    """This function copies ``x`` to the device ``target``.

    If ``x`` is already on ``target``, the function returns ``x`` unchanged.

    Args:
        x (tensor): The tensor to copy.
        target (torch.device): The destination device.
        dev2dev_safe (bool): The result of :func:`is_dev2dev_safe` for these
            devices.  If False, the copy goes through CPU memory.  The function
            then warns once per process.
    """
    if dev2dev_safe:
        return x.to(target)
    global _warned_host_bounce
    if not _warned_host_bounce:
        _warned_host_bounce = True
        warnings.warn(
            "Direct device-to-device transfer failed the round-trip probe on "
            "this hardware; routing cross-device transfers through host "
            "memory.  This is correct but slower.", stacklevel=2)
    return torch.as_tensor(x.detach().cpu().numpy()).to(target)


#: The number of bytes of one partial back projection that ``sum_band_to_owner``
#: copies in one step.
REDUCE_SLAB_BYTES = 256 * 2 ** 20


def reduce_slab_rows(num_rows, row_bytes):
    """This function returns the number of rows of a partial back projection
    that :func:`sum_band_to_owner` copies in one step.  The argument ``row_bytes``
    is the number of bytes in one row.  The memory estimate in ``_memory_estimate``
    also uses this function.
    """
    # The result is a loop step, so it must be at least 1.
    if row_bytes <= 0:
        return max(1, int(num_rows))
    return max(1, min(int(num_rows), int(REDUCE_SLAB_BYTES) // int(row_bytes)))


def sum_band_to_owner(partials, owner, dev2dev_safe=True):
    """This function copies the partial back projections to ``owner`` and sums
    them.

    When the views are divided among the devices, each device computes a
    partial back projection of a contiguous range of slices.  The device that
    holds those slices is ``owner``.  This function sums the partials on
    ``owner`` a group of rows at a time.  This limits the extra memory on
    ``owner`` to one group of rows per source device.  The function sums the
    partials in the order given and does not modify them.

    Args:
        partials (list of tensor): One partial back projection per device.  All
            have the same shape.
        owner (torch.device): The device that holds the slices.  The sum is
            computed and left on this device.
        dev2dev_safe (bool): This value is passed to :func:`move_shard`.
    """
    if len(partials) == 1:
        return move_shard(partials[0], owner, dev2dev_safe=dev2dev_safe)
    total = move_shard(partials[0], owner, dev2dev_safe=dev2dev_safe)
    if total is partials[0]:
        # move_shard returned the caller's tensor.  The sum goes into a copy so
        # that the caller's tensor is not modified.
        total = total.clone()
    num_rows = int(total.shape[0])
    row_bytes = (total.numel() // max(1, num_rows)) * total.element_size()
    step = reduce_slab_rows(num_rows, row_bytes)
    for start in range(0, num_rows, step):
        stop = min(start + step, num_rows)
        # A partial has shape (pixels, slices) with the slices contiguous.  So each
        # group of rows is contiguous, and each transfer is a single copy.
        slabs = [move_shard(p[start:stop], owner, dev2dev_safe=dev2dev_safe)
                 for p in partials[1:]]
        rows = total[start:stop]
        for slab in slabs:
            rows.add_(slab)
        # The copies must be released before the next list is built.  Otherwise the
        # memory in use doubles.
        slabs = None
    return total


def transfer_cylinder_batch(shard_tensors, p0, p1, target, dev2dev_safe=True):
    """This function collects the voxels at pixels ``[p0:p1]`` over all slices
    onto the device ``target``.

    The slices are divided among the devices.  The function copies rows
    ``[p0:p1]`` of each shard to ``target`` with :func:`move_shard`.  It then
    concatenates the copies in slice order.  A shard that holds no slices
    contributes a piece of width zero.

    Args:
        shard_tensors (sequence of tensor): The shards of the reconstruction, in
            slice order.  Each shard has shape (num_pixels, local_slices).
        p0 (int): The first pixel of the batch.
        p1 (int): One past the last pixel of the batch.
        target (torch.device): The device that holds the views to project.
        dev2dev_safe (bool): This value is passed to :func:`move_shard`.

    Returns:
        tensor: A tensor of shape (p1 - p0, total_slices) on ``target``.
    """
    pieces = [move_shard(t[p0:p1], target, dev2dev_safe=dev2dev_safe)
              for t in shard_tensors]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


# Each CUDA device has one extra stream for copies, so a copy and a projection can run
# at the same time.  The lock protects only the creation of a stream.
_COPY_STREAMS = {}
_COPY_STREAM_LOCK = threading.Lock()


def copy_stream(device):
    """This function returns the copy stream for ``device``.  It returns None
    for a device that is not a CUDA device.  The callers then copy
    synchronously.
    """
    device = torch.device(device)
    if device.type != 'cuda':
        return None
    index = (device.index if device.index is not None
             else torch.cuda.current_device())
    stream = _COPY_STREAMS.get(index)
    if stream is None:
        with _COPY_STREAM_LOCK:
            stream = _COPY_STREAMS.get(index)
            if stream is None:
                stream = torch.cuda.Stream(device=index)
                _COPY_STREAMS[index] = stream
    return stream


def _transfer_stream_devices(shard_tensors, target):
    """This function returns the CUDA devices that one transfer uses, sorted by
    device index.  These are the devices of the shards and the target."""
    seen = {}
    for dev in [t.device for t in shard_tensors] + [torch.device(target)]:
        if dev.type == 'cuda':
            index = (dev.index if dev.index is not None
                     else torch.cuda.current_device())
            seen[index] = torch.device('cuda', index)
    return [seen[index] for index in sorted(seen)]


def open_copy_streams(devices):
    """This function makes each copy stream wait for the compute stream of its
    device.  As a result, no copy reads a shard before the kernel that writes the
    shard has finished.  The caller calls this function once per forward
    projection.
    """
    for dev in devices:
        stream = copy_stream(dev)
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(torch.device(dev)))


def close_copy_streams(devices):
    """This function makes each compute stream wait for the copy stream of its
    device.  It is the counterpart of :func:`open_copy_streams`.  As a result, no
    later update overwrites a shard while a copy is still reading it.
    """
    for dev in devices:
        stream = copy_stream(dev)
        if stream is not None:
            torch.cuda.current_stream(torch.device(dev)).wait_stream(stream)


def transfer_cylinder_batch_async(shard_tensors, p0, p1, target,
                                  dev2dev_safe=True):
    """This function runs :func:`transfer_cylinder_batch` on the copy streams.
    The result is the same.  The transfer can run at the same time as the
    projection of an earlier batch.

    Returns:
        (tensor, ready): The batch of voxels, and a CUDA event that is recorded
        after the copies finish.  On a device that is not a CUDA device, the
        event is None and the copies are finished when the function returns.
    """
    stream = copy_stream(target)
    if stream is None:
        return transfer_cylinder_batch(shard_tensors, p0, p1, target,
                                       dev2dev_safe), None
    # Both ends of every copy must use a copy stream.  Otherwise the copy waits
    # behind the projections on the compute stream.
    with contextlib.ExitStack() as stack:
        for dev in _transfer_stream_devices(shard_tensors, target):
            stack.enter_context(torch.cuda.stream(copy_stream(dev)))
        cylinder = transfer_cylinder_batch(shard_tensors, p0, p1, target,
                                           dev2dev_safe)
        ready = torch.cuda.Event()
        ready.record(stream)
    # The compute stream reads the batch.  Without this line, the allocator could
    # reuse the memory of the batch while the projection still reads the batch.
    cylinder.record_stream(torch.cuda.current_stream(torch.device(target)))
    return cylinder, ready


def wait_for_cylinder_batch(target, ready):
    """This function makes the compute stream of ``target`` wait until a batch
    has arrived.  It waits on the event of that batch only.  As a result, the
    transfer of the next batch can continue during the projection of this batch.
    """
    if ready is not None:
        torch.cuda.current_stream(torch.device(target)).wait_event(ready)


# Running one thread per device.
def device_pool(n):
    """This function returns a thread pool that several :func:`run_per_device`
    calls can share.  The caller uses the pool as a context manager and passes it
    as ``executor=``.
    """
    return ThreadPoolExecutor(max_workers=n)


def run_per_device(devices, worker_fn, executor=None):
    """This function runs worker_fn once per device, each in its own thread.

    The function returns the results in device order, so result[i] belongs to
    devices[i].  The function does not synchronize the devices.  A caller that
    needs the values must synchronize.  With a single device, the function runs
    worker_fn on the calling thread.  A worker must wrap each Triton launch in
    ``with torch.cuda.device(...)``, because the current CUDA device of a new
    thread is 0.

    Args:
        devices (sequence): The devices to run on, with one thread per device.
        worker_fn (callable): The function worker_fn(i, device), which returns a
            result.
        executor (ThreadPoolExecutor, optional): A thread pool to use, such as
            one from :func:`device_pool`.  If None, the function creates a new
            pool and shuts it down before it returns.
    """
    devices = list(devices)
    n = len(devices)
    if n == 1:
        return [worker_fn(0, devices[0])]

    def _run(i):
        return worker_fn(i, devices[i])

    if executor is not None:
        return list(executor.map(_run, range(n)))
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(_run, range(n)))


def exchange_qggmrf_halos(recon_shards, dev2dev_safe=True):
    """This function exchanges the boundary slices that the qGGMRF prior needs
    when the slices are divided among devices.

    Each shard receives the last slice of the shard to its left and the first
    slice of the shard to its right.  Each received slice is a (num_pixels,)
    tensor on the device of the receiving shard.  At an edge of the volume, and
    next to a shard that holds no slices, the function returns None.  The prior
    treats None as a reflected boundary.

    Args:
        recon_shards (Shards): The reconstruction, divided along the slice axis.
            Each tensor has shape (num_pixels, local_slices).
        dev2dev_safe (bool): This value is passed to :func:`move_shard`.

    Returns:
        (left_halos, right_halos): Two lists in device order.  Each entry is a
        (num_pixels,) tensor, or None at an edge of the volume or next to a
        shard that holds no slices.
    """
    tensors = recon_shards.tensors
    devs = recon_shards.placement.devices
    n = len(tensors)
    # A boundary slice is sent only when the shards on both sides hold slices.
    # Shards with no slices are always last.
    joined =[t.shape[1] > 0 and u.shape[1] > 0
              for t, u in zip(tensors[:-1], tensors[1:])]
    left = [None] + [move_shard(tensors[i][:, -1].contiguous(), devs[i + 1],
                                dev2dev_safe) if joined[i] else None
                     for i in range(n - 1)]
    right = [move_shard(tensors[i + 1][:, 0].contiguous(), devs[i],
                        dev2dev_safe) if joined[i] else None
             for i in range(n - 1)] + [None]
    return left, right

"""This module holds the projection loop that all geometries share, and the torch.compile code.

The class ``Projectors`` loops over batches of views.  It limits the size of the temporary
arrays of each batch, and it keeps one compiled copy of each projection function per device.
Each geometry model supplies its own projection functions, which project one batch of views.
For a torch function, ``view_batch_size`` sets the number of views in a batch.  A hand-written
kernel states its own memory cost per view in its ``_view_batch_cost`` attribute.
"""

import logging
import os
import threading

import torch

from . import _sharding

_F32 = torch.float32

# ── torch.compile support ─────────────────────────────────────────────────────
# When a compile fails, the function runs in eager mode, and the error is recorded in
# _COMPILE_ERRORS.


class _DropAutotuneNotice(logging.Filter):
    """This class is used to suppress the compiler message that a GPU is too small for one
    optional tuning mode."""

    def filter(self, record):
        return 'Not enough SMs' not in record.getMessage()


logging.getLogger('torch._inductor.utils').addFilter(_DropAutotuneNotice())

_COMPILE_CACHE = {}
_COMPILE_ERRORS = {}
# Triton and inductor compilation is not thread safe, so this lock lets one thread
# compile at a time.
_GLOBAL_COMPILE_LOCK = threading.Lock()

#: The smallest recompile limit for each function.  mbirtorch sets this limit before it
#: compiles.  Torch counts recompiles for each code object, so the copies of one function on
#: different devices share the limit.  After the limit is reached, calls run slowly in eager mode.
_RECOMPILE_LIMIT_FLOOR = 64


def _raise_recompile_budget():
    """This function raises the torch recompile limit for each function on the calling thread.

    If the environment variable ``MBIRTORCH_RECOMPILE_LIMIT`` is set, the function sets the
    limit to its value.  Otherwise, the function sets the limit to at least
    ``_RECOMPILE_LIMIT_FLOOR``.
    """
    import torch._dynamo.config as dynamo_config

    override = os.environ.get('MBIRTORCH_RECOMPILE_LIMIT')
    if override:
        limit = int(override)
    else:
        limit = max(int(dynamo_config.recompile_limit),
                    _RECOMPILE_LIMIT_FLOOR)
    dynamo_config.recompile_limit = limit
    dynamo_config.cache_size_limit = limit


def _shape_key(args, kwargs):
    parts = []
    for a in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        if torch.is_tensor(a):
            parts.append(('T', str(a.device)) + tuple(a.shape))
        elif isinstance(a, tuple):
            for t in a:
                parts.append(((('T', str(t.device)) + tuple(t.shape))
                              if torch.is_tensor(t) else ('o', str(t)[:32])))
        else:
            parts.append(('o', str(a)[:32]))
    return tuple(parts)


def maybe_compile(fn, enabled, instance_key=None):
    """This function returns a compiled form of ``fn`` when ``enabled`` is True.  Otherwise, it
    returns ``fn`` itself.

    The function caches one compiled form for each pair (``fn``, ``instance_key``).  The thread
    of each device passes its device index as ``instance_key``.  If ``fn`` has the attribute
    ``_mbirtorch_no_compile``, the function returns ``fn`` unchanged.  If the first call raises
    an error and an eager call then succeeds, the function records the error in
    ``_COMPILE_ERRORS`` and runs ``fn`` in eager mode from then on.
    """
    if not enabled or getattr(fn, '_mbirtorch_no_compile', False):
        return fn
    cache_key = fn if instance_key is None else (fn, instance_key)
    if cache_key in _COMPILE_CACHE:
        return _COMPILE_CACHE[cache_key]
    _raise_recompile_budget()
    compiled = torch.compile(fn)
    state = {"impl": compiled}
    seen_keys = set()

    def guarded(*args, **kwargs):
        key = _shape_key(args, kwargs)
        if key in seen_keys:
            return state["impl"](*args, **kwargs)
        # This is the first call at this shape, so it may compile.  The lock prevents two
        # threads from compiling at the same time.
        with _GLOBAL_COMPILE_LOCK:
            _raise_recompile_budget()
            try:
                out = state["impl"](*args, **kwargs)
            except Exception as e:                            # noqa: BLE001
                # If the compiler failed, this eager call succeeds, and fn stays eager from
                # then on.  If the input is invalid, this call raises the error again.
                out = fn(*args, **kwargs)
                _COMPILE_ERRORS[f"{fn.__module__}.{fn.__name__}"] = \
                    f"{type(e).__name__}: {e}"[:400]
                state["impl"] = fn
            seen_keys.add(key)
            return out

    guarded.__name__ = f"compiled_{fn.__name__}"
    _COMPILE_CACHE[cache_key] = guarded
    return guarded


# The wrappers below pad a call with fewer pixels than ``min_compiled_pixel_width``.  Inductor
# compiles the fused parallel-beam projection functions incorrectly for a single pixel.


def _callable_name(fn, fallback):
    """This function returns a readable name for a wrapped callable."""
    return getattr(fn, '__name__', fallback)


def forward_at_min_pixel_width(compiled, min_width):
    """This function wraps a forward projection function so that it pads a batch with fewer
    than ``min_width`` pixels to ``min_width`` pixels.

    The padded pixels repeat the last pixel index and have the value zero.  They add exactly
    0.0, so the result is identical to that of the unpadded call.
    """
    def forward_padded(values, pixel_indices, *args, **kwargs):
        width = int(pixel_indices.shape[0])
        if width == 0 or width >= min_width:
            return compiled(values, pixel_indices, *args, **kwargs)
        pad = min_width - width
        wide_values = torch.cat(
            [values, values.new_zeros((pad,) + tuple(values.shape[1:]))])
        wide_indices = torch.cat([pixel_indices,
                                  pixel_indices[-1:].repeat(pad)])
        return compiled(wide_values, wide_indices, *args, **kwargs)

    forward_padded.__name__ = f'padded_{_callable_name(compiled, "forward")}'
    return forward_padded


def back_at_min_pixel_width(compiled, min_width):
    """This function wraps a back projection function so that it pads a batch with fewer than
    ``min_width`` pixels to ``min_width`` pixels.

    The padded pixels repeat the last pixel index.  The wrapper removes the extra output rows.
    The remaining rows are exactly those of the unpadded call.
    """
    def back_padded(sino_batch, pixel_indices, *args, **kwargs):
        width = int(pixel_indices.shape[0])
        if width == 0 or width >= min_width:
            return compiled(sino_batch, pixel_indices, *args, **kwargs)
        pad = min_width - width
        wide_indices = torch.cat([pixel_indices,
                                  pixel_indices[-1:].repeat(pad)])
        block = compiled(sino_batch, wide_indices, *args, **kwargs)
        # The clone gives the output its own memory, so the padded block can be freed.
        return block[:width].clone()

    back_padded.__name__ = f'padded_{_callable_name(compiled, "back")}'
    return back_padded


def compile_serialized():
    """This function returns the compile lock as a context manager.  All threads of the process
    share this lock.  The lock is meant for hand-written kernels::

        with compile_serialized():
            my_triton_kernel[grid](...)     # The first launch compiles and autotunes.

    ``maybe_compile`` already takes this lock for torch.compile.  The lock should surround
    only the launches that compile.  If the lock were used as a decorator, every call would wait
    for the lock.
    """
    return _GLOBAL_COMPILE_LOCK




class Projectors:
    """This class is used to run the sparse forward and back projections of one model in
    batches of views.  It handles the loop over views and the memory limits, which do not
    depend on the geometry.

    The geometry enters only through the projection functions of the model, which
    ``_view_batch_bodies`` and ``_view_batch_args`` supply.  The forward and back projections
    compute the same centers for each (view, pixel) pair.  As a result, the two projections
    are exact adjoints.
    """

    # The limit on the bytes of the (Vb, P, cols) temporary arrays of a torch projection
    # function in one batch.  The estimate counts one such array per view, although the
    # functions hold two to five at once.
    VIEW_BATCH_TRANSIENT_BUDGET_BYTES = 2 * 2**30
    VIEW_BATCH_TRANSIENT_FLOOR_BYTES = 256 * 2**20
    VIEW_BATCH_SINO_MULTIPLE = 8
    # The default view batch size of a torch projection function when model.view_batch_size
    # is None.  A hand-written kernel supplies its own default instead.
    VIEW_BATCH_BODY_DEFAULT = 64

    def _transient_budget_bytes(self, n_devices=None):
        if self.model.torch_device.type == 'cpu':
            return self.VIEW_BATCH_TRANSIENT_BUDGET_BYTES
        num_views, num_rows, num_channels = self.model.get_params('sinogram_shape')
        # When the views are divided among devices, each device projects only its own views.
        # The limit therefore scales with the number of views on one device.  ``n_devices``
        # is a device count that the caller wants to test.
        n_dev = (self.model.sino_placement.n_devices if n_devices is None
                 else int(n_devices))
        local_views = -(-int(num_views) // n_dev)
        sino_bytes = local_views * num_rows * num_channels * 4
        return max(self.VIEW_BATCH_TRANSIENT_FLOOR_BYTES,
                   min(self.VIEW_BATCH_TRANSIENT_BUDGET_BYTES,
                       self.VIEW_BATCH_SINO_MULTIPLE * sino_bytes))

    def __init__(self, model):
        # The geometry supplies its projection functions as module-level functions.  A bound
        # method would keep the model alive in the module-level compile cache.
        self.model = model
        fwd_body, back_body = model._view_batch_bodies()
        use_compile = model.compile_enabled
        n_dev = model.sino_placement.n_devices
        min_width = int(getattr(model, 'min_compiled_pixel_width', 1))

        def bind(body, pad_narrow, i):
            """This function returns the compiled projection function for one device.

            If the model sets a minimum pixel width and the projection function was compiled,
            ``bind`` also wraps the projection function to pad narrow batches.
            """
            bound = maybe_compile(body, use_compile, instance_key=i)
            if min_width > 1 and bound is not body:
                bound = pad_narrow(bound, min_width)
            return bound

        self._fwd_body_per_dev = [
            bind(fwd_body, forward_at_min_pixel_width, i)
            for i in range(n_dev)]
        self._back_body_per_dev = [
            bind(back_body, back_at_min_pixel_width, i)
            for i in range(n_dev)]
        # The view parameters are read from the current params each time the projectors are
        # built.  move_shard copies them to each device.
        view_params_name = model.get_params('view_params_name')
        view_params = torch.as_tensor(model.get_params(view_params_name),
                                      dtype=_F32, device=model.torch_device)
        self._view_params_per_dev = [
            _sharding.move_shard(view_params, dev, model.dev2dev_safe)
            for dev in model.sino_placement.devices]
        self.view_params_array = self._view_params_per_dev[0]

    def _effective_view_batch(self, body, num_pixels, band_cols, args):
        """This method returns the number of views in a batch for one call of ``body``.

        The number is the default batch size, reduced if needed so that the temporary arrays
        of one batch fit within the memory limit.
        """
        return self.view_batch_bytes(body, num_pixels, band_cols, args)[0]

    def view_batch_bytes(self, body, num_pixels, band_cols, args,
                         n_devices=None):
        """This method returns ``(view_batch, bytes_per_view)`` for one call of ``body``.

        ``view_batch`` is the number of views that one call of ``body`` projects.  The memory
        estimate in ``_memory_estimate`` uses both numbers.  ``bytes_per_view`` does not include
        the output array or the array that accumulates partial results.  The memory estimate
        adds those arrays itself.

        Args:
            body: The projection function in use, either a kernel or a torch function.
            num_pixels (int): The number P of pixels that this call projects.
            band_cols (int): The number of columns in the call.  For the forward projection,
                this is the number of slices.  For the back projection, it is the number of
                rows in the local sinogram.
            args (dict): The arguments that the geometry passes to each call.
            n_devices (int, optional): A device count to use for the memory limit in place of
                the current device count of the model.  Only the memory estimate uses this
                argument.  Defaults to None.
        """
        nominal = self.model.view_batch_size
        cost = getattr(body, '_view_batch_cost', None)
        if cost is None:
            cols = self.model._transient_cols(band_cols)
            bytes_per_view = num_pixels * cols * 4
            if nominal is None:
                nominal = self.VIEW_BATCH_BODY_DEFAULT
        else:
            bytes_per_view, view_chunk = cost(num_pixels, band_cols, args)
            if nominal is None:
                nominal = view_chunk
        budget = self._transient_budget_bytes(n_devices=n_devices)
        cap = budget // max(1, int(bytes_per_view))
        return max(1, min(int(nominal), int(cap))), int(bytes_per_view)

    def sparse_forward_project_view_range(self, band_values, pixel_indices,
                                          view_range, slice_start=0,
                                          dev_index=0, plan=None,
                                          accumulate_into=None):
        """This method forward projects voxel values into the sinogram block of one range of
        views.

        Each device holds one contiguous range of views.  The projection function of the
        geometry computes the projection and sets the order of the output axes.  This method
        loops over batches of views, applies the memory limit, and assembles the output.  The
        single-device projection calls this method with the range (0, num_views).

        Args:
            band_values: A (P, cols) tensor on the device of this view range.  Each row holds
                the voxels of one pixel across all slices, or across a range of slices.
            pixel_indices: A (P,) int64 tensor on the same device.
            view_range: The half-open range (v0, v1) of global view indices.
            slice_start (int): The global index of the first slice in ``band_values``.  This
                argument is used by geometries such as cone beam, in which a detector row does
                not correspond to one slice.  For other geometries it must be 0.
            dev_index (int): The index of the compiled copy of the projection function to use.
            plan: This argument is not used.
            accumulate_into: An existing block with the shape of the output, or None.  If a
                block is given, the method adds the projection to it and returns it.  If None,
                the method allocates a new block.

        Returns:
            A tensor of shape (v1 - v0, rows, num_channels) on the device of the input.  For
            geometries such as cone beam, rows is the number of detector rows.  For other
            geometries, rows equals the number of slices in ``band_values``.
        """
        m = self.model
        v0, v1 = view_range
        args = m._view_batch_args()
        fwd_body = self._fwd_body_per_dev[dev_index]
        vb_size = self._effective_view_batch(fwd_body, pixel_indices.shape[0],
                                             band_values.shape[-1], args)
        view_params = self._view_params_per_dev[dev_index]
        out = accumulate_into
        # With accumulate_into, every batch is added, including the first, because the block
        # already holds the results of earlier calls.
        adding = out is not None
        for v in range(v0, v1, vb_size):
            view_params_batch = view_params[v:min(v + vb_size, v1)]
            block = fwd_body(
                band_values, pixel_indices, view_params_batch,
                slice_start=slice_start, plan=plan, **args)
            if out is None:
                out = torch.empty((v1 - v0,) + tuple(block.shape[1:]),
                                  dtype=block.dtype, device=block.device)
            rows = slice(v - v0, v - v0 + block.shape[0])
            # The view batches cover disjoint rows of the block, so each row is written
            # exactly once.
            if adding:
                out[rows].add_(block)
            else:
                out[rows] = block
        return out

    def sparse_back_project_view_range(self, local_sino, pixel_indices,
                                       view_range, coeff_power=1,
                                       slice_start=0, band_slices=None,
                                       dev_index=0, plan=None):
        """This method back projects the sinogram of one range of views onto the voxels at the
        given pixels.  It is the adjoint of :meth:`sparse_forward_project_view_range`.

        Args:
            local_sino: A (v1 - v0, rows, channels) tensor that holds the views of this range.
                For geometries in which each detector row corresponds to one slice, it can hold
                a range of rows.  For geometries such as cone beam, it holds all rows.
            pixel_indices: A (P,) int64 tensor on the same device.
            view_range: The half-open range (v0, v1) of global view indices.
            coeff_power (int): The value 1, or the value 2 for the Hessian diagonal.
            slice_start (int) / band_slices (int or None): The first slice and the number of
                slices to compute.  Geometries such as cone beam use these arguments.  For
                other geometries they must keep their default values.
            dev_index (int): The index of the compiled copy of the projection function to use.
            plan: This argument is not used.

        Returns:
            A tensor of shape (P, slices) on the device of the input.  The number of slices is
            ``band_slices`` when it is given.
        """
        m = self.model
        v0, v1 = view_range
        args = m._view_batch_args()
        back_body = self._back_body_per_dev[dev_index]
        vb_size = self._effective_view_batch(back_body, pixel_indices.shape[0],
                                             local_sino.shape[1], args)
        view_params = self._view_params_per_dev[dev_index]
        out = None
        for v in range(v0, v1, vb_size):
            view_params_batch = view_params[v:min(v + vb_size, v1)]
            block = back_body(
                local_sino[v - v0:v - v0 + view_params_batch.shape[0]],
                pixel_indices, view_params_batch, coeff_power=coeff_power,
                slice_start=slice_start, band_slices=band_slices, plan=plan,
                **args)
            if out is None:
                out = block
            else:
                out.add_(block)
            # Releasing the block here means the loop holds two output-sized arrays instead of
            # three.
            block = None
        return out

    def _sparse_forward_project_single_device(self, voxel_values, pixel_indices):
        """This method forward projects the voxels at the given pixels into a full sinogram on
        device 0.  ``TomographyModel.sparse_forward_project`` calls this method."""
        m = self.model
        num_views = int(m.get_params('sinogram_shape')[0])
        voxel_values = torch.as_tensor(voxel_values, dtype=_F32,
                                       device=m.torch_device)
        pixel_indices = torch.as_tensor(pixel_indices, dtype=torch.int64,
                                        device=m.torch_device)
        return self.sparse_forward_project_view_range(
            voxel_values, pixel_indices, (0, num_views), dev_index=0)

    def _sparse_back_project_single_device(self, sinogram, pixel_indices,
                                           coeff_power=1):
        """This method back projects a full sinogram onto the voxels at ``pixel_indices`` on
        device 0.  ``TomographyModel.sparse_back_project`` calls this method.

        Args:
            sinogram: A tensor of shape (num_views, num_det_rows, num_det_channels).
            pixel_indices: A (P,) tensor of indices into the flattened (rows, cols) grid.
            coeff_power (int): The method back projects with the entries A_ij ** coeff_power.
                The value 2 gives the Hessian diagonal.

        Returns:
            A (P, num_slices) tensor.  Each row holds the voxels of one pixel across all slices.
        """
        m = self.model
        num_views = int(m.get_params('sinogram_shape')[0])
        sinogram = torch.as_tensor(sinogram, dtype=_F32, device=m.torch_device)
        pixel_indices = torch.as_tensor(pixel_indices, dtype=torch.int64,
                                        device=m.torch_device)
        return self.sparse_back_project_view_range(
            sinogram, pixel_indices, (0, num_views), coeff_power=coeff_power,
            dev_index=0)

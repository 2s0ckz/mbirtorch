"""This module checks whether the hand-written Triton kernels can run.  It also checks
whether each kernel gives the same values as the torch code that it replaces.

Every check returns (usable, reason) and never raises an error.  The first check compiles
and runs a trivial kernel once per process.  Each kernel check, such as
:func:`cone_back_kernel_usable`, compares one kernel with its torch code on a tiny problem.
The kernel check runs on the device that will run the kernel.  When a check fails, the
caller uses the torch code.  The ``_view_batch_bodies`` method of each geometry chooses
which code runs.
"""

import os
import threading

import torch

# Setting this environment variable to 1 makes mbirtorch use the torch code for every kernel.
DISABLE_ENV_VAR = 'MBIRTORCH_DISABLE_TRITON'

# mbirtorch does not read the three names below.  They are defined so that an old script that
# imports them does not fail.
ENABLE_FWD_ENV_VAR = 'MBIRTORCH_ENABLE_TRITON_FWD'
ENABLE_PBACK_ENV_VAR = 'MBIRTORCH_ENABLE_TRITON_PBACK'
ENABLE_PFWD_ENV_VAR = 'MBIRTORCH_ENABLE_TRITON_PFWD'

# The relative tolerance of the kernel checks.  It detects a kernel that was compiled
# incorrectly.  It allows a difference of a few units in the last place.
SELF_CHECK_REL_TOL = 1e-4

# The result of the first check is cached, because it cannot change within a process.
_PROBE_RESULT = None

# The results of the kernel checks are cached by device string.  Each result depends on the
# device and its compiler, not on the model that asks.
_CONE_BACK_RESULTS = {}
_CONE_FWD_RESULTS = {}
_PARALLEL_BACK_RESULTS = {}
_PARALLEL_FWD_RESULTS = {}
_MULTIAXIS_BACK_RESULTS = {}
_MULTIAXIS_FWD_RESULTS = {}

# A kernel check builds its own small model, and that model asks this module which code to
# use.  While a check runs in a thread, every call in that thread answers that the torch code
# is used.  Calls in other threads are not affected.  The lock lets only one check run at a time.
_SELF_CHECK_STATE = threading.local()
_SELF_CHECK_LOCK = threading.RLock()


def triton_available():
    """This function returns (usable, reason).  The value usable says whether mbirtorch may use
    the hand-written Triton kernels.  The function caches the result for the process.

    Returns:
        (bool, str): If usable is True, reason describes the Triton version and the GPU.  If
        usable is False, reason gives the cause.  The possible causes are these: the disable
        variable is set, CUDA is not available, triton cannot be imported, or a trivial
        kernel failed to compile, failed to run, or returned the wrong value.
    """
    global _PROBE_RESULT
    if _PROBE_RESULT is None:
        _PROBE_RESULT = _probe_triton()
    return _PROBE_RESULT


def _probe_triton():
    """This function compiles and runs a trivial Triton kernel and returns (usable, reason).

    The function never raises an error.
    """
    if os.environ.get(DISABLE_ENV_VAR, '0') == '1':
        result = (False, f'disabled by {DISABLE_ENV_VAR}=1')
    else:
        try:
            if not torch.cuda.is_available():
                result = (False, 'not a CUDA platform (torch.cuda is '
                                 'unavailable)')
            else:
                import triton
                import triton.language as tl

                @triton.jit
                def probe_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
                    offsets = tl.arange(0, BLOCK)
                    tl.store(out_ptr + offsets,
                             tl.load(x_ptr + offsets) + tl.load(y_ptr + offsets))

                block = 32
                x = torch.ones(block, dtype=torch.float32, device='cuda')
                y = torch.full((block,), 2.0, dtype=torch.float32, device='cuda')
                out = torch.empty_like(x)
                probe_kernel[(1,)](x, y, out, BLOCK=block)
                if bool(torch.all(out == 3.0)):
                    result = (True, f'available (triton {triton.__version__}, '
                                    f'{torch.cuda.get_device_name(0)})')
                else:
                    result = (False, 'probe kernel returned a wrong value')
        except Exception as e:                                    # noqa: BLE001
            result = (False, f'probe kernel failed to compile/run: {e}')
    return result


def cone_back_kernel_usable(model):
    """This function returns (usable, reason).  The value usable says whether the Triton cone
    back projection may replace the torch version on ``model.torch_device``.

    The function first requires :func:`triton_available` to pass.  It then compares the two
    versions on a tiny problem with coeff_power 1 and 2.  The largest relative difference
    must be at most :data:`SELF_CHECK_REL_TOL`.  The function caches the result for each
    device.  With several devices, the check runs on device 0.
    """
    return _kernel_usable(model, _CONE_BACK_RESULTS, _cone_back_self_check)


def cone_forward_kernel_usable(model):
    """This function returns (usable, reason).  The value usable says whether the Triton cone
    forward projection may replace the torch version on ``model.torch_device``.

    The function runs the checks of :func:`cone_back_kernel_usable` on all slices and on an
    interior range of slices.
    """
    return _kernel_usable(model, _CONE_FWD_RESULTS, _cone_forward_self_check)


def parallel_back_kernel_usable(model):
    """This function returns (usable, reason).  The value usable says whether the Triton
    parallel back projection may replace the torch version on ``model.torch_device``.

    The function runs the checks of :func:`cone_back_kernel_usable` on all detector rows and
    on an interior range of rows.
    """
    return _kernel_usable(model, _PARALLEL_BACK_RESULTS,
                          _parallel_back_self_check)


def parallel_forward_kernel_usable(model):
    """This function returns (usable, reason).  The value usable says whether the Triton
    parallel forward projection may replace the torch version on ``model.torch_device``.

    The function runs the checks of :func:`cone_back_kernel_usable` on all slices and on an
    interior range of slices.
    """
    return _kernel_usable(model, _PARALLEL_FWD_RESULTS,
                          _parallel_forward_self_check)


def multiaxis_back_kernel_usable(model):
    """This function returns (usable, reason).  The value usable says whether the Triton
    multiaxis back projection may replace the torch version on ``model.torch_device``.

    The function runs the checks of :func:`cone_back_kernel_usable` on all slices and on an
    interior range of slices.
    """
    return _kernel_usable(model, _MULTIAXIS_BACK_RESULTS,
                          _multiaxis_back_self_check)


def multiaxis_forward_kernel_usable(model):
    """This function returns (usable, reason).  The value usable says whether the Triton
    multiaxis forward projection may replace the torch version on ``model.torch_device``.

    The function runs the checks of :func:`cone_back_kernel_usable` on all slices and on an
    interior range of slices.
    """
    return _kernel_usable(model, _MULTIAXIS_FWD_RESULTS,
                          _multiaxis_forward_self_check)


def _kernel_usable(model, cache, self_check):
    """This function returns (usable, reason) from three checks in order.

    First, no kernel check may be running in this thread.  Second, :func:`triton_available` must pass.
    Third, the kernel check for this device must pass.  The function caches the third check.
    """
    if getattr(_SELF_CHECK_STATE, 'active', False):
        result = (False, 'kernel self-check in progress (its own probe model '
                         'uses the torch bodies)')
    else:
        probe_usable, probe_reason = triton_available()
        if not probe_usable:
            result = (False, probe_reason)
        else:
            device_key = str(model.torch_device)
            with _SELF_CHECK_LOCK:
                if device_key not in cache:
                    cache[device_key] = self_check(device_key)
                result = cache[device_key]
    return result


def _cone_self_check_problem(device_key):
    """This function returns (model, pixel_indices, view_params, kwargs) for a small cone
    beam test problem.

    The problem has four views, a nonzero cone angle, and an odd number of pixels.
    """
    import numpy as np

    from .cone_beam import ConeBeamModel
    from .vcd_utils import gen_full_indices

    cell = (4, 10, 10)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    model = ConeBeamModel(cell, angles, source_detector_dist=4 * cell[2],
                          source_iso_dist=2 * cell[2], 
                          compile_mode='off')
    model.configure_devices(devices=[device_key])
    device = model.torch_device
    pixel_indices = torch.as_tensor(
        gen_full_indices(model.get_params('recon_shape')),
        dtype=torch.int64, device=device)[:-1]
    view_params = torch.as_tensor(model.get_params('view_params_array'),
                                  dtype=torch.float32, device=device)
    return model, pixel_indices, view_params, model._view_batch_args()


def _parallel_self_check_problem(device_key):
    """This function returns (model, pixel_indices, view_params, kwargs) for a small parallel
    beam test problem.

    The view angles of the problem cover a half turn.
    """
    import numpy as np

    from .parallel_beam import ParallelBeamModel
    from .vcd_utils import gen_full_indices

    cell = (4, 10, 10)
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    model = ParallelBeamModel(cell, angles, 
                              compile_mode='off')
    model.configure_devices(devices=[device_key])
    device = model.torch_device
    pixel_indices = torch.as_tensor(
        gen_full_indices(model.get_params('recon_shape')),
        dtype=torch.int64, device=device)[:-1]
    view_params = torch.as_tensor(model.get_params('angles'),
                                  dtype=torch.float32, device=device)
    return model, pixel_indices, view_params, model._view_batch_args()


def _multiaxis_self_check_problem(device_key):
    """This function returns (model, pixel_indices, view_params, kwargs) for a small multiaxis
    test problem.

    The elevation angles of the problem are nonzero and less than 45 degrees.
    """
    import numpy as np

    from .multiaxis_parallel import MultiAxisParallelModel
    from .vcd_utils import gen_full_indices

    cell = (4, 10, 10)
    azimuth = np.linspace(0, np.pi, cell[0], endpoint=False)
    elevation = np.linspace(-0.4, 0.4, cell[0])
    model = MultiAxisParallelModel(cell,
                                   np.stack([azimuth, elevation], axis=1),
                                   compile_mode='off')
    model.configure_devices(devices=[device_key])
    device = model.torch_device
    pixel_indices = torch.as_tensor(
        gen_full_indices(model.get_params('recon_shape')),
        dtype=torch.int64, device=device)[:-1]
    view_params = torch.as_tensor(model.get_params('angles'),
                                  dtype=torch.float32, device=device)
    return model, pixel_indices, view_params, model._view_batch_args()


def _rel_diff(kernel_out, reference):
    """This function returns the largest relative difference between a kernel output and the
    output of its torch version."""
    return (float((kernel_out - reference).abs().max())
            / max(float(reference.abs().max()), 1e-12))


def _cone_back_self_check(device_key):
    """This function compares the cone back kernel with its torch version on one device.

    The function never raises an error.
    """
    _SELF_CHECK_STATE.active = True
    try:
        from .cone_beam import _cone_back_view_batch
        from .triton_cone import _cone_back_view_batch_triton

        model, pixel_indices, view_params, args = _cone_self_check_problem(
            device_key)
        # A private generator keeps this check from advancing the global random state,
        # which the tests with fixed seeds depend on.
        generator = torch.Generator().manual_seed(0)
        sinogram = torch.rand(tuple(model.get_params('sinogram_shape')),
                              generator=generator).to(model.torch_device)

        worst_rel = 0.0
        for coeff_power in (1, 2):
            reference = _cone_back_view_batch(sinogram, pixel_indices,
                                              view_params,
                                              coeff_power=coeff_power, **args)
            kernel_out = _cone_back_view_batch_triton(
                sinogram, pixel_indices, view_params,
                coeff_power=coeff_power, **args)
            worst_rel = max(worst_rel, _rel_diff(kernel_out, reference))
        result = _self_check_verdict('cone back', device_key, worst_rel)
    except Exception as e:                                        # noqa: BLE001
        result = (False, f'cone back self-check failed to run: '
                         f'{type(e).__name__}: {e}')
    finally:
        _SELF_CHECK_STATE.active = False
    return result


def _cone_forward_self_check(device_key):
    """This function compares the cone forward kernel with its torch version on one device.

    The function never raises an error.
    """
    _SELF_CHECK_STATE.active = True
    try:
        from .cone_beam import _cone_forward_view_batch
        from .triton_cone import _cone_forward_view_batch_triton

        model, pixel_indices, view_params, args = _cone_self_check_problem(
            device_key)
        num_slices = int(args['num_slices'])
        generator = torch.Generator().manual_seed(0)
        values = torch.rand((int(pixel_indices.shape[0]), num_slices),
                            generator=generator).to(model.torch_device)

        worst_rel = 0.0
        # The check runs on all slices and then on an interior range of slices, which
        # tests the use of slice_start.
        interior = max(1, num_slices // 3)
        bands = ((0, num_slices), (interior, num_slices - 2 * interior))
        for slice_start, band_len in bands:
            band_values = values[:, slice_start:slice_start + band_len]
            reference = _cone_forward_view_batch(band_values, pixel_indices,
                                                 view_params,
                                                 slice_start=slice_start, **args)
            kernel_out = _cone_forward_view_batch_triton(
                band_values, pixel_indices, view_params,
                slice_start=slice_start, **args)
            worst_rel = max(worst_rel, _rel_diff(kernel_out, reference))
        result = _self_check_verdict('cone forward', device_key, worst_rel)
    except Exception as e:                                        # noqa: BLE001
        result = (False, f'cone forward self-check failed to run: '
                         f'{type(e).__name__}: {e}')
    finally:
        _SELF_CHECK_STATE.active = False
    return result


def _parallel_back_self_check(device_key):
    """This function compares the parallel back kernel with its torch version on one device.

    The function never raises an error.
    """
    _SELF_CHECK_STATE.active = True
    try:
        from .parallel_beam import _parallel_back_view_batch
        from .triton_parallel import _parallel_back_view_batch_triton

        model, pixel_indices, view_params, args = _parallel_self_check_problem(
            device_key)
        # A private generator keeps this check from advancing the global random state,
        # which the tests with fixed seeds depend on.
        generator = torch.Generator().manual_seed(0)
        sinogram = torch.rand(tuple(model.get_params('sinogram_shape')),
                              generator=generator).to(model.torch_device)

        worst_rel = 0.0
        # In parallel beam, each detector row corresponds to one slice.  For this reason,
        # the test of a range of rows only needs to slice the sinogram.
        num_rows = int(sinogram.shape[1])
        interior = max(1, num_rows // 3)
        bands = ((0, num_rows), (interior, num_rows - 2 * interior))
        for coeff_power in (1, 2):
            for row_start, band_rows in bands:
                band = sinogram[:, row_start:row_start + band_rows]
                reference = _parallel_back_view_batch(
                    band, pixel_indices, view_params,
                    coeff_power=coeff_power, **args)
                kernel_out = _parallel_back_view_batch_triton(
                    band, pixel_indices, view_params,
                    coeff_power=coeff_power, **args)
                worst_rel = max(worst_rel, _rel_diff(kernel_out, reference))
        result = _self_check_verdict('parallel back', device_key, worst_rel)
    except Exception as e:                                        # noqa: BLE001
        result = (False, f'parallel back self-check failed to run: '
                         f'{type(e).__name__}: {e}')
    finally:
        _SELF_CHECK_STATE.active = False
    return result


def _parallel_forward_self_check(device_key):
    """This function compares the parallel forward kernel with its torch version on one device.

    The function never raises an error.
    """
    _SELF_CHECK_STATE.active = True
    try:
        from .parallel_beam import _parallel_forward_view_batch
        from .triton_parallel import _parallel_forward_view_batch_triton

        model, pixel_indices, view_params, args = _parallel_self_check_problem(
            device_key)
        num_slices = int(model.get_params('recon_shape')[2])
        generator = torch.Generator().manual_seed(0)
        values = torch.rand((int(pixel_indices.shape[0]), num_slices),
                            generator=generator).to(model.torch_device)

        worst_rel = 0.0
        # The forward projection receives a range of slices as the columns of the values.
        # Each range of slices produces the matching detector rows.
        interior = max(1, num_slices // 3)
        bands = ((0, num_slices), (interior, num_slices - 2 * interior))
        for slice_start, band_len in bands:
            band_values = values[:, slice_start:slice_start + band_len]
            reference = _parallel_forward_view_batch(band_values,
                                                     pixel_indices,
                                                     view_params, **args)
            kernel_out = _parallel_forward_view_batch_triton(
                band_values, pixel_indices, view_params, **args)
            worst_rel = max(worst_rel, _rel_diff(kernel_out, reference))
        result = _self_check_verdict('parallel forward', device_key, worst_rel)
    except Exception as e:                                        # noqa: BLE001
        result = (False, f'parallel forward self-check failed to run: '
                         f'{type(e).__name__}: {e}')
    finally:
        _SELF_CHECK_STATE.active = False
    return result


def _multiaxis_back_self_check(device_key):
    """This function compares the multiaxis back kernel with its torch version on one device.

    The function never raises an error.
    """
    _SELF_CHECK_STATE.active = True
    try:
        from .multiaxis_parallel import _multiaxis_back_view_batch
        from .triton_multiaxis import _multiaxis_back_view_batch_triton

        model, pixel_indices, view_params, args = _multiaxis_self_check_problem(
            device_key)
        # A private generator keeps this check from advancing the global random state,
        # which the tests with fixed seeds depend on.
        generator = torch.Generator().manual_seed(0)
        sinogram = torch.rand(tuple(model.get_params('sinogram_shape')),
                              generator=generator).to(model.torch_device)

        worst_rel = 0.0
        # The interior range of slices tests the use of band_slices.  Its length is not a
        # multiple of the padded launch width of the kernel.
        num_slices = int(args['num_slices'])
        interior = max(1, num_slices // 3)
        bands = ((0, num_slices), (interior, num_slices - 2 * interior))
        for coeff_power in (1, 2):
            for slice_start, band_slices in bands:
                reference = _multiaxis_back_view_batch(
                    sinogram, pixel_indices, view_params,
                    coeff_power=coeff_power, slice_start=slice_start,
                    band_slices=band_slices, **args)
                kernel_out = _multiaxis_back_view_batch_triton(
                    sinogram, pixel_indices, view_params,
                    coeff_power=coeff_power, slice_start=slice_start,
                    band_slices=band_slices, **args)
                worst_rel = max(worst_rel, _rel_diff(kernel_out, reference))
        result = _self_check_verdict('multiaxis back', device_key, worst_rel)
    except Exception as e:                                        # noqa: BLE001
        result = (False, f'multiaxis back self-check failed to run: '
                         f'{type(e).__name__}: {e}')
    finally:
        _SELF_CHECK_STATE.active = False
    return result


def _multiaxis_forward_self_check(device_key):
    """This function compares the multiaxis forward kernel with its torch version on one device.

    The function never raises an error.
    """
    _SELF_CHECK_STATE.active = True
    try:
        from .multiaxis_parallel import _multiaxis_forward_view_batch
        from .triton_multiaxis import _multiaxis_forward_view_batch_triton

        model, pixel_indices, view_params, args = _multiaxis_self_check_problem(
            device_key)
        num_slices = int(args['num_slices'])
        generator = torch.Generator().manual_seed(0)
        values = torch.rand((int(pixel_indices.shape[0]), num_slices),
                            generator=generator).to(model.torch_device)

        worst_rel = 0.0
        # The check runs on all slices and then on an interior range of slices, which
        # tests the use of slice_start.
        interior = max(1, num_slices // 3)
        bands = ((0, num_slices), (interior, num_slices - 2 * interior))
        for slice_start, band_len in bands:
            band_values = values[:, slice_start:slice_start + band_len]
            reference = _multiaxis_forward_view_batch(
                band_values, pixel_indices, view_params,
                slice_start=slice_start, **args)
            kernel_out = _multiaxis_forward_view_batch_triton(
                band_values, pixel_indices, view_params,
                slice_start=slice_start, **args)
            worst_rel = max(worst_rel, _rel_diff(kernel_out, reference))
        result = _self_check_verdict('multiaxis forward', device_key,
                                     worst_rel)
    except Exception as e:                                        # noqa: BLE001
        result = (False, f'multiaxis forward self-check failed to run: '
                         f'{type(e).__name__}: {e}')
    finally:
        _SELF_CHECK_STATE.active = False
    return result


def _self_check_verdict(name, device_key, worst_rel):
    """This function converts the largest relative difference of a kernel check to
    (usable, reason)."""
    if worst_rel <= SELF_CHECK_REL_TOL:
        result = (True, f'{name} kernel matches the torch body on '
                        f'{device_key} (rel {worst_rel:.1e} <= '
                        f'{SELF_CHECK_REL_TOL:.0e})')
    else:
        result = (False, f'{name} kernel differs from the torch body on '
                         f'{device_key} (rel {worst_rel:.1e} > '
                         f'{SELF_CHECK_REL_TOL:.0e})')
    return result


def _reset_probe_cache():
    """This function clears the cached result of :func:`triton_available`.

    A change to the disable variable takes effect only after this call.
    """
    global _PROBE_RESULT
    _PROBE_RESULT = None


def _reset_self_check_cache():
    """This function clears the cached results of the kernel checks for all devices."""
    _CONE_BACK_RESULTS.clear()
    _CONE_FWD_RESULTS.clear()
    _PARALLEL_BACK_RESULTS.clear()
    _PARALLEL_FWD_RESULTS.clear()
    _MULTIAXIS_BACK_RESULTS.clear()
    _MULTIAXIS_FWD_RESULTS.clear()

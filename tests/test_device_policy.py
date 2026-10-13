"""The automatic device count and the memory check that limits it.

A CUDA model spreads a reconstruction across the devices that can hold their
share.  These tests check the RULE rather than the hardware: the selection, the
validation, the count set by MBIRTORCH_NUM_DEVICES, and the readable failure
all run on CPU, by driving the policy with fabricated device lists and budgets.  The
multi-device VALUE gates live in test_sharding.py, which is where the seeded
n>1 parity patterns already are.
"""


import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _memory_estimate, _sharding
from mbirtorch._memory_estimate import MemoryCheckError

GB = 2 ** 30

# Sinogram shapes at which the minimum problem sizes are measured, each named
# for its view count.  The comment beside each is its sinogram element count.


def make_model(shape=(8, 6, 8), device='cpu', **kwargs):
    angles = np.linspace(0, np.pi, shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(shape, angles, **kwargs)
    model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    return model


@pytest.fixture(autouse=True)
def kernel_declared_projection(monkeypatch):
    """Price the projection the way the CUDA model these tests stand in for
    would price it.

    A CUDA parallel or cone model binds the hand-written kernel bodies, and
    each of those declares what one of its views holds.  On CPU no kernel is
    available, so the same model binds the general torch bodies instead, and
    the memory estimate counts many more bytes for each view of a torch body
    (``_memory_estimate.TORCH_PROJECTOR_VIEW_SLABS``).  These tests are about
    the device-count RULE, not about either byte count, so they hold the
    projection bytes at the kernel-declared value.  Otherwise the capacity
    arithmetic they drive would be a different model's.
    """
    monkeypatch.setattr(_memory_estimate, 'torch_projector_directions',
                        lambda model: ())


@pytest.fixture
def no_size_limit(monkeypatch):
    """Turn off the minimum problem sizes.

    The minimum sizes keep small problems on one device, and the toy shapes
    these tests use are far below every one of them.  A test whose subject is
    the CAPACITY rule therefore has to opt out, exactly as a user would, or it
    would be measuring two rules at once.  The effect of the minimum sizes on
    the chosen count is tested below, at the sizes where they were measured.
    """
    monkeypatch.setenv('MBIRTORCH_LIMIT_GPUS_BY_SIZE', '0')


@pytest.fixture
def no_device_count_env(monkeypatch):
    """Clear the device count that the suite sets with MBIRTORCH_NUM_DEVICES.

    The conftest fixture fixes every test to one device, which is exactly what
    keeps the suite deterministic on a multi-GPU host.  The tests that
    exercise the SEARCH have to opt out of it, and doing so explicitly keeps
    the reach of that setting visible.
    """
    monkeypatch.delenv('MBIRTORCH_NUM_DEVICES', raising=False)


def as_automatic(model, num_devices):
    """Make a CPU model behave like an eligible CUDA model with
    ``num_devices`` visible, so the rule can be exercised without a GPU.

    The fake devices carry INDICES, so a test can give different devices
    different budgets and exercise the heterogeneous case.
    """
    model.device_layout_is_automatic = True
    model._candidate_devices = lambda n: [torch.device('cpu', i)
                                          for i in range(n)]
    return num_devices


# ── the selection rule ───────────────────────────────────────────────────────
def test_the_largest_count_that_fits_is_chosen(monkeypatch, no_device_count_env,
                                                no_size_limit):
    """Two budgets through the same selection call.

    With room on every device the largest valid count wins.  Per-device peaks
    SHRINK as the count grows, so no uniform budget can allow two devices and
    refuse four; what refuses four is one device without room.  Every device
    in a candidate set must pass its own budget, so a busy `cuda:2` sends the
    rule back to two devices rather than letting it start a run only three
    devices could hold.
    """
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)

    roomy = make_model((16, 8, 16))
    as_automatic(roomy, 4)
    monkeypatch.setattr(_memory_estimate, 'device_budget_bytes',
                        lambda d: 64 * GB)
    roomy._apply_device_policy()
    assert roomy.sino_placement.n_devices == 4

    # Devices 2 and 3 are nearly full; 0 and 1 are free.
    crowded = make_model((16, 8, 16))
    as_automatic(crowded, 4)
    monkeypatch.setattr(
        _memory_estimate, 'device_budget_bytes',
        lambda d: 1024 if (d.index or 0) >= 2 else 64 * GB)
    crowded._apply_device_policy()
    assert crowded.sino_placement.n_devices == 2

    # The empty-shard rules gate the count before the estimate does: 3 views
    # and 3 slices over 4 devices leaves device 3 with neither.
    idle = make_model((3, 6, 8))
    idle.set_params(no_warning=True, recon_shape=(6, 6, 3))
    as_automatic(idle, 4)
    monkeypatch.setattr(_memory_estimate, 'device_budget_bytes',
                        lambda d: 64 * GB)
    assert not idle._layout_is_valid(['cpu'] * 4)
    idle._apply_device_policy()
    assert idle.sino_placement.n_devices <= 3


# ── what turns the rule off ──────────────────────────────────────────────────
def test_a_count_the_user_sets_is_used(monkeypatch, no_device_count_env):
    """A device list given to configure_devices is used as given.
    MBIRTORCH_NUM_DEVICES decides the count, the count is capped by what is
    visible, and it holds through a whole reconstruction.  This is the suite's
    own determinism guarantee on a multi-GPU host."""
    model = make_model()
    model.configure_devices(devices=['cpu', 'cpu'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    # A budget of zero would fail any memory check; an explicit layout does not
    # consult one at all.
    monkeypatch.setattr(_memory_estimate, 'device_budget_bytes', lambda d: 0)
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 2

    monkeypatch.setattr(_memory_estimate, 'device_budget_bytes',
                        lambda d: 64 * GB)
    model = make_model((16, 8, 16))
    as_automatic(model, 4)
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '2')
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 2      # not the 4 that would fit

    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    capped = make_model((16, 8, 16))
    as_automatic(capped, 2)
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '8')
    capped._apply_device_policy()
    assert capped.sino_placement.n_devices == 2

    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '1')
    fixed = make_model()
    as_automatic(fixed, 4)
    sinogram = np.zeros((8, 6, 8), dtype=np.float32)
    sinogram[:, 3, 4] = 1.0
    np.random.seed(0)
    fixed.recon(sinogram, max_iterations=2)
    assert fixed.sino_placement.n_devices == 1


# ── the doomed run ───────────────────────────────────────────────────────────
def test_a_doomed_run_fails_before_allocating_anything(monkeypatch, no_device_count_env):
    """No count fits, including one, so the answer to 'which count' is
    'none'.  That must arrive as a readable error rather than as a
    reconstruction that dies later inside the allocator, and it must arrive
    through the public entry as well as the internal one."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setattr(_memory_estimate, 'device_budget_bytes',
                        lambda d: 1024)          # a kilobyte: nothing fits

    model = make_model((16, 8, 16))
    as_automatic(model, 4)
    with pytest.raises(MemoryCheckError):
        model._apply_device_policy()

    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    through_recon = make_model((16, 8, 16))
    as_automatic(through_recon, 2)
    sinogram = np.zeros((16, 8, 16), dtype=np.float32)
    sinogram[:, 4, 8] = 1.0
    np.random.seed(0)
    with pytest.raises(MemoryCheckError):
        through_recon.recon(sinogram, max_iterations=1)


# ── helpers for the tests below ─────────────────────────────────────────────
def with_four_visible(monkeypatch, model, budget=64 * GB):
    """Four visible devices, every one of them with ample room, so the ONLY
    thing that can hold the count down is the minimum problem size."""
    as_automatic(model, 4)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setattr(_memory_estimate, 'device_budget_bytes',
                        lambda d: budget)
    return model


# ── the device choice is kept ────────────────────────────────────────────────
# The automatic choice is made once per model and kept.  The model records the
# (sinogram_shape, recon_shape) pair it decided from, later calls reuse the
# layout while those shapes hold, and only a shape change re-decides.  The
# budget function below raises when it is called, so a passing test proves
# that a call after the choice ran no search.
def poison_budgets(monkeypatch):
    def no_budget(_device):
        raise AssertionError('a call after the device choice consulted a device budget')
    monkeypatch.setattr(_memory_estimate, 'device_budget_bytes', no_budget)


def test_a_model_redecides_its_devices_only_when_its_shapes_change(
        monkeypatch, no_device_count_env, no_size_limit):
    """The automatic choice is made once and kept.  The budget that raises
    is the proof that no second search ran, either on a later policy call or
    after a recompile-flagged parameter that leaves the shapes alone.  A
    shape change does make a new choice."""
    model = make_model((16, 8, 16))
    with_four_visible(monkeypatch, model)
    model._apply_device_policy()
    placement = model.sino_placement
    assert placement.n_devices == 4
    poison_budgets(monkeypatch)
    model._apply_device_policy()
    # The same layout, and the SAME placement object: a re-decision would
    # reinstall the placements and invalidate every Shards a caller holds.
    assert model.sino_placement is placement
    # Re-deciding on every recompile would clear the device choice on a
    # detector-offset edit, and on the sigma_noise the denoiser sets at every
    # call.
    model.set_params(no_warning=True, det_channel_offset=0.5)
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 4

    # Two of the four devices fill up.  A model whose device choice is made
    # ignores that by design; a shape change must not, because the decision's
    # inputs are new.
    monkeypatch.setattr(
        _memory_estimate, 'device_budget_bytes',
        lambda d: 1024 if (d.index or 0) >= 2 else 64 * GB)
    model.set_params(no_warning=True, sinogram_shape=(16, 12, 16))
    model._apply_device_policy()
    assert model.sino_placement.n_devices == 2


# ── the direct reconstructions ───────────────────────────────────────────────
# recon_fbp and recon_fdk choose the devices themselves, so a direct
# reconstruction spreads across the devices instead of landing whole on the
# first one.


def _automatic_cone(shape=(8, 6, 8)):
    angles = np.linspace(0, 2 * np.pi, shape[0], endpoint=False)
    return mbirtorch.ConeBeamModel(shape, angles,
                                   source_detector_dist=4.0 * shape[2],
                                   source_iso_dist=2.0 * shape[2])




def _automatic_on_cpu(make, verbose=0):
    """A model the caller has never placed, put on the CPU the way the
    automatic path itself does.

    ``_install_device_layout`` carries no policy, so the layout stays the
    library's to choose, while the fabricated CUDA visibility below cannot
    pull real allocations onto a device this host lacks.
    """
    model = make()
    model.set_params(no_warning=True, verbose=verbose)
    model._install_device_layout(['cpu'])
    assert model.device_layout_is_automatic is True
    return model


def _impulse_sinogram(model):
    shape = tuple(int(s) for s in model.get_params('sinogram_shape'))
    sinogram = np.zeros(shape, dtype=np.float32)
    sinogram[:, shape[1] // 2, shape[2] // 2] = 1.0
    return sinogram


def test_a_bare_recon_direct_chooses_the_devices_and_spreads(
        monkeypatch, no_device_count_env, no_size_limit):
    """A direct reconstruction (FDK here) on a model with no explicit layout
    uses the devices that fit, exactly as recon does, rather than running
    whole on the first device."""
    model = _automatic_on_cpu(_automatic_cone)
    with_four_visible(monkeypatch, model)
    recon = model.recon_fdk(_impulse_sinogram(model))
    assert model.sino_placement.n_devices == 4
    assert recon.shape == tuple(model.get_params('recon_shape'))
    assert np.all(np.isfinite(recon))


# ── two models sharing one device layout ─────────────────────────────────────
# A Plug-and-Play or ADMM loop alternates prox_map on a reconstruction model
# with denoise on a QGGMRFDenoiser over the same volume.  Both accept and
# return the device form, so the volume can stay on the devices for the whole
# loop -- but only if the two models place recon-like arrays the same way.
# configure_devices(like=...) is how they are made to agree, and the tests
# below drive the handoff on two virtual CPU devices, in both directions.
PAIR_SHAPE = (12, 8, 20)      # recon shape (20, 20, 8): 8 slices, split 4 + 4


def paired_ct_and_denoiser(sino_shape=PAIR_SHAPE, devices=('cpu', 'cpu')):
    """A parallel-beam model and a denoiser built at its recon shape, placed
    on the same (virtual) devices -- the two-line idiom the configure_devices
    docstring gives."""
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    ct_model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    ct_model.configure_devices(devices=list(devices))
    ct_model.set_params(no_warning=True, verbose=0)

    denoiser = mbirtorch.QGGMRFDenoiser(ct_model.get_params('recon_shape'))
    denoiser.configure_devices(like=ct_model)
    denoiser.set_params(no_warning=True, verbose=0)
    return ct_model, denoiser


def pair_phantom(ct_model):
    """A small block phantom in the model's recon shape."""
    recon_shape = tuple(ct_model.get_params('recon_shape'))
    phantom = np.zeros(recon_shape, dtype=np.float32)
    r0, c0, s0 = [max(1, n // 4) for n in recon_shape]
    phantom[r0:-r0, c0:-c0, s0:-s0] = 1.0
    return phantom


def test_a_denoiser_placed_like_a_ct_model_takes_its_sharded_volume():
    """The handoff the whole seam exists for, on two virtual CPU devices.

    The two models are configured separately, so their recon placements are
    distinct OBJECTS naming the same devices, axis and slice count.  Equality
    rather than identity is what lets the denoiser accept a volume the
    reconstruction model placed, with no gather to the host in between.  The
    return leg closes the loop: what denoise hands back in the device form is
    a valid prox_input for the reconstruction model.
    """
    ct_model, denoiser = paired_ct_and_denoiser()
    assert denoiser.recon_placement == ct_model.recon_placement
    assert denoiser.recon_placement is not ct_model.recon_placement

    phantom = pair_phantom(ct_model)
    sinogram = ct_model.forward_project(phantom)
    placed = ct_model._shard_recon(phantom)
    assert isinstance(placed, _sharding.Shards)

    denoised, _ = denoiser.denoise(placed, sigma_noise=0.1, max_iterations=1,
                                   stop_threshold_change_pct=0.0,
                                   logfile_path=None, print_logs=False,
                                   output_sharded=True)
    assert isinstance(denoised, _sharding.Shards)
    assert denoised.placement == ct_model.recon_placement
    assert ([tuple(t.shape) for t in denoised.tensors]
            == [tuple(t.shape) for t in placed.tensors])
    assert np.isfinite(denoised.gather()).all()

    np.random.seed(0)
    recon, _ = ct_model.prox_map(denoised, sinogram, sigma_prox=0.5,
                                 init_recon=phantom, max_iterations=1,
                                 stop_threshold_change_pct=0.0,
                                 logfile_path=None, print_logs=False,
                                 output_sharded=True)
    assert isinstance(recon, _sharding.Shards)
    assert recon.placement == ct_model.recon_placement
    # The reconstruction comes back as a volume, one 3-D block per device,
    # split on the slice axis exactly as the input was.
    assert ([tuple(t.shape) for t in recon.tensors]
            == [tuple(t.shape) for t in denoised.tensors])
    assert np.isfinite(recon.gather()).all()


# ── the prepared sinogram, straight into a reconstruction ────────────────────
# prepare_sino_for_devices divides a sinogram across the devices once, and the
# reconstruction entries take that divided form as it stands.  A Plug-and-Play
# loop can therefore prepare its sinogram once and pay the host-to-device
# transfer once, rather than on every prox_map call.  The tests below drive
# that on two virtual CPU devices, against the same runs on host arrays, at the
# small shape and phantom the section above already builds.


def model_on_devices(sino_shape, num_devices):
    """A parallel-beam model whose sinogram is divided across ``num_devices``
    virtual CPU devices."""
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(devices=['cpu'] * num_devices)
    model.set_params(no_warning=True, verbose=0)
    return model


def view_shards(model, sinogram):
    """A sinogram in the divided device form, one tensor per device.

    This is what prepare_sino_for_devices returns, built here directly so that
    a single-device placement gives shards too: that call hands back a plain
    tensor when there is only one device, and the subsample arithmetic below
    is about the divided form at every device count.
    """
    return _sharding.Shards(
        [torch.as_tensor(sinogram[start:end]) for _d, (start, end)
         in model.sino_placement.shard_ranges()], model.sino_placement)


def max_relative_difference(out, ref):
    """The comparison the value tests in this file use: the largest pointwise
    difference as a fraction of the reference's largest magnitude."""
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


def run_recon(model, sinogram, weights=None, max_iterations=2):
    """One short reconstruction, seeded so that two runs draw the same pixel
    partitions and differ only in the form their sinogram arrived in."""
    np.random.seed(0)
    recon, _ = model.recon(sinogram, weights=weights,
                           max_iterations=max_iterations,
                           stop_threshold_change_pct=0.0,
                           logfile_path=None, print_logs=False)
    return recon


def test_a_divided_sinogram_subsamples_exactly_like_a_whole_one():
    """The statistics that set the regularization parameters run on an
    evenly-spaced subsample of the views, and taking that subsample from the
    divided form is data movement rather than an approximation.  It therefore
    has to give the very same views in the very same order as subsampling the
    assembled sinogram.

    41 views over 2 devices are subsampled with stride 2, and the first
    device holds one view more, so the second shard starts on an odd view.
    """
    num_views, num_devices = 41, 2
    model = model_on_devices((num_views, 4, 6), num_devices)
    whole = np.random.RandomState(num_views).rand(
        num_views, 4, 6).astype(np.float32)
    divided = model.subsample_views(view_shards(model, whole))
    assert np.array_equal(divided, model.subsample_views(whole))


def test_a_prepared_sinogram_reconstructs_like_a_host_one():
    """The point of the whole seam: recon takes the divided form directly and
    returns what the same call on the host array returns, weights included --
    those are placed by the same call and travel the same path.

    Not bit-for-bit -- the per-device sums this model runs on CPU vary a
    little from run to run whatever their input was -- so the comparison is
    the relative one the value tests in this file use.
    """
    model = model_on_devices(PAIR_SHAPE, 2)
    phantom = pair_phantom(model)
    sinogram = model.forward_project(phantom)
    prepared = model.prepare_sino_for_devices(sinogram)
    assert isinstance(prepared, _sharding.Shards)

    from_host = run_recon(model, sinogram)
    from_prepared = run_recon(model, prepared)
    assert max_relative_difference(from_prepared, from_host) <= 1e-4

    weights = 0.5 + np.random.RandomState(5).rand(
        *PAIR_SHAPE).astype(np.float32)
    prepared, prepared_weights = model.prepare_sino_for_devices(
        sinogram, weights=weights)
    assert isinstance(prepared_weights, _sharding.Shards)

    from_host = run_recon(model, sinogram, weights=weights)
    from_prepared = run_recon(model, prepared, weights=prepared_weights)
    assert max_relative_difference(from_prepared, from_host) <= 1e-4

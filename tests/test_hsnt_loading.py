"""Tests for the hsnt loader: a dead open-beam pixel, bad values, smoothing and dose in any blocks of bins, and the
working set of load_dataset."""
import tracemalloc
import warnings

import h5py
import numpy as np
import pytest
import tifffile

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt.loading import (_dead_mask, _smooth_open_beam, _stack_to_transmission, convert_to_hdf5,
                                    load_dataset)


def test_an_open_beam_of_floating_point_residue_holds_no_count():
    """A dead pixel whose processing left 1e-12 in its open beam: a count there is divided by the bin's median open
    beam, as for an open beam of zero, not by the residue (a transmission of 2e12), and the smoothing's dead mask
    counts the pixel as dead."""
    rng = np.random.default_rng(4)
    ob = rng.poisson(30.0, (64, 8)).astype(np.float32)
    counts = rng.poisson(15.0, (64, 8)).astype(np.float32)
    ob[5], counts[5] = 1e-12, 2.0
    T, _, info = _stack_to_transmission(counts, "counts", open_beam=ob)
    assert T[5].max() < 0.1 and T.max() < 2 and info["open_beam_zero_frac"] == pytest.approx(1 / 64)
    total = rng.poisson(1e4, (1, 8, 8)).astype(np.float64)
    total[0, 2, 3] = 8 * 13 * 1e-12
    dead = _dead_mask(total)
    assert dead is not None and dead.sum() == 1 and dead[0, 2, 3, 0]


def _cleaning_warnings(run):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = run()
    return out, sorted(str(w.message) for w in caught if "% of the entries" in str(w.message))


def test_the_loader_treats_bad_values_as_dehydrate_does(tmp_path):
    """Negative, NaN and infinite transmissions, and NaN, -inf and overflowing attenuations, load and convert (one bin
    per block) to the transmission dehydrate makes of the same array, with the same warnings; +inf attenuation is
    zero transmission, with no warning. Negative counts divided by the open beam are clipped as well."""
    rng = np.random.default_rng(3)
    T = rng.uniform(0.05, 1.0, (1, 8, 8, 12)).astype(np.float32)
    T[0, 0, :3, 1], T[0, 1, 0, 5], T[0, 2, 1, 7], T[0, 3, 2, 9] = -0.2, np.nan, np.inf, -np.inf
    A = -np.log(np.abs(T))
    A[0, 4, 4, 2], A[0, 5, 5, 3], A[0, 6, 6, 4], A[0, 7, 7, 6] = np.nan, -np.inf, np.inf, -100.0
    for itype, data in (("transmission", T), ("attenuation", A)):
        path, conv = str(tmp_path / f"{itype}.h5"), str(tmp_path / f"{itype}_conv.h5")
        hsnt.export_hsnt_data_hdf5(path, data, hsnt.create_hsnt_metadata(dataset_type=itype))
        (ref, _), expected = _cleaning_warnings(lambda: hsnt.denoise._to_transmission(data, itype))
        assert len(expected) == (2 if itype == "transmission" else 1) and ref.min() == 0.0
        loaded, said = _cleaning_warnings(lambda: load_dataset(path, memory_budget_mib=1e-3).T)
        assert np.array_equal(loaded, ref) and said == expected
        _, said = _cleaning_warnings(lambda: convert_to_hdf5(path, output=conv, as_type="transmission", block_bins=1))
        assert np.array_equal(load_dataset(conv).T, ref) and said == expected
    counts, ob = rng.poisson(1.0, (64, 12)).astype(np.float32) - 1.0, np.full((64, 12), 2.0, dtype=np.float32)
    (Tc, _, info), said = _cleaning_warnings(lambda: _stack_to_transmission(counts, "counts", open_beam=ob))
    assert np.mean(counts < 0) > 0.2 and info["negative_frac"] == pytest.approx(np.mean(counts < 0))
    assert Tc.min() == 0.0 and not said


def _write_stack(directory, counts):
    """A TIFF stack of (rows, cols, bins) counts, one image per bin."""
    directory.mkdir(parents=True)
    for k in range(counts.shape[-1]):
        tifffile.imwrite(directory / f"wave_idx_{k:05d}.tif", counts[:, :, k].astype(np.float32))
    return str(directory)


def _converted(tmp_path, name, sample, ob, **kw):
    out, _, info = convert_to_hdf5(sample, output=str(tmp_path / f"{name}.h5"), open_beam=[ob], **kw)
    with h5py.File(out) as f:
        return f["data"][()], f.attrs["dose"], f["open_beam_dose"][()], info


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_smoothing_and_the_dose_are_the_same_in_any_blocks_of_bins(tmp_path):
    """At one count per pixel and bin over 32 bins, converted in blocks of 1 bin and of all 32: the same smoothed open
    beam, so the same data and dose (the median open beam about 1). A pixel dead in a block of one bin but not in the
    others keeps its zero count; one dead in every bin is left out, and recorded.

    At one count per pixel and bin the smoothed open beam keeps the raw mean, and 8 counts per pixel are too few to
    call a pixel without one dead; a pixel zero in every bin of a bright open beam is left out and filled from its
    neighbors, which it does not lower.

    An open beam of 40 counts in 31 bins and 5 in 29: the dose is the median of the per-bin doses, about 40, in
    blocks of 7 bins and of all 60, as a range of the converted file takes it and as load_dataset finds it."""
    rng = np.random.default_rng(11)
    ob_counts = rng.poisson(1.0, (48, 48, 32))
    ob_counts[10, 10] = 0
    sample = _write_stack(tmp_path / "sample", rng.poisson(0.6, (48, 48, 32)))
    ob = _write_stack(tmp_path / "ob", ob_counts)
    T1, dose1, per_bin1, info = _converted(tmp_path, "one", sample, ob, open_beam_smoothing=3, block_bins=1)
    T32, dose32, per_bin32, _ = _converted(tmp_path, "all", sample, ob, open_beam_smoothing=3, block_bins=32)
    assert np.array_equal(T1, T32) and dose1 == dose32 and np.array_equal(per_bin1, per_bin32)
    assert 0.85 < dose1 < 1.1 and info["open_beam_smoothing"]["dead_pixels"] == 1

    # Smoothing at one count per pixel keeps the raw mean, and a pixel zero in
    # every bin of a bright open beam is left out and filled from its neighbors.
    rng = np.random.default_rng(0)
    ob = rng.poisson(1.0, size=(1, 128, 128, 8)).astype(np.float32)
    dead = _dead_mask(ob.sum(-1, dtype=np.float64))
    assert dead is None and _smooth_open_beam(ob.copy(), 3, dead).mean() / ob.mean() == pytest.approx(1, abs=0.01)
    ob = rng.poisson(100.0, size=(1, 32, 32, 8)).astype(np.float32)
    ob[0, 10, 10] = 0
    dead = _dead_mask(ob.sum(-1, dtype=np.float64))
    smoothed = _smooth_open_beam(ob, 3, dead)
    assert int(dead.sum()) == 1 and abs(smoothed[0, 10, 10].mean() / 100 - 1) < 0.05 and \
        abs(smoothed[0, 9:12, 9:12].mean() / 100 - 1) < 0.03

    # The dose is the median of the per-bin doses in any blocks of bins.
    rng = np.random.default_rng(12)
    lam = np.where(np.arange(60) < 31, 40.0, 5.0)
    sample = _write_stack(tmp_path / "sample_dose", rng.poisson(0.7 * lam, (16, 16, 60)))
    ob = _write_stack(tmp_path / "ob_dose", rng.poisson(lam, (16, 16, 60)))
    _, dose7, per_bin, _ = _converted(tmp_path, "b7", sample, ob, block_bins=7)
    _, dose60, _, _ = _converted(tmp_path, "b60", sample, ob, block_bins=60)
    assert dose7 == dose60 == pytest.approx(np.median(per_bin)) and dose7 == pytest.approx(40, rel=0.05)
    assert load_dataset(sample, open_beam=[ob]).dose == pytest.approx(dose7)
    assert load_dataset(str(tmp_path / "b7.h5"), wave_range=(0, 59)).dose == pytest.approx(dose7)


def test_load_dataset_stays_within_its_memory_budget(tmp_path):
    """Transmissions of (64, 64, 64, 100) load within 1.5 x 16 MiB + 8 MiB of traced memory beyond T at a 16 MiB
    budget."""
    MiB = 2**20
    budget = 16
    path = str(tmp_path / "T4.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.random.default_rng(6).random((64, 64, 64, 100), dtype=np.float32) * 1.2)
        f.create_dataset("dataset_type", data=np.bytes_("transmission"))
    tracemalloc.start()
    try:
        ds = load_dataset(path, memory_budget_mib=budget)
        excess = tracemalloc.get_traced_memory()[1] - ds.T.nbytes
    finally:
        tracemalloc.stop()
    assert excess <= (1.5 * budget + 8) * MiB, excess / MiB

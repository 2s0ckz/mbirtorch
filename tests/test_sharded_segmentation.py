"""Gate for the sharded path of segment_plastic_metal: run the same volume
through the function whole and split into shards, and require the same
answer.  Identical is the right bar for THIS volume: the counts are exact
integers and the masks are pure thresholding, so a threshold moves only if
the sharded binning displaces a count across the boundary the DP picks.  That
is a measured expectation, not a guarantee -- the device-side binning is
float32 and truncating where numpy's is float64 with an edge-correction pass
(see _sharded_masked_histogram) -- so a future volume that trips it is a
tolerance question, not a bug in this gate.
"""

import numpy as np
import pytest
import torch

import mbirtorch.preprocess as mtp
from mbirtorch import _sharding


def _test_volume():
    """A small volume with three intensity classes and an odd slice count,
    so a 2-shard split gives shards of 6 and 5 slices."""
    rng = np.random.default_rng(3)
    vol = rng.uniform(0.0, 0.02, size=(40, 40, 11)).astype(np.float32)
    vol[10:30, 10:30, 2:9] += 0.05      # plastic
    vol[18:22, 18:22, 3:8] += 0.2       # metal
    return vol


def _as_shards(vol, n_shards):
    """Split the volume's slice axis over n CPU shards, as the engine does."""
    placement = _sharding.Placement(['cpu'] * n_shards, axis=-1,
                                    axis_len=vol.shape[2])
    tensors = [torch.as_tensor(vol[:, :, s0:s1].copy())
               for _dev, (s0, s1) in placement.shard_ranges()]
    return _sharding.Shards(tensors, placement)


def test_sharded_segmentation_matches_unsharded():
    vol = _test_volume()
    ref_p, ref_m, ref_ps, ref_ms = mtp.segment_plastic_metal(
        torch.as_tensor(vol), num_metal=1)

    shards = _as_shards(vol, 2)
    p, m, ps, ms = mtp.segment_plastic_metal(shards, num_metal=1)

    # Masks come back sharded on the same layout as the input.
    assert isinstance(p, _sharding.Shards) and p.placement is shards.placement
    assert np.array_equal(p.gather(), ref_p.numpy())
    assert np.array_equal(m[0].gather(), ref_m[0].numpy())
    # Scale factors agree to float rounding (sums accumulate per shard).
    assert ps == pytest.approx(ref_ps, rel=1e-5)
    assert ms[0] == pytest.approx(ref_ms[0], rel=1e-5)

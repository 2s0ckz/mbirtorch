"""Tests of the one-dimensional clustering that groups the translations of a scan.

Each test builds values from known positions plus random positioning error, shuffles
them, and checks the number of clusters, the labels, and the centers.  The tests run in
well under a second.
"""

import numpy as np
import pytest

from mbirtorch.utilities import _cluster_1d

SEED = 0


def make_values(positions, views_per_position, error, rng):
    """This function returns shuffled values at the given positions plus uniform error in
    [-error, error], and the true label of each value."""
    positions = np.asarray(positions, dtype=np.float64)
    values = np.concatenate([p + rng.uniform(-error, error, views_per_position)
                             for p in positions])
    labels = np.repeat(np.arange(len(positions)), views_per_position)
    order = rng.permutation(len(values))
    return values[order], labels[order]


@pytest.mark.parametrize("positions, views_per_position, error", [
    (np.arange(5), 3, 0.05),            # 5 stops one unit apart, small error
    (np.arange(5), 3, 0.2),             # error near the limit of the method
    (np.arange(5), 3, 0.0),             # no error at all
    (np.arange(5) * 3.0, 3, 0.05),      # step larger than one unit
    (np.arange(200), 40, 0.02),         # a large scan
    (np.arange(100), 2, 0.05),          # only two views per stop
    ([0.0, 5.0], 10, 0.05),             # only two stops
])
def test_clusters_on_a_grid_are_found_exactly(positions, views_per_position, error):
    rng = np.random.default_rng(SEED)
    values, true_labels = make_values(positions, views_per_position, error, rng)
    centers, labels = _cluster_1d(values)
    assert len(centers) == len(positions)
    assert np.all(np.diff(centers) > 0)
    assert np.array_equal(labels, true_labels)
    assert np.abs(centers - np.asarray(positions, dtype=np.float64)).max() <= error


def test_exact_repeats_with_one_roundoff_value_stay_grouped():
    rng = np.random.default_rng(SEED)
    values, _ = make_values(np.arange(5), 3, 0.0, rng)
    values = np.append(values, 2.0001)
    centers, labels = _cluster_1d(values)
    assert len(centers) == 5
    assert labels[-1] == 2


@pytest.mark.parametrize("values", [
    np.full(20, 3.0),                                        # one stop, exact repeats
    3.0 + np.random.default_rng(SEED).uniform(-0.05, 0.05, 20),  # one stop with error
    np.arange(7) * 2.0,                                      # one view per stop
    np.random.default_rng(SEED).uniform(0.0, 100.0, 50),     # random positions
    np.array([0.0, 1.0, 2.0]),                               # three values
    np.array([4.0]),                                         # one value
])
def test_values_without_two_kinds_of_differences_are_one_cluster(values):
    centers, labels = _cluster_1d(values)
    assert len(centers) == 1
    assert np.all(labels == 0)
    assert np.isclose(centers[0], values.mean())


def test_labels_are_in_the_input_order():
    values = np.array([10.0, 0.0, 10.02, 0.01, 20.0, 19.99, 0.02, 10.01])
    centers, labels = _cluster_1d(values)
    assert np.allclose(centers, [0.01, 10.01, 19.995])
    assert np.array_equal(labels, [1, 0, 1, 0, 2, 2, 0, 1])

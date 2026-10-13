"""A staleness check of the minimum problem sizes that cannot run is logged
rather than raised, so a missing file mid-refactor never takes a
reconstruction down with it.

The selection RULE these numbers feed is tested in test_device_policy.py.
"""

from mbirtorch import _min_problem_sizes as sizes


# ── the staleness report (loud, never fatal) ─────────────────────────────────
def test_a_staleness_check_that_cannot_run_is_logged_rather_than_raised(
        monkeypatch):
    """A missing file mid-refactor must not take a reconstruction down with
    it.  The note reports that the minimum sizes could not be checked, which
    is not the same claim as 'up to date', and nothing raises."""
    def exploding():
        raise FileNotFoundError('mbirtorch/projectors.py')

    monkeypatch.setattr(sizes, 'stale_cost_inputs', exploding)
    monkeypatch.setattr(sizes, 'STALE_SINCE', None)
    monkeypatch.setattr(sizes, '_DRIFT_CHECK', None)

    note = sizes.stale_note()
    assert 'could not be checked' in note, note
    assert 'FileNotFoundError' in note, note

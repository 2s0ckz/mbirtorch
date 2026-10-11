"""The minimum problem sizes, the checksum that protects them, and the rule
that decides which device counts are allowed.

The minimum sizes are a MEASUREMENT of where each device count starts paying
for itself.  Two things fail HARD here:

  * TAMPERING -- TABLE_CHECKSUM binds the minimum sizes, the recorded code
    hashes and STALE_SINCE, so hand-editing a hash to silence the staleness
    note fails here.
  * PROVENANCE -- minimum sizes that rise with the device count: the check
    that makes a row a measurement rather than an assertion.

A staleness check that cannot run is logged rather than raised, so a missing
file mid-refactor never takes a reconstruction down with it.

The selection RULE these numbers feed is tested in test_device_policy.py.
"""

from mbirtorch import _min_problem_sizes as sizes

REFRESH = 'python dev_scripts/measure_min_problem_sizes.py'


def install_a_table_with_an_unmeasured_row(monkeypatch):
    """A synthetic two-row table: a finite n=2 minimum size and an n=4
    unmeasured row.  An unmeasured row has ``elements`` None, because no
    minimum size was ever found for that count.

    The rule outlives the data.  The unmeasured rows of the shipped table come
    and go with new measurements, so the tests below read the rule off a table
    built here rather than off whichever numbers happen to ship.
    """
    def row(count, elements, shape, bracket, note):
        return sizes.MinSizeRow(family='synthetic', count=count, elements=elements,
                                shape=shape, against=count // 2, bracket=bracket,
                                spread=0.01, gpu=sizes.MEASURED_GPU,
                                config=sizes.MEASURED_CONFIG, measured='2026-08-10',
                                commit='a880d9c', largest_tested=1_023_934_464,
                                note=note)

    table = {
        ('synthetic', 2): row(
            2, 88_080_384, (512, 448, 384),
            sizes.Bracket(losing_shape=(384, 336, 288), losing_speedup=0.64,
                       winning_shape=(512, 448, 384), winning_speedup=1.23),
            'synthetic finite minimum size'),
        ('synthetic', 4): row(
            4, None, None,
            sizes.Bracket(losing_shape=(1024, 1008, 992), losing_speedup=0.92,
                       winning_shape=None, winning_speedup=None),
            'synthetic unmeasured row: no minimum size at or below the '
            '1024-class shape'),
    }
    monkeypatch.setattr(sizes, 'MIN_SIZES', table)
    return table


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


# ── the checksum (this still fails hard) ────────────────────────────────────
def test_the_min_sizes_hashes_and_staleness_stamp_move_as_one_unit(monkeypatch):
    """Now that a changed code input only warns, hand-editing a hash is the
    cheap way to silence the note while leaving minimum sizes that were never
    measured again.  The checksum binds all three, so that shortcut fails HERE
    -- and this one is not a warning.  The binding is then exercised rather
    than asserted: forging a recorded hash must move the checksum."""
    assert sizes.table_checksum() == sizes.TABLE_CHECKSUM, (
        'mbirtorch/_min_problem_sizes.py was edited without going through the '
        'measurement script: MIN_SIZES, MEASURED_CODE_HASHES and STALE_SINCE are '
        'bound by TABLE_CHECKSUM and must be written together.\n'
        '    {} --update-hashes            (after a new measurement)'.format(REFRESH))

    forged = dict(sizes.MEASURED_CODE_HASHES)
    forged['projectors.py'] = '0' * 64
    monkeypatch.setattr(sizes, 'MEASURED_CODE_HASHES', forged)
    assert sizes.table_checksum() != sizes.TABLE_CHECKSUM


# ── provenance ───────────────────────────────────────────────────────────────
def test_finite_min_sizes_rise_with_the_device_count():
    assert sizes.monotone_violations() == []


# ── the accessor ─────────────────────────────────────────────────────────────
def test_multiaxis_min_sizes_split_by_count():
    """Multiaxis allows two devices from the 512-class and four only from
    the 1024-class, and refuses both below their minimum sizes.

    The history in brief.  Both rows were unmeasured rows until 2026-08-20,
    when the recompile remedy cleared them and the mg48 measurement placed
    both minimum sizes at the 512-class.  The Triton multiaxis kernels then
    cut the one-device time about four-fold.  The mg56 measurement on that
    tree (with the mg55 1024-class reading) kept the two-device minimum size
    at the 512-class and RAISED the four-device minimum size to the
    1024-class.  Small shapes no longer amortize a four-way fan-out when one
    device is this fast.  The notes of the rows and multigpu_findings.md
    sections 1.45 and 1.46 in the plans repository carry the measurements.
    Counts with no row of their own (3, and anything above 4) use the n=4
    row.
    """
    import mbirtorch

    assert mbirtorch.MultiAxisParallelModel._size_table_key == 'multiaxis'
    at_the_512_class = sizes.sinogram_elements((512, 448, 384))
    at_the_1024_class = sizes.sinogram_elements((1024, 1008, 992))
    below_every_min_size = sizes.sinogram_elements((384, 336, 288))
    ok, why = sizes.allowed('multiaxis', 2, at_the_512_class)
    assert ok, why
    for count in (2, 3, 4, 8):
        ok, why = sizes.allowed('multiaxis', count, at_the_1024_class)
        assert ok, (count, why)
    for count in (3, 4, 8):
        ok, why = sizes.allowed('multiaxis', count, at_the_512_class)
        assert not ok, (count, why)
        assert 'below the minimum problem size' in why, why
    for count in (2, 3, 4, 8):
        ok, why = sizes.allowed('multiaxis', count, below_every_min_size)
        assert not ok, (count, why)
        assert 'below the minimum problem size' in why, why
        assert 'configure_devices' in why, why


def test_the_allowed_rule_for_counts_model_types_and_unmeasured_rows(monkeypatch):
    """The allowed() rule in one place.

    One device always passes.  A count with no row of its own uses the row of
    the next measured count above it, and the reason names that row rather
    than the count asked about.  A model class that sets no table key uses the
    parallel minimum sizes and is told so.  A model type the table has never
    heard of allows every count rather than refusing it.  An unmeasured row
    refuses its count at every size, including sizes above the largest one
    tested, and the refusal names that limit.
    """
    for family in (None, 'parallel', 'cone', 'a-family-with-no-rows'):
        ok, why = sizes.allowed(family, 1, 1)
        assert ok and 'always allowed' in why

    # The conservative direction: n=3 uses the n=4 row, and so does any count
    # above 4, since minimum sizes rise with the count.
    four = sizes.MIN_SIZES[('parallel', 4)]
    for count in (3, 4, 5, 8):
        assert sizes.row_for_count('parallel', count) is four
    assert sizes.row_for_count('parallel', 2) is sizes.MIN_SIZES[('parallel', 2)]
    _ok, why = sizes.allowed('parallel', 3, 1000)
    assert 'parallel n=4 row, which n=3 uses' in why

    for count in (2, 3, 4):
        for size in (1_376_256, 1_023_934_464):
            assert (sizes.allowed(None, count, size)[0]
                    == sizes.allowed('parallel', count, size)[0])
    _ok, why = sizes.allowed(None, 2, 1_376_256)
    assert 'sets no _size_table_key' in why and 'parallel minimum sizes apply' in why

    # Only a model class reaches a model type with no rows, and only by
    # setting a table key with no rows.  The measurement script reports such a
    # class as work to do.  The name below is invented: every shipped model
    # type has rows.
    ok, why = sizes.allowed('a-family-with-no-rows', 4, 1)
    assert ok and 'no minimum problem sizes are measured' in why

    # The unmeasured row is read off a synthetic table, so the assertion tests
    # the rule rather than whichever rows happen to be unmeasured in the
    # shipped table.  This replaces MIN_SIZES, so it comes last.
    install_a_table_with_an_unmeasured_row(monkeypatch)
    for size in (1, 88_080_384, 1_023_934_464, 10 ** 12):
        ok, why = sizes.allowed('synthetic', 4, size)
        assert not ok
        assert 'unmeasured row' in why and 'largest size tested' in why


def test_a_min_size_allows_exactly_at_its_own_value():
    row = sizes.MIN_SIZES[('parallel', 2)]
    assert sizes.allowed('parallel', 2, row.elements)[0]
    assert not sizes.allowed('parallel', 2, row.elements - 1)[0]

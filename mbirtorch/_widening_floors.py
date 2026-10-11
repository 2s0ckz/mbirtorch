"""This module holds a table of minimum problem sizes for multi-GPU reconstruction.
For each model type and device count, the table gives the smallest problem size at which
the automatic device choice uses that many devices.

The problem size is the number of sinogram elements, ``prod(sinogram_shape)``.  Below
the minimum size, mbirtorch uses that device count only when no other count fits in
memory.  mbirtorch always allows one device.  A device count with no table row uses the
row of the next larger measured count.  A row with ``elements=None`` has no measured
minimum size.  The sizes are measured, and the program
``dev_scripts/refresh_widening_floors.py`` writes them.
"""

import hashlib
import inspect
import os
from collections import namedtuple

#: Setting this environment variable to '0', 'false', 'no', or 'off' turns off the minimum
#: size check.
GUARD_ENV_VAR = 'MBIRTORCH_WIDENING_GUARD'
_GUARD_OFF_VALUES = ('0', 'false', 'no', 'off')

#: The model type used for a model that has no ``_floor_family`` attribute.
DEFAULT_FAMILY = 'parallel'

#: The hardware and the run settings used to measure every row of the table.
MEASURED_GPU = 'NVIDIA H100 80GB HBM3 (4 per node)'
MEASURED_CONFIG = ('warm median of 3 seeded 3-iteration VCD recons, cold '
                   'pass discarded, package-default subset schedule, Triton '
                   'kernels on, torch.compile auto')

#: The sizes were measured only with MEASURED_CONFIG.  Other iteration counts and
#: subset schedules were not measured.
MEASUREMENT_CAVEAT = ('validated at MEASURED_CONFIG only; a different '
                      'iteration count or subset schedule is outside the '
                      'measured envelope')

#: The minimum speedup for a device count.  A count is allowed at the smallest measured
#: size where it is faster than the best smaller allowed count by this factor.  The
#: speedup must also exceed the run-to-run spread.
ADMISSION_MARGIN = 1.15

#: The two measured shapes on either side of a minimum size, with their speedups.
#: ``losing_cell`` is the largest (views, rows, channels) shape where the count was slower.
#: ``winning_cell`` is the smallest shape where the count was faster.  Either shape is
#: None when it was not measured.  If ``winning_cell`` is None, then ``elements`` is None.
Bracket = namedtuple('Bracket', ('losing_cell', 'losing_speedup',
                                 'winning_cell', 'winning_speedup'))

#: One table row.  ``elements`` is the minimum size in sinogram elements, or None when
#: no minimum size was measured.  ``cell`` is the sinogram shape where the minimum size
#: was measured.  ``against`` is the device count used for comparison.  ``spread`` is the
#: run-to-run spread of the time.
Floor = namedtuple('Floor', ('family', 'count', 'elements', 'cell', 'against',
                             'bracket', 'spread', 'gpu', 'config', 'measured',
                             'commit', 'largest_tested', 'note'))

# ── the measured table ───────────────────────────────────────────────────────
# Shapes are (views, rows, channels).  The notes name each shape by its view count:
# 384-class (384, 336, 288), 512-class (512, 448, 384), 768-class (768, 672, 576), and
# 1024-class (1024, 1008, 992).  For the denoiser, the shapes are image shapes.
FLOORS = {
    ('cone', 2): Floor(
        family='cone', count=2, elements=88_080_384, cell=(512, 448, 384),
        against=1,
        bracket=Bracket(losing_cell=(384, 336, 288), losing_speedup=0.88,
                        winning_cell=(512, 448, 384), winning_speedup=1.31),
        spread=0.06817, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-22', commit='c024ec9',
        largest_tested=297_271_296,
        note='the full refresh on the multiaxis-kernel tree (mg56, job '
             '15435735): 0.88x at the 384-class, 1.31x at the 512-class, '
             '1.65x at the 768-class.  The floor holds where mg48 placed '
             'it; the multiaxis kernels share helper modules with this '
             'family and moved nothing here, which is what this '
             'reproduction verifies'),
    ('cone', 4): Floor(
        family='cone', count=4, elements=297_271_296, cell=(768, 672, 576),
        against=2,
        bracket=Bracket(losing_cell=None, losing_speedup=None,
                        winning_cell=(768, 672, 576), winning_speedup=1.15),
        spread=0.01318, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-22', commit='c024ec9',
        largest_tested=1_023_934_464,
        note='mg56 (job 15435735): the floor MOVES DOWN one class.  The '
             '768-class read 1.15x against two devices, clearing the '
             'margin it had missed at every measurement since the width '
             'padding landed, and the 1024-class read 1.64x.  No losing '
             'cell was measured below it this run; mg48 read 1.12x at the '
             'same cell, so the movement is at the margin, not a step'),
    ('denoiser', 2): Floor(
        family='denoiser', count=2, elements=None, cell=None,
        against=1,
        bracket=Bracket(losing_cell=(1024, 1008, 992), losing_speedup=0.93,
                        winning_cell=None, winning_speedup=None),
        spread=0.0426, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-22', commit='c024ec9',
        largest_tested=1_023_934_464,
        note='still a sentinel after the denoiser performance commit '
             '(mg56, job 15435735): 0.73x at the 512-class, 0.87x at the '
             '768-class, 0.93x at the 1024-class.  Read in IMAGE VOXELS: '
             'the denoiser sinogram shape is its image shape.  The timed '
             'call is denoise with sigma_noise supplied, at the shared '
             'warm-median protocol.  The ratio keeps rising with size '
             '(mg48 read 0.67x at the top cell), so a larger ladder may '
             'yet find admission; capacity still widens a denoise that '
             'cannot fit one device'),
    ('denoiser', 4): Floor(
        family='denoiser', count=4, elements=None, cell=None,
        against=1,
        bracket=Bracket(losing_cell=(1024, 1008, 992), losing_speedup=0.90,
                        winning_cell=None, winning_speedup=None),
        spread=0.03603, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-22', commit='c024ec9',
        largest_tested=1_023_934_464,
        note='a sentinel like n=2 (mg56, job 15435735): 0.55x, 0.75x, '
             '0.90x across the probes, measured against n=1 because no '
             'smaller denoiser count is admitted.  Read in image voxels; '
             'the n=2 note records the protocol.  Nearer parity at the '
             'top cell than mg48 read (0.61x), same rising trend'),
    ('multiaxis', 2): Floor(
        family='multiaxis', count=2, elements=88_080_384,
        cell=(512, 448, 384), against=1,
        bracket=Bracket(losing_cell=(384, 336, 288), losing_speedup=0.86,
                        winning_cell=(512, 448, 384), winning_speedup=1.38),
        spread=0.0608, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-22', commit='c024ec9',
        largest_tested=297_271_296,
        note='the first refresh with the multiaxis Triton kernels '
             'selected (mg56, job 15435735): 0.86x at the 384-class, '
             '1.38x at the 512-class, 1.72x at the 768-class.  The knee '
             'did not move when the kernels landed, because the '
             'one-device and two-device walls fell together (about '
             'four-fold; multigpu_findings.md sections 1.45 and 1.46).  '
             'This run also measured the 384-class loss that mg48\'s '
             'sentinel probes could not, so the bracket is complete for '
             'the first time'),
    ('multiaxis', 4): Floor(
        family='multiaxis', count=4, elements=1_023_934_464,
        cell=(1024, 1008, 992), against=2,
        bracket=Bracket(losing_cell=(768, 672, 576), losing_speedup=1.09,
                        winning_cell=(1024, 1008, 992), winning_speedup=1.64),
        spread=0.0608, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-22', commit='c024ec9',
        largest_tested=1_023_934_464,
        note='placed by hand at the 1024-class, on two instruments.  mg56 '
             '(job 15435735) read 0.70x at the 512-class and a thin 1.09x '
             'at the 768-class, under the 1.15x margin, with nothing '
             'larger on its ladder.  The winning reading is mg55 (job '
             '15434826), the same seeded warm-median protocol at the '
             '1024-class, where four devices beat two at 1.64x with 0.1 '
             'percent spread (multigpu_findings.md section 1.46).  The '
             'coarse rule rounds the thin 768-class win up one class and '
             'lands on the same cell.  The floor RISES from the '
             'torch-body era (mg48 read wins from the 512-class up) '
             'because the kernels cut the one-device wall about '
             'four-fold, so small cells no longer amortize the fan-out'),
    ('parallel', 2): Floor(
        family='parallel', count=2, elements=297_271_296,
        cell=(768, 672, 576), against=1,
        bracket=Bracket(losing_cell=(512, 448, 384), losing_speedup=1.09,
                        winning_cell=(768, 672, 576), winning_speedup=1.46),
        spread=0.1124, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-22', commit='c024ec9',
        largest_tested=1_023_934_464,
        note='mg56 (job 15435735): 1.09x at the 512-class inside an 11.2 '
             'percent spread, which does not count as a win; 1.46x at the '
             '768-class, 1.59x at the 1024-class.  The floor holds where '
             'mg40 moved it on the channel-sorted kernel'),
    ('parallel', 4): Floor(
        family='parallel', count=4, elements=1_023_934_464,
        cell=(1024, 1008, 992), against=2,
        bracket=Bracket(losing_cell=(768, 672, 576), losing_speedup=0.82,
                        winning_cell=(1024, 1008, 992), winning_speedup=1.40),
        spread=0.05224, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-22', commit='c024ec9',
        largest_tested=1_023_934_464,
        note='mg56 (job 15435735): 0.82x at the 768-class against two '
             'devices and 1.40x at the 1024-class; the floor holds at the '
             '1024-class, reproducing mg48 (0.83x / 1.34x) within the '
             'spreads'),
    ('translation', 2): Floor(
        family='translation', count=2, elements=364_800_000,
        cell=(256, 950, 1500), against=1,
        bracket=Bracket(losing_cell=(256, 475, 750), losing_speedup=0.64,
                        winning_cell=(256, 950, 1500), winning_speedup=1.19),
        spread=0.04742, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-22', commit='c024ec9',
        largest_tested=1_459_200_000,
        note='mg56 (job 15435735): 0.64x, 1.19x, 1.25x across the '
             'production-anchored cells, reproducing mg48 to the second '
             'decimal; the floor holds at the middle cell.  The probe '
             'cells are NOT the shared ladder: production-anchored '
             'translation scans on a fixed 16x16 grid with the detector '
             'and the spacing scaled together; the floor reads in '
             'sinogram elements as usual'),
    ('translation', 4): Floor(
        family='translation', count=4, elements=1_459_200_000,
        cell=(256, 1900, 3000), against=2,
        bracket=Bracket(losing_cell=(256, 950, 1500), losing_speedup=0.79,
                        winning_cell=(256, 1900, 3000), winning_speedup=1.15),
        spread=0.02454, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-22', commit='c024ec9',
        largest_tested=1_459_200_000,
        note='mg56 (job 15435735): 0.79x at half scale against two '
             'devices and 1.15x at the production scan; the floor holds '
             'at the production cell, now measured against n=2 where mg48 '
             'measured against n=1.  The cells are the production-anchored '
             'scans the n=2 note names'),
}

# ── the code the sizes were measured against ─────────────────────────────────
#: The files and methods that determine the projection time of every model type.  The
#: hash of a method covers its source code.  The hash of a file covers the whole file.
_SHARED_COST_INPUTS = (
    'projectors.py', '_sharding.py', '_utils.py',
    'TomographyModel._sparse_forward_project_sharded',
    'TomographyModel._sparse_forward_project_cylinders',
    'TomographyModel._sparse_back_project_sharded',
)

#: The files and methods that determine the projection time of each model type.
#: ``triton_cone.py`` holds functions that the other kernel files import.  The
#: denoiser uses no projector.
FAMILY_COST_INPUTS = {
    'parallel': _SHARED_COST_INPUTS + ('triton_parallel.py', 'triton_cone.py',
                                       'parallel_beam.py'),
    'cone': _SHARED_COST_INPUTS + ('triton_cone.py', 'cone_beam.py'),
    'multiaxis': _SHARED_COST_INPUTS + ('triton_multiaxis.py',
                                        'triton_cone.py',
                                        'multiaxis_parallel.py'),
    'translation': _SHARED_COST_INPUTS + ('translation_model.py',),
    'denoiser': ('denoising.py', 'qggmrf.py'),
}

#: Every file and method above, separated into files and methods.
_ALL_COST_INPUTS = sorted({name for names in FAMILY_COST_INPUTS.values()
                           for name in names})
COST_INPUT_FILES = tuple(name for name in _ALL_COST_INPUTS
                         if not name.startswith('TomographyModel.'))
COST_INPUT_METHODS = tuple(name.split('.', 1)[1] for name in _ALL_COST_INPUTS
                           if name.startswith('TomographyModel.'))

#: The sha256 of each file and method at the time the sizes were measured.
#: ``refresh_widening_floors.py --bless`` prints new values.
BLESSED_COST_HASHES = {
    'TomographyModel._sparse_back_project_sharded':
        'f73fa28f32d2393fac3f06139d8ddeccc55260fceb28bbca7f102334839fb5a4',
    'TomographyModel._sparse_forward_project_cylinders':
        '3107ed365cd4d06340996901a9bb0b8440b53cae55195234d89a454535bef48f',
    'TomographyModel._sparse_forward_project_sharded':
        'f4389b004622d77eb74be81a9c6a61628007e1558f657d0a09b7bf790e69266d',
    '_sharding.py':
        'f7b9cd3b117a880f76415a9923dbdc2171766a922a3fea575934af88dd8fea8c',
    '_utils.py':
        '33908e7e7bc48741456064e14cb044a163a3ed70f2c4f1c49e9cd38db4f387bb',
    'cone_beam.py':
        'db669d0eeea29b0b0ec06db2d016cad6ff841f02f309d565e48c98b98da6a903',
    'denoising.py':
        'f5f9ea4e18668d2437d6ee038da728eadcb2772deba3573c5368c61c2b5d0013',
    'multiaxis_parallel.py':
        '965eb36581ff8cc19d7f9acef8015209cbbc188b6fe05163d07ef28ae04b9230',
    'parallel_beam.py':
        'd923a542a9f0cdd612d88e0edc233f7de5c114b73da48bb938ea2e9e022df1a4',
    'projectors.py':
        '05293d4ba18dc9fa56e50c3773909304fe8e635edd6256f19594d95066daec5f',
    'qggmrf.py':
        '75ce00d82f10650bf8d231b3b4beaf62ba2e7164f255b82f854403893afd53de',
    'translation_model.py':
        '07f4b9eb5527e66139341847192913c9afd30a8abf10e8e40ef9c07cbb21c520',
    'triton_cone.py':
        'a4d8350b350cd34a358bb54c33ae3d408f5b1d0131044d0d88b462c5cbaf2dbc',
    'triton_multiaxis.py':
        'be7ba7dd0291669ec12207a563886acd867edc0a431fca4eb8a0e790a3debbab',
    'triton_parallel.py':
        '79c4aaaa071c03567f505978a65ed9a183908d6c11342b59ac5668fb10ad0d55',
}

#: A date that marks the sizes as needing a new measurement, or None.  The refresh
#: program writes None.  A person may set a date, for example after a hardware change.
STALE_SINCE = None

#: The sha256 of FLOORS, BLESSED_COST_HASHES, and STALE_SINCE together.  A hand edit to
#: any of them makes a test fail.  ``refresh_widening_floors.py --bless`` prints this value.
TABLE_CHECKSUM = 'e564710cec9ba7d789f059de69d700ec0c6d74c8b5525d70c78a0e2a3fbaca32'


# ── the environment variable ─────────────────────────────────────────────────
def guard_enabled():
    """This function returns True when the minimum size check is on.  The function
    reads the environment variable on each call."""
    return os.environ.get(GUARD_ENV_VAR, '').strip().lower() \
        not in _GUARD_OFF_VALUES


# ── the check for changed code ───────────────────────────────────────────────
#: The saved result of the hash check, ``(changed_names, failure)``.  The value is None
#: before the first check.
_DRIFT_CHECK = None

#: The program that measures new sizes.  Every notice names this program.
_REFRESH_COMMAND = 'dev_scripts/refresh_widening_floors.py'


def _drift_check():
    """This function returns ``(names, failure)``.  ``names`` lists the files and methods
    whose hash has changed.  ``failure`` is an error message when the check could not run.
    The function runs the check once per process and saves the result.
    """
    global _DRIFT_CHECK
    if _DRIFT_CHECK is None:
        try:
            _DRIFT_CHECK = (tuple(name for name, _want, _got
                                  in stale_cost_inputs()), None)
        except Exception as exc:                                  # noqa: BLE001
            _DRIFT_CHECK = ((), '{}: {}'.format(type(exc).__name__, exc))
    return _DRIFT_CHECK


def stale_note():
    """This function returns a one-line notice for the device log, or None.  The notice
    says that the sizes need a new measurement.  The function returns a notice when the
    projection code changed, when :data:`STALE_SINCE` is set, or when the check could not
    run.  mbirtorch still uses the sizes in all these cases.
    """
    drifted, failure = _drift_check()
    parts = []
    if failure is not None:
        parts.append('the widening speed floors could not be checked against '
                     'the projection-cost code ({}), so whether they are '
                     'stale is unknown'.format(failure))
    elif drifted:
        families_hit = _families_priced_by(drifted)
        parts.append('the widening speed floors were measured against '
                     'projection-cost code that has since changed ({}, '
                     'pricing the {} floors), so they may no longer '
                     'describe it'.format(', '.join(drifted),
                                          ', '.join(families_hit)))
    if STALE_SINCE is not None:
        parts.append('the widening speed floors are marked stale since {}: '
                     'the floors were not re-measured when they were '
                     'recorded as owing a measurement'.format(STALE_SINCE))
    if not parts:
        return None
    return '{}; re-measuring with {} clears this'.format('; '.join(parts),
                                                         _REFRESH_COMMAND)


# ── the size measure ─────────────────────────────────────────────────────────
def sinogram_elements(sinogram_shape):
    """This function returns ``prod(sinogram_shape)`` as an int.  The table measures
    problem size with this number."""
    total = 1
    for extent in sinogram_shape:
        total *= int(extent)
    return int(total)


def _millions(elements):
    return '{:.1f}M'.format(elements / 1e6)


# ── the lookup ───────────────────────────────────────────────────────────────
def families():
    """This function returns every model type that has rows in the table."""
    return sorted({family for family, _count in FLOORS})


def _measured_counts(family):
    return sorted(count for fam, count in FLOORS if fam == family)


def governing_floor(family, count):
    """This function returns the table row that applies to ``count`` devices for the model
    type ``family``.  A count with no row uses the row of the next larger measured count.  If
    no larger count was measured, the function returns the row of the largest count.  The
    function returns None when the table has no rows for ``family``.
    """
    counts = _measured_counts(family)
    if not counts:
        return None
    above = [c for c in counts if c >= count]
    return FLOORS[(family, above[0] if above else counts[-1])]


def admitted(family, count, sino_elements):
    """This function decides whether the automatic device choice may use ``count``
    devices at this problem size.  The function returns the decision and the reason.

    Args:
        family (str or None): The model type, such as ``'parallel'`` or ``'cone'``.  None
            means that the model has no ``_floor_family`` attribute, and then the function
            uses the rows for ``'parallel'``.
        count (int): The device count to check.
        sino_elements (int): The problem size, ``prod(sinogram_shape)``.

    Returns:
        (bool, str): True if the count is allowed, and the reason as text for the log.
    """
    resolved = DEFAULT_FAMILY if family is None else family
    substituted = ('' if family is not None else
                   ' (this model names no _floor_family, so the {} floors '
                   'apply)'.format(DEFAULT_FAMILY))
    size = _millions(sino_elements)

    if count <= 1:
        return True, 'a single device is always admitted'

    floor = governing_floor(resolved, count)
    if floor is None:
        return True, ('no speed floors are measured for the {} family, so '
                      'every count is admitted'.format(resolved))

    inherited = ('' if floor.count == count else
                 ', which n={} inherits'.format(count))
    named = 'the {} n={} floor{}'.format(resolved, floor.count, inherited)

    if floor.elements is None:
        return False, (
            'held by the speed floor: {} is a sentinel -- no admission point '
            'has been measured at or below {} sinogram elements, the largest '
            'size tested; configure_devices(num_devices={}) '
            'overrides'.format(named, _millions(floor.largest_tested), count)
            + substituted)
    if sino_elements >= floor.elements:
        return True, ('{} sinogram elements >= {} ({})'.format(
            size, _millions(floor.elements), named) + substituted)
    return False, (
        'held by the speed floor: {} sinogram elements < {} ({}); '
        'configure_devices(num_devices={}) overrides'.format(
            size, _millions(floor.elements), named, count) + substituted)


def fallback_reason(family, count, sino_elements):
    """This function returns the log note for a device count that mbirtorch chose below
    its minimum size.  mbirtorch makes this choice only when no allowed count fits in memory.
    """
    resolved = DEFAULT_FAMILY if family is None else family
    floor = governing_floor(resolved, count)
    if floor is None or floor.elements is None:
        detail = ('no admission point is measured for {} n={} at or below {} '
                  'sinogram elements'.format(
                      resolved, count,
                      _millions(floor.largest_tested) if floor else 0))
    else:
        detail = '{} sinogram elements < {}'.format(
            _millions(sino_elements), _millions(floor.elements))
    return ('chosen past its speed floor because no admitted count fits: '
            '{} (the {} n={} floor)'.format(detail, resolved, floor.count
                                            if floor else count))


# ── checks on the table ──────────────────────────────────────────────────────
def monotone_violations():
    """This function returns the model types whose measured minimum size decreases
    as the device count increases.  The module asserts that there are none."""
    bad = []
    for family in families():
        finite = [(count, FLOORS[(family, count)].elements)
                  for count in _measured_counts(family)
                  if FLOORS[(family, count)].elements is not None]
        for (low, low_floor), (high, high_floor) in zip(finite, finite[1:]):
            if high_floor < low_floor:
                bad.append('{}: n={} floor {} < n={} floor {}'.format(
                    family, high, high_floor, low, low_floor))
    return bad


assert not monotone_violations(), \
    'the widening floors are not monotone in count: ' \
    + '; '.join(monotone_violations())


# ── the hashes and the checksum ──────────────────────────────────────────────
def cost_input_hashes():
    """This function returns the sha256 of every file and method that determines the
    projection time.  The keys are the same as in :data:`BLESSED_COST_HASHES`."""
    from .tomography_model import TomographyModel

    here = os.path.dirname(os.path.abspath(__file__))
    digests = {}
    for name in COST_INPUT_FILES:
        with open(os.path.join(here, name), 'rb') as handle:
            digests[name] = hashlib.sha256(handle.read()).hexdigest()
    for name in COST_INPUT_METHODS:
        source = inspect.getsource(getattr(TomographyModel, name))
        digests['TomographyModel.' + name] = hashlib.sha256(
            source.encode('utf-8')).hexdigest()
    return digests


def stale_cost_inputs():
    """This function returns the files and methods whose hash differs from the recorded
    hash.  The result is a list of ``(name, recorded, actual)`` tuples."""
    actual = cost_input_hashes()
    stale = []
    for name in sorted(set(actual) | set(BLESSED_COST_HASHES)):
        want = BLESSED_COST_HASHES.get(name)
        got = actual.get(name)
        if want != got:
            stale.append((name, want, got))
    return stale


def _families_priced_by(input_names):
    """This function returns the model types whose list in :data:`FAMILY_COST_INPUTS`
    includes any of the given names."""
    moved = set(input_names)
    return sorted(family for family, names in FAMILY_COST_INPUTS.items()
                  if moved & set(names))


def stale_families():
    """This function returns the model types whose recorded hashes no longer match
    the code.  By default, the refresh program measures these model types again and
    replaces their rows."""
    return _families_priced_by(name for name, _want, _got
                               in stale_cost_inputs())


def table_checksum():
    """This function returns the sha256 of FLOORS, BLESSED_COST_HASHES, and STALE_SINCE
    together.  A hand edit to any of them makes a test fail.
    """
    payload = '\n'.join([repr(sorted(FLOORS.items())),
                         repr(sorted(BLESSED_COST_HASHES.items())),
                         repr(STALE_SINCE)])
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def bless_lines(stale_since=None):
    """This function returns new source text for BLESSED_COST_HASHES, STALE_SINCE, and
    TABLE_CHECKSUM.  A person pastes the text into this module.  ``stale_since`` is None
    after a refresh.  A person may pass a date instead.
    """
    hashes = cost_input_hashes()
    lines = ['BLESSED_COST_HASHES = {']
    for name, digest in sorted(hashes.items()):
        lines.append("    '{}':".format(name))
        lines.append("        '{}',".format(digest))
    lines.append('}')
    lines.append('')
    lines.append('STALE_SINCE = {!r}'.format(stale_since))
    lines.append('')
    payload = '\n'.join([repr(sorted(FLOORS.items())),
                         repr(sorted(hashes.items())),
                         repr(stale_since)])
    lines.append("TABLE_CHECKSUM = '{}'".format(
        hashlib.sha256(payload.encode('utf-8')).hexdigest()))
    return '\n'.join(lines)

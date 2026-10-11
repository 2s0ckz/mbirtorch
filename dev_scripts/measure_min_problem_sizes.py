"""This program measures the minimum problem sizes for multi-GPU reconstruction.

It prints a new table for a person to paste into ``mbirtorch/_min_problem_sizes.py``.
That module holds a measurement, not a constant.  For each model type and device count,
a table row gives the smallest sinogram size at which that device count is faster than
the best smaller count.  A change to the projection code can change these sizes.  Examples
are a new kernel, a new chunk size, or a new limit on temporary memory.

No one is required to run this program.  mbirtorch detects changed projection code
automatically.  The first time a process checks the minimum problem sizes, mbirtorch
hashes the projection code.  ``_min_problem_sizes.stale_note()`` then names the files
that changed, in every automatic device choice.  ``tests/test_min_problem_sizes.py``
reports the same files, and it still passes.  A new measurement is work to schedule.  The
note goes away after a person runs this program and pastes its output.

This program is the only writer of three things that must change together.  The three
things are the table rows, the record of how the rows were measured, and the recorded
hashes of the projection code.  The program prints all three as one block with a
checksum.  If a person edits a hash by hand to remove the note, the checksum test fails.
That test always fails when the checksum does not match.

This program uses the following terms.  A test shape is one sinogram shape to measure
for one model type.  An arm is one timed run of one test shape at one device count.
``TEST_SHAPES`` is the fixed list of test shapes that a measurement uses.  Most model
types use ``TEST_SHAPES``.  A model type whose problems ``TEST_SHAPES`` does not describe
has its own list in ``TEST_SHAPES_BY_MODEL``.  An unmeasured row is a table row whose
``elements`` is None.  No minimum problem size has been found yet for its device count.

The method is the same for each test shape and each device count.  The program runs one
reconstruction and discards its time.  It then reports the median time of 3 seeded
3-iteration reconstructions.  Every arm runs in a new subprocess.  The subprocess sets
``MBIRTORCH_NUM_DEVICES`` to the device count of the arm.  This variable fixes the count
and keeps the model on the automatic device choice.  A count changed inside one process
would keep the compiled kernels and the allocator state of the previous arm.  The
program generates one sinogram for each test shape, on one device.  Every arm of that
test shape loads the same sinogram.  Otherwise, the arms would reconstruct different
arrays, and the comparison would not be controlled.

The denoiser differs from the other model types in three ways, because it has no
projectors.  Its input is a noisy phantom rather than a forward projection.  Its timed
call is ``denoise`` rather than ``recon``.  Its device count is set with
``configure_devices`` rather than with the environment variable.  ``denoise`` makes no
automatic device-count decision, and the environment variable acts only through that
decision.  Its test shape is an image shape, because QGGMRFDenoiser sets its sinogram
shape equal to its image shape.

The multiaxis model type uses ``TEST_SHAPES``.  Its models are built as mg18 built them.
Each view has two angles.  The azimuths are evenly spaced over half a turn, and the
elevations are swept across +/- 0.5 radians.  The elevation range is part of what a
multiaxis test shape measures, because the automatic geometry sizes the reconstruction
from it.

The translation model type has its own test shapes.  ``TEST_SHAPES`` does not describe a
translation scan.  Production runs a 16x16 grid of translations, which is 256 views,
against a large detector.  The three translation test shapes keep that grid and halve the
detector and the spacing together.  As a result, all three scan the same object at three
resolutions.  The largest is the production shape that mg18 measured.  A sinogram shape
alone does not determine a translation model the way an angle list does.  For this
reason, ``TRANSLATION_SPECS`` records the grid and the spacing for every translation test
shape, and ``_build_model`` reads them.  The generator also differs.  The Shepp-Logan
phantom is all zeros on a volume only a few voxels deep, and the --smoke translation
shape reconstructs such a volume.  In that case, the generator uses a seeded uniform
phantom and records that it did so.

The first translation measurement compares both counts against one device.  The rule
below reads the allowed counts from the current table, and translation has no measured
smaller count there yet.  After its rows are pasted in, the next measurement compares
n=4 against n=2, provided that n=2 gained a minimum problem size.  Cone n=4 reached its
comparison count in the same way.

The rule for a minimum problem size is as follows.  A device count is compared against
the best smaller allowed count, not always against n=1.  For example, parallel n=4 must
beat n=2.  The allowed counts are read from the current table, so the comparison count
can change between measurements.  Cone n=4 was measured against n=1 while cone n=2 had
no minimum problem size.  The 2026-08-10 measurement gave cone n=2 a minimum problem
size, so the next cone n=4 measurement compares against n=2.  A test shape wins only
under a coarse rule, set on 2026-08-19.  The speedup must reach ``MIN_SPEEDUP`` (1.15x),
and it must exceed 1.0x by more than the run-to-run spread of that test shape.  The
minimum problem size is the smallest measured test shape that wins.  A smaller win that
does not reach ``MIN_SPEEDUP`` moves the minimum problem size up to the next test shape.
If such a win occurs at the largest measured test shape, this run cannot place the
minimum problem size, and the program says so.  When the crossover falls between two
test shapes, the program takes the larger test shape.

A measurement can be limited to some model types with ``--families``.  A full
measurement covers every model type and costs about 12 GPU-hours.  Most of that time
measures rows whose projection code did not change.  ``--families cone,parallel``
measures only the named model types.  ``--families`` with no value measures exactly the
model types whose recorded hashes changed (``_min_problem_sizes.stale_families()``).  The
rows of every other model type are carried forward.  The program prints them unchanged in
the paste, with their measurement records unchanged.  In this way, the program remains
the only writer of the whole table, and the checksum still covers it.  Carrying a model
type forward assumes that its projection code did not change.  Each model type has its
own list of code files, so the program can check this assumption.  A limited run
refuses to start if a model type that it would carry forward has changed code.  For this
reason, a change to a shared file requires a full measurement.  A change to the files of
one model type allows a limited measurement.

The program chooses the test shapes to run as follows.  For a row with a minimum problem
size, it runs three test shapes: the one below the minimum problem size, the one at it,
and the one above it.  For an unmeasured row, it runs the three largest test shapes of
that model type.  In ``TEST_SHAPES``, these are the 512-, 768-, and 1024-class shapes.  A
model type with no table rows at all uses the same test shapes.  See
``UNTABLED_FAMILIES``.  An unmeasured row gains a minimum problem size when its count
exceeds 1.0x by more than the measured spread.  The program then prints the proposed
minimum problem size.  The program does not measure sizes above the ``largest_tested``
value of a row, and it reports them as unmeasured.

Run:
    python dev_scripts/measure_min_problem_sizes.py                  # a 4-GPU node
    python dev_scripts/measure_min_problem_sizes.py --families       # changed model types only
    python dev_scripts/measure_min_problem_sizes.py --families cone,parallel
    python dev_scripts/measure_min_problem_sizes.py --plan           # list the arms, then exit
    python dev_scripts/measure_min_problem_sizes.py --smoke          # quick test shapes on the CPU
    python dev_scripts/measure_min_problem_sizes.py --update-hashes  # print new hashes only

``--plan``, ``--smoke``, and ``--update-hashes`` need no GPU.  The full run needs CUDA.
The run time depends on the plan.  The largest test shapes at three device counts take
hours by themselves.  For this reason, run ``--plan`` first and estimate the time from
the list of arms that it prints.  ``--accept-stale`` is accepted and does nothing.

Environment:
    REFRESH_PYTHON      interpreter for the arm subprocesses (default: this one)
    REFRESH_RESULTS     scratch directory for the staged sinograms and rows
    REFRESH_ITERATIONS  VCD iterations per recon (default 3)
    REFRESH_REPEATS     timed repeats after the discarded first run (default 3)
"""

import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbirtorch import _min_problem_sizes as sizes                      # noqa: E402

# ── configuration ────────────────────────────────────────────────────────────
TORCH_PYTHON = os.environ.get('REFRESH_PYTHON', sys.executable)
RESULTS_DIR = os.environ.get(
    'REFRESH_RESULTS',
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '_min_problem_size_runs'))
ITERATIONS = int(os.environ.get('REFRESH_ITERATIONS', '3'))
WARM_REPEATS = int(os.environ.get('REFRESH_REPEATS', '3'))
SEED = 13

#: The test shapes of every measurement, roughly geometric, largest last.  A minimum
#: problem size is always the size of one of these shapes, so a measurement reports it
#: by naming a test shape.
TEST_SHAPES = [(128, 112, 96), (192, 168, 144), (256, 224, 192), (384, 336, 288),
               (512, 448, 384), (768, 672, 576), (1024, 1008, 992)]
#: The test shapes at which an unmeasured row is measured.  They are the three largest
#: shapes in TEST_SHAPES.  A model type with its own test shapes uses its own three
#: largest instead.  See ``measure_shapes``.
UNMEASURED_ROW_SHAPES = [(512, 448, 384), (768, 672, 576), (1024, 1008, 992)]
#: The quick test shapes are small, so that --smoke runs every code path in seconds on
#: a CPU.
QUICK_TEST_SHAPES = [(8, 12, 16), (12, 12, 16), (16, 12, 16)]

#: The translation grid and spacing for each translation test shape:
#: ``shape -> (num_x, num_z, x_spacing, z_spacing)``.  A translation scan
#: moves the object across a fixed source and detector on a grid, so its
#: views are grid positions rather than angles, and the same sinogram shape
#: could come from many different grids.  The grid is therefore written down
#: here rather than derived, and ``_build_model`` reads it.  The three
#: measured shapes keep production's 16x16 grid and halve the detector and the
#: spacing together; the largest is the production shape mg18 measured, and
#: its grid, spacing and source distances are mg18's.
TRANSLATION_SPECS = {
    (256, 475, 750): (16, 16, 6.0, 4.0),
    (256, 950, 1500): (16, 16, 12.0, 8.0),
    (256, 1900, 3000): (16, 16, 24.0, 16.0),
    (16, 40, 32): (4, 4, 3.0, 2.0),        # the --smoke shape, also mg18's
}
#: The translation test shapes, and the quick test shape that --smoke uses for them.
TRANSLATION_TEST_SHAPES = [(256, 475, 750), (256, 950, 1500), (256, 1900, 3000)]
TRANSLATION_QUICK_TEST_SHAPES = [(16, 40, 32)]

#: The model types that have their own test shapes, and the quick test shapes that
#: --smoke uses for them.  A model type belongs here when TEST_SHAPES does not describe
#: its problems.  A translation scan has few views and a large detector, so no shape in
#: TEST_SHAPES resembles one.
TEST_SHAPES_BY_MODEL = {'translation': TRANSLATION_TEST_SHAPES}
QUICK_TEST_SHAPES_BY_MODEL = {'translation': TRANSLATION_QUICK_TEST_SHAPES}

#: The model types that have no table rows yet, and the device counts to measure for
#: each.  The MIN_SIZES table gains a model type only when a person pastes the output
#: of this program.  For this reason, a model type that has never been measured cannot
#: be planned from the table, so it is named here instead.  The first measurement of it
#: prints its first rows.  Remove a model type from here in the same change that pastes
#: its rows in.  After that, it is planned from the table like every other model type.
#: The denoiser left this set on 2026-08-16, when the mg16 measurement gave it its first
#: rows, which were unmeasured rows.  Translation entered on 2026-08-17 and left the same
#: day, when the mg22 measurement gave it its first unmeasured rows and gave
#: TranslationModel its table key.
UNTABLED_FAMILIES = {}

#: The quantity that a minimum problem size counts, for a model type whose quantity is
#: not the plain sinogram element count.  The program prints this text with the plan and
#: again under the pasted rows.  The text has to end up in the note of the row, and a
#: person must write the note.
FAMILY_METRIC_NOTES = {
    'denoiser': ('QGGMRFDenoiser sets its sinogram shape equal to its image '
                 'shape, so a denoiser minimum problem size is counted in IMAGE '
                 'VOXELS.  Every other model type counts sinogram elements.  '
                 "Say so in the row's note."),
    'translation': ('a translation minimum problem size is counted in sinogram '
                    'elements like the other projection models, but its test '
                    'shapes are not TEST_SHAPES.  They are translation scans '
                    'based on production: a fixed 16x16 translation grid, with '
                    'the detector and the spacing scaled together.  Say so '
                    "in the row's note and name the shape, so that nobody "
                    'reads a translation minimum problem size as a size in '
                    'TEST_SHAPES.'),
}


def planned_rows():
    """This function returns every ``(family, count)`` that a measurement covers.

    These are the rows of the table, plus the counts named for the model types that have
    no table rows."""
    rows = set(sizes.MIN_SIZES)
    for family, counts in UNTABLED_FAMILIES.items():
        rows.update((family, int(count)) for count in counts)
    return sorted(rows)


def resolve_scope(families_arg):
    """The families this run measures: None for every family, else a sorted
    list.  Raises SystemExit on an unknown name, and on a scope that would
    CARRY a family whose cost inputs moved -- carrying a family asserts its
    costs did not change, and a drifted family fails that assertion, so the
    refusal happens here, before any GPU work.
    """
    all_families = sorted({family for family, _count in planned_rows()})
    stale = sizes.stale_families()
    if families_arg is None:
        return None
    if families_arg == 'stale':
        chosen = list(stale)
        if not chosen:
            raise SystemExit(
                'no cost input has moved since the hashes were recorded, so '
                'a stale-scoped refresh has nothing to measure.  Name '
                'families explicitly (--families cone,parallel) for a '
                'deliberate re-measure, or run without --families for the '
                'full refresh.')
    else:
        chosen = sorted({token.strip() for token in families_arg.split(',')
                         if token.strip()})
        unknown = [name for name in chosen if name not in all_families]
        if unknown:
            raise SystemExit(
                'unknown model type{}: {}.  The model types a measurement can '
                'cover are: {}.'.format(
                    '' if len(unknown) == 1 else 's', ', '.join(unknown),
                    ', '.join(all_families)))
    carried_stale = [name for name in stale if name not in chosen]
    if carried_stale:
        raise SystemExit(
            'refusing the scoped refresh: the cost inputs of {} moved, and '
            'this scope would carry {} rows forward as if their costs were '
            'unchanged.  Add {} to --families, or run the full '
            'refresh.'.format(', '.join(carried_stale),
                              'their' if len(carried_stale) > 1 else 'its',
                              ', '.join(carried_stale)))
    return chosen


def elements(shape):
    return int(shape[0]) * int(shape[1]) * int(shape[2])


def shape_for_size(target_elements, test_shapes):
    for shape in test_shapes:
        if elements(shape) == target_elements:
            return shape
    return None


def test_shapes_for(family, smoke=False):
    """This function returns the test shapes at which ``family`` is measured.

    Most model types use TEST_SHAPES.  A model type named in TEST_SHAPES_BY_MODEL uses its
    own test shapes instead."""
    if smoke:
        return QUICK_TEST_SHAPES_BY_MODEL.get(family, QUICK_TEST_SHAPES)
    return TEST_SHAPES_BY_MODEL.get(family, TEST_SHAPES)


def measure_shapes(test_shapes):
    """This function returns the test shapes at which a row with no minimum problem size
    is measured.

    These are the largest shapes in ``test_shapes``.  For TEST_SHAPES, they are exactly
    UNMEASURED_ROW_SHAPES."""
    return list(test_shapes[-len(UNMEASURED_ROW_SHAPES):])


# ── the plan ─────────────────────────────────────────────────────────────────
def comparison_count(family, count):
    """This function returns the best smaller allowed count, which ``count`` has to beat.

    The function reads the count from the table.  An unmeasured row never allows its
    count, so that count is never the comparison.  A row that gains a minimum problem size
    becomes the comparison at the next measurement, as cone n=2 did on 2026-08-10.
    """
    for n in range(count - 1, 1, -1):
        row = sizes.MIN_SIZES.get((family, n))
        # Only a count with its own row and a measured minimum problem size can be the
        # comparison.  A count that uses the row of a larger count was never measured
        # itself, and an unmeasured row never allows its count.
        if row is not None and row.elements is not None:
            return n
    return 1


def build_plan(smoke=False, scope=None):
    """``[{family, count, shape, counts, role}, ...]`` in run order, plus the
    families that have no measured rows at all.  ``scope`` limits the plan to
    the named families; None plans every family."""
    rows = []
    for (family, count) in planned_rows():
        if scope is not None and family not in scope:
            continue
        test_shapes = test_shapes_for(family, smoke)
        table_row = sizes.MIN_SIZES.get((family, count))
        against = comparison_count(family, count)
        arms = sorted({1, against, count})
        if table_row is None or table_row.elements is None:
            # No minimum problem size is known.  A row that has never been measured and
            # an unmeasured row are measured the same way, at the largest test shapes,
            # because neither one says where its crossover is.
            shapes = measure_shapes(test_shapes)
            role = ('first measurement' if table_row is None
                    else 'unmeasured row')
        else:
            at = shape_for_size(table_row.elements, test_shapes)
            if at is None:                       # --smoke, or changed test shapes
                at = test_shapes[len(test_shapes) // 2]
            index = test_shapes.index(at)
            shapes = test_shapes[max(0, index - 1):index + 2]
            role = 'bracket'
        for shape in shapes:
            rows.append(dict(family=family, count=count, shape=list(shape),
                             counts=arms, role=role, against=against,
                             shape_elements=elements(shape)))
    return rows


def unmeasured_families():
    """This function returns the model classes whose automatic device count uses table
    rows that were never measured for them.  The keys are the table keys that the
    classes declare.

    A geometry reaches this state in two ways.  Both are work to do:

      * it declares a ``_size_table_key`` for which the table has no rows, keyed here
        under that name; or
      * it declares no table key, and so it uses the ``sizes.DEFAULT_FAMILY`` rows, keyed
        here under None.

    A newly ported geometry arrives in the second case.
    """
    import mbirtorch
    from mbirtorch.tomography_model import TomographyModel

    seen, known = {}, set(sizes.families())
    for name in dir(mbirtorch):
        cls = getattr(mbirtorch, name)
        if not (isinstance(cls, type) and issubclass(cls, TomographyModel)):
            continue
        # The base class is not a geometry.  A subclass that does not reconstruct
        # through the shared VCD loop never reaches the automatic device-count
        # decision.  QGGMRFDenoiser subclasses TomographyModel but refuses recon, so
        # the minimum problem sizes never apply to it here.
        if cls is TomographyModel or cls.recon is not TomographyModel.recon:
            continue
        family = getattr(cls, '_size_table_key', None)
        if family is None or family not in known:
            # An exported alias and its class are the same object, so record
            # the class's own name once rather than once per exported name.
            seen.setdefault(family, set()).add(cls.__name__)
    return {family: sorted(names) for family, names in seen.items()}


def print_plan(plan, smoke, scope=None):
    test_shapes = QUICK_TEST_SHAPES if smoke else TEST_SHAPES
    print('minimum problem size measurement plan ({}), interpreter {}'.format(
        'SMOKE, CPU' if smoke else 'CUDA', TORCH_PYTHON))
    if scope is not None:
        carried = sorted({family for family, _count in planned_rows()}
                         - set(scope))
        print('  SCOPED: measuring {}; carrying verbatim: {}'.format(
            ', '.join(scope), ', '.join(carried) or 'nothing'))
    print('  test shapes: ' + ', '.join(
        '{} ({:,})'.format(c, elements(c)) for c in test_shapes))
    # A model type with its own test shapes prints them too, so the plan never shows a
    # shape that is absent from the lists printed above it.
    for family in sorted({row['family'] for row in plan} & set(TEST_SHAPES_BY_MODEL)):
        print('  {} test shapes: '.format(family) + ', '.join(
            '{} ({:,})'.format(c, elements(c))
            for c in test_shapes_for(family, smoke)))
    print()
    # The family column is wide enough for the longest family name, so a long
    # one does not push the rest of the row out of alignment.
    header = '{:>11}{:>4}{:>9}{:>20}{:>16}{:>18}'.format(
        'family', 'n', 'against', 'shape', 'sino elements', 'role')
    print(header)
    for row in plan:
        print('{:>11}{:>4}{:>9}{:>20}{:>16,}{:>18}'.format(
            row['family'], row['count'], row['against'], str(tuple(row['shape'])),
            row['shape_elements'], row['role']))
    arms = sum(len(row['counts']) for row in plan)
    shapes = {(row['family'], tuple(row['shape'])) for row in plan}
    print('\n{} rows, {} distinct shapes, {} timed arms + {} generators'.format(
        len(plan), len(shapes), arms, len(shapes)))
    for (family, count) in sorted({(r['family'], r['count']) for r in plan}):
        table_row = sizes.MIN_SIZES.get((family, count))
        # Compare against the largest test shape of this model type, because a model
        # type with its own test shapes never runs the largest shape in TEST_SHAPES.
        top = elements(test_shapes_for(family, smoke)[-1])
        # Count the shapes that this row actually got, because a short list of test
        # shapes gives fewer than three.
        probes = sum(1 for r in plan
                     if r['family'] == family and r['count'] == count)
        if table_row is None:
            print('  NOTE {} n={}: the MIN_SIZES table has no row for this '
                  'model type yet, so this run is looking for its first '
                  'minimum problem size at the {} shapes above.  Until a row is '
                  'pasted in, the {} rows apply to it.'.format(
                      family, count, probes, sizes.DEFAULT_FAMILY))
        elif table_row.elements is None:
            print('  NOTE {} n={}: UNMEASURED ROW.  Measured at the {} shapes '
                  'above; sizes above largest_tested ({:,} sinogram '
                  'elements) are NOT measured by this plan.'.format(
                      family, count, probes, table_row.largest_tested))
        elif table_row.elements >= top:
            print('  NOTE {} n={}: its minimum problem size is the LARGEST test '
                  'shape, so the bracket has no shape above it.  A measurement '
                  'can confirm the minimum problem size, but it cannot lower it '
                  'without a larger test shape.'.format(family, count))
    planned = {row['family'] for row in plan}
    for family in sorted(planned & set(FAMILY_METRIC_NOTES)):
        print('  METRIC {}: {}'.format(family, FAMILY_METRIC_NOTES[family]))
    missing = unmeasured_families()
    if missing:
        # The None key sorts first.  A class that declares no table key uses the
        # default rows without any message, so it is the case to read first.
        for family, classes in sorted(
                missing.items(),
                key=lambda item: (item[0] is not None, item[0] or '')):
            if family is None:
                print('  NEEDS MEASUREMENT: {} declare no table key, so the '
                      '{} rows set their automatic device count.'
                      .format(', '.join(classes), sizes.DEFAULT_FAMILY))
            else:
                print('  NEEDS MEASUREMENT: table key {!r} is declared by {} '
                      'but has no rows; the {} rows currently apply.'
                      .format(family, ', '.join(classes), sizes.DEFAULT_FAMILY))
    else:
        print('  every model class uses table rows measured for it.')


# ── the worker: one arm, one subprocess ──────────────────────────────────────
#: The noise standard deviation the denoiser family is measured at, as a
#: fraction of the phantom's unit dynamic range.  It is passed to ``denoise``
#: rather than estimated, so no arm spends time on the host-side estimate,
#: which does no device work and would add a size-dependent constant to every
#: reading.
DENOISE_SIGMA = 0.1


def _build_model(family, shape, device, n_dev=None):
    """The model one arm times.

    ``n_dev`` is the arm's device count.  It matters only for the denoiser
    family, which is placed explicitly; see the denoiser branch below.
    """
    import numpy as np

    import mbirtorch

    num_views, _rows, num_channels = shape
    if family == 'cone':
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirtorch.ConeBeamModel(
            tuple(shape), angles, source_detector_dist=4.0 * num_channels,
            source_iso_dist=2.0 * num_channels)
    elif family == 'parallel':
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(tuple(shape), angles)
    elif family == 'multiaxis':
        # Two angles per view: azimuth around the object, elevation (tilt)
        # out of the plane.  These are the geometry's own test defaults, and
        # mg18 measured this family with them -- azimuths evenly spaced over
        # half a turn, elevations swept across +/- 0.5 radians.  The
        # elevation range sets the recon shape: the automatic geometry
        # divides the detector height by the smallest |cos(elevation)| and
        # clamps that divisor at 0.1, so a range wide enough to reach the
        # clamp would inflate the slice count roughly tenfold.  0.5 radians
        # is far from the clamp.
        azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
        elevation = np.linspace(-0.5, 0.5, num_views)
        model = mbirtorch.MultiAxisParallelModel(
            tuple(shape), np.stack([azimuth, elevation], axis=1))
    elif family == 'translation':
        # A translation scan moves the object across a fixed source and
        # detector on a grid, so its views are grid positions rather than
        # angles.  The shape does not say what that grid is, so the grid and
        # the spacing come from TRANSLATION_SPECS.  The construction below is
        # mg18's, including the source distances.
        spec = TRANSLATION_SPECS.get(tuple(shape))
        if spec is None:
            raise ValueError(
                'measure_min_problem_sizes has no translation grid recorded '
                'for shape {}.  Add it to TRANSLATION_SPECS before measuring '
                'that shape.'.format(tuple(shape)))
        num_x, num_z, x_spacing, z_spacing = spec
        vectors = mbirtorch.gen_translation_vectors(
            num_x, num_z, x_spacing=x_spacing, z_spacing=z_spacing)
        if vectors.shape[0] != num_views:
            raise ValueError(
                'translation shape {} has {} views, but its {}x{} grid gives '
                '{} translations.'.format(tuple(shape), num_views, num_x,
                                          num_z, vectors.shape[0]))
        # Both distances are half the smaller detector extent, as in mg18.
        source_dist = min(shape[1], shape[2]) / 2
        model = mbirtorch.TranslationModel(
            tuple(shape), vectors, source_detector_dist=source_dist,
            source_iso_dist=source_dist)
    elif family == 'denoiser':
        # The denoiser's input is an image, not a sinogram, and its
        # sinogram_shape is set equal to that image shape, so the shape IS the
        # image shape here.  Its minimum problem sizes are therefore counted in
        # image voxels.  Every other model type counts sinogram elements, and
        # the rows that this program prints have to record that.
        model = mbirtorch.QGGMRFDenoiser(tuple(shape))
        if device == 'cuda':
            # The device count of the arm is set explicitly.  The denoiser
            # rows were first measured this way on 2026-08-16, before denoise
            # used the automatic device choice.  An explicit device list still
            # fixes the count, because mbirtorch never changes a device list
            # that the user sets.  The environment variable would also work,
            # because it skips the minimum problem sizes.  The explicit list
            # keeps the measurement method the same across measurements.
            # Each arm reports realized_devices, so the run output shows a
            # count that did not take effect.
            model.configure_devices(
                devices=['cuda:{}'.format(i) for i in range(n_dev or 1)])
    else:
        # Falling through to parallel beam here would time parallel beam and
        # record the result under this model type's name.  The minimum problem
        # size would then be wrong, and nothing would show the error.
        raise ValueError(
            'measure_min_problem_sizes cannot build a model for model type '
            '{!r}.  Add its geometry to _build_model before measuring it.'
            .format(family))
    if device != 'cuda':
        # This applies to CPU and MPS only.  MBIRTORCH_NUM_DEVICES acts only on
        # CUDA, because the automatic choice stops early below two visible
        # devices.  For this reason, --smoke places the model by hand.  On
        # CUDA, nothing is configured, so the model stays on the automatic
        # device choice, which is where the environment variable acts.  The
        # denoiser branch above is the exception, and it is placed there.
        model.configure_devices(devices=[device])
    model.set_params(no_warning=True, verbose=0)
    return model


def _input_path(family, shape):
    """Where one shape's staged input array lives: a sinogram for the
    projection families, a noisy image for the denoiser."""
    return os.path.join(RESULTS_DIR, '_sino_{}_{}x{}x{}.npy'.format(
        family, *shape))


def generate(cfg):
    """Stage one input array per shape, at ONE device, so every arm of that
    shape works on the same array.

    A projection family's input is the forward projection of a phantom.  The
    denoiser has no projectors, so its input is that phantom plus seeded
    gaussian noise -- the array a denoiser is actually given.
    """
    import numpy as np

    import mbirtorch

    model = _build_model(cfg['family'], cfg['shape'],
                         'cpu' if cfg['device'] != 'cuda' else 'cuda',
                         n_dev=1)
    recon_shape = tuple(model.get_params('recon_shape'))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    # The shepp-logan builder places its ellipsoids as fractions of the
    # volume, and on a volume only a few voxels deep every one of them can
    # miss, leaving the phantom all zeros.  The --smoke translation shape
    # reconstructs a volume that shallow.  An all-zero phantom forward
    # projects to an all-zero sinogram, so every arm of that shape would time
    # a reconstruction of nothing.  A seeded uniform volume has the same
    # shape and a comparable dynamic range, and the generator row records
    # that it was used.
    phantom_fallback = None
    if float(np.max(phantom)) == 0.0:
        phantom = np.asarray(np.random.RandomState(SEED).rand(*recon_shape),
                             dtype=np.float32)
        phantom_fallback = 'seeded uniform (shepp-logan returned all zeros)'
    if cfg['family'] == 'denoiser':
        noise = np.random.RandomState(SEED).randn(*recon_shape)
        staged = np.asarray(phantom + DENOISE_SIGMA * noise, dtype=np.float32)
    else:
        staged = np.asarray(_to_numpy(model.forward_project(phantom)),
                            dtype=np.float32)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(_input_path(cfg['family'], cfg['shape']),
            np.ascontiguousarray(staged))
    return dict(cfg, role='generator', recon_shape=list(recon_shape),
                sino_shape=list(staged.shape),
                phantom_fallback=phantom_fallback)


def _to_numpy(x):
    """The one host exit.  ``Shards.gather()`` ALREADY returns numpy, so a
    gather is never followed by ``.detach()`` -- re-detaching one is a
    recorded way to lose every multi-device row in a run."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    if callable(getattr(x, 'gather', None)) and hasattr(x, 'placement'):
        return x.gather()
    return (x.detach().cpu().numpy()
            if callable(getattr(x, 'detach', None)) else np.asarray(x))


def measure(cfg):
    """One arm: cold pass discarded, then the warm median of REPEATS runs."""
    import numpy as np
    import torch

    family = cfg['family']
    model = _build_model(family, cfg['shape'], cfg['device'],
                         n_dev=cfg.get('n_dev'))
    staged = np.load(_input_path(family, cfg['shape']))
    weights = (None if family == 'denoiser' else
               np.exp(-staged / (2 * np.max(staged))).astype(np.float32))

    def one():
        np.random.seed(SEED)
        if family == 'denoiser':
            # The denoiser's own entry point; recon raises on it.  The
            # iteration count and the exact-iteration stop match the
            # reconstruction call below, so both families are timed over the
            # same amount of work per run.
            out, _info = model.denoise(staged, sigma_noise=DENOISE_SIGMA,
                                       max_iterations=ITERATIONS,
                                       stop_threshold_change_pct=0.0)
        else:
            out, _info = model.recon(staged, weights=weights,
                                     max_iterations=ITERATIONS,
                                     stop_threshold_change_pct=0.0)
        if cfg['device'] == 'cuda':
            # Both placements name the same device list; the recon one is
            # named because it is the one the denoiser divides its image on.
            for device in model.recon_placement.devices:
                torch.cuda.synchronize(device)
        return _to_numpy(out)

    start = time.perf_counter()
    out = one()
    cold = time.perf_counter() - start
    warm = []
    for _ in range(WARM_REPEATS):
        start = time.perf_counter()
        out = one()
        warm.append(time.perf_counter() - start)
    median = statistics.median(warm)
    return dict(cfg, role='arm', cold_s=cold, warm_all=warm, warm_s=median,
                spread=(max(warm) - min(warm)) / median,
                realized_devices=len(model.recon_placement.devices),
                recon_checksum=float(np.sum(np.abs(out), dtype=np.float64)))


# ── the runner ───────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The environment that DEFINES an arm, set explicitly so nothing is
    inherited from the submitting shell."""
    env = dict(os.environ)
    env.pop('MBIRTORCH_MEMORY_CALIBRATION', None)
    env.pop('MBIRTORCH_NUM_DEVICES', None)
    env.pop('MBIRTORCH_LIMIT_GPUS_BY_SIZE', None)   # MBIRTORCH_NUM_DEVICES skips it
    env['MBIRTORCH_DISABLE_TRITON'] = '0'       # production: kernels on
    if cfg.get('n_dev') and cfg['device'] == 'cuda':
        env['MBIRTORCH_NUM_DEVICES'] = str(cfg['n_dev'])
    return env


def run_one(cfg, tag):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, '_cfg_{}.json'.format(tag))
    out_path = os.path.join(RESULTS_DIR, '_out_{}.json'.format(tag))
    with open(cfg_path, 'w') as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    proc = subprocess.run([TORCH_PYTHON, '-u', os.path.abspath(__file__),
                           '_worker', cfg_path, out_path], env=arm_env(cfg))
    if proc.returncode != 0 and not os.path.exists(out_path):
        return dict(cfg, error='worker exited {}'.format(proc.returncode))
    with open(out_path) as handle:
        return json.load(handle)


def _worker_main(cfg_path, out_path):
    with open(cfg_path) as handle:
        cfg = json.load(handle)
    try:
        row = generate(cfg) if cfg['role'] == 'generator' else measure(cfg)
    except Exception:                                             # noqa: BLE001
        row = dict(cfg, error=traceback.format_exc()[-3000:])
    with open(out_path, 'w') as handle:
        json.dump(row, handle)


def run_plan(plan, device):
    """Every generator, then every arm.  Returns ``{(family, shape, n): row}``."""
    shapes = sorted({(row['family'], tuple(row['shape'])) for row in plan})
    for family, shape in shapes:
        cfg = dict(family=family, shape=list(shape), device=device,
                   role='generator')
        tag = 'gen_{}_{}x{}x{}'.format(family, *shape)
        print('  generator {} {} ...'.format(family, shape), flush=True)
        row = run_one(cfg, tag)
        if row.get('error'):
            print('    ERROR: {}'.format(str(row['error'])[:400]))
        elif row.get('phantom_fallback'):
            print('    phantom: {}'.format(row['phantom_fallback']))

    measured = {}
    arms = sorted({(row['family'], tuple(row['shape']), n)
                   for row in plan for n in row['counts']})
    for index, (family, shape, n_dev) in enumerate(arms):
        cfg = dict(family=family, shape=list(shape), device=device,
                   role='arm', n_dev=n_dev)
        tag = 'arm_{}_{}x{}x{}_n{}'.format(family, *shape, n_dev)
        print('  [{}/{}] {} {} n={} ...'.format(index + 1, len(arms), family,
                                                shape, n_dev), flush=True)
        row = run_one(cfg, tag)
        measured[(family, shape, n_dev)] = row
        if row.get('error'):
            print('    ERROR: {}'.format(str(row['error'])[:400]))
        else:
            print('    warm {:.3f}s  spread {:.1%}  realized {} device(s)'
                  .format(row['warm_s'], row['spread'],
                          row['realized_devices']))
    return measured


# ── the analysis ─────────────────────────────────────────────────────────────
def speedup(measured, family, shape, count, against):
    """This function returns the median time at ``against`` divided by the median time at
    ``count``.  A value above 1.0 means that the larger count is faster."""
    a = measured.get((family, shape, against))
    b = measured.get((family, shape, count))
    if not a or not b or a.get('error') or b.get('error'):
        return None, None
    spread = max(a['spread'], b['spread'])
    return a['warm_s'] / b['warm_s'], spread


def verdict(plan, measured):
    """This function returns, for each table row, the speedup at each test shape and the
    winning test shape.

    A test shape wins only when the speedup reaches ``sizes.MIN_SPEEDUP`` and exceeds 1.0x by
    more than the run-to-run spread of that test shape.  The required speedup keeps a
    minimum problem size valid across small changes in hardware and shape.  The spread
    condition prevents one noisy test shape from moving a minimum problem size.  A win
    that exceeds the spread but not ``sizes.MIN_SPEEDUP`` is recorded as ``thin``, so the
    printout can list it.
    """
    out = {}
    # The rows come from the plan rather than the table, so a model type with no
    # table rows is judged too.
    for (family, count) in sorted({(r['family'], r['count']) for r in plan}):
        shapes = [tuple(row['shape']) for row in plan
                  if row['family'] == family and row['count'] == count]
        against = comparison_count(family, count)
        rows, winner = [], None
        for shape in shapes:
            ratio, spread = speedup(measured, family, shape, count, against)
            clears_spread = ratio is not None and ratio - 1.0 > spread
            wins = clears_spread and ratio >= sizes.MIN_SPEEDUP
            rows.append((shape, ratio, spread, wins))
            if wins and winner is None:
                winner = shape
        thin = [shape for shape, ratio, spread, wins in rows
                if not wins and ratio is not None and ratio - 1.0 > spread
                and (winner is None or elements(shape) < elements(winner))]
        out[(family, count)] = dict(against=against, rows=rows, winner=winner,
                                    thin=thin)
    return out


def print_verdict(verdicts):
    for (family, count), record in sorted(verdicts.items()):
        table_row = sizes.MIN_SIZES.get((family, count))
        print('\n===== {} n={}  (crossover against n={}) ====='.format(
            family, count, record['against']))
        for shape, ratio, spread, wins in record['rows']:
            if ratio is None:
                print('  {:>20} {:>16,}   (no measurement)'.format(
                    str(shape), elements(shape)))
                continue
            if wins:
                label = 'WINS'
            elif ratio - 1.0 > spread and ratio < sizes.MIN_SPEEDUP:
                label = ('thin: clears the spread, under the {:.2f}x '
                         'margin'.format(sizes.MIN_SPEEDUP))
            else:
                label = 'loses'
            print('  {:>20} {:>16,}   {:.3f}x  spread {:.1%}   {}'.format(
                str(shape), elements(shape), ratio, spread, label))
        largest = max(elements(c) for c, _r, _s, _w in record['rows'])
        winner = record['winner']
        proposed = elements(winner) if winner else None
        # A model type with no table row is in the same state as an unmeasured
        # row, because no minimum problem size is known for either one.  The two
        # are reported separately, so the reader can tell a first measurement
        # from a new measurement.
        if winner is None and record['thin']:
            print('  NO SHAPE REACHES THE {:.2f}x SPEEDUP, but {} exceed the '
                  'spread.  A thin win moves the minimum problem size up to '
                  'the next test shape, and the shape above the largest '
                  'measured shape is unmeasured.  This run cannot place the '
                  'minimum problem size.  Run again with larger test shapes, '
                  'or a person decides what the row becomes.'.format(
                      sizes.MIN_SPEEDUP,
                      ', '.join(str(c) for c in record['thin'])))
        elif winner is None and table_row is None:
            print('  no minimum problem size in these shapes -> {} n={} still '
                  'has no row; sizes above {:,} elements are unmeasured.'.format(
                      family, count, largest))
        elif winner is None and table_row.elements is None:
            print('  no minimum problem size in these shapes -> the row stays '
                  'an UNMEASURED ROW; sizes above {:,} sinogram elements are '
                  'still unmeasured.'.format(largest))
        elif winner is None:
            print('  NOTHING WON: the minimum problem size is above every shape '
                  'measured ({:,} sinogram elements).  Run again with larger '
                  'test shapes.  This run cannot place a minimum problem '
                  'size.'.format(largest))
        elif table_row is None:
            print('  FIRST MINIMUM PROBLEM SIZE: n={} beats 1.0x by more than '
                  'its spread at {}.  Proposed minimum problem size: {:,} '
                  'elements.  Paste the row below, then remove {!r} from '
                  'UNTABLED_FAMILIES.'.format(count, winner, proposed, family))
        elif table_row.elements is None:
            print('  UNMEASURED ROW MEASURED: n={} beats 1.0x by more than its '
                  'spread at {}.  Proposed minimum problem size: {:,} sinogram '
                  'elements.'.format(count, winner, proposed))
        elif proposed != table_row.elements:
            print('  MINIMUM PROBLEM SIZE CHANGES: {:,} -> {:,} sinogram '
                  'elements.'.format(table_row.elements, proposed))
        else:
            print('  minimum problem size unchanged at {:,} sinogram '
                  'elements.'.format(proposed))
        if winner is not None and record['thin']:
            print('  thin wins below the {:.2f}x speedup were not counted at: '
                  '{}.'.format(sizes.MIN_SPEEDUP,
                               ', '.join(str(c) for c in record['thin'])))


_ROW_TEMPLATE = """    ('{family}', {count}): MinSizeRow(
        family='{family}', count={count}, elements={elements}, shape={shape},
        against={against},
        bracket=Bracket(losing_shape={low_shape}, losing_speedup={low_x},
                        winning_shape={shape}, winning_speedup={win_x}),
        spread={spread:.4g}, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='{today}', commit='{commit}',
        largest_tested={largest},
        note='...'),"""


def _source_chunks(text, width=60):
    """``text`` split at spaces into pieces at most ``width`` characters,
    each keeping its trailing space, so joining the pieces reproduces the
    text exactly.  Used to print a carried row's note as wrapped source."""
    chunks, current = [], ''
    for word in text.split(' '):
        candidate = word if not current else current + ' ' + word
        if current and len(candidate) > width:
            chunks.append(current + ' ')
            current = word
        else:
            current = candidate
    chunks.append(current)
    assert ''.join(chunks) == text
    return chunks


def _render_carried_row(family, count, table_row):
    """One CURRENT table row, printed back in source form.  Every field
    passes through verbatim -- elements, bracket, spread, dates, commit, and
    the note -- so a scoped run's paste never claims a measurement that did
    not happen."""
    gpu = ('MEASURED_GPU' if table_row.gpu == sizes.MEASURED_GPU
           else repr(table_row.gpu))
    config = ('MEASURED_CONFIG' if table_row.config == sizes.MEASURED_CONFIG
              else repr(table_row.config))
    elements_text = ('{:_}'.format(table_row.elements)
                     if table_row.elements is not None else 'None')
    largest_text = ('{:_}'.format(table_row.largest_tested)
                    if table_row.largest_tested is not None else 'None')
    lines = [
        "    ('{}', {}): MinSizeRow(".format(family, count),
        "        family='{}', count={}, elements={}, shape={},".format(
            family, count, elements_text, table_row.shape),
        "        against={},".format(table_row.against),
        "        bracket=Bracket(losing_shape={}, losing_speedup={},".format(
            table_row.bracket.losing_shape, table_row.bracket.losing_speedup),
        "                        winning_shape={}, winning_speedup={}),".format(
            table_row.bracket.winning_shape, table_row.bracket.winning_speedup),
        "        spread={}, gpu={}, config={},".format(
            table_row.spread, gpu, config),
        "        measured={!r}, commit={!r},".format(
            table_row.measured, table_row.commit),
        "        largest_tested={},".format(largest_text),
    ]
    chunks = _source_chunks(table_row.note)
    lines.append("        note={!r}".format(chunks[0])
                 + ('' if len(chunks) == 1 else ''))
    for chunk in chunks[1:]:
        lines.append("             {!r}".format(chunk))
    lines[-1] += '),'
    return '\n'.join(lines)


def print_table(verdicts, commit, carried_families=()):
    """The paste-ready replacement for MIN_SIZES, provenance included.  Measured
    families print from this run's verdicts with their notes left to write;
    carried families print their CURRENT rows verbatim, so the paste is
    always the whole table."""
    today = datetime.date.today().isoformat()
    print('\n' + '=' * 78)
    print('PASTE INTO mbirtorch/_min_problem_sizes.py (MIN_SIZES, then the three')
    print('bound constants printed by --update-hashes).  All of it, or none of it.')
    if carried_families:
        print('measured this run: {}.  carried verbatim (their cost inputs '
              'are unmoved): {}.'.format(
                  ', '.join(sorted({f for f, _c in verdicts})),
                  ', '.join(sorted(carried_families))))
    print('=' * 78 + '\nMIN_SIZES = {')
    # Carried rows keep the CURRENT table's own ordering, printed first so
    # the paste groups this run's measured rows together at the end.
    for (family, count), table_row in sizes.MIN_SIZES.items():
        if family in carried_families:
            print(_render_carried_row(family, count, table_row))
    for (family, count), record in sorted(verdicts.items()):
        winner, rows = record['winner'], record['rows']
        found = elements(winner) if winner else None
        # The bracket's low side is the largest shape BELOW the winner that
        # lost.  A shape above the winner may also read under 1.0x on noise,
        # and calling that the losing side would invert the bracket.
        below = [(c, r) for c, r, _s, wins in rows if r and not wins
                 and (found is None or elements(c) < found)]
        low = below[-1] if below else (None, None)
        win_ratio = next((r for c, r, _s, _w in rows if c == winner), None)
        print(_ROW_TEMPLATE.format(
            family=family, count=count, against=record['against'],
            elements='{:_}'.format(found) if found else 'None',
            shape=winner if winner else 'None',
            low_shape=low[0] if low[0] else 'None',
            low_x='{:.2f}'.format(low[1]) if low[1] else 'None',
            win_x='{:.2f}'.format(win_ratio) if win_ratio else 'None',
            spread=max([s for _c, _r, s, _w in rows if s is not None] or [0.0]),
            today=today, commit=commit,
            # This field is the largest shape of THIS run, never that of the
            # old row.  The field exists to prevent a claim about a size that
            # this measurement did not measure.
            largest='{:_}'.format(max(elements(c)
                                      for c, _r, _s, _w in rows))))
    print('}')
    print('\nCheck MEASURED_GPU and MEASURED_CONFIG still describe this run, '
          'and rewrite each note by hand: a note is the one field a machine '
          'cannot fill in.')
    for family in sorted({f for f, _c in verdicts} & set(FAMILY_METRIC_NOTES)):
        print('\n{}: {}'.format(family, FAMILY_METRIC_NOTES[family]))
    new = sorted({f for f, _c in verdicts} & set(UNTABLED_FAMILIES))
    if new:
        print('\nThese families are new to the table: {}.  Drop each from '
              "UNTABLED_FAMILIES in this script once its rows are pasted in, "
              'so it is planned from the table afterwards.'.format(
                  ', '.join(new)))


def head_commit():
    try:
        out = subprocess.run(['git', 'log', '-1', '--format=%h'],
                             capture_output=True, text=True, timeout=10,
                             cwd=os.path.dirname(os.path.dirname(
                                 os.path.abspath(__file__))))
        return out.stdout.strip() or 'unknown'
    except Exception:                                             # noqa: BLE001
        return 'unknown'


def print_hash_paste_help(stale):
    """This function prints the steps to paste the block that --update-hashes printed, and
    the steps to check the paste.

    Nothing here writes _min_problem_sizes.py.  Every value in that file sits
    beside a hand-written comment recording what measured it and when, and a
    writer that rewrote the file would have to preserve those comments or
    silently drop them.  So the paste is by hand, and these are the steps.
    """
    print('\n' + '-' * 78)
    print('HOW TO PASTE')
    print('-' * 78)
    print('In mbirtorch/_min_problem_sizes.py, replace three things with the')
    print('block above:')
    print()
    print('  1. the whole MEASURED_CODE_HASHES = {...} dict.  Usually only a')
    print('     few entries differ, but replacing the dict whole is always')
    print('     right and takes the same time;')
    print('  2. the STALE_SINCE = ... line;')
    print('  3. the TABLE_CHECKSUM = ... value.  The file wraps it after the')
    print("     '=' with a backslash; either form is fine.")
    print()
    print('Leave MIN_SIZES alone.  --update-hashes measures nothing, so no row and no')
    print('row note moves.  A full refresh prints its own MIN_SIZES block to')
    print('paste beside these three.')
    print()
    print('Then extend the comment above MEASURED_CODE_HASHES to say WHEN you')
    print('re-recorded and WHY.  That comment is the only place a reader can')
    print('learn whether a hash moved because the minimum problem sizes were')
    print('measured again or')
    print('because someone edited a comment in a priced file.  No hash')
    print('distinguishes the two, and the difference decides whether the')
    print('measurements above still describe the code.')
    if stale:
        print('  Here it needs to cover: {}.'.format(
            ', '.join(name for name, _w, _g in stale)))
    print()
    print('Finally, check that the paste took:')
    print()
    print('    python -c "import mbirtorch._min_problem_sizes as sizes; '
          'print(sizes.stale_note())"')
    print('    python -m pytest tests/test_min_problem_sizes.py')
    print()
    print('The first should print None.  A string instead means the hashes')
    print('did not all land.  A test failure naming TABLE_CHECKSUM means the')
    print('checksum and the values it binds disagree, which is exactly what')
    print('that check exists to catch: paste all three, or none of them.')


def do_update_hashes():
    """This function prints new values of MEASURED_CODE_HASHES, STALE_SINCE, and
    TABLE_CHECKSUM for a person to paste into ``mbirtorch/_min_problem_sizes.py``.  It
    measures nothing."""
    stale = sizes.stale_cost_inputs()
    print('projection-cost inputs that moved since the hashes were last '
          'recorded: {}'.format(
              ', '.join(name for name, _w, _g in stale) or 'none'))
    if stale:
        print('model types that use those inputs: {}'.format(
            ', '.join(sizes.stale_families())))
    print('\nRecording the current hashes after a measurement run.  If you')
    print('have NOT just re-measured, doing this hides a real change from')
    print('the automatic staleness note; leave the old hashes in place')
    print('instead and let the note stand until the minimum problem sizes')
    print('are measured again.\n')
    print('=' * 78)
    print('PASTE INTO mbirtorch/_min_problem_sizes.py.  All of it, or none of')
    print('it: the checksum binds these three constants together.')
    print('=' * 78)
    print(sizes.hash_update_lines(stale_since=None))
    print_hash_paste_help(stale)
    return 0


def do_accept_stale():
    """``--accept-stale``, kept parseable and made a no-op.

    It used to stamp STALE_SINCE so the test would go green.  The test no
    longer goes red over drift and the note no longer needs a stamp, so the
    flag has nothing left to do.
    """
    print('--accept-stale: no longer needed -- drift is detected and logged '
          'automatically; run a real refresh to clear the note.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='measure the minimum problem sizes for multi-GPU '
                    'reconstruction')
    parser.add_argument('--plan', action='store_true',
                        help='print the arm list and exit (no GPU needed)')
    parser.add_argument('--smoke', action='store_true',
                        help='run the quick test shapes on the CPU to check '
                             'that every code path works (no GPU needed)')
    parser.add_argument('--update-hashes', action='store_true',
                        help='print new values of MEASURED_CODE_HASHES, '
                             'STALE_SINCE, and TABLE_CHECKSUM to paste into '
                             'mbirtorch/_min_problem_sizes.py; measures '
                             'nothing (no GPU needed)')
    parser.add_argument('--accept-stale', action='store_true',
                        help='accepted and ignored: staleness is detected and '
                             'logged automatically, so there is nothing to '
                             'acknowledge by hand')
    parser.add_argument('--families', nargs='?', const='stale', default=None,
                        metavar='NAMES',
                        help='measure only these model types (comma-'
                             'separated); with no value, exactly the model types '
                             'whose recorded cost inputs moved.  Every other '
                             "family's rows are carried into the paste "
                             'verbatim, and the run refuses a scope that '
                             'would carry a family whose inputs moved')
    args = parser.parse_args(argv)

    # Checked before --update-hashes: the pair used to mean "record the hashes
    # without measuring", and someone typing it from memory should be told it
    # is unnecessary rather than quietly re-record the hashes for real.
    if args.accept_stale:
        return do_accept_stale()
    if args.update_hashes:
        return do_update_hashes()

    scope = resolve_scope(args.families)
    carried = (() if scope is None else
               sorted({family for family, _count in planned_rows()}
                      - set(scope)))
    plan = build_plan(smoke=args.smoke, scope=scope)
    if args.plan:
        print_plan(plan, args.smoke, scope)
        return 0

    device = 'cpu' if args.smoke else 'cuda'
    if not args.smoke:
        import torch
        if not torch.cuda.is_available():
            print('a real refresh needs CUDA; use --smoke to test the '
                  'plumbing on this machine.')
            return 2
        print('{} CUDA device(s) visible'.format(torch.cuda.device_count()))
    else:
        print('SMOKE: quick test shapes on the CPU.  MBIRTORCH_NUM_DEVICES '
              'acts only on CUDA, so every arm uses ONE device here and the '
              'speedups are all 1.0x.  This run checks the code paths '
              '(batching, subprocess environment, analysis, table printing), '
              'not the minimum problem sizes.')

    print_plan(plan, args.smoke, scope)
    print('\nrunning ...')
    started = time.time()
    measured = run_plan(plan, device)
    verdicts = verdict(plan, measured)
    print_verdict(verdicts)
    print_table(verdicts, head_commit(), carried)
    print('\nthen: python dev_scripts/measure_min_problem_sizes.py --update-hashes')
    print('elapsed {:.1f} min; rows under {}'.format(
        (time.time() - started) / 60, RESULTS_DIR))
    return 0


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '_worker':
        _worker_main(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main())

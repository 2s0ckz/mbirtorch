"""Gate for copy_ct_model.

copy_ct_model keeps the parent's reconstruction geometry and re-derives only
the axes whose inputs changed.  The test here covers the case where only the
views change, on cone and parallel geometries: the pitch, the aspect ratio, the recon
shape and the slice offset must all survive the copy, with no warning.
"""

import warnings

import numpy as np
import pytest

import mbirtorch


def _cone_model(num_det_rows=24, num_det_cols=48, helical_z_shifts=None):
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ConeBeamModel((8, num_det_rows, num_det_cols), angles, source_detector_dist=200.0,
                                    source_iso_dist=100.0, helical_z_shifts=helical_z_shifts,
                                    compile_mode='off')
    model.set_params(no_warning=True, verbose=0)
    return model


def _parallel_model(num_det_rows=24, num_det_cols=48):
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    model = mbirtorch.ParallelBeamModel((8, num_det_rows, num_det_cols), angles, compile_mode='off')
    model.set_params(no_warning=True, verbose=0)
    return model


def _shape(model):
    return tuple(int(n) for n in model.get_params('recon_shape'))


def _copy_warnings(caught):
    return [w for w in caught if 'copy_ct_model' in str(w.message)]


@pytest.mark.parametrize('make_model', [_cone_model, _parallel_model], ids=['cone', 'parallel'])
def test_copy_keeps_the_parents_geometry_when_only_the_views_change(make_model):
    """A plain copy, and a copy over a subset of the views, keep every hand-set value: the pitch, the
    aspect ratio, the recon shape and (cone) the slice offset, and warn about nothing."""
    parent = make_model()
    required, optional, _ = parent.get_all_params()
    rows, cols, slices = _shape(parent)
    hand_set = dict(delta_voxel=0.74 * float(parent.get_params('delta_voxel')), voxel_row_aspect=1.25,
                    recon_shape=(rows + 1, cols - 3, max(1, slices - 1)))
    if 'recon_slice_offset' in optional:
        hand_set['recon_slice_offset'] = 1.4
    parent.set_params(no_warning=True, **hand_set)

    view_key = 'translation_vectors' if 'translation_vectors' in required else 'angles'
    subset = np.asarray(required[view_key])[::2]
    subset_kwargs = {'new_' + view_key: subset}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        copies = [mbirtorch.copy_ct_model(parent), mbirtorch.copy_ct_model(parent, **subset_kwargs)]
    assert not _copy_warnings(caught)
    for copy, num_views in zip(copies, (8, 4)):
        assert tuple(copy.get_params('sinogram_shape')) == (num_views, 24, 48)
        assert _shape(copy) == hand_set['recon_shape']
        for name in ('delta_voxel', 'voxel_row_aspect', 'recon_slice_offset'):
            if name in hand_set:
                assert float(copy.get_params(name)) == pytest.approx(hand_set[name])

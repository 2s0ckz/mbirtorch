"""export_recon_hdf5 and import_recon_hdf5: the file layout, the round trip of the volume and
the recon_dict, old files, and the viewer's reader and writer."""
import warnings

import h5py
import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _sharding, utilities
from mbirtorch.viewers import slice_figure


def _volume(shape=(6, 5, 4), seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _recon_dict():
    return {'recon_params': {'iterations': 10, 'angles': np.linspace(0, np.pi, 2000),
                             'sharpness': np.float32(0.5), 'nested': {'a': [1, 2], 'b': 'text'}},
            'model_params': {'recon_shape': (6, 5, 4), 'delta_voxel': 1.0},
            'recon_log': 'line one\nline two',
            'notes': 'a note'}


def _assert_dict_round_trip(out, ref):
    assert set(out) == set(ref)
    assert out['recon_log'] == ref['recon_log'] and out['notes'] == ref['notes']
    assert out['recon_params']['iterations'] == 10
    assert np.array_equal(out['recon_params']['angles'], ref['recon_params']['angles'])
    assert out['recon_params']['angles'].dtype == np.float64
    assert out['recon_params']['sharpness'] == pytest.approx(0.5)
    assert out['recon_params']['nested'] == {'a': [1, 2], 'b': 'text'}
    assert out['model_params'] == {'recon_shape': [6, 5, 4], 'delta_voxel': 1.0}


def _shard_on(vol, axis, n):
    """The volume split into n CPU shards along the axis."""
    pl = _sharding.Placement(['cpu'] * n, axis=axis, axis_len=vol.shape[axis])
    tensors = []
    for _d, (s0, s1) in pl.shard_ranges():
        cut = [slice(None)] * vol.ndim
        cut[axis] = slice(s0, s1)
        tensors.append(torch.as_tensor(vol[tuple(cut)].copy()))
    return _sharding.Shards(tensors, pl)


@pytest.mark.parametrize('kind', ['numpy', 'sharded'])
def test_round_trip_3d(tmp_path, monkeypatch, kind):
    """Save and load give the same volume and recon_dict, from a numpy array and
    from shards, in several slabs.  The file is right handed and self describing,
    and the slice viewer reads and writes the same layout.  The sharded writers
    write exactly what writing the gathered volume writes, on both sharding axes."""
    monkeypatch.setattr(utilities, '_HDF5_SLAB_BYTES', 64)     # several slabs for a small volume
    vol = _volume()
    recon = vol if kind == 'numpy' else _shard_on(vol, 2, 2)
    path = str(tmp_path / f'{kind}.h5')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mbirtorch.export_recon_hdf5(path, recon, _recon_dict())
        out, out_dict = mbirtorch.import_recon_hdf5(path)
    assert out.shape == vol.shape and out.dtype == vol.dtype
    assert np.array_equal(out, vol)
    _assert_dict_round_trip(out_dict, _recon_dict())

    if kind == 'sharded':
        # _save_array_hdf5 with shards on the view axis writes the whole-volume file.
        ref_path, out_path = str(tmp_path / 'ref_save.h5'), str(tmp_path / 'sharded_save.h5')
        utilities._save_array_hdf5(ref_path, vol, 'recon')
        utilities._save_array_hdf5(out_path, _shard_on(vol, 0, 2), 'recon')
        ref, _ = utilities._load_array_hdf5(ref_path)
        saved, _ = utilities._load_array_hdf5(out_path)
        assert saved.shape == ref.shape and saved.dtype == ref.dtype and np.array_equal(saved, ref)
        return

    # The file layout: (slice, col, row), one chunk per slice, self describing.
    with h5py.File(path, 'r') as f:
        assert list(f.keys()) == ['recon']
        dset = f['recon']
        assert dset.shape == (4, 5, 6)                      # (slice, col, row)
        assert dset.chunks == (1, 5, 6)
        assert np.array_equal(dset[()], np.transpose(vol, (2, 1, 0)))
        assert dset.attrs['format'] == 'mbirtorch_recon_v2'
        assert dset.attrs['axes'] == 'slice,col,row'
        assert dset.attrs['mbirtorch_version'] == mbirtorch.__version__

    # The viewer's reader permutes by the axes attribute, so it shows (row, col, slice).
    array, data_dict = slice_figure.VolumeStack.read_file_array(path)
    assert np.array_equal(array, vol)
    assert data_dict['axes'] == 'slice,col,row' and data_dict['notes'] == 'a note'
    # The viewer's writer produces a file import_recon_hdf5 reads back unchanged.
    saved = str(tmp_path / 'from_viewer.h5')
    slice_figure._save_data_hdf5(saved, vol, 'recon', {'notes': 'from the viewer', 'axes': 'stale'})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out, out_dict = mbirtorch.import_recon_hdf5(saved)
    assert np.array_equal(out, vol)
    assert out_dict['notes'] == 'from the viewer'


def test_round_trip_4d(tmp_path):
    vol = _volume((3, 6, 5, 4))
    path = str(tmp_path / 'recon4d.h5')
    mbirtorch.export_recon_hdf5(path, vol, {'notes': 'four'})
    with h5py.File(path, 'r') as f:
        assert f['recon'].shape == (3, 4, 5, 6)             # (time, slice, col, row)
        assert f['recon'].attrs['axes'] == 'time,slice,col,row'
    out, out_dict = mbirtorch.import_recon_hdf5(path)
    assert np.array_equal(out, vol)
    assert out_dict == {'notes': 'four'}


def test_old_files_are_read_by_dataset_name(tmp_path):
    vol = _volume()
    # An old export file: dataset 'recon' in (slice, col, row) with text attributes and no format tag.
    old_export = str(tmp_path / 'old_export.h5')
    with h5py.File(old_export, 'w') as f:
        d = f.create_dataset('recon', data=np.transpose(vol, (2, 1, 0)))
        d.attrs['notes'] = 'old'
    with pytest.warns(UserWarning, match='old layout'):
        out, out_dict = mbirtorch.import_recon_hdf5(old_export)
    assert np.array_equal(out, vol) and out_dict == {'notes': 'old'}
    # An old viewer file: dataset 'volume' in (row, col, slice).
    old_viewer = str(tmp_path / 'old_viewer.h5')
    with h5py.File(old_viewer, 'w') as f:
        f.create_dataset('volume', data=vol)
    with pytest.warns(UserWarning, match='read as stored'):
        out, out_dict = mbirtorch.import_recon_hdf5(old_viewer)
    assert np.array_equal(out, vol) and out_dict == {}

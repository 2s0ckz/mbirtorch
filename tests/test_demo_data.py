"""Gates for the 3D and 4D demo-data generators."""

import numpy as np

import mbirtorch


def test_gen_demo_data_is_finite_float32_for_every_model_type():
    """Each generator must return a host numpy float32 phantom and a finite,
    non-empty sinogram of the shape its geometry implies.  The cube phantom
    itself must be torch float32: mbirjax's jnp.array downcasts to float32,
    while torch.as_tensor keeps numpy's float64, which doubles the memory and
    mps cannot hold at all."""
    import torch
    assert mbirtorch.gen_cube_phantom((8, 8, 4)).dtype == torch.float32

    phantom, sino, params = mbirtorch.gen_demo_data(
        model_type='cone', object_type='cube', num_views=12,
        num_det_rows=24, num_det_channels=32)
    # Host numpy float32 for either object type, as the docstring promises.
    assert isinstance(phantom, np.ndarray) and phantom.dtype == np.float32
    assert phantom.max() > 0 and np.isfinite(np.asarray(sino)).all()

    phantom, sino, params = mbirtorch.gen_demo_data(
        model_type='translation', object_type='cube',
        num_det_rows=24, num_det_channels=32)
    assert isinstance(phantom, np.ndarray) and phantom.dtype == np.float32
    assert sino.shape[0] == params['translation_vectors'].shape[0]
    assert phantom.max() > 0 and np.isfinite(np.asarray(sino)).all()
    assert np.asarray(sino).max() > 0

    phantom, sino, params = mbirtorch.gen_demo_data(
        model_type='multiaxis', elevation_degrees=25.0, object_type='cube',
        num_views=8, num_det_rows=16, num_det_channels=24)
    assert sino.shape == (8, 16, 24)
    assert params['angles'].shape == (8, 2)
    assert np.allclose(params['angles'][:, 1], np.deg2rad(25.0))
    assert np.isfinite(np.asarray(sino)).all() and np.asarray(sino).max() > 0


def test_demo_data_4d_projects_each_step_from_its_own_object():
    """The sinogram has one view per angle, and the views of a step equal the projection of
    that step's object, so the scan is of a moving object."""
    num_views, num_steps = 24, 4
    phantom_4d, sinogram, params = mbirtorch.gen_demo_data_4d(
        num_views=num_views, num_rotations=1, num_det_rows=8, num_det_channels=32,
        num_steps=num_steps)
    assert phantom_4d.shape == (num_steps, 32, 32, 8)
    assert sinogram.shape == (num_views, 8, 32) and sinogram.dtype == np.float32
    assert params['angles'].shape == (num_views,)
    assert np.array_equal(params['step_of_view'], np.repeat(np.arange(num_steps), num_views // num_steps))

    ct_model = mbirtorch.ParallelBeamModel(sinogram.shape, params['angles'])
    ct_model.set_params(verbose=0)
    for step in (0, num_steps - 1):
        views = params['step_of_view'] == step
        expected = ct_model.forward_project(phantom_4d[step])[views]
        assert np.allclose(sinogram[views], expected, atol=1e-5)
    # The scan is of a moving object: the middle views are not the first object's projection.
    middle = params['step_of_view'] == num_steps // 2
    assert not np.allclose(sinogram[middle], ct_model.forward_project(phantom_4d[0])[middle], atol=1e-3)

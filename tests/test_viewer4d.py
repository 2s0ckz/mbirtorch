"""Crash guards for the 4D viewer (mbirtorch/viewers/slice_figure4d.py), run
headlessly on Agg: the window renders, frames step and play, and the mbirtorch
wrapper opens on a tensor and saves GIFs.
"""


import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

import mbirtorch
from mbirtorch.viewers.slice_figure4d import SliceViewer4D


def make_volume(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


def shifting_square(num_frames=12, n=32, depth=4):
    """A square that shifts along y by up to 2 pixels, repeating every 6 frames."""
    volume = np.zeros((num_frames, n, n, depth), dtype=np.float32)
    for t in range(num_frames):
        shift = int(round(2 * np.cos(2 * np.pi * t / 6)))
        volume[t, 10:22, 10 + shift:22 + shift, :] = 1.0
    return volume


# ---------------------------------------------------------------------------
# The window (SliceViewer4D), headless on Agg
# ---------------------------------------------------------------------------


def _press_key(fig, key):
    fig.canvas.callbacks.process('key_press_event',
                                 KeyEvent('key_press_event', fig.canvas, key))


@pytest.fixture
def make_viewer():
    created = []

    def _make(*datasets, **kwargs):
        viewer = SliceViewer4D(*datasets, **kwargs)
        viewer.fig.canvas.draw()
        created.append(viewer)
        return viewer

    yield _make
    for viewer in created:
        plt.close(viewer.fig)


class TestWindow:
    def test_window_renders_with_frame_titles(self, make_viewer, tmp_path):
        viewer = make_viewer(shifting_square(), make_volume((32, 32, 4)),
                             slice_label=['moving', 'static'])
        assert viewer.axes[0].get_title().splitlines()[0] == 'moving: t = 0, z = 2'
        assert viewer.axes[1].get_title().splitlines()[0] == 'static: z = 2'
        path = str(tmp_path / 'viewer4d.png')
        viewer.fig.savefig(path)
        # The panel region must not be blank: a row through the square has edges.
        from matplotlib.image import imread
        pixels = imread(path)
        ax_bbox = viewer.axes[0].bbox
        height = viewer.fig.canvas.get_width_height()[1]
        row = int(height - (ax_bbox.y0 + ax_bbox.y1) / 2)
        panel_row = pixels[row, int(ax_bbox.x0):int(ax_bbox.x1), :3]
        assert panel_row.std() > 0.05


    def test_frame_keys_and_playback(self, make_viewer):
        viewer = make_viewer(shifting_square(), slice_label='moving')
        _press_key(viewer.fig, '.')
        assert viewer.stack.master_frame == 1
        _press_key(viewer.fig, ',')
        assert viewer.stack.master_frame == 0

        _press_key(viewer.fig, ' ')
        assert viewer.playing
        # The timer does not fire on Agg, so one playback step is taken by hand.
        viewer._play_step()
        assert viewer.stack.master_frame == 1
        np.testing.assert_array_equal(np.asarray(viewer.images[0].get_array()),
                                      viewer.stack.slice_image(0))
        _press_key(viewer.fig, ' ')
        assert not viewer.playing
        assert viewer.frame_slider.val == 1
        assert viewer.axes[0].get_title().splitlines()[0] == 'moving: t = 1, z = 2'

        # Playback loops at the end.
        viewer.frame_slider.set_val(11)
        _press_key(viewer.fig, ' ')
        viewer._play_step()
        assert viewer.stack.master_frame == 0
        _press_key(viewer.fig, ' ')
        assert not viewer.playing


class TestMovie:


    def test_wrapper_writes_one_gif_frame_per_position(self, tmp_path):
        """The public wrapper opens on a 4D tensor and saves one GIF frame per position."""
        from PIL import Image
        import torch
        import mbirtorch.viewers.slice_figure as viewer_module

        volume = torch.as_tensor(shifting_square())
        with pytest.warns(UserWarning, match='non-interactive'):
            viewer = mbirtorch.slice_viewer4d(volume, block=False)
        try:
            np.testing.assert_array_equal(viewer.stack.original_data[0], volume.numpy())
            assert viewer.stack.frame_counts == [12]
            assert viewer.movie_fn is mbirtorch.save_volume_as_gif
            viewer._write_gifs('frame', str(tmp_path))
            viewer._write_gifs('slice', str(tmp_path))
            with Image.open(tmp_path / 'volume_x-y_along-t_z2.gif') as gif:
                assert gif.n_frames == 12
            with Image.open(tmp_path / 'volume_x-y_along-z_t0.gif') as gif:
                assert gif.n_frames == 4
        finally:
            plt.close(viewer.fig)
            if viewer in viewer_module._NONBLOCKING_VIEWERS:
                viewer_module._NONBLOCKING_VIEWERS.remove(viewer)

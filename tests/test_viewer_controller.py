"""Crash guard for the slice viewer, run headlessly on Agg.

The viewer is built over three volumes, rendered to a PNG, and
the image area is sampled to confirm that pixels were actually drawn.
"""

import os

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

from mbirtorch.viewers.slice_figure import SliceViewer


def make_volume(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


@pytest.fixture
def make_viewer():
    created = []

    def _make(*datasets, **kwargs):
        viewer = SliceViewer(*datasets, **kwargs)
        viewer.fig.canvas.draw()
        created.append(viewer)
        return viewer

    yield _make
    for viewer in created:
        plt.close(viewer.fig)

# ---------------------------------------------------------------------------
# Blitting and snapshots
# ---------------------------------------------------------------------------

class TestRendering:
    def test_snapshot_contains_image_pixels(self, make_viewer, tmp_path):
        n_volumes = 3
        volumes = [make_volume((32, 32, 8), s) for s in range(n_volumes)]
        viewer = make_viewer(*volumes, title='snapshot test')
        path = str(tmp_path / f'snapshot_{n_volumes}.png')
        viewer.fig.savefig(path)
        assert os.path.getsize(path) > 30_000
        # The panel region must not be blank: sample the image area.
        from matplotlib.image import imread
        pixels = imread(path)
        ax_bbox = viewer.axes[0].bbox
        height = viewer.fig.canvas.get_width_height()[1]
        row = int(height - (ax_bbox.y0 + ax_bbox.y1) / 2)
        col_range = slice(int(ax_bbox.x0), int(ax_bbox.x1))
        panel_row = pixels[row, col_range, :3]
        assert panel_row.std() > 0.05  # random noise, not a flat fill

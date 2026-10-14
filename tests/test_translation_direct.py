"""A value test for the direct reconstruction of the translation geometry.

The test reconstructs a simulated strip with ``recon_direct_tct`` and checks that the
values have the right scale.  The problem is small, so the test runs on the CPU in a
few seconds.
"""

import numpy as np
import pytest

import mbirtorch

# The geometry has magnification 2 and a translation step of 2 ALU.  Neither value is 1,
# so a missing or extra factor of the magnification or of the step changes the result.
SOURCE_ISO_DIST = 16.0
SOURCE_DETECTOR_DIST = 32.0
X_STEP = 2.0
NUM_X_TRANSLATIONS = 39
Z_STEP = 2.0
NUM_Z_TRANSLATIONS = 4
NUM_DET_ROWS = 8
# The detector half width is 1.5 times the source to detector distance.  So the measured
# rays make angles up to atan(1.5), or 56 degrees, with the y axis.
NUM_DET_CHANNELS = 96

# The voxels are 0.5 ALU wide and 1 ALU deep.  The volume is 8 ALU deep, 16 ALU wide,
# and 8 ALU high.
RECON_SHAPE = (8, 32, 16)
DELTA_VOXEL = 0.5
VOXEL_ROW_ASPECT = 2.0

# The strip is 4 columns (2 ALU) wide.  It fills every row and every slice.
STRIP_WIDTH = 4
STRIP_VALUE = 1.0

# The bounds on the mean value of the reconstructed strip, relative to STRIP_VALUE.
VALUE_LOWER_BOUND = 0.8
VALUE_UPPER_BOUND = 1.0


def test_direct_reconstruction_of_a_strip_has_the_right_scale():
    """The direct reconstruction of a strip has a mean value near the strip value.

    The strip is narrow in x and runs the full depth of the volume along y, which is close
    to the ray direction.  Most of its Fourier content therefore lies inside the range of
    measured ray angles.  The scan is a limited angle scan, so the reconstruction loses the
    rest of this content and its mean is somewhat below the strip value.  The mean was 0.90
    when this test was written, and it changed by about 2% for x steps of 1, 2, and 3 ALU.

    The bounds detect a scale error of about 12% or more.  A missing or extra factor of the
    magnification, the translation step, the source to detector distance, or pi moves the
    mean far outside them.  The mean is taken away from the edges of the volume.
    """
    translation_vectors = mbirtorch.gen_translation_vectors(
        NUM_X_TRANSLATIONS, NUM_Z_TRANSLATIONS, x_spacing=X_STEP, z_spacing=Z_STEP)
    sinogram_shape = (translation_vectors.shape[0], NUM_DET_ROWS, NUM_DET_CHANNELS)
    model = mbirtorch.TranslationModel(sinogram_shape, translation_vectors,
                                       source_detector_dist=SOURCE_DETECTOR_DIST,
                                       source_iso_dist=SOURCE_ISO_DIST)
    model.configure_devices(devices=["cpu"])
    model.set_params(no_warning=True, verbose=0, recon_shape=RECON_SHAPE,
                     delta_voxel=DELTA_VOXEL, voxel_row_aspect=VOXEL_ROW_ASPECT)

    num_rows, num_cols, num_slices = RECON_SHAPE
    first_col = num_cols // 2 - STRIP_WIDTH // 2
    strip_cols = slice(first_col, first_col + STRIP_WIDTH)
    phantom = np.zeros(RECON_SHAPE, dtype=np.float32)
    phantom[:, strip_cols, :] = STRIP_VALUE

    sinogram = np.asarray(model.forward_project(phantom))
    recon = model.recon_direct_tct(sinogram)

    assert recon.shape == RECON_SHAPE
    assert np.all(np.isfinite(recon))
    inner_slices = slice(num_slices // 4, 3 * num_slices // 4)
    strip_mean = float(recon[1:-1, strip_cols, inner_slices].mean()) / STRIP_VALUE
    print(f"mean of the reconstructed strip / strip value = {strip_mean:.3f}")
    assert VALUE_LOWER_BOUND < strip_mean < VALUE_UPPER_BOUND, strip_mean


def test_direct_reconstruction_off_grid_warns_and_returns_zeros():
    """With one x translation the scan is not on a regular grid.  Both direct methods warn
    and return zeros of the right shape instead of raising."""
    translation_vectors = mbirtorch.gen_translation_vectors(
        1, NUM_Z_TRANSLATIONS, x_spacing=X_STEP, z_spacing=Z_STEP)
    sinogram_shape = (translation_vectors.shape[0], NUM_DET_ROWS, NUM_DET_CHANNELS)
    model = mbirtorch.TranslationModel(sinogram_shape, translation_vectors,
                                       source_detector_dist=SOURCE_DETECTOR_DIST,
                                       source_iso_dist=SOURCE_ISO_DIST)
    model.configure_devices(devices=["cpu"])
    model.set_params(no_warning=True, verbose=0, recon_shape=RECON_SHAPE,
                     delta_voxel=DELTA_VOXEL, voxel_row_aspect=VOXEL_ROW_ASPECT)
    sinogram = np.ones(sinogram_shape, dtype=np.float32)

    with pytest.warns(UserWarning, match="not on a regular grid"):
        recon = model.recon_direct_tct(sinogram)
    assert recon.shape == RECON_SHAPE
    assert not np.any(recon)

    with pytest.warns(UserWarning, match="not on a regular grid"):
        filtered = model.direct_filter_tct(sinogram)
    assert filtered.shape == sinogram_shape
    assert not np.any(filtered)

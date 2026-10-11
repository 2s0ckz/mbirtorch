"""This module holds the geometry formulas that the four scan geometries share.  The projectors
use these functions.  The method :meth:`mbirtorch.TomographyModel.project_points` also uses
them.  The docstring of that method states the voxel coordinate convention.

Let (u, v) be a point on the detector.  The coordinate u is measured along the channels, and v is
measured along the rows.  Both are measured from the point where the central ray meets the
detector.  The point (u, v) has these fractional indices:

    channel = (u + det_channel_offset) / delta_det_channel + (num_det_channels - 1) / 2
    row     = (v + det_row_offset) / delta_det_row + (num_det_rows - 1) / 2
"""

import torch

_F32 = torch.float32


def pixel_xy(pixel_indices, num_rows, num_cols, delta_voxel, delta_voxel_row):
    """This function returns the in-plane position (x_tilde, y_tilde) of each recon pixel
    before rotation.  The position is returned as two (P,) float32 tensors.  Each entry of
    ``pixel_indices`` is the flat index i * num_cols + j.
    """
    row_index = (pixel_indices // num_cols).to(_F32)
    col_index = (pixel_indices % num_cols).to(_F32)
    # The row index i gives y, and the column index j gives x.
    y_tilde = delta_voxel_row * (row_index - (num_rows - 1) / 2.0)
    x_tilde = delta_voxel * (col_index - (num_cols - 1) / 2.0)
    return x_tilde, y_tilde


def rotate_about_z(x_tilde, y_tilde, cosine, sine):
    """This function rotates positions about the z axis, with one rotation for each view.
    The inputs ``x_tilde`` and ``y_tilde`` have shape (P,), and ``cosine`` and ``sine`` have
    shape (Vb, 1).  A positive angle turns +x toward +y.  The function returns x and y, each
    with shape (Vb, P).
    """
    x = cosine * x_tilde[None, :] - sine * y_tilde[None, :]
    y = sine * x_tilde[None, :] + cosine * y_tilde[None, :]
    return x, y


def channel_index(u, delta_det_channel, det_channel_offset, num_channels):
    """This function returns the fractional channel index of the detector coordinate ``u``.
    A positive ``det_channel_offset`` moves the image of an object toward higher channel
    indices.
    """
    det_center_channel = (num_channels - 1) / 2.0
    return (u + det_channel_offset) / delta_det_channel + det_center_channel


def row_index(v, delta_det_row, det_row_offset, num_rows):
    """This function returns the fractional row index of the detector coordinate ``v``.
    It uses the same formula as :func:`channel_index`, applied to rows."""
    det_center_row = (num_rows - 1) / 2.0
    return (v + det_row_offset) / delta_det_row + det_center_row

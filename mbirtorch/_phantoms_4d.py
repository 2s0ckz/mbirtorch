"""This module generates moving phantoms for 4D reconstruction demos and tests.

Each moving phantom is a function ``object(phantom_shape, t)`` that returns the volume at time
``t``.  The time ``t`` runs from 0 at the start of the scan to 1 at the end.  The dictionary
``MOVING_PHANTOMS`` lists the phantom functions by name.  Each volume has shape
(num_rows, num_cols, num_slices), and the slice axis is the rotation axis.
"""

import numpy as np


def _rack_and_pinion(phantom_shape, t, rotation_degrees=45.0, num_teeth=8):
    """
    This function returns a rack and pinion phantom at time ``t``.

    The rack is a toothed bar along the slice axis.  The pinion is a toothed wheel beside the bar
    on a horizontal axle.  The wheel turns by ``rotation_degrees * t``, and the bar slides so that
    the teeth stay meshed.  The radius of the wheel is one sixth of the number of columns.  The
    bar has attenuation 1.0, and the wheel has attenuation 0.7.  The volume should have more
    slices than rows or columns.

    Args:
        phantom_shape (tuple of int): (num_rows, num_cols, num_slices).
        t (float): Time in [0, 1].
        rotation_degrees (float, optional): The total rotation of the wheel over the scan, in
            degrees.  Defaults to 45.
        num_teeth (int, optional): The number of teeth on the wheel.  Defaults to 8.

    Returns:
        numpy.ndarray: The volume at time ``t``, float32, shape ``phantom_shape``.
    """
    num_rows, num_cols, num_slices = phantom_shape
    radius = num_cols / 6.0                   # Wheel radius, measured to the base of the teeth.
    tooth_depth = radius / 3.0
    bar_width = radius / 2.0                  # Measured along the columns.
    thickness = num_rows / 5.0                # Thickness of bar and wheel along the rows.
    pitch = 2.0 * np.pi * radius / num_teeth  # Tooth spacing along the bar.
    angle = np.deg2rad(rotation_degrees) * t

    # The bar and wheel are centered in the volume.  The bar is on the low-column side of the wheel.
    extent = bar_width + 2.0 * tooth_depth + 1.0 + 2.0 * radius
    bar_col = (num_cols - 1) / 2.0 - extent / 2.0 + bar_width / 2.0
    wheel_col = bar_col + bar_width / 2.0 + 2.0 * tooth_depth + 1.0 + radius
    center_row = (num_rows - 1) / 2.0
    wheel_slice = (num_slices - 1) / 2.0

    rows = np.arange(num_rows, dtype=np.float32)[:, None, None]
    cols = np.arange(num_cols, dtype=np.float32)[None, :, None]
    slices = np.arange(num_slices, dtype=np.float32)[None, None, :]
    in_thickness = np.abs(rows - center_row) <= thickness / 2.0

    # A positive turn of the wheel moves the bar toward lower slice index.
    bar_shift = -radius * angle
    in_bar = np.abs(cols - bar_col) <= bar_width / 2.0
    on_bar_teeth = ((cols > bar_col + bar_width / 2.0)
                    & (cols <= bar_col + bar_width / 2.0 + tooth_depth)
                    & (np.mod(slices - bar_shift, pitch) < pitch / 2.0))
    bar = in_thickness & (in_bar | on_bar_teeth)

    rho = np.sqrt((cols - wheel_col) ** 2 + (slices - wheel_slice) ** 2)
    phi = np.arctan2(slices - wheel_slice, cols - wheel_col)
    tooth_angle = 2.0 * np.pi / num_teeth
    on_wheel_teeth = ((rho > radius) & (rho <= radius + tooth_depth)
                      & (np.mod(phi - angle, tooth_angle) < tooth_angle / 2.0))
    wheel = in_thickness & ((rho <= radius) | on_wheel_teeth)

    volume = np.zeros(phantom_shape, dtype=np.float32)
    volume[np.broadcast_to(bar, phantom_shape)] = 1.0
    volume[np.broadcast_to(wheel, phantom_shape)] = 0.7
    return volume


# The moving phantom functions by name.  Each function takes (phantom_shape, t).
MOVING_PHANTOMS = {
    'rack-and-pinion': _rack_and_pinion,
}


def _gen_moving_phantom(object_type, phantom_shape, num_steps, **options):
    """
    This function returns the volumes of a moving phantom at ``num_steps`` times.

    The function divides the scan into ``num_steps`` equal intervals.  It returns the volume at the
    middle of each interval.

    Args:
        object_type (str): A name in ``MOVING_PHANTOMS``, such as 'rack-and-pinion'.
        phantom_shape (tuple of int): (num_rows, num_cols, num_slices).
        num_steps (int): The number of time intervals.
        **options: Keyword arguments passed to the phantom function.

    Returns:
        numpy.ndarray: The volumes, float32, shape (num_steps,) + phantom_shape.
    """
    try:
        phantom = MOVING_PHANTOMS[object_type]
    except KeyError:
        raise ValueError(f'object_type must be one of {sorted(MOVING_PHANTOMS)}; got {object_type!r}.')
    if num_steps < 1:
        raise ValueError(f'num_steps must be at least 1; got {num_steps}.')
    times = (np.arange(num_steps) + 0.5) / num_steps
    return np.stack([phantom(phantom_shape, t, **options) for t in times])

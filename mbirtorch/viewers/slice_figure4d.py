"""This module provides an interactive viewer for 4D volumes.  The viewer extends the slice
viewer in ``slice_figure``.

The classes ``VolumeStack4D`` and ``SliceViewer4D`` subclass the two classes of
``slice_figure`` and add a time axis.  A 4D array has axes ``(t, x, y, z)``.  The viewer has a
second slider, called the frame slider.  In a spatial plane, the frame slider sets the frame.
In a space-time plane, it sets the position along the second hidden spatial axis.  At import,
the module loads only numpy and the base matplotlib package.
"""

import copy
import os
import re
import sys

import matplotlib
import numpy as np

from . import slice_figure as sf
from .slice_figure import Mode, SliceViewer, VolumeStack

__all__ = ['SliceViewer4D', 'VolumeStack4D', 'slice_viewer4d']

# AXIS_NAMES holds the names of the axes of a 4D volume, in axis order.
AXIS_NAMES = ('t', 'x', 'y', 'z')

# PLANES lists the planes a panel can show, as pairs of axes in (t, x, y, z) order.
# PLANE_LABELS holds the labels of their radio buttons.  The first three planes are spatial
# planes.  The last three are space-time planes.
PLANES = ((1, 2), (1, 3), (2, 3), (0, 1), (0, 2), (0, 3))
PLANE_LABELS = tuple('{}-{}'.format(AXIS_NAMES[a], AXIS_NAMES[b]) for a, b in PLANES)

ROI_PLOT_FONT_SIZE = 8
ROI_PLOT_HINT = 'Draw an ROI on an image to plot its mean against frame'

# ROWS names the two slider rows that set positions.  The slice row holds the slice slider.  The frame
# row holds the frame slider.  Each row has a Play button and a GIF button.
ROWS = ('slice', 'frame')

# The inherited in-figure dialogs place their parts at fixed fractions of the figure height.
# These fractions assume the slice viewer's figure height of 8 inches.
DIALOG_LAYOUT_HEIGHT = 8.0


class VolumeStack4D(VolumeStack):
    """This class is used to hold the data of the 4D viewer.  It uses numpy only.

    The class holds 2D, 3D, and 4D arrays.  A 4D array has axes ``(t, x, y, z)``.  A 2D or 3D
    array is fixed in time.  Each volume shows a spatial plane (x-y, x-z, y-z) or a
    space-time plane (t-x, t-y, t-z).  The slice slider and the frame slider set the
    positions along the two hidden axes.

    Args:
        datasets (sequence of ndarray or None): One or more 2D, 3D, or 4D arrays.  A 2D
            array gets a trailing axis of length 1.  None is replaced by a volume of zeros.
        data_dicts (None, dict, or list of dict/None, optional): Dicts of strings that
            describe the volumes.
        vmin (float, optional): The display minimum.  Defaults to the minimum over every
            voxel of every volume.
        vmax (float, optional): The display maximum.  Defaults to the maximum over every
            voxel of every volume.
        slice_label (str or list of str, optional): The label at the start of each panel
            title.  Defaults to no label.
        slice_axis (int or list of int, optional): The axis to slice along.  Each array
            counts its own axes, so the value is 1, 2, or 3 for a 4D array and 0, 1, or 2
            for a 3D array.  Defaults to z.  If the arrays mix 3D and 4D, give a list with
            one entry per volume.
    """

    def __init__(self, datasets, data_dicts=None, vmin=None, vmax=None,
                 slice_label=None, slice_axis=None):
        datasets = list(datasets)
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")
        self.n_volumes = len(datasets)
        self.labels = self._normalize_labels(slice_label)
        self.data_dicts = self._normalize_data_dicts(data_dicts)

        # original_data holds each array as passed in, and Save writes it.  frames holds
        # each volume as (t, x, y, z), with one frame for 2D or 3D input.  display_axes
        # holds the axes of each volume in the order rows, columns, slice axis, second axis.
        # data holds the displayed 3D view, which the inherited slicing and ROI methods use.
        self.original_data = []
        self.frames = []
        for dataset in datasets:
            if dataset is None:
                dataset = np.zeros(sf.PLACEHOLDER_SHAPE)
            dataset = np.asarray(dataset)
            if dataset.ndim == 2:
                dataset = dataset[..., np.newaxis]
            elif dataset.ndim not in (3, 4):
                raise ValueError("Each input data must be a 2D, 3D, or 4D array")
            self.original_data.append(dataset)
            self.frames.append(self._as_frames(dataset))

        self.display_axes = self._initial_display_axes(slice_axis)
        self._difference_info = [None] * self.n_volumes

        # All volumes share the frame index.  Each spatial axis stores its frame slider
        # position as a fraction of its length.  So volumes of different lengths move in
        # proportion, as they do with the slice slider.
        self.master_frame = 0
        self._second_fractions = {1: 0.5, 2: 0.5, 3: 0.5}
        self.cur_seconds = [0] * self.n_volumes
        self.data = [None] * self.n_volumes
        for i in range(self.n_volumes):
            self._refresh_view(i)

        # Each volume opens at its own middle slice.  The shared slider starts at the
        # middle slice of volume 0, as in the 3D viewer.
        self.cur_slices = [d.shape[2] // 2 for d in self.data]
        self.master_index = self.cur_slices[0]

        self.vmin, self.vmax = self.resolve_range(vmin, vmax)

    # --- Input normalization ---

    @staticmethod
    def _as_frames(array):
        """This method returns ``array`` as a 4D view.  A 3D array becomes a single frame."""
        return array if array.ndim == 4 else array[np.newaxis]

    def _normalize_labels(self, slice_label):
        # A 4D panel title gives its position, so the default label is empty.
        return super()._normalize_labels('' if slice_label is None else slice_label)

    def _initial_display_axes(self, slice_axis):
        # Each array counts its own axes, so axis 3 of a 4D array and axis 2 of a 3D
        # array both mean z.  A volume opens on the spatial plane normal to its slice axis.
        if slice_axis is None:
            slice_axis = [None] * self.n_volumes
        elif isinstance(slice_axis, (int, np.integer)):
            slice_axis = [slice_axis] * self.n_volumes
        slice_axis = list(slice_axis)
        if len(slice_axis) != self.n_volumes:
            raise ValueError(
                "slice_axis must be a single int or a list of ints of the "
                "same length as the number of datasets")
        display_axes = []
        for axis, dataset in zip(slice_axis, self.original_data):
            if axis is None:
                axis = 3
            elif dataset.ndim == 4:
                if int(axis) not in (1, 2, 3):
                    raise ValueError("slice_axis of a 4D array must be 1, 2, or 3")
                axis = int(axis)
            else:
                if int(axis) not in (0, 1, 2):
                    raise ValueError("slice_axis must be 0, 1, or 2")
                axis = int(axis) + 1
            rows, columns = (a for a in (1, 2, 3) if a != axis)
            display_axes.append([rows, columns, axis, 0])
        return display_axes

    # --- Planes and the two hidden axes ---

    def plane(self, i):
        """This method returns the plane of volume ``i`` as a sorted pair of axes."""
        return tuple(sorted(self.display_axes[i][:2]))

    def shows_time(self, i):
        """This method returns True if volume ``i`` shows a space-time plane."""
        return 0 in self.display_axes[i][:2]

    @property
    def slice_axes(self):
        """This property is the list of the slice axes of the volumes, in (t, x, y, z)
        numbering."""
        return [axes[2] for axes in self.display_axes]

    @property
    def second_axis(self):
        """This property is the axis of the frame slider, in (t, x, y, z) numbering.  The
        axis is the same for every volume.  In a spatial plane it is t.  In a space-time
        plane it is the second hidden spatial axis.
        """
        return self.display_axes[0][3]

    @property
    def second_count(self):
        """This property is the number of positions of the frame slider."""
        axis = self.second_axis
        return max(frames.shape[axis] for frames in self.frames)

    @property
    def second_position(self):
        """This property is the position of the frame slider.  The position is a frame
        index or an index along a spatial axis.
        """
        axis = self.second_axis
        if axis == 0:
            return self.master_frame
        return int(round(self._second_fractions[axis] * (self.second_count - 1)))

    def _second_index(self, i):
        axis = self.display_axes[i][3]
        if axis == 0:
            return min(self.master_frame, self.frames[i].shape[0] - 1)
        return int(round(self._second_fractions[axis] * (self.frames[i].shape[axis] - 1)))

    def _source(self, i):
        # In a space-time plane, a one-frame volume is repeated over every frame of the
        # longest volume without a copy.  So it appears constant in time.
        frames = self.frames[i]
        if frames.shape[0] == 1 and self.shows_time(i):
            frames = np.broadcast_to(frames, (self.max_frames,) + frames.shape[1:])
        return frames

    def _refresh_view(self, i):
        self.cur_seconds[i] = self._second_index(i)
        self.data[i] = np.transpose(self._source(i), self.display_axes[i])[..., self.cur_seconds[i]]

    def _refresh_views(self):
        for i in range(self.n_volumes):
            self._refresh_view(i)

    def _update_second_positions(self):
        changed = []
        for i in range(self.n_volumes):
            if self._second_index(i) != self.cur_seconds[i]:
                self._refresh_view(i)
                changed.append(i)
        return changed

    def set_plane(self, indices, plane):
        """This method shows ``plane`` in the given volumes.  The plane is a pair of axes
        in (t, x, y, z) numbering.  The method returns True if any volume changed.  If t is
        hidden, the frame slider controls t.  The slice position keeps its fraction of the
        axis length.
        """
        plane = tuple(sorted(int(a) for a in plane))
        hidden = [a for a in range(4) if a not in plane]
        previous_slice_axis = self.display_axes[indices[0]][2]
        if 0 in hidden:
            slice_axis, second_axis = hidden[1], 0
        elif previous_slice_axis in hidden:
            slice_axis = previous_slice_axis
            second_axis = hidden[0] if hidden[1] == slice_axis else hidden[1]
        else:
            slice_axis, second_axis = hidden[1], hidden[0]
        new_axes = [plane[0], plane[1], slice_axis, second_axis]
        changed = [i for i in indices if self.display_axes[i] != new_axes]
        if not changed:
            return False
        fraction = self.master_fraction
        for i in changed:
            self.display_axes[i] = list(new_axes)
            self._refresh_view(i)
        self._set_master_fraction(fraction)
        return True

    def transpose(self, i):
        """This method swaps the two displayed axes of volume ``i``."""
        axes = self.display_axes[i]
        axes[0], axes[1] = axes[1], axes[0]
        self._refresh_view(i)

    # --- Frames and the frame slider ---

    @property
    def frame_counts(self):
        """This property is the list of the frame counts of the volumes.  The count is 1
        for a 2D or 3D volume."""
        return [frames.shape[0] for frames in self.frames]

    @property
    def max_frames(self):
        """This property is the number of frames of the longest volume."""
        return max(self.frame_counts)

    @property
    def cur_frames(self):
        """This property is the list of the frames that the volumes show in a spatial plane."""
        return [min(self.master_frame, count - 1) for count in self.frame_counts]

    def has_time(self, i):
        """This method returns True if volume ``i`` changes with the frame index."""
        return self.original_data[i].ndim == 4 or self.frames[i].shape[0] > 1

    def holds_last_frame(self, i):
        """This method returns True if volume ``i`` has fewer frames than the frame index
        needs."""
        return self.has_time(i) and self.master_frame > self.frames[i].shape[0] - 1

    def set_frame(self, index):
        """This method sets the shared frame index and returns the volumes whose image
        changed.  The index is clipped to the length of the longest volume.  In a spatial
        plane, a shorter volume shows its last frame.  In a space-time plane, no image
        changes.
        """
        self.master_frame = int(np.clip(int(np.round(index)), 0, self.max_frames - 1))
        return self._update_second_positions()

    def set_second(self, index):
        """This method moves the frame slider to ``index`` and returns the volumes whose
        image changed.  In a spatial plane, the method sets the frame.  In a space-time
        plane, it sets the position along the second hidden spatial axis.
        """
        axis = self.second_axis
        if axis == 0:
            return self.set_frame(index)
        count = self.second_count
        index = int(np.clip(int(np.round(index)), 0, count - 1))
        self._second_fractions[axis] = index / (count - 1) if count > 1 else 0.0
        return self._update_second_positions()

    # --- Intensity range ---

    def data_range(self):
        """This method returns (min, max) over every frame of every volume."""
        lo = min(float(np.min(frames)) for frames in self.frames)
        hi = max(float(np.max(frames)) for frames in self.frames)
        return lo, hi

    # --- Difference images ---

    def can_difference(self, baseline_index, comparison_index):
        """This method returns True if the two volumes can be subtracted.  The volumes
        must be distinct and have the same x, y, z shape.  They must also have the same
        number of frames, or one of them must have one frame."""
        if baseline_index == comparison_index:
            return False
        base_shape = self.frames[baseline_index].shape
        comparison_shape = self.frames[comparison_index].shape
        return (base_shape[1:] == comparison_shape[1:]
                and (base_shape[0] == comparison_shape[0]
                     or 1 in (base_shape[0], comparison_shape[0])))

    def apply_difference(self, baseline_index, comparison_index, use_abs=False):
        """This method replaces volume ``baseline_index`` by (comparison - baseline) in
        every frame.  If ``use_abs`` is True, it uses the absolute value.  A 3D volume is
        subtracted from every frame of a 4D volume.  :meth:`restore` brings back the
        original data and label.
        """
        if not self.can_difference(baseline_index, comparison_index):
            raise ValueError(
                "Difference requires two distinct volumes with the same x, y, z shape")
        # The frames are stored in their original axis order, so the two volumes need
        # no reordering.  A one-frame volume broadcasts over the frames.
        difference = self.frames[comparison_index] - self.frames[baseline_index]
        if use_abs:
            difference = np.abs(difference)
        self._difference_info[baseline_index] = {
            'comparison_index': comparison_index,
            'use_abs': use_abs,
            'prev_label': self.labels[baseline_index],
        }
        self.frames[baseline_index] = difference
        self._refresh_views()
        if use_abs:
            label_prepend = 'abs(Image {} minus current): '.format(comparison_index)
        else:
            label_prepend = 'Image {} minus current: '.format(comparison_index)
        self.labels[baseline_index] = label_prepend + self.labels[baseline_index]

    def restore(self, i):
        """This method resets volume ``i`` to its original data and ends any difference."""
        self.frames[i] = self._as_frames(self.original_data[i])
        self.set_frame(self.master_frame)
        self._refresh_views()
        info = self._difference_info[i]
        if info is not None:
            self.labels[i] = info['prev_label']
        self._difference_info[i] = None

    # --- ROI mean against frame ---

    def roi_frame_means(self, i, x, y, radius):
        """This method returns the mean of volume ``i`` in a circle at the current slice
        for each frame.

        The volume must show a spatial plane.  The values ``x`` and ``y`` are the data
        coordinates (column, row) of the circle center.  The method returns an array with
        one value per frame.  It returns None if no pixel center falls inside the circle.
        """
        rows, columns, slice_axis, _second_axis = self.display_axes[i]
        # This view holds the displayed plane of every frame, with axes (frame, row, column).
        planes = np.transpose(self.frames[i], [0, rows, columns, slice_axis])[
            ..., self.cur_slices[i]]
        ny, nx = planes.shape[1:]
        yv, xv = np.ogrid[:ny, :nx]
        mask = (xv - x) ** 2 + (yv - y) ** 2 <= radius ** 2
        if not mask.any():
            return None
        return planes[:, mask].mean(axis=1)

    # --- Movies ---

    def movie_frame_count(self, i, axis):
        """This method returns the number of frames in the movie of volume ``i`` along the
        hidden axis ``axis``.
        """
        return self._source(i).shape[axis]

    def movie_view(self, i, axis):
        """This method returns the ``save_volume_as_gif`` arguments for a movie of the
        view of volume ``i``.

        The argument ``axis`` is one of the two hidden axes of the volume, in (t, x, y, z)
        numbering.  The movie plays along this axis.  The method returns
        ``(volume, frame_axis, slice_axis, slice_index)``.  Each movie frame matches the
        panel, even if the panel is transposed.
        """
        rows, columns, slice_axis, second_axis = self.display_axes[i]
        volume = self._source(i)
        if rows > columns:
            volume = np.swapaxes(volume, rows, columns)
        if axis == second_axis:
            return volume, second_axis, slice_axis, self.cur_slices[i]
        return volume, slice_axis, second_axis, self.cur_seconds[i]

    # --- File load ---

    def load_array(self, image_index, new_array, data_dict=None):
        """This method replaces volume ``image_index`` by a loaded 2D, 3D, or 4D array and
        returns [image_index].  For an array of any other dimension, it raises ValueError.
        The method ends any difference in this volume.  It moves the slice position to the
        middle slice of the volume.
        """
        new_array = np.asarray(new_array)
        if new_array.ndim == 2:
            new_array = new_array[..., np.newaxis]
        if new_array.ndim not in (3, 4):
            raise ValueError("Loaded array must be 2D, 3D, or 4D")
        i = image_index
        self.original_data[i] = new_array
        self.frames[i] = self._as_frames(new_array)
        self.data_dicts[i] = data_dict
        axes = self.display_axes[i]
        self.display_axes[i] = sorted(axes[:2]) + axes[2:]
        if self._difference_info[i] is not None:
            self.labels[i] = self._difference_info[i]['prev_label']
            self._difference_info[i] = None
        self.set_frame(self.master_frame)
        self._refresh_views()

        depth = self.data[i].shape[2]
        middle_fraction = (depth // 2) / (depth - 1) if depth > 1 else 0.0
        self._set_master_fraction(middle_fraction)
        return [i]


class SliceViewer4D(SliceViewer):
    """This class is used to view 4D volumes interactively in matplotlib.  It has every
    feature of the slice viewer.

    The class adds a choice of six planes, the frame slider, Play and GIF buttons for both
    sliders, and a plot of the ROI mean against frame.  In a spatial plane, the frame slider
    sets the frame.  In a space-time plane, each image row is one frame.  The space key plays
    and pauses.  The comma and period keys step the frame slider.  :meth:`show` displays the
    figure.

    Args:
        *datasets (ndarray or None): One or more 2D, 3D, or 4D arrays to display.  A 4D
            array has axes ``(t, x, y, z)``.  A 2D or 3D array stays fixed in time.
        data_dicts (None, dict, or list of dict/None, optional): Dicts of strings that
            describe the volumes.  The viewer can display them.
        title (str, optional): The figure title.  Defaults to ''.
        vmin (float, optional): The minimum display intensity.  Defaults to the minimum
            over every voxel of every volume.
        vmax (float, optional): The maximum display intensity.  Defaults to the maximum
            over every voxel of every volume.
        slice_label (str or list of str, optional): The label at the start of each
            panel title.  Defaults to no label.
        slice_axis (int or list of int, optional): The axis to slice along.  Each array
            counts its own axes, so the value is 1, 2, or 3 for a 4D array and 0, 1, or 2
            for a 3D array.  Defaults to z.  If the arrays mix 3D and 4D, give a list.
        cmap (str, optional): The colormap.  Defaults to 'gray'.
        show_instructions (bool, optional): If True, the viewer shows the "Press h for
            help" hint.  Defaults to True.
        save_fn (callable, optional): A replacement for the built-in HDF5 writer.  The
            viewer calls ``save_fn(file_path, array, array_name, attributes_dict)``.
        fps (float, optional): The playback speed in frames per second.  The GIFs use
            the same speed.  Defaults to 5.
        movie_fn (callable, optional): The writer for the GIF buttons.  The viewer calls
            it with the arguments of mbirtorch's ``save_volume_as_gif``, as in
            ``movie_fn(volume, filename, frame_axis=..., slice_axis=..., slice_index=...,
            vmin=..., vmax=..., fps=...)``.  None leaves out the GIF buttons.  Defaults to
            None.
    """

    def __init__(self, *datasets, data_dicts=None, title='', vmin=None, vmax=None,
                 slice_label=None, slice_axis=None, cmap='gray',
                 show_instructions=True, save_fn=None, fps=5, movie_fn=None):
        if fps <= 0:
            raise ValueError('fps must be positive; got {}.'.format(fps))
        sf._load_pyplot()
        self.stack = VolumeStack4D(datasets, data_dicts=data_dicts, vmin=vmin,
                                   vmax=vmax, slice_label=slice_label,
                                   slice_axis=slice_axis)
        self.title = title
        self.cmap = cmap
        self.show_instructions = show_instructions
        self.save_fn = save_fn if save_fn is not None else sf._save_data_hdf5
        self.fps = float(fps)
        self.movie_fn = movie_fn

        self._init_interaction_state()
        self._build_figure()
        self._connect_events()

    # --- Construction ---

    def _init_interaction_state(self):
        super()._init_interaction_state()
        self.playing = False
        # The row that plays while playing is True.  It is 'slice' or 'frame'.
        self._play_row = None
        self._play_timer = None
        self._play_regions = None
        self.frame_slider = None
        # These dicts hold, for each row, the Play button, all buttons, and the label that
        # playback shows in place of the slider value.
        self.play_buttons = {}
        self._row_buttons = {}
        self._play_labels = {}
        self.roi_plot_ax = None
        self._roi_lines = []
        self._roi_marker = None
        self._roi_hint = None

    def _build_figure(self):
        n = self.stack.n_volumes
        self.fig = sf.plt.figure(figsize=(6 * n, 10.5))
        self.fig.suptitle(self.title)
        # The grid rows are the panels, the plane radio buttons, the sliders, and the ROI
        # plot.  The slider area holds the slice slider, the frame row, and the intensity slider.
        self.gs = sf.gridspec.GridSpec(nrows=4, ncols=n, height_ratios=[14, 3, 3.5, 4],
                                       left=0.12, right=0.95, top=0.92,
                                       bottom=0.05, hspace=0.3, figure=self.fig)
        self._slider_rows = sf.gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=self.gs[2, :], hspace=0.6)
        self.axes = [None] * n
        self.caxes = [None] * n
        self.images = [None] * n
        self._create_panels()
        self._create_tooltips()
        self._create_axis_row()
        self._create_slice_slider()
        self._create_row_controls('slice')
        self._set_row_visible('slice', self.stack.max_slices > 1)
        self._create_intensity_slider()
        self._create_frame_row()
        self._create_roi_plot()
        self._label_slice_slider()

        # This opaque rectangle is drawn under partial redraws.  It is animated and kept
        # out of the layout, so full draws and saves with a tight bounding box omit it.
        self._clear_rect = sf.Rectangle((0, 0), 1, 1,
                                        facecolor=self.fig.get_facecolor(),
                                        edgecolor='none', animated=True,
                                        transform=sf.IdentityTransform())
        self._clear_rect.set_in_layout(False)
        self.fig.add_artist(self._clear_rect)

        if self.show_instructions:
            self.fig.text(0.01, 0.25, sf.multiline('Press h', 'for help'),
                          fontdict={'color': 'red'})

    @staticmethod
    def _slider_cells(spec):
        # A slider row has three parts: the buttons and label, the slider, and the value.
        # Every slider row and the ROI plot use these proportions.
        return sf.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec,
                                                   width_ratios=[2.5, 8.0, 2.0])

    def _slider_slot(self, row):
        # The inherited sliders ask for row 2 (slice) and row 3 (intensity).  Here they
        # are the first and last rows of the slider area, and the frame row is between them.
        block_row = {2: 0, 3: 2}[row]
        return self._slider_cells(self._slider_rows[block_row, 0])[0, 1]

    def _title_text(self, i):
        stack = self.stack
        head = stack.labels[i].strip()
        if head and not head.endswith(':'):
            head += ':'
        rows, columns, slice_axis, second_axis = stack.display_axes[i]
        # The title gives the position along each hidden axis.  The title omits the frame
        # of a volume that does not change in time.
        positions = {slice_axis: stack.cur_slices[i]}
        if second_axis != 0 or stack.has_time(i):
            positions[second_axis] = stack.cur_seconds[i]
        parts = []
        for axis in sorted(positions):
            part = '{} = {}'.format(AXIS_NAMES[axis], positions[axis])
            if axis == 0 and stack.holds_last_frame(i):
                part += ' (last)'
            parts.append(part)
        return sf.multiline(
            '{} {}'.format(head, ', '.join(parts)).strip(),
            'Shape: {}, plane {}-{}'.format(stack.original_data[i].shape,
                                            AXIS_NAMES[rows], AXIS_NAMES[columns]))

    def _rebuild_axis_radios(self):
        # This method follows the inherited method, with one radio button per plane.
        for ax in self._radio_axes:
            ax.remove()
        self._radio_axes = []
        self.axis_radios = []

        def make_radio(slot, i):
            ax = self.fig.add_subplot(slot)
            ax.set_title("Plane", loc='left', fontsize=sf.SLICE_AXIS_FONT_SIZE)
            radio = sf.RadioButtons(ax, labels=list(PLANE_LABELS),
                                    active=PLANES.index(self.stack.plane(i)),
                                    radio_props={'s': [sf.SLICE_AXIS_RADIO_SIZE]})
            for label in radio.labels:
                label.set_fontsize(sf.SLICE_AXIS_LABEL_FONT_SIZE)
            self._radio_axes.append(ax)
            self.axis_radios.append(radio)
            return radio

        if self.sync_axes or self.stack.n_volumes == 1:
            radio = make_radio(self._radio_slots[0], 0)
            radio.on_clicked(
                lambda label: self._on_axis_selected(None, PLANES[PLANE_LABELS.index(label)]))
        else:
            for i in range(self.stack.n_volumes):
                radio = make_radio(self._radio_slots[i], i)
                radio.on_clicked(
                    lambda label, i=i: self._on_axis_selected(
                        i, PLANES[PLANE_LABELS.index(label)]))

    def _label_slice_slider(self):
        # The slice slider is labeled with its axis when every panel slices the same axis.
        if self.slice_slider is None:
            return
        axes = set(self.stack.slice_axes)
        self.slice_slider.label.set_text(
            AXIS_NAMES[axes.pop()] if len(axes) == 1 else 'Slice')

    def _create_row_controls(self, row):
        """This method adds the Play and GIF buttons and the playback label of row ``row``.
        The row is 'slice' or 'frame'.  The method returns the gridspec of the row.
        """
        cells = self._slider_cells(self._slider_rows[ROWS.index(row), 0])
        left = sf.gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=cells[0, 0], width_ratios=[1.0, 0.8, 0.7], wspace=0.15)
        play_button = sf.Button(self.fig.add_subplot(left[0, 0]), 'Play')
        play_button.on_clicked(lambda _event: self._toggle_play(row))
        buttons = [play_button]
        if self.movie_fn is not None:
            gif_button = sf.Button(self.fig.add_subplot(left[0, 1]), 'GIF')
            gif_button.on_clicked(lambda _event: self._on_gif_button(row))
            buttons.append(gif_button)
        for button in buttons:
            button.label.set_fontsize(sf.STRIP_FONT_SIZE)
        self.play_buttons[row] = play_button
        self._row_buttons[row] = buttons
        label_ax = self.fig.add_subplot(cells[0, 2])
        label_ax.axis('off')
        self._play_labels[row] = label_ax.text(0.02, 0.5, '', va='center', fontsize=10,
                                               visible=False, transform=label_ax.transAxes)
        return cells

    def _set_row_visible(self, row, visible):
        for button in self._row_buttons[row]:
            button.ax.set_visible(visible)

    def _create_frame_row(self):
        # The frame row is the middle row of the slider area.
        cells = self._create_row_controls('frame')
        self._frame_slider_ax = self.fig.add_subplot(cells[0, 1])
        self._update_second_slider()

    def _make_frame_slider(self):
        stack = self.stack
        self.frame_slider = sf.Slider(self._frame_slider_ax,
                                      label=AXIS_NAMES[stack.second_axis], valmin=0,
                                      valmax=stack.second_count - 1,
                                      valinit=stack.second_position,
                                      valstep=1, valfmt='%0.0f')
        self.frame_slider.drawon = False
        self.frame_slider.on_changed(self._on_frame_slider)

    def _update_second_slider(self):
        """This method updates the label, bounds, value, and visibility of the frame slider
        to match its current axis."""
        stack = self.stack
        count = stack.second_count
        visible = count > 1
        self._frame_slider_ax.set_visible(visible)
        self._set_row_visible('frame', visible)
        if self.roi_plot_ax is not None:
            # The ROI plot shows values against frame, so it appears in a spatial plane only.
            roi_visible = visible and stack.second_axis == 0
            self.roi_plot_ax.set_visible(roi_visible)
            if roi_visible:
                self.roi_plot_ax.set_xlim(0, count - 1)
        if not visible:
            return
        if self.frame_slider is None:
            self._make_frame_slider()
            return
        self.frame_slider.label.set_text(AXIS_NAMES[stack.second_axis])
        self.frame_slider.valmax = count - 1
        self.frame_slider.ax.set_xlim(0, count - 1)
        self.frame_slider.set_val(stack.second_position)

    def _create_roi_plot(self):
        # The plot spans the slider columns, so its frame axis lines up with the frame slider.
        self.roi_plot_ax = self.fig.add_subplot(self._slider_cells(self.gs[3, :])[0, 1])
        ax = self.roi_plot_ax
        ax.set_xlabel('t', fontsize=ROI_PLOT_FONT_SIZE)
        ax.set_ylabel('ROI mean', fontsize=ROI_PLOT_FONT_SIZE)
        ax.tick_params(labelsize=ROI_PLOT_FONT_SIZE)
        ax.set_xlim(0, max(self.stack.max_frames - 1, 1))
        self._roi_marker = ax.axvline(self.stack.master_frame, color='0.5', lw=1)
        self._roi_hint = ax.text(0.5, 0.5, ROI_PLOT_HINT, transform=ax.transAxes,
                                 ha='center', va='center', color='0.5',
                                 fontsize=ROI_PLOT_FONT_SIZE)
        ax.set_visible(self.stack.second_axis == 0 and self.stack.max_frames > 1)

    # --- The plot of the ROI mean against frame ---

    def _update_roi_plot(self):
        """This method recomputes the ROI mean against frame for every volume."""
        ax = self.roi_plot_ax
        for line in self._roi_lines:
            line.remove()
        self._roi_lines = []
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        circle = self.circles[0] if self.circles else None
        # The plot is computed in a spatial plane only.  The code checks the plane of the
        # stack, because a plane change moves the slice slider and calls this method before
        # the plot is hidden.
        if circle is not None and self.stack.second_axis == 0 and ax.get_visible():
            x, y = circle.center
            radius = circle.get_radius()
            last_frame = self.stack.max_frames - 1
            for i in range(self.stack.n_volumes):
                means = self.stack.roi_frame_means(i, x, y, radius)
                if means is None:
                    continue
                if means.size == 1:
                    # A volume with one frame is fixed in time, so its plot is a flat line.
                    frames, means = [0, last_frame], [means[0], means[0]]
                else:
                    frames = np.arange(means.size)
                label = self.stack.labels[i].strip() or 'Image {}'.format(i)
                # Each volume keeps the same color when the plot is recomputed.
                (line,) = ax.plot(frames, means, lw=1.2, label=label,
                                  color='C{}'.format(i))
                self._roi_lines.append(line)
        self._roi_hint.set_visible(not self._roi_lines)
        if self._roi_lines:
            ax.relim()
            ax.autoscale_view(scalex=False)
            ax.legend(fontsize=ROI_PLOT_FONT_SIZE, loc='upper left',
                      bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    def _move_roi_marker(self):
        frame = self.stack.master_frame
        self._roi_marker.set_xdata([frame, frame])

    def _remove_roi_graphics(self):
        super()._remove_roi_graphics()
        if self.roi_plot_ax is not None:
            self._update_roi_plot()

    # --- Planes ---

    def _on_axis_selected(self, volume_index, plane):
        self._pause_if_playing()
        if volume_index is not None and 0 in plane and not self.sync_axes:
            # All panels share one frame slider.  So a space-time plane is shown in all
            # panels, and the panels share one set of radio buttons.
            self.sync_axes = True
            self._rebuild_axis_radios()
            volume_index = None
        indices = (list(range(self.stack.n_volumes)) if volume_index is None
                   else [volume_index])
        if not self.stack.set_plane(indices, plane):
            return
        # A space-time image stretches to fill its panel, so short movies stay
        # readable.  A spatial image keeps square pixels.
        for i in indices:
            self.axes[i].set_aspect('auto' if 0 in plane else 'equal')
        self._reset_view(volume_index)
        self.refresh()
        self._update_slice_slider()
        self._label_slice_slider()
        self._update_second_slider()
        self._reset_navigation()
        self._display_roi_stats(force=True)
        self._update_roi_plot()
        self.fig.canvas.draw_idle()

    def _toggle_couple_axes(self):
        self.sync_axes = not self.sync_axes
        self._rebuild_axis_radios()
        if self.sync_axes:
            # Every volume takes the plane of volume 0.
            self._on_axis_selected(None, self.stack.plane(0))
        self.fig.canvas.draw_idle()

    def _menu_items(self, i):
        items = super()._menu_items(i)
        if self.stack.second_axis != 0:
            # A space-time plane is shown in every panel, so the panels must stay coupled.
            items = [item for item in items if item[0] != 'Decouple slice axes']
        return items

    def _reset_view(self, volume_index=None):
        super()._reset_view(volume_index)
        # In a space-time plane, every panel spans the frames of the longest volume.  So the
        # frames of all panels line up.  The code sets _in_sync_callback, as the inherited
        # method does.
        indices = (range(self.stack.n_volumes) if volume_index is None
                   else [volume_index])
        span = (-0.5, self.stack.max_frames - 0.5)
        already_syncing = self._in_sync_callback
        self._in_sync_callback = True
        try:
            for i in indices:
                rows, columns = self.stack.display_axes[i][:2]
                if rows == 0:
                    self.axes[i].set_ylim(span[1], span[0])
                elif columns == 0:
                    self.axes[i].set_xlim(*span)
        finally:
            self._in_sync_callback = already_syncing

    # --- The two slider rows, their axes, and the panels that change along them ---

    def _row_slider(self, row):
        return self.slice_slider if row == 'slice' else self.frame_slider

    def _row_axis(self, row, i):
        """This method returns the axis of volume ``i`` that the slider of the row moves.
        The axis is in (t, x, y, z) numbering.
        """
        slice_axis, second_axis = self.stack.display_axes[i][2:]
        return slice_axis if row == 'slice' else second_axis

    def _row_position(self, row):
        """This method returns the position and the number of positions of the slider of the
        row."""
        stack = self.stack
        if row == 'slice':
            return stack.master_index, stack.max_slices
        return stack.second_position, stack.second_count

    def _moving_panels(self, row):
        """This method returns the panels whose view changes along the axis of the row."""
        return [i for i in range(self.stack.n_volumes)
                if self.stack.movie_frame_count(i, self._row_axis(row, i)) > 1]

    # --- GIF buttons ---

    def _gif_file_name(self, row, i):
        # An example name is init_x-y_along-z_t3.gif.  It gives the label, the plane, the
        # axis of the movie, and the position along the other hidden axis.
        stack = self.stack
        slice_axis, second_axis = stack.display_axes[i][2:]
        axis = self._row_axis(row, i)
        fixed_axis, fixed_index = ((second_axis, stack.cur_seconds[i]) if axis == slice_axis
                                   else (slice_axis, stack.cur_slices[i]))
        label = re.sub(r'[^A-Za-z0-9_-]+', '_', stack.labels[i]).strip('_') or 'volume'
        plane = PLANE_LABELS[PLANES.index(stack.plane(i))]
        name = '{}_{}_along-{}'.format(label, plane, AXIS_NAMES[axis])
        # The name omits the frame of a volume that does not change in time.
        if fixed_axis != 0 or stack.has_time(i):
            name += '_{}{}'.format(AXIS_NAMES[fixed_axis], fixed_index)
        return name + '.gif'

    def _on_gif_button(self, row):
        chosen = self._native_choose_folder(self._last_dir)
        if chosen is sf._NATIVE_UNAVAILABLE:
            self._open_gif_dialog(row)
        elif chosen is not None:
            self._write_gifs(row, chosen)

    def _native_choose_folder(self, directory):
        """This method returns a folder chosen in the macOS folder panel.  It returns None if
        the user cancels.  On other systems it returns ``_NATIVE_UNAVAILABLE``.
        """
        if (sys.platform != 'darwin'
                or matplotlib.get_backend().lower() in sf.NONINTERACTIVE_BACKENDS):
            return sf._NATIVE_UNAVAILABLE
        import subprocess

        def quoted(text):
            return text.replace('\\', '\\\\').replace('"', '\\"')

        script = ('POSIX path of (choose folder with prompt "Save the GIFs in" '
                  f'default location POSIX file "{quoted(directory)}")')
        try:
            result = subprocess.run(['osascript', '-e', script],
                                    capture_output=True, text=True)
        except OSError:
            return sf._NATIVE_UNAVAILABLE
        if result.returncode != 0:
            if 'canc' in (result.stderr or '').lower():
                return None  # The user cancelled the panel.
            return sf._NATIVE_UNAVAILABLE
        return result.stdout.strip() or None

    def _open_gif_dialog(self, row):
        """This method opens a dialog in the figure.  The dialog has a text box for the folder
        of the GIFs."""
        self._open_dialog('gif')
        x0, y0, w, h = self._dialog_panel(6.0, 1.9)
        self._dialog_text('title', (x0 + 0.02 * w, y0 + h - 0.05),
                          'Save the GIFs in a folder', fontweight='bold')
        self._dialog_textbox('path', 'Folder ',
                             (x0 + 0.11 * w, y0 + 0.48 * h, 0.86 * w, 0.2 * h),
                             self._last_dir)
        self._dialog_text('error', (x0 + 0.02 * w, y0 + 0.33 * h), '', color='red')
        self._dialog_button('Save', (x0 + 0.58 * w, y0 + 0.06 * h, 0.18 * w, 0.2 * h),
                            lambda: self._gif_dialog_accept(row))
        self._dialog_button('Cancel', (x0 + 0.79 * w, y0 + 0.06 * h, 0.17 * w, 0.2 * h),
                            self._close_dialog)
        self.fig.canvas.draw_idle()

    def _gif_dialog_accept(self, row):
        folder = os.path.expanduser(self._dialog['widgets']['path'].text.strip())
        if not folder:
            self._dialog_error('Enter a folder')
            return
        self._write_gifs(row, folder)

    def _write_gifs(self, row, folder):
        """This method writes one GIF into ``folder`` for each panel that changes along the
        axis of the row.  The method does not overwrite an existing file.  Instead, the new
        GIF gets a numbered name.
        """
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            # For example, the path is an existing file.
            self._file_error(f"Cannot save the GIFs in {folder}: {e.strerror or e}")
            return
        written = []
        for i in self._moving_panels(row):
            path = self._unused_path(os.path.join(folder, self._gif_file_name(row, i)))
            volume, frame_axis, slice_axis, slice_index = self.stack.movie_view(
                i, self._row_axis(row, i))
            vmin, vmax = self.images[i].get_clim()
            try:
                self.movie_fn(volume, path, frame_axis=frame_axis, slice_axis=slice_axis,
                              slice_index=slice_index, vmin=vmin, vmax=vmax, fps=self.fps)
            except Exception as e:
                self._file_error(f"Failed to save a GIF: {e}")
                return
            written.append(os.path.basename(path))
        self._last_dir = os.path.abspath(folder)
        self._close_dialog(draw=False)
        count = '1 GIF' if len(written) == 1 else '{} GIFs'.format(len(written))
        self._show_message(True, message=sf.multiline(
            'Saved {} in {}:'.format(count, folder), *written, 'Press Esc to dismiss.'))

    @staticmethod
    def _unused_path(path):
        """This method returns ``path`` if no file exists there.  Otherwise it returns
        ``path`` with a number before its extension.
        """
        stem, extension = os.path.splitext(path)
        number = 2
        while os.path.exists(path):
            path = '{}_{}{}'.format(stem, number, extension)
            number += 1
        return path

    # --- Frame slider, stepping, and playback ---

    def _on_frame_slider(self, value):
        self._pause_if_playing()
        self.stack.set_second(value)
        # Every title gives the positions along the hidden axes.
        self.refresh()
        self._display_roi_stats()
        if self.stack.second_axis == 0:
            self._move_roi_marker()
        self.fig.canvas.draw_idle()

    def _step_frame(self, step):
        self._pause_if_playing()
        stack = self.stack
        self.frame_slider.set_val(
            int(np.clip(stack.second_position + step, 0, stack.second_count - 1)))

    def _toggle_play(self, row='frame'):
        """This method plays the slider of the row, or pauses it if it is playing.  Starting
        one row stops the other row.
        """
        playing_row = self._play_row
        if self.playing:
            self._stop_playback()
        if playing_row != row:
            self._start_playback(row)

    def _pause_if_playing(self):
        if self.playing:
            self._stop_playback()

    def _playback_regions(self, row):
        """This method returns the (bbox, artists) pairs that playback of the row redraws in
        each step.
        """
        regions = []
        for i in self._moving_panels(row):
            # The ROI circle and its statistics sit above the image, so they are
            # redrawn after it.
            artists = [self.images[i]] + [artist for artist in
                                          (self.circles[i], self.stats_texts[i])
                                          if artist is not None]
            regions.append((self.axes[i].bbox, artists))
        label = self._play_labels[row]
        regions.append((label.axes.bbox, [label]))
        # The marker of the ROI plot follows the frame.
        if row == 'frame' and self.roi_plot_ax.get_visible():
            regions.append((self.roi_plot_ax.bbox, [self._roi_marker]))
        return regions

    def _row_label_text(self, row):
        # The text gives the slider name and its position, as in "t = 3".
        return '{} = {}'.format(self._row_slider(row).label.get_text(),
                                self._row_position(row)[0])

    def _start_playback(self, row):
        slider = self._row_slider(row)
        if slider is None or self._dialog is not None or self._row_position(row)[1] < 2:
            return
        self.playing = True
        self._play_row = row
        self._hide_tooltips()
        self.play_buttons[row].label.set_text('Pause')
        # On a canvas that can blit, playback redraws only the images, the ROI graphics,
        # the playback label, and the plot marker.  The code marks these artists as animated.
        # This keeps the macosx backend from redrawing the whole figure on each step.  A canvas that
        # cannot blit, such as WebAgg, redraws the whole figure.
        canvas = self.fig.canvas
        blit = getattr(canvas, 'supports_blit', False)
        regions = self._playback_regions(row)
        if blit:
            for _bbox, artists in regions:
                for artist in artists:
                    artist.set_animated(True)
        slider.valtext.set_visible(False)
        label = self._play_labels[row]
        label.set_text(self._row_label_text(row))
        label.set_visible(True)
        canvas.draw()
        if blit:
            self._play_regions = [(bbox, canvas.copy_from_bbox(bbox), artists)
                                  for bbox, artists in regions]
        else:
            self._play_regions = None
        self._blit_playback()
        self._play_timer = canvas.new_timer(interval=max(1, int(round(1000 / self.fps))))
        self._play_timer.add_callback(self._play_step)
        self._play_timer.start()

    def _play_step(self):
        """This method advances playback by one position.  At the end, it returns to the
        start."""
        if not self.playing:
            return
        row = self._play_row
        position, count = self._row_position(row)
        if row == 'slice':
            changed = self.stack.set_master_index((position + 1) % count)
        else:
            changed = self.stack.set_second((position + 1) % count)
        for i in changed:
            self.images[i].set_data(self.stack.slice_image(i))
        self._play_labels[row].set_text(self._row_label_text(row))
        if row == 'frame':
            self._move_roi_marker()
        self._blit_playback()

    def _blit_playback(self):
        canvas = self.fig.canvas
        if self._play_regions is None:
            canvas.draw_idle()
            return
        for bbox, background, artists in self._play_regions:
            canvas.restore_region(background)
            for artist in artists:
                self.fig.draw_artist(artist)
            canvas.blit(bbox)
        try:
            canvas.flush_events()
        except NotImplementedError:
            pass

    def _stop_playback(self):
        row = self._play_row
        if self._play_timer is not None:
            self._play_timer.stop()
            self._play_timer = None
        self.playing = False
        self._play_row = None
        # Only a canvas that can blit has animated artists and saved regions.
        if self._play_regions is not None:
            for _bbox, _background, artists in self._play_regions:
                for artist in artists:
                    artist.set_animated(False)
        self._play_regions = None
        self._play_labels[row].set_visible(False)
        slider = self._row_slider(row)
        slider.valtext.set_visible(True)
        self.play_buttons[row].label.set_text('Play')
        # Playback does not update the slider, the titles, or the ROI plot, so they are
        # updated here.  Setting the slider moves the marker of the ROI plot or recomputes
        # its curves.
        slider.set_val(self._row_position(row)[0])
        self.refresh()
        self._display_roi_stats(force=True)
        self.fig.canvas.draw_idle()

    # --- Event wiring: playback stops before anything else changes the figure ---

    def _connect_events(self):
        super()._connect_events()
        canvas = self.fig.canvas
        # A resize makes the saved backgrounds the wrong size.
        canvas.mpl_connect('resize_event', lambda _event: self._pause_if_playing())
        canvas.mpl_connect('close_event', self._on_close)

    def _on_close(self, _event):
        # The window is going away, so only the timer is stopped.
        if self._play_timer is not None:
            self._play_timer.stop()
            self._play_timer = None
        self.playing = False
        self._play_row = None

    def _on_button_press(self, event):
        # A Play button handles its own clicks, which pause or switch the playback.
        if event.inaxes not in [button.ax for button in self.play_buttons.values()]:
            self._pause_if_playing()
        super()._on_button_press(event)

    def _on_motion(self, event):
        # Hover tooltips would change the figure under the playing images.
        if self.playing:
            return
        super()._on_motion(event)

    def _on_release(self, event):
        roi_changed = self.mode in (Mode.DRAW_ROI, Mode.MOVE_ROI, Mode.RESIZE_ROI)
        super()._on_release(event)
        if roi_changed:
            self._update_roi_plot()
            self.fig.canvas.draw_idle()

    def _on_key(self, event):
        # The space key pauses any playback, or plays the frame row.  The comma and period
        # keys step the frame row.
        if self._dialog is None:
            if event.key == ' ' and (self.playing or self.frame_slider is not None):
                if self.playing:
                    self._stop_playback()
                else:
                    self._start_playback('frame')
                return
            if event.key in (',', '.') and self.frame_slider is not None:
                self._step_frame(-1 if event.key == ',' else 1)
                return
        self._pause_if_playing()
        super()._on_key(event)

    def _show_message(self, show, message_type=None, message=None):
        if show and message_type == 'help':
            message = sf.multiline(
                'Left-click and drag on an image for an ROI',
                'Right-click an image for the menu',
                'Right-click the intensity slider or press Set range '
                'for exact bounds',
                'Plane t-x, t-y, or t-z shows one line of the volume in every frame',
                'Play and GIF beside a slider play its axis and save it as GIFs',
                'Press [space] to play or pause, [,] and [.] to step the frame row',
                'Press [esc] to remove ROI/messages/dialogs',
                'Close the window to quit')
            message_type = None
        super()._show_message(show, message_type=message_type, message=message)

    def _open_menu_dialog(self, items, event):
        # Mouse events give physical pixels, but the inherited method expects logical
        # pixels.  So the position is converted here for screens such as Retina.
        ratio = self.fig.canvas.device_pixel_ratio
        if ratio != 1:
            event = copy.copy(event)
            event.x, event.y = event.x / ratio, event.y / ratio
        super()._open_menu_dialog(items, event)

    # A dialog is laid out as if the figure were DIALOG_LAYOUT_HEIGHT inches tall.  Each
    # part is then mapped onto this figure, centered vertically.  So the dialog has the
    # same size and spacing in inches as in the slice viewer.

    def _dialog_panel(self, width_in, height_in):
        fig_w, fig_h = self.fig.get_size_inches()
        layout_h = min(fig_h, DIALOG_LAYOUT_HEIGHT)
        self._dialog['vertical_scale'] = layout_h / fig_h
        w = min(width_in / fig_w, 0.92)
        h = min(height_in / layout_h, 0.88)
        x0, y0 = (1 - w) / 2, (1 - h) / 2
        panel = self.fig.add_axes(self._to_figure((x0, y0, w, h)),
                                  zorder=sf.DIALOG_ZORDER + 1, facecolor='white')
        panel.set_xticks([])
        panel.set_yticks([])
        self._dialog['axes'].append(panel)
        self._dialog['panel_ax'] = panel
        return x0, y0, w, h

    def _to_figure(self, rect):
        """This method maps a rectangle from the dialog coordinates to figure fractions."""
        scale = self._dialog.get('vertical_scale', 1.0)
        x, y, w, h = rect
        return (x, 0.5 + (y - 0.5) * scale, w, h * scale)

    def _dialog_button(self, label, rect, callback, key=None):
        return super()._dialog_button(label, self._to_figure(rect), callback, key=key)

    def _dialog_textbox(self, name, label, rect, initial):
        return super()._dialog_textbox(name, label, self._to_figure(rect), initial)

    def _dialog_text(self, name, xy, text, **kwargs):
        x, y, _w, _h = self._to_figure((xy[0], xy[1], 0.0, 0.0))
        return super()._dialog_text(name, (x, y), text, **kwargs)

    def _main_widgets(self):
        widgets = super()._main_widgets()
        if self.frame_slider is not None:
            widgets.append(self.frame_slider)
        for buttons in self._row_buttons.values():
            widgets.extend(buttons)
        return widgets

    def _update_slice_slider(self):
        super()._update_slice_slider()
        self._set_row_visible('slice', self.stack.max_slices > 1)

    # --- Actions that change what the ROI plot shows ---

    def _on_slice_slider(self, value):
        self._pause_if_playing()
        super()._on_slice_slider(value)
        if self._roi_lines:
            self._update_roi_plot()
            self.fig.canvas.draw_idle()

    def _on_intensity_slider(self, value):
        self._pause_if_playing()
        super()._on_intensity_slider(value)

    def _on_transpose_button(self, i):
        super()._on_transpose_button(i)
        self._update_roi_plot()
        self.fig.canvas.draw_idle()

    def _apply_difference(self, baseline_index, comparison_index, use_abs):
        super()._apply_difference(baseline_index, comparison_index, use_abs)
        self.refresh()
        self._update_second_slider()
        self._update_roi_plot()
        self.fig.canvas.draw_idle()

    def _on_restore(self, i):
        super()._on_restore(i)
        self.refresh()
        self._update_second_slider()
        self._update_roi_plot()
        self.fig.canvas.draw_idle()

    def _finish_load(self, i, path, name):
        super()._finish_load(i, path, name)
        self._label_slice_slider()
        self._update_second_slider()
        self._update_roi_plot()
        self.fig.canvas.draw_idle()


def slice_viewer4d(*datasets, data_dicts=None, title='', vmin=None, vmax=None,
                   slice_label=None, slice_axis=None, cmap='gray',
                   show_instructions=True, block=True, save_fn=None, fps=5,
                   movie_fn=None):
    """This function opens an interactive viewer for one or more 4D volumes.

    The function builds a :class:`SliceViewer4D`, shows it, and returns it.  The viewer has
    every feature of :func:`slice_viewer`.  It adds a frame slider, space-time planes, Play
    and GIF buttons, and a plot of the ROI mean against frame.

    Args:
        *datasets (ndarray or None): One or more 2D, 3D, or 4D arrays to display.  A 4D
            array has axes ``(t, x, y, z)``.  A 2D or 3D array stays fixed in time.
        data_dicts (None, dict, or list of dict/None, optional): Dicts of strings that
            describe the volumes.  The viewer can display them.
        title (str, optional): The figure title.  Defaults to ''.
        vmin (float, optional): The minimum display intensity.  Defaults to the minimum
            over every voxel of every volume.
        vmax (float, optional): The maximum display intensity.  Defaults to the maximum
            over every voxel of every volume.
        slice_label (str or list of str, optional): The label at the start of each
            panel title.  Defaults to no label.
        slice_axis (int or list of int, optional): The axis to slice along.  Each array
            counts its own axes, so the value is 1, 2, or 3 for a 4D array and 0, 1, or 2
            for a 3D array.  Defaults to z.  If the arrays mix 3D and 4D, give a list.
        cmap (str, optional): The colormap.  Defaults to 'gray'.
        show_instructions (bool, optional): If True, the viewer shows the "Press h for
            help" hint.  Defaults to True.
        block (bool, optional): If True, the function waits until the window is closed.
            If False, it leaves the window open and returns at once, as in
            :func:`slice_viewer`.  Defaults to True.
        save_fn (callable, optional): A replacement for the built-in HDF5 writer.  The
            viewer calls ``save_fn(file_path, array, array_name, attributes_dict)``.
        fps (float, optional): The playback speed in frames per second.  The GIFs use
            the same speed.  Defaults to 5.
        movie_fn (callable, optional): The writer for the GIF buttons.  The viewer calls
            it with the arguments of mbirtorch's ``save_volume_as_gif``.  None leaves out
            the GIF buttons.  ``mbirtorch.slice_viewer4d`` passes ``save_volume_as_gif``.
            Defaults to None.

    Returns:
        SliceViewer4D: The viewer object.
    """
    viewer = SliceViewer4D(*datasets, data_dicts=data_dicts, title=title,
                           vmin=vmin, vmax=vmax, slice_label=slice_label,
                           slice_axis=slice_axis, cmap=cmap,
                           show_instructions=show_instructions, save_fn=save_fn,
                           fps=fps, movie_fn=movie_fn)
    viewer.show(block=block)
    # A nonblocking viewer is added to the list of open slice viewers.  A later blocking
    # call of either viewer closes the viewers in this list.
    if not block:
        sf._NONBLOCKING_VIEWERS.append(viewer)
        return viewer
    for nonblocking_viewer in sf._NONBLOCKING_VIEWERS:
        sf.plt.close(nonblocking_viewer.fig)
    sf._NONBLOCKING_VIEWERS.clear()
    if matplotlib.get_backend() == 'TkAgg':
        import gc
        gc.collect()
    return viewer

"""This module provides an interactive multi-volume slice viewer.

``VolumeStack`` holds the volumes and the display state as numpy arrays.  ``SliceViewer``
displays them with matplotlib.  The module imports numpy and matplotlib.  It imports h5py only
to read and write files.  A package that uses this viewer wraps ``slice_viewer`` to convert its
own tensors and data dicts.
"""

import enum
import os
import sys
import time
import warnings

import numpy as np
import matplotlib  # Only the base package.  Import does not load a GUI toolkit.

__all__ = ['SliceViewer', 'VolumeStack', 'slice_viewer']

# The shape of the placeholder volume shown for a dataset passed as None.
PLACEHOLDER_SHAPE = (20, 20, 20)

# --- Appearance constants ---
TOOLTIP_FONT_SIZE = 9
TOOLTIP_BOX_ALPHA = 0.9
TOOLTIP_OFFSET = (10, 10)
TOOLTIP_TEXT = "Click and drag to move\nClick edge to resize\nPress Esc to remove"

CIRCLE_COLOR = 'red'
CIRCLE_LINEWIDTH = 2
CIRCLE_ALPHA = 1.0
CIRCLE_FILL = False

SLICE_AXIS_FONT_SIZE = 9
SLICE_AXIS_LABEL_FONT_SIZE = 8
SLICE_AXIS_RADIO_SIZE = 30

STRIP_FONT_SIZE = 8
STATS_FONT_SIZE = 12
DIALOG_FONT_SIZE = 9
DIALOG_BODY_FONT_SIZE = 8
DIALOG_ZORDER = 50
DIALOG_PAGE_LINES = 30

ROI_STATS_THROTTLE_S = 0.3

# These backends have no interactive window.  On these backends, show() warns and returns.
NONINTERACTIVE_BACKENDS = {'agg', 'pdf', 'ps', 'svg', 'template', 'cairo'}

# The viewer redraws only the changed regions on these backends.  The macosx backend
# reports blit support, but it repaints the whole window on each blit.
BLIT_BACKENDS = {'agg', 'tkagg'}

# The file extensions that the file-load browser offers.  Save always writes .h5.
FILE_EXTENSIONS = ('.npy', '.npz', '.h5', '.hdf5')
FILE_DIALOG_ROWS = 11

# The native file dialog functions return this value when no native dialog can run.
# The caller then uses the file browser inside the figure.
_NATIVE_UNAVAILABLE = object()


def multiline(*lines):
    return '\n'.join(lines)


class Mode(enum.Enum):
    """This class is used to list the interaction modes of the viewer.  Exactly one mode is active
    at a time."""
    IDLE = enum.auto()
    DRAW_ROI = enum.auto()
    MOVE_ROI = enum.auto()
    RESIZE_ROI = enum.auto()
    SELECT_COMPARISON = enum.auto()


# _load_pyplot sets these names when the first viewer is constructed.  For this reason,
# importing this module does not choose a matplotlib backend or load a GUI toolkit.
plt = None
gridspec = None
make_axes_locatable = None
Slider = RangeSlider = RadioButtons = Button = TextBox = None
Bbox = None
Rectangle = None
IdentityTransform = None


def _load_pyplot():
    """This function imports pyplot and the widget classes the first time it is called."""
    global plt, gridspec, make_axes_locatable, Slider, RangeSlider, \
        RadioButtons, Button, TextBox, Bbox, Rectangle, IdentityTransform
    if plt is not None:
        return
    import matplotlib.pyplot as _plt
    from matplotlib import gridspec as _gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable as _mal
    from matplotlib.widgets import (Slider as _Slider, RangeSlider as _RangeSlider,
                                    RadioButtons as _RadioButtons,
                                    Button as _Button, TextBox as _TextBox)
    from matplotlib.transforms import (Bbox as _Bbox,
                                       IdentityTransform as _IdentityTransform)
    from matplotlib.patches import Rectangle as _Rectangle
    plt = _plt
    gridspec = _gridspec
    make_axes_locatable = _mal
    Slider, RangeSlider = _Slider, _RangeSlider
    RadioButtons = _RadioButtons
    Button, TextBox = _Button, _TextBox
    Bbox, Rectangle, IdentityTransform = _Bbox, _Rectangle, _IdentityTransform


class VolumeStack:
    """This class is used to hold the volumes and the display state of the slice viewer.

    The class uses only numpy and imports no GUI code.  The display state includes the axis
    permutation of each volume, the shared slice position, the intensity range, the difference
    images, and the data dicts.  The shared slice position is scaled to the depth of each volume.
    If the caller passes both ``vmin`` and ``vmax``, the class does not need to scan every voxel.

    Args:
        datasets (sequence of ndarray or None): One or more 2D or 3D arrays.  A 2D array
            gets a third axis of length 1.  A None entry becomes a placeholder volume of zeros.
        data_dicts (None, dict, or list of dict/None, optional): The dicts of strings that go
            with the volumes.  A single dict is accepted only for a single volume.  Otherwise,
            the list must have one entry for each volume.
        vmin (float, optional): The display minimum.  Defaults to the minimum over all volumes.
        vmax (float, optional): The display maximum.  Defaults to the maximum over all volumes.
        slice_label (str or list of str, optional): The label or labels for the slice title.
            Defaults to "Slice".
        slice_axis (int or list of int, optional): The axis to slice along (0, 1, or 2).  It can
            be one value for all volumes or a list with one value for each volume.  Defaults to 2.
    """

    def __init__(self, datasets, data_dicts=None, vmin=None, vmax=None,
                 slice_label=None, slice_axis=None):
        datasets = list(datasets)
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")
        self.n_volumes = len(datasets)

        self.axes_perms = self._normalize_slice_axes(slice_axis)
        self.labels = self._normalize_labels(slice_label)
        self.data_dicts = self._normalize_data_dicts(data_dicts)

        # original_data holds each volume in the axis order of the input.  data holds the
        # permuted view that is displayed, with the slice axis last.
        self.original_data = []
        self.data = []
        for i, dataset in enumerate(datasets):
            if dataset is None:
                dataset = np.zeros(PLACEHOLDER_SHAPE)
            dataset = np.asarray(dataset)
            if dataset.ndim == 2:
                dataset = dataset[..., np.newaxis]
            elif dataset.ndim != 3:
                raise ValueError("Each input data must be a 2D or 3D array")
            self.original_data.append(dataset)
            self.data.append(np.transpose(dataset, self.axes_perms[i]))

        # Each entry is None, or a dict for a difference image with the keys
        # 'comparison_index', 'use_abs', and 'prev_label'.
        self._difference_info = [None] * self.n_volumes

        # Each volume opens at its own middle slice.  The shared slider starts at the
        # middle slice of volume 0.
        self.cur_slices = [d.shape[2] // 2 for d in self.data]
        self.master_index = self.cur_slices[0]

        self.vmin, self.vmax = self.resolve_range(vmin, vmax)

    # --- Input normalization ---

    @staticmethod
    def perm_from_slice_axis(slice_axis):
        """This method returns the display permutation for a slice axis.

        The permutation lists the two in-plane axes in increasing order and then the slice
        axis.  For example, slice axis 0 gives [1, 2, 0].
        """
        slice_axis = int(slice_axis)
        if slice_axis not in (0, 1, 2):
            raise ValueError("slice_axis must be 0, 1, or 2")
        return sorted({0, 1, 2} - {slice_axis}) + [slice_axis]

    def _normalize_slice_axes(self, slice_axis):
        if slice_axis is None or isinstance(slice_axis, (int, np.integer)):
            axis = 2 if slice_axis is None else int(slice_axis)
            slice_axes = [axis] * self.n_volumes
        else:
            slice_axes = [int(s) for s in slice_axis]
            if len(slice_axes) != self.n_volumes:
                raise ValueError(
                    "slice_axis must be a single int or a list of ints of the "
                    "same length as the number of datasets")
        return [self.perm_from_slice_axis(s) for s in slice_axes]

    def _normalize_labels(self, slice_label):
        if slice_label is None or isinstance(slice_label, str):
            return ["Slice" if slice_label is None else slice_label] * self.n_volumes
        labels = [str(label) for label in slice_label]
        if len(labels) != self.n_volumes:
            raise ValueError(
                "slice_label must be a single string or a list of strings of "
                "the same length as the number of datasets")
        return labels

    def _normalize_data_dicts(self, data_dicts):
        if data_dicts is None:
            return [None] * self.n_volumes
        if isinstance(data_dicts, dict):
            data_dicts = [data_dicts]
        data_dicts = list(data_dicts)
        if len(data_dicts) != self.n_volumes or \
                not all(isinstance(d, dict) or d is None for d in data_dicts):
            raise ValueError(
                "data_dicts must be single dict or a list of dicts of the "
                "same length as the number of datasets")
        return data_dicts

    # --- Slice position ---
    # The master index is the shared slice position.  Each volume shows the slice at the
    # same fraction of its own depth.

    @property
    def max_slices(self):
        """This property is the largest depth of any volume in display orientation."""
        return max(d.shape[2] for d in self.data)

    @property
    def slice_counts(self):
        """This property is the list of the displayed depths of the volumes."""
        return [d.shape[2] for d in self.data]

    @property
    def slice_axes(self):
        """This property is the list of the slice axes of the volumes.

        The slice axis of a volume is the last entry of its permutation.
        """
        return [perm[-1] for perm in self.axes_perms]

    @property
    def master_fraction(self):
        """This property is the master index as a fraction of the largest depth.

        The fraction is in [0, 1].
        """
        if self.max_slices > 1:
            return self.master_index / (self.max_slices - 1)
        return 0.0

    def set_master_index(self, index):
        """This method sets the shared slice position and returns the indices of the volumes
        whose slice changed.

        The method clips the master index to the range 0 to max_slices - 1.  Each volume then
        shows slice ``round(fraction * (depth - 1))``.  In this way, a volume with fewer slices
        moves in proportion.
        """
        index = int(np.clip(int(np.round(index)), 0, self.max_slices - 1))
        self.master_index = index
        return self._recompute_slices()

    def _set_master_fraction(self, fraction):
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if self.max_slices > 1:
            self.master_index = int(np.round(fraction * (self.max_slices - 1)))
        else:
            self.master_index = 0
        return self._recompute_slices()

    def _recompute_slices(self):
        fraction = self.master_fraction
        changed = []
        for i, d in enumerate(self.data):
            new_slice = int(np.round(fraction * (d.shape[2] - 1)))
            if new_slice != self.cur_slices[i]:
                self.cur_slices[i] = new_slice
                changed.append(i)
        return changed

    def slice_image(self, i):
        """This method returns the 2D array currently displayed for volume ``i``."""
        return self.data[i][:, :, self.cur_slices[i]]

    # --- Axis permutations ---

    def set_perm(self, i, new_perm):
        """This method changes the orientation of volume ``i``.  It returns True if the
        permutation changed.

        ``new_perm`` is either a slice axis (int) or a full permutation of (0, 1, 2).  The
        method keeps ``master_fraction`` unchanged.  For this reason, every volume stays at the
        same fraction of its new depth.
        """
        if isinstance(new_perm, (int, np.integer)):
            new_perm = self.perm_from_slice_axis(new_perm)
        new_perm = [int(p) for p in new_perm]
        if sorted(new_perm) != [0, 1, 2]:
            raise ValueError("Permutation must be a permutation of (0, 1, 2)")
        if new_perm == self.axes_perms[i]:
            return False

        fraction = self.master_fraction
        inverse_perm = np.argsort(self.axes_perms[i])
        unpermuted = np.transpose(self.data[i], inverse_perm)
        self.axes_perms[i] = new_perm
        self.data[i] = np.transpose(unpermuted, new_perm)
        self._set_master_fraction(fraction)
        return True

    def transpose(self, i):
        """This method swaps the two in-plane axes of volume ``i``."""
        perm = list(self.axes_perms[i])
        perm[0], perm[1] = perm[1], perm[0]
        self.set_perm(i, perm)

    # --- Intensity range ---

    def data_range(self):
        """This method returns (min, max) over the current data of all volumes."""
        lo = min(float(np.min(d)) for d in self.data)
        hi = max(float(np.max(d)) for d in self.data)
        return lo, hi

    def resolve_range(self, vmin=None, vmax=None):
        """This method returns the display bounds after it fills missing bounds from the data.

        If the bounds are equal, the method moves them apart by a small amount.  If vmin > vmax,
        the method raises a ValueError.
        """
        if vmin is None or vmax is None:
            data_lo, data_hi = self.data_range()
            vmin = data_lo if vmin is None else vmin
            vmax = data_hi if vmax is None else vmax
        vmin, vmax = float(vmin), float(vmax)
        if vmin > vmax:
            raise ValueError("Minimum must be less than maximum")
        if vmin == vmax:
            eps = 1e-6
            scale = float(np.clip(eps * np.abs(vmax), a_min=eps, a_max=None))
            vmin, vmax = vmin - scale, vmax + scale
        return vmin, vmax

    def set_range(self, vmin=None, vmax=None):
        """This method sets the display range.  A bound that is None comes from the data."""
        self.vmin, self.vmax = self.resolve_range(vmin, vmax)
        return self.vmin, self.vmax

    # --- Difference images ---

    def is_difference(self, i):
        """This method returns True if volume ``i`` shows a difference image."""
        return self._difference_info[i] is not None

    def can_difference(self, baseline_index, comparison_index):
        """This method returns True if the two volumes can form a difference image.

        The two indices must differ, and the original shapes must be equal.  The orientations
        can differ, because :meth:`apply_difference` reorients the comparison volume to match
        the baseline volume.
        """
        return (baseline_index != comparison_index
                and self.original_data[baseline_index].shape
                == self.original_data[comparison_index].shape)

    def apply_difference(self, baseline_index, comparison_index, use_abs=False):
        """This method replaces volume ``baseline_index`` with (comparison - baseline).

        If ``use_abs`` is True, the method uses the absolute value of the difference.  The
        method uses the displayed data.  If the orientations differ, it reorients the comparison
        volume to match the baseline volume.  :meth:`restore` brings back the original data and
        label.
        """
        if not self.can_difference(baseline_index, comparison_index):
            raise ValueError(
                "Difference requires two distinct volumes with the same shape")
        comparison = self.data[comparison_index]
        if self.axes_perms[comparison_index] != self.axes_perms[baseline_index]:
            inverse_perm = np.argsort(self.axes_perms[comparison_index])
            comparison = np.transpose(np.transpose(comparison, inverse_perm),
                                      self.axes_perms[baseline_index])
        difference = comparison - self.data[baseline_index]
        if use_abs:
            difference = np.abs(difference)
        self._difference_info[baseline_index] = {
            'comparison_index': comparison_index,
            'use_abs': use_abs,
            'prev_label': self.labels[baseline_index],
        }
        self.data[baseline_index] = difference
        if use_abs:
            label_prepend = 'abs(Image {} minus current): '.format(comparison_index)
        else:
            label_prepend = 'Image {} minus current: '.format(comparison_index)
        self.labels[baseline_index] = label_prepend + self.labels[baseline_index]

    def restore(self, i):
        """This method resets volume ``i`` to its original data.  This ends any difference image."""
        self.data[i] = np.transpose(self.original_data[i], self.axes_perms[i])
        info = self._difference_info[i]
        if info is not None:
            self.labels[i] = info['prev_label']
        self._difference_info[i] = None

    # --- ROI statistics ---

    def roi_stats(self, i, x, y, radius):
        """This method returns statistics of the current slice of volume ``i`` inside a circle.

        ``x`` and ``y`` are the data coordinates (column, row) of the circle center.  The method
        returns a dict with the keys 'mean', 'std', 'min', and 'max'.  It returns None if no
        pixel center falls inside the circle.
        """
        slice_2d = self.slice_image(i)
        ny, nx = slice_2d.shape[:2]
        yv, xv = np.ogrid[:ny, :nx]
        mask = (xv - x) ** 2 + (yv - y) ** 2 <= radius ** 2
        values = slice_2d[mask]
        if values.size == 0:
            return None
        return {'mean': float(np.mean(values)), 'std': float(np.std(values)),
                'min': float(np.min(values)), 'max': float(np.max(values))}

    # --- File load ---

    @staticmethod
    def list_file_arrays(file_path):
        """This method lists the arrays in a file, for use in a dialog that chooses one.

        For .npz and .h5/.hdf5 files, the method returns (names, shapes).  For a .npy file, it
        returns None, because the file holds one unnamed array.  For other extensions, it
        raises a ValueError.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.npy':
            return None
        if ext == '.npz':
            with np.load(file_path) as array_dict:
                names = list(array_dict.files)
                shapes = [array_dict[name].shape for name in names]
            return names, shapes
        if ext in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(file_path, 'r') as f:
                names = list(f.keys())
                shapes = [f[name].shape for name in names]
            return names, shapes
        raise ValueError("Unsupported file type: {}".format(ext))

    @staticmethod
    def read_file_array(file_path, name=None):
        """This method reads one array from a file and returns (array, data_dict).

        For .npz and .h5/.hdf5 files, ``name`` selects the array.  If ``name`` is None, the
        method reads the first array.  For HDF5 files, the data dict holds the dataset
        attributes converted to strings.  For the other formats, the data dict is None.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.npy':
            return np.load(file_path), None
        if ext == '.npz':
            with np.load(file_path) as array_dict:
                key = array_dict.files[0] if name is None else name
                return array_dict[key], None
        if ext in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(file_path, 'r') as f:
                key = list(f.keys())[0] if name is None else name
                dataset = f[key]
                data_dict = {
                    str(k): v.decode() if isinstance(v, bytes) else str(v)
                    for k, v in dataset.attrs.items()
                } or None
                array = dataset[()]
            # If the file states its axis order, the array is permuted to the display order.
            # The display order is (row, col, slice) or (time, row, col, slice).
            axes = (data_dict or {}).get('axes')
            if axes:
                names = axes.split(',')
                wanted = [n for n in ('time', 'row', 'col', 'slice') if n in names]
                if sorted(names) == sorted(wanted) and len(names) == array.ndim:
                    array = np.transpose(array, [names.index(n) for n in wanted])
            return array, data_dict
        raise ValueError("Unsupported file type: {}".format(ext))

    def load_array(self, image_index, new_array, data_dict=None):
        """This method replaces one or more volumes with a loaded array.  It returns the indices
        of the replaced volumes.

        A 2D or 3D array replaces volume ``image_index``.  A 4D array replaces volumes 0 to k-1
        with its 3D parts along the last axis, where k = min(last dimension, n_volumes).  In
        this case, ``image_index`` becomes min(image_index, k - 1).  For any other number of
        dimensions, the method raises a ValueError.  A replaced volume is no longer a
        difference image.  The slice position moves to the middle slice of volume
        ``image_index``.
        """
        new_array = np.asarray(new_array)
        if new_array.ndim == 2:
            new_array = new_array[..., np.newaxis]
        if new_array.ndim == 3:
            replaced = [image_index]
            self.original_data[image_index] = new_array
        elif new_array.ndim == 4:
            if new_array.shape[-1] == 0:
                raise ValueError("Loaded 4D array must contain at least one volume")
            num_volumes_to_load = min(new_array.shape[-1], self.n_volumes)
            replaced = list(range(num_volumes_to_load))
            for j in replaced:
                self.original_data[j] = new_array[..., j]
            image_index = min(image_index, num_volumes_to_load - 1)
        else:
            raise ValueError("Loaded array must be 2D, 3D, or 4D")

        self.data_dicts[image_index] = data_dict
        # Volume image_index returns to the standard display order for its slice axis.
        # Other replaced volumes keep their permutations.
        self.axes_perms[image_index] = self.perm_from_slice_axis(
            self.axes_perms[image_index][-1])
        for j in replaced:
            if self._difference_info[j] is not None:
                self.labels[j] = self._difference_info[j]['prev_label']
                self._difference_info[j] = None
            self.data[j] = np.transpose(self.original_data[j], self.axes_perms[j])

        depth = self.data[image_index].shape[2]
        middle_fraction = (depth // 2) / (depth - 1) if depth > 1 else 0.0
        self._set_master_fraction(middle_fraction)
        return replaced


def _save_data_hdf5(file_path, array, array_name='recon', attributes_dict=None):
    """This function writes one HDF5 dataset with string attributes.  It is the default save
    function of the viewer.

    The file format matches ``export_recon_hdf5``.  The dataset is in (slice, col, row) or
    (time, slice, col, row) order.  The data dict is stored as text attributes.
    """
    import h5py
    array = np.asarray(array)
    axes = {3: 'slice,col,row', 4: 'time,slice,col,row'}.get(array.ndim)
    if axes is not None:
        array = np.transpose(array, {3: (2, 1, 0), 4: (0, 3, 2, 1)}[array.ndim])
    directory = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(directory, exist_ok=True)
    with h5py.File(file_path, 'w') as f:
        dataset = f.create_dataset(array_name, data=array)
        for key, value in (attributes_dict or {}).items():
            dataset.attrs[str(key)] = str(value)
        if axes is not None:
            dataset.attrs['format'] = 'mbirtorch_recon_v2'
            dataset.attrs['axes'] = axes


# Native Tk dialogs.  Each dialog creates a hidden Tk root when it is called and destroys
# the root before it returns.  For this reason, importing this module does not load a GUI toolkit.

def _tk_menu(labels):
    """This function shows a popup menu at the mouse pointer and returns the chosen label.

    The function returns None if the user closes the menu without a choice.  The user can close
    the menu with Escape or by clicking outside it.
    """
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    result = {'label': None}
    try:
        top = tkinter.Toplevel(root)
        top.overrideredirect(True)
        listbox = tkinter.Listbox(
            top, height=len(labels), activestyle='none', relief='solid',
            borderwidth=1, highlightthickness=0, exportselection=False)
        for label in labels:
            listbox.insert('end', label)
        listbox.configure(width=max(len(label) for label in labels) + 2)
        listbox.pack()

        def choose(_event=None):
            selection = listbox.curselection()
            if selection:
                result['label'] = labels[selection[0]]
            top.destroy()

        def highlight(event):
            listbox.selection_clear(0, 'end')
            listbox.selection_set(listbox.nearest(event.y))

        listbox.bind('<ButtonRelease-1>', choose)
        listbox.bind('<Return>', choose)
        listbox.bind('<Motion>', highlight)
        top.bind('<Escape>', lambda _event: top.destroy())
        top.geometry(f'+{root.winfo_pointerx()}+{root.winfo_pointery()}')
        top.attributes('-topmost', True)
        top.lift()
        listbox.focus_force()
        # The popup closes when it loses focus.  The binding is made 200 ms later,
        # so that it cannot fire while the popup is being created.
        top.after(200, lambda: top.winfo_exists() and top.bind(
            '<FocusOut>', lambda _event: top.destroy()))
        root.wait_window(top)
    finally:
        root.destroy()
    return result['label']


def _tk_range_dialog(current_vmin, current_vmax):
    """This function shows a modal dialog in which the user enters the min and max.

    The function returns ('apply', min_text, max_text) for Apply, ('data',) for Data range, or
    None for Cancel.  A blank entry keeps the current bound.
    """
    import tkinter
    from tkinter import ttk

    root = tkinter.Tk()
    root.withdraw()
    result = {'value': None}
    try:
        top = tkinter.Toplevel(root)
        top.title('Set intensity range')
        top.resizable(False, False)
        frame = ttk.Frame(top, padding=12)
        frame.grid()
        hint = (f'Blank keeps the current bound '
                f'({current_vmin:.6g}, {current_vmax:.6g}).')
        ttk.Label(frame, text=hint).grid(column=0, row=0, columnspan=3,
                                         sticky='w', pady=(0, 8))
        ttk.Label(frame, text='Min').grid(column=0, row=1, sticky='w')
        min_entry = ttk.Entry(frame, width=18)
        min_entry.grid(column=1, row=1, columnspan=2, sticky='we', pady=2)
        ttk.Label(frame, text='Max').grid(column=0, row=2, sticky='w')
        max_entry = ttk.Entry(frame, width=18)
        max_entry.grid(column=1, row=2, columnspan=2, sticky='we', pady=2)

        def apply(_event=None):
            result['value'] = ('apply', min_entry.get(), max_entry.get())
            top.destroy()

        def data_range():
            result['value'] = ('data',)
            top.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(column=0, row=3, columnspan=3, pady=(10, 0))
        ttk.Button(buttons, text='Apply',
                   command=apply).grid(column=0, row=0, padx=3)
        ttk.Button(buttons, text='Data range',
                   command=data_range).grid(column=1, row=0, padx=3)
        ttk.Button(buttons, text='Cancel',
                   command=top.destroy).grid(column=2, row=0, padx=3)
        top.bind('<Return>', apply)
        top.bind('<Escape>', lambda _event: top.destroy())
        top.attributes('-topmost', True)
        top.lift()
        min_entry.focus_force()
        root.wait_window(top)
    finally:
        root.destroy()
    return result['value']


# The keys that the dict editor always shows when the user saves.
DICT_EDITOR_KEYS = ('model_params', 'notes', 'recon_log', 'recon_params')


def _tk_dict_editor(data_dict):
    """This function shows a modal editor for the data dict that Save writes.

    The function returns ('save', edited_dict) for Save.  The edited dict leaves out blank
    entries.  The function returns ('as-is',) to save the dict unchanged, or None to cancel
    the save.
    """
    import tkinter
    from tkinter import ttk

    existing = {str(k): str(v) for k, v in (data_dict or {}).items()}
    keys = sorted(set(DICT_EDITOR_KEYS) | set(existing))
    values = {key: existing.get(key, '') for key in keys}

    root = tkinter.Tk()
    root.withdraw()
    result = {'value': None}
    try:
        top = tkinter.Toplevel(root)
        top.title('Edit data dict before saving')
        frame = ttk.Frame(top, padding=10)
        frame.grid(sticky='nsew')
        listbox = tkinter.Listbox(frame, height=max(len(keys), 4),
                                  exportselection=False)
        for key in keys:
            listbox.insert('end', key)
        listbox.grid(column=0, row=0, sticky='ns', padx=(0, 8))
        text = tkinter.Text(frame, width=64, height=18, wrap='none')
        text.grid(column=1, row=0, sticky='nsew')
        state = {'key': keys[0]}
        listbox.selection_set(0)
        text.insert('1.0', values[state['key']])

        def store_current():
            values[state['key']] = text.get('1.0', 'end-1c')

        def on_select(_event=None):
            selection = listbox.curselection()
            if not selection or keys[selection[0]] == state['key']:
                return
            store_current()
            state['key'] = keys[selection[0]]
            text.delete('1.0', 'end')
            text.insert('1.0', values[state['key']])

        listbox.bind('<<ListboxSelect>>', on_select)

        def save():
            store_current()
            edited = {key: value for key, value in values.items() if value}
            result['value'] = ('save', edited)
            top.destroy()

        def save_as_is():
            result['value'] = ('as-is',)
            top.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(column=0, row=1, columnspan=2, pady=(8, 0))
        ttk.Button(buttons, text='Save',
                   command=save).grid(column=0, row=0, padx=4)
        ttk.Button(buttons, text='Save without editing',
                   command=save_as_is).grid(column=1, row=0, padx=4)
        ttk.Button(buttons, text='Cancel save',
                   command=top.destroy).grid(column=2, row=0, padx=4)
        top.bind('<Escape>', lambda _event: top.destroy())
        top.attributes('-topmost', True)
        top.lift()
        root.wait_window(top)
    finally:
        root.destroy()
    return result['value']


class SliceViewer:
    """This class is used to display 2D and 3D arrays in an interactive matplotlib slice viewer.

    The viewer shows one panel for each volume.  The panels share a slice slider and an
    intensity slider.  A right-click on a panel opens a menu of actions for the image in that
    panel.  A left-drag draws a circular ROI and shows its statistics.  The 'h' key shows help.
    The constructor builds the figure, and :meth:`show` displays the figure.  ``self.stack``
    holds the :class:`VolumeStack`.

    Args:
        *datasets (ndarray or None): One or more 2D or 3D arrays to display.  A 2D array gets
            a third axis of length 1.  A None entry becomes a placeholder array of zeros.
        data_dicts (None, dict, or list of dict/None, optional): The dicts of strings that go
            with the volumes.  The user can view them in the viewer.
        title (str, optional): The figure title.  Defaults to ''.
        vmin (float, optional): The minimum display intensity.  Defaults to the minimum over
            all datasets.
        vmax (float, optional): The maximum display intensity.  Defaults to the maximum over
            all datasets.
        slice_label (str or list of str, optional): The label or labels shown before the slice
            number in each panel title.  Defaults to "Slice".
        slice_axis (int or list of int, optional): The axis to slice along (0, 1, or 2).
            Defaults to the last axis (2).
        cmap (str, optional): The colormap.  Defaults to 'gray'.
        show_instructions (bool, optional): If True, the viewer shows the "Press h for help"
            hint.  Defaults to True.
        save_fn (callable, optional): A function that replaces the built-in HDF5 writer.  The
            viewer calls it as ``save_fn(file_path, array, array_name, attributes_dict)``.
            Defaults to a minimal h5py writer.
    """

    def __init__(self, *datasets, data_dicts=None, title='', vmin=None,
                 vmax=None, slice_label=None, slice_axis=None, cmap='gray',
                 show_instructions=True, save_fn=None):
        _load_pyplot()
        self.stack = VolumeStack(datasets, data_dicts=data_dicts, vmin=vmin,
                                 vmax=vmax, slice_label=slice_label,
                                 slice_axis=slice_axis)
        self.title = title
        self.cmap = cmap
        self.show_instructions = show_instructions
        self.save_fn = save_fn if save_fn is not None else _save_data_hdf5

        # The interaction state must be set before any widget or callback is created.
        # Otherwise a callback could see a partly built viewer.
        self._init_interaction_state()
        self._build_figure()
        self._connect_events()

    # --- Construction ---

    def _init_interaction_state(self):
        n = self.stack.n_volumes
        self.mode = Mode.IDLE
        self._mode_data = {}
        # sync_limits and sync_axes are user settings.  _in_sync_callback is not a
        # setting.  It blocks a callback from running inside another callback.
        self.sync_limits = True
        self.sync_axes = len(set(self.stack.slice_axes)) == 1
        self._in_sync_callback = False
        # Each volume has one ROI circle and one statistics text.
        self.circles = [None] * n
        self.stats_texts = [None] * n
        self.tooltips = []
        # These two values limit how often the ROI statistics are updated.
        self._last_stats_time = 0.0
        self._stats_timer = None
        self._dialog = None
        self._message_artist = None
        # Blitting starts after the first full draw.
        self.enable_blit = True
        self._renderer_ready = False
        self._clear_rect = None
        self._last_blit_regions = {}
        self._last_dir = os.getcwd()

    def _build_figure(self):
        n = self.stack.n_volumes
        self.fig = plt.figure(figsize=(6 * n, 8))
        self.fig.suptitle(self.title)
        self.gs = gridspec.GridSpec(nrows=4, ncols=n,
                                    height_ratios=[15, 1.7, 1, 1],
                                    left=0.12, right=0.95, top=0.90,
                                    bottom=0.05, hspace=0.5, figure=self.fig)
        self.axes = [None] * n
        self.caxes = [None] * n
        self.images = [None] * n
        self._create_panels()
        self._create_tooltips()
        self._create_axis_row()
        self._create_slice_slider()
        self._create_intensity_slider()

        # This opaque rectangle is drawn under each partial redraw.  It is marked
        # animated, so a full draw does not draw it.
        self._clear_rect = Rectangle((0, 0), 1, 1,
                                     facecolor=self.fig.get_facecolor(),
                                     edgecolor='none', animated=True,
                                     transform=IdentityTransform())
        self.fig.add_artist(self._clear_rect)

        if self.show_instructions:
            self.fig.text(0.01, 0.25, multiline('Press h', 'for help'),
                          fontdict={'color': 'red'})

    def _title_text(self, i):
        stack = self.stack
        return multiline(
            f"{stack.labels[i]} {stack.cur_slices[i]}",
            f"Shape: {stack.original_data[i].shape}, "
            f"Axes: {np.array(stack.axes_perms[i])}")

    def _create_panels(self):
        stack = self.stack
        for i in range(stack.n_volumes):
            ax = self.fig.add_subplot(self.gs[0, i])
            img = ax.imshow(stack.slice_image(i), cmap=self.cmap,
                            aspect='equal', vmin=stack.vmin, vmax=stack.vmax)
            # _reset_view sets the axis limits.  Autoscale is off, because set_extent would
            # otherwise apply the extent of one volume to the other volumes.
            ax.set_autoscale_on(False)
            ax.set_title(self._title_text(i), fontsize=10)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            self.fig.colorbar(img, cax=cax, orientation='vertical')
            ax.zorder = 2
            cax.zorder = 1
            self.axes[i] = ax
            self.caxes[i] = cax
            self.images[i] = img

    def _create_tooltips(self):
        # This is the only place that creates the tooltips, and it runs once.
        self.tooltips = [
            ax.annotate(TOOLTIP_TEXT, xy=(0, 0), xytext=TOOLTIP_OFFSET,
                        textcoords='offset points', ha='left',
                        fontsize=TOOLTIP_FONT_SIZE,
                        bbox=dict(boxstyle='round', fc='w',
                                  alpha=TOOLTIP_BOX_ALPHA),
                        arrowprops=dict(arrowstyle='->'), visible=False)
            for ax in self.axes
        ]

    # --- Slice-axis radio buttons and the Set range button ---

    def _create_axis_row(self):
        n = self.stack.n_volumes
        # Each grid position is split into a part for the radio buttons and a spare part.
        # The last spare part holds the Set range button.
        self._radio_slots = []
        spare_slots = []
        for i in range(n):
            sub = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=self.gs[1, i], wspace=0.3)
            self._radio_slots.append(sub[0, 0])
            spare_slots.append(sub[0, 1])

        self.axis_radios = []
        self._radio_axes = []
        self._rebuild_axis_radios()

        range_ax = self.fig.add_subplot(spare_slots[-1])
        self.range_button = Button(range_ax, 'Set range')
        self.range_button.label.set_fontsize(STRIP_FONT_SIZE)
        self.range_button.on_clicked(lambda _event: self._open_range_dialog())

    def _rebuild_axis_radios(self):
        for ax in self._radio_axes:
            ax.remove()
        self._radio_axes = []
        self.axis_radios = []

        def make_radio(slot, active):
            ax = self.fig.add_subplot(slot)
            ax.set_title("Slice axis", loc='left',
                         fontsize=SLICE_AXIS_FONT_SIZE)
            radio = RadioButtons(ax, labels=["0", "1", "2"], active=active,
                                 radio_props={'s': [SLICE_AXIS_RADIO_SIZE]})
            for label in radio.labels:
                label.set_fontsize(SLICE_AXIS_LABEL_FONT_SIZE)
            self._radio_axes.append(ax)
            self.axis_radios.append(radio)
            return radio

        if self.sync_axes or self.stack.n_volumes == 1:
            radio = make_radio(self._radio_slots[0],
                               self.stack.slice_axes[0])
            radio.on_clicked(
                lambda label: self._on_axis_selected(None, int(label)))
        else:
            for i in range(self.stack.n_volumes):
                radio = make_radio(self._radio_slots[i],
                                   self.stack.slice_axes[i])
                radio.on_clicked(
                    lambda label, i=i: self._on_axis_selected(i, int(label)))

    # --- Sliders ---

    def _slider_slot(self, row):
        # The slider is inset within its row.  This keeps the label on the left and the
        # value on the right inside the figure.
        sub = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=self.gs[row, :], width_ratios=[1.7, 8.0, 2.0])
        return sub[0, 1]

    def _create_slice_slider(self):
        self._slice_slider_ax = self.fig.add_subplot(self._slider_slot(2))
        self.slice_slider = None
        if self.stack.max_slices > 1:
            self._make_slice_slider()
        else:
            self._slice_slider_ax.set_visible(False)

    def _make_slice_slider(self):
        self.slice_slider = Slider(self._slice_slider_ax, label="Slice",
                                   valmin=0,
                                   valmax=self.stack.max_slices - 1,
                                   valinit=self.stack.master_index,
                                   valstep=1, valfmt='%0.0f')
        self.slice_slider.drawon = False
        self.slice_slider.on_changed(self._on_slice_slider)
        self._slice_slider_ax.set_visible(True)

    @staticmethod
    def _intensity_valfmt(vmin, vmax):
        log_range = np.log10(vmax - vmin)
        digits = max(-int(np.round(log_range)) + 2, 0)
        return '%0.' + str(digits) + 'f'

    def _create_intensity_slider(self):
        ax = self.fig.add_subplot(self._slider_slot(3))
        stack = self.stack
        self.intensity_slider = RangeSlider(
            ax=ax, label="Intensity range", valmin=stack.vmin,
            valmax=stack.vmax, valinit=(stack.vmin, stack.vmax),
            valfmt=self._intensity_valfmt(stack.vmin, stack.vmax))
        self.intensity_slider.drawon = False
        self.intensity_slider.on_changed(self._on_intensity_slider)

    # --- Event wiring ---

    def _connect_events(self):
        canvas = self.fig.canvas
        canvas.mpl_connect('button_press_event', self._on_button_press)
        canvas.mpl_connect('motion_notify_event', self._on_motion)
        canvas.mpl_connect('button_release_event', self._on_release)
        canvas.mpl_connect('key_press_event', self._on_key)
        canvas.mpl_connect('draw_event', self._on_draw_event)
        for ax in self.axes:
            ax.callbacks.connect(
                'xlim_changed', lambda a: self._sync_limits_from(a, 'x'))
            ax.callbacks.connect(
                'ylim_changed', lambda a: self._sync_limits_from(a, 'y'))
        self._patch_toolbar()

    def _patch_toolbar(self):
        """This method changes the toolbar so that Home, Back, and Forward restore the view of
        each panel separately.

        On the macosx backend, the method also prevents the cancellation of a fast drag zoom.
        """
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if toolbar is None:
            return
        import types

        stock_update_view = type(toolbar)._update_view

        def _update_view(tb):
            already_syncing = self._in_sync_callback
            self._in_sync_callback = True
            try:
                stock_update_view(tb)
            finally:
                self._in_sync_callback = already_syncing

        toolbar._update_view = types.MethodType(_update_view, toolbar)

        # In standard matplotlib, Home does nothing if there are no saved views.
        # In this viewer, Home then resets each panel to its full extent.
        stock_home = type(toolbar).home

        def home(tb, *args):
            if tb._nav_stack() is None:
                self._reset_view()
                self.fig.canvas.draw_idle()
                return
            stock_home(tb, *args)

        toolbar.home = types.MethodType(home, toolbar)

        if matplotlib.get_backend().lower() == 'macosx':
            stock_drag_zoom = type(toolbar).drag_zoom

            def drag_zoom(tb, event):
                if tb._zoom_info is not None and not event.buttons:
                    event.buttons = frozenset({tb._zoom_info.button})
                stock_drag_zoom(tb, event)

            toolbar.drag_zoom = types.MethodType(drag_zoom, toolbar)

    def _on_draw_event(self, _event):
        self._renderer_ready = True

    def _toolbar_active(self):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        return bool(getattr(toolbar, 'mode', ''))

    def _sync_limits_from(self, src_ax, which):
        # This runs for both the x and y limits.  sync_limits is the user setting.
        # _in_sync_callback blocks a call from inside another call.
        if not self.sync_limits or self._in_sync_callback:
            return
        if src_ax not in self.axes:
            return
        self._in_sync_callback = True
        try:
            lim = src_ax.get_xlim() if which == 'x' else src_ax.get_ylim()
            for ax in self.axes:
                if ax is not src_ax:
                    (ax.set_xlim if which == 'x' else ax.set_ylim)(lim)
        finally:
            self._in_sync_callback = False
        self.fig.canvas.draw_idle()

    # --- Mouse button handling ---

    def _volume_index_of(self, ax):
        try:
            return self.axes.index(ax)
        except ValueError:
            return None

    def _on_button_press(self, event):
        if self._dialog is not None:
            # A click outside an open context menu closes the menu.  The widgets of
            # every other dialog handle their own events.
            if (self._dialog['kind'] == 'menu'
                    and event.inaxes not in self._dialog.get('menu_axes', ())):
                self._close_dialog()
            return
        if event.button == 3:
            # A right-click works even while the pan or zoom tool of the toolbar is active.
            if event.inaxes is self.intensity_slider.ax:
                self._open_range_dialog()
                return
            i = self._volume_index_of(event.inaxes)
            if i is not None and self.mode is not Mode.SELECT_COMPARISON:
                self._open_context_menu(i, event)
            return
        if event.button != 1:
            return
        # The choice of a comparison image takes priority over the toolbar tools.  While the
        # instructions are shown, a left-click selects an image even if pan or zoom is on.
        if self.mode is Mode.SELECT_COMPARISON:
            i = self._volume_index_of(event.inaxes)
            if i is not None:
                self._complete_difference(i)
            return
        if self._toolbar_active():
            return
        i = self._volume_index_of(event.inaxes)
        if i is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        # A click on the edge of a circle resizes the circle.  A click inside a circle
        # moves the circle.  A click anywhere else starts a new ROI.
        for circle in self.circles:
            if circle is None:
                continue
            x, y = circle.center
            radius = circle.get_radius()
            dist = np.hypot(event.xdata - x, event.ydata - y)
            if abs(dist - radius) <= 0.1 * radius:
                self._set_mode(Mode.RESIZE_ROI, anchor=(x, y))
                return
            if dist <= radius:
                self._set_mode(Mode.MOVE_ROI,
                               offset=(event.xdata - x, event.ydata - y))
                return
        self._remove_roi_graphics()
        self._set_mode(Mode.DRAW_ROI, start=(event.xdata, event.ydata))
        for j, ax in enumerate(self.axes):
            circle = plt.Circle((event.xdata, event.ydata), 0,
                                color=CIRCLE_COLOR, lw=CIRCLE_LINEWIDTH,
                                fill=CIRCLE_FILL, alpha=CIRCLE_ALPHA)
            self.circles[j] = circle
            ax.add_patch(circle)
        self._hide_tooltips()
        self._partial_redraw()

    def _set_mode(self, mode, **data):
        self.mode = mode
        self._mode_data = data

    def _on_motion(self, event):
        if self._dialog is not None or self._toolbar_active():
            return
        if event.inaxes not in self.axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.mode is Mode.IDLE or self.mode is Mode.SELECT_COMPARISON:
            self._update_tooltips(event)
            return
        if self.mode is Mode.DRAW_ROI:
            x0, y0 = self._mode_data['start']
            radius = np.hypot(event.xdata - x0, event.ydata - y0)
            for circle in self.circles:
                circle.set_radius(radius)
        elif self.mode is Mode.MOVE_ROI:
            dx, dy = self._mode_data['offset']
            center = (event.xdata - dx, event.ydata - dy)
            for circle in self.circles:
                circle.center = center
        elif self.mode is Mode.RESIZE_ROI:
            x0, y0 = self._mode_data['anchor']
            radius = np.hypot(event.xdata - x0, event.ydata - y0)
            for circle in self.circles:
                circle.set_radius(radius)
        self._display_roi_stats()
        self._partial_redraw()

    def _on_release(self, _event):
        if self.mode in (Mode.DRAW_ROI, Mode.MOVE_ROI, Mode.RESIZE_ROI):
            self._set_mode(Mode.IDLE)
            # The final update always runs, so that the statistics are current after a fast release.
            self._display_roi_stats(force=True)
            self._partial_redraw()

    def _on_key(self, event):
        if self._dialog is not None:
            if event.key == 'escape':
                self._close_dialog()
            return
        handlers = {'h': lambda: self._show_message(True, message_type='help'),
                    'escape': self._on_escape}
        handler = handlers.get(event.key)
        if handler is not None:
            handler()

    def _on_escape(self):
        self._show_message(False)
        self._set_mode(Mode.IDLE)
        self._remove_roi_graphics()
        self.fig.canvas.draw_idle()

    # --- ROI graphics, tooltips, and statistics ---

    def _remove_roi_graphics(self):
        for artist in self.circles + self.stats_texts:
            if artist is not None:
                artist.remove()
        n = self.stack.n_volumes
        self.circles = [None] * n
        self.stats_texts = [None] * n
        self._hide_tooltips()

    def _hide_tooltips(self):
        for tooltip in self.tooltips:
            tooltip.set_visible(False)

    def _update_tooltips(self, event):
        changed = []
        for i, circle in enumerate(self.circles):
            if circle is None:
                continue
            x, y = circle.center
            radius = circle.get_radius()
            dist = np.hypot(event.xdata - x, event.ydata - y)
            visible = dist <= 1.1 * radius and event.inaxes is self.axes[i]
            if visible:
                self.tooltips[i].xy = circle.center
            if visible != self.tooltips[i].get_visible():
                self.tooltips[i].set_visible(visible)
                changed.append(i)
        if changed:
            self._partial_redraw(changed)

    def _display_roi_stats(self, force=False):
        if all(circle is None for circle in self.circles):
            return
        now = time.monotonic()
        if not force and now - self._last_stats_time < ROI_STATS_THROTTLE_S:
            # This timer runs one final update after a rapid series of moves.
            self._schedule_trailing_stats()
            return
        self._last_stats_time = now
        for i, circle in enumerate(self.circles):
            if circle is None:
                continue
            x, y = circle.center
            stats = self.stack.roi_stats(i, x, y, circle.get_radius())
            if stats is None:
                continue
            text = multiline(
                f"(µ, σ)=({stats['mean']:.3g}, {stats['std']:.3g})",
                f"(min, max)=({stats['min']:.3g}, {stats['max']:.3g})")
            if self.stats_texts[i] is not None:
                self.stats_texts[i].set_text(text)
            else:
                self.stats_texts[i] = self.axes[i].text(
                    0.05, 0.95, text, transform=self.axes[i].transAxes,
                    fontsize=STATS_FONT_SIZE, va='top',
                    bbox=dict(facecolor='white', alpha=1.0))

    def _schedule_trailing_stats(self):
        if self._stats_timer is not None:
            return
        timer = self.fig.canvas.new_timer(
            interval=int(ROI_STATS_THROTTLE_S * 1000) + 50)
        timer.single_shot = True
        timer.add_callback(self._trailing_stats_fire)
        timer.start()
        self._stats_timer = timer

    def _trailing_stats_fire(self):
        self._stats_timer = None
        self._display_roi_stats(force=True)
        self._partial_redraw()

    # --- Messages ---

    def _show_message(self, show, message_type=None, message=None):
        if self._message_artist is not None:
            self._message_artist.remove()
            self._message_artist = None
        if show:
            if message_type == 'help':
                message = multiline(
                    'Left-click and drag on an image for an ROI',
                    'Right-click an image for the menu',
                    'Right-click the intensity slider or press Set range '
                    'for exact bounds',
                    'Press [esc] to remove ROI/messages/dialogs',
                    'Close the window to quit')
            elif message_type == 'difference':
                message = multiline(
                    'Select another image of the same shape',
                    'or press [esc] to exit')
            if message is None:
                return
            self._message_artist = self.fig.text(
                0.25, 0.5, message, ha='left', va='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9), zorder=20)
        self.fig.canvas.draw_idle()

    # --- In-place refresh ---

    def refresh(self, volume_index=None):
        """This method copies the data, extent, and title of each volume from the stack into its
        panel."""
        indices = (range(self.stack.n_volumes) if volume_index is None
                   else [volume_index])
        for i in indices:
            slice_2d = self.stack.slice_image(i)
            self.images[i].set_data(slice_2d)
            h, w = slice_2d.shape
            self.images[i].set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
            self.axes[i].set_title(self._title_text(i), fontsize=10)

    def _reset_view(self, volume_index=None):
        # _in_sync_callback is set here.  A reset gives each volume its own full extent.
        # The synchronization must not copy this extent to panels with a different shape.
        indices = (range(self.stack.n_volumes) if volume_index is None
                   else [volume_index])
        already_syncing = self._in_sync_callback
        self._in_sync_callback = True
        try:
            for i in indices:
                h, w = self.stack.slice_image(i).shape
                self.axes[i].set_xlim(-0.5, w - 0.5)
                self.axes[i].set_ylim(h - 0.5, -0.5)
        finally:
            self._in_sync_callback = already_syncing

    def _update_slice_slider(self):
        """This method updates the slice slider after the depths of the volumes change.

        The method updates the range, the value, and the visibility of the slider.
        """
        max_slices = self.stack.max_slices
        if max_slices <= 1:
            if self.slice_slider is not None:
                self._slice_slider_ax.set_visible(False)
            return
        if self.slice_slider is None:
            self._make_slice_slider()
            return
        self._slice_slider_ax.set_visible(True)
        self.slice_slider.valmax = max_slices - 1
        self.slice_slider.ax.set_xlim(0, max_slices - 1)
        self.slice_slider.set_val(self.stack.master_index)

    # --- Slider callbacks ---

    def _on_slice_slider(self, value):
        changed = self.stack.set_master_index(value)
        for i in changed:
            self.refresh(i)
        if changed:
            self._display_roi_stats()
        self._partial_redraw(changed, widgets=('slice',))

    def _on_intensity_slider(self, value):
        for img in self.images:
            img.set_clim(value[0], value[1])
        self._partial_redraw(widgets=('intensity',))

    # --- Slice-axis selection and coupling ---

    def _reset_navigation(self):
        """This method clears the saved views of the toolbar.

        The saved views are out of date after a change of slice axis, a transpose, or a load.
        """
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if toolbar is not None:
            try:
                toolbar.update()
            except Exception:
                pass

    def _on_axis_selected(self, volume_index, axis):
        indices = (range(self.stack.n_volumes) if volume_index is None
                   else [volume_index])
        any_changed = False
        for i in indices:
            if self.stack.set_perm(i, axis):
                any_changed = True
                self._reset_view(i)
        if not any_changed:
            return
        self.refresh()  # The new depth can change the slice of any volume.
        self._update_slice_slider()
        self._reset_navigation()
        self.fig.canvas.draw_idle()

    def _toggle_couple_axes(self):
        self.sync_axes = not self.sync_axes
        self._rebuild_axis_radios()
        if self.sync_axes:
            # Every volume uses the slice axis of volume 0.
            self._on_axis_selected(None, self.stack.slice_axes[0])
        self.fig.canvas.draw_idle()

    def _toggle_couple_zoom(self):
        self.sync_limits = not self.sync_limits

    # --- The context menu ---

    def _in_process_tk_ok(self):
        """This method returns True if it is safe to open a Tk window in this process.

        It is safe only with the TkAgg backend.
        """
        return matplotlib.get_backend().lower() == 'tkagg'

    def _menu_items(self, i):
        """This method returns the (label, callback) pairs for the context menu of volume ``i``.

        The Cancel item has the callback None, and it closes the menu.
        """
        stack = self.stack
        items = []
        if stack.n_volumes > 1:
            items.append((('Decouple' if self.sync_axes else 'Couple')
                          + ' slice axes', self._toggle_couple_axes))
            items.append((('Decouple' if self.sync_limits else 'Couple')
                          + ' pan/zoom', self._toggle_couple_zoom))
            if stack.is_difference(i):
                items.append(('Restore original image',
                              lambda i=i: self._on_restore(i)))
            else:
                items.append(('Replace with difference image',
                              lambda i=i: self._on_difference_button(i)))
                items.append(('Replace with error image',
                              lambda i=i: self._on_error_button(i)))
        items += [
            ('Show data dict', lambda i=i: self._on_dict_button(i)),
            ('Transpose image', lambda i=i: self._on_transpose_button(i)),
            ('Load', lambda i=i: self._on_load_button(i)),
            ('Save data to h5', lambda i=i: self._on_save_button(i)),
            ('Reset', lambda i=i: self._on_reset_button(i)),
            ('Cancel', None),
        ]
        return items

    def _open_context_menu(self, i, event):
        items = self._menu_items(i)
        if self._in_process_tk_ok():
            def run():
                try:
                    chosen = _tk_menu([label for label, _cb in items])
                except Exception:
                    self._open_menu_dialog(items, event)
                    return
                for label, callback in items:
                    if label == chosen and callback is not None:
                        callback()
            # Tk runs the menu after this callback returns, not inside this callback.
            try:
                self.fig.canvas.manager.window.after(10, run)
                return
            except Exception:
                pass
            run()
            return
        self._open_menu_dialog(items, event)

    def _open_menu_dialog(self, items, event):
        """This method opens a popup menu inside the figure at the click position.

        The viewer uses this menu on backends other than TkAgg.
        """
        dialog = self._open_dialog('menu', dim=False)
        canvas_w, canvas_h = self.fig.canvas.get_width_height()
        fx, fy = event.x / canvas_w, event.y / canvas_h
        fig_w, fig_h = self.fig.get_size_inches()
        longest = max(len(label) for label, _cb in items)
        w = min(max(1.9, 0.078 * longest) / fig_w, 0.6)
        row_h = 0.26 / fig_h
        h = row_h * len(items) + 0.012
        x0 = float(np.clip(fx, 0.005, 1 - w - 0.005))
        y0 = float(np.clip(fy - h, 0.005, 1 - h - 0.005))
        panel = self.fig.add_axes((x0, y0, w, h), zorder=DIALOG_ZORDER + 1,
                                  facecolor='white')
        panel.set_xticks([])
        panel.set_yticks([])
        dialog['axes'].append(panel)
        dialog['panel_ax'] = panel
        menu_axes = []
        for j, (label, callback) in enumerate(items):
            button = self._dialog_button(
                label, (x0 + 0.004, y0 + h - (j + 1) * row_h + 0.004,
                        w - 0.008, row_h - 0.004),
                lambda callback=callback: self._menu_dispatch(callback),
                key=f'item{j}')
            button.label.set_fontsize(DIALOG_FONT_SIZE)
            menu_axes.append(button.ax)
        dialog['menu_axes'] = menu_axes
        self.fig.canvas.draw_idle()

    def _menu_dispatch(self, callback):
        self._close_dialog(draw=False)
        if callback is not None:
            callback()
        else:
            self.fig.canvas.draw_idle()

    # --- Actions on one image ---

    def _on_difference_button(self, i, use_abs=False):
        # This code does not change the toolbar state.  On the macosx backend, a change of
        # pan or zoom in code makes the native buttons disagree with the matplotlib mode.
        if self.stack.is_difference(i):
            self._on_restore(i)
            return
        if self.stack.n_volumes == 2:
            other = 1 - i
            if self.stack.can_difference(i, other):
                self._apply_difference(i, other, use_abs)
                return
        self._set_mode(Mode.SELECT_COMPARISON, baseline_index=i,
                       use_abs=use_abs)
        self._show_message(True, message_type='difference')

    def _on_error_button(self, i):
        if self.stack.is_difference(i):
            return  # The menu offers Restore for a difference image, so this does nothing.
        self._on_difference_button(i, use_abs=True)

    def _complete_difference(self, comparison_index):
        baseline_index = self._mode_data['baseline_index']
        use_abs = self._mode_data['use_abs']
        if not self.stack.can_difference(baseline_index, comparison_index):
            self._show_message(True, message_type='difference')
            return  # The viewer stays in the mode that selects a comparison image.
        self._set_mode(Mode.IDLE)
        self._show_message(False)
        self._apply_difference(baseline_index, comparison_index, use_abs)

    def _apply_difference(self, baseline_index, comparison_index, use_abs):
        self.stack.apply_difference(baseline_index, comparison_index, use_abs)
        self.refresh(baseline_index)
        self.fig.canvas.draw_idle()

    def _on_restore(self, i):
        self.stack.restore(i)
        self.refresh(i)
        self.fig.canvas.draw_idle()

    def _on_transpose_button(self, i):
        self.stack.transpose(i)
        self._reset_view(i)
        self.refresh()
        self._update_slice_slider()
        self._reset_navigation()
        self.fig.canvas.draw_idle()

    def _on_reset_button(self, i):
        # This resets the zoom, the ROI, the messages, and any unfinished selection.
        self._reset_view(None if self.sync_limits else i)
        self._remove_roi_graphics()
        self._set_mode(Mode.IDLE)
        self._show_message(False)
        self.fig.canvas.draw_idle()

    # A modal dialog inside the figure is a set of axes with a high zorder.  Its backdrop covers
    # the figure, so event.inaxes is always a dialog axes.  The widgets underneath get no clicks.

    def _main_widgets(self):
        widgets = list(self.axis_radios)
        widgets.append(self.range_button)
        if self.slice_slider is not None:
            widgets.append(self.slice_slider)
        widgets.append(self.intensity_slider)
        return widgets

    def _open_dialog(self, kind, dim=True):
        self._close_dialog(draw=False)
        backdrop = self.fig.add_axes((0, 0, 1, 1), zorder=DIALOG_ZORDER,
                                     facecolor=(1, 1, 1, 0.6) if dim
                                     else 'none')
        backdrop.set_xticks([])
        backdrop.set_yticks([])
        self._dialog = {'kind': kind, 'axes': [backdrop], 'widgets': {},
                        'texts': {}, 'state': {}}
        # Matplotlib widgets respond to clicks by position, not by zorder.  For this reason, the
        # widgets under the dialog are disabled with the .active flag, which ignore() checks.
        # The flag is set directly, because RadioButtons.set_active has another meaning.
        for widget in self._main_widgets():
            widget.active = False
        return self._dialog

    def _close_dialog(self, draw=True):
        if self._dialog is None:
            return
        for ax in self._dialog['axes']:
            ax.remove()
        self._dialog = None
        for widget in self._main_widgets():
            widget.active = True
        if draw:
            self.fig.canvas.draw_idle()

    def _dialog_panel(self, width_in, height_in):
        """This method adds a centered white panel to the dialog and returns its (x0, y0, w, h).

        The values are fractions of the figure size.  The caller uses them to place the widgets
        inside the panel.
        """
        fig_w, fig_h = self.fig.get_size_inches()
        w = min(width_in / fig_w, 0.92)
        h = min(height_in / fig_h, 0.88)
        x0, y0 = (1 - w) / 2, (1 - h) / 2
        panel = self.fig.add_axes((x0, y0, w, h), zorder=DIALOG_ZORDER + 1,
                                  facecolor='white')
        panel.set_xticks([])
        panel.set_yticks([])
        self._dialog['axes'].append(panel)
        self._dialog['panel_ax'] = panel
        return x0, y0, w, h

    def _dialog_button(self, label, rect, callback, key=None):
        ax = self.fig.add_axes(rect, zorder=DIALOG_ZORDER + 2)
        button = Button(ax, label)
        button.label.set_fontsize(DIALOG_FONT_SIZE)
        button.on_clicked(lambda _event: callback())
        self._dialog['axes'].append(ax)
        self._dialog['widgets'][key if key is not None else label] = button
        return button

    def _dialog_textbox(self, name, label, rect, initial):
        ax = self.fig.add_axes(rect, zorder=DIALOG_ZORDER + 2)
        textbox = TextBox(ax, label, initial=initial, textalignment='left')
        textbox.label.set_fontsize(DIALOG_FONT_SIZE)
        textbox.text_disp.set_fontsize(DIALOG_FONT_SIZE)
        textbox.text_disp.set_clip_on(True)  # A long path is cut off at the edge of the box.
        self._dialog['axes'].append(ax)
        self._dialog['widgets'][name] = textbox
        return textbox

    def _dialog_text(self, name, xy, text, **kwargs):
        # The texts belong to the panel axes.  For this reason, they are drawn above the
        # opaque panel, and they are removed with the panel.
        kwargs.setdefault('fontsize', DIALOG_FONT_SIZE)
        panel = self._dialog['panel_ax']
        artist = panel.text(xy[0], xy[1], text,
                            transform=self.fig.transFigure, **kwargs)
        self._dialog['texts'][name] = artist
        return artist

    # --- Intensity range dialog ---

    def _open_range_dialog(self):
        if self._in_process_tk_ok():
            def run():
                try:
                    result = _tk_range_dialog(self.stack.vmin,
                                              self.stack.vmax)
                except Exception:
                    self._open_range_dialog_infigure()
                    return
                self._handle_range_result(result)
            try:
                self.fig.canvas.manager.window.after(10, run)
                return
            except Exception:
                pass
            run()
            return
        self._open_range_dialog_infigure()

    def _handle_range_result(self, result):
        """This method applies the result of the Tk range dialog.

        If the input is not valid, the method shows the error as a message in the figure.
        """
        if result is None:
            return
        try:
            if result[0] == 'data':
                self.stack.set_range(None, None)
            else:
                self._parse_and_set_range(result[1], result[2])
        except ValueError as e:
            self._show_message(
                True, message=f"{e}. Press Esc to dismiss.")
            return
        self._set_intensity_bounds()
        self.fig.canvas.draw_idle()

    def _parse_and_set_range(self, min_text, max_text):
        """This method converts the text of the min and max entries to numbers and sets the
        range.

        Both range dialogs use this method.  A blank entry keeps the current bound.
        """
        try:
            vmin = (self.stack.vmin if not min_text.strip()
                    else float(min_text))
            vmax = (self.stack.vmax if not max_text.strip()
                    else float(max_text))
        except ValueError:
            raise ValueError(
                'Enter numbers, or leave blank to keep a bound')
        self.stack.set_range(vmin, vmax)

    def _open_range_dialog_infigure(self):
        stack = self.stack
        dialog = self._open_dialog('range')
        x0, y0, w, h = self._dialog_panel(4.6, 2.6)
        self._dialog_text('title', (x0 + 0.02 * w, y0 + h - 0.06),
                          'Set intensity range', fontweight='bold')
        # The fields start empty, so the user can type a new bound directly.  A
        # blank field keeps its current bound.
        self._dialog_text('hint', (x0 + 0.02 * w, y0 + h - 0.11),
                          f'Blank keeps the current bound '
                          f'({stack.vmin:.6g}, {stack.vmax:.6g}).')
        box_w, box_h = 0.45 * w, 0.14 * h
        self._dialog_textbox('min', 'Min ',
                             (x0 + 0.14 * w, y0 + 0.52 * h, box_w, box_h), '')
        self._dialog_textbox('max', 'Max ',
                             (x0 + 0.14 * w, y0 + 0.32 * h, box_w, box_h), '')
        self._dialog_text('error', (x0 + 0.02 * w, y0 + 0.22 * h), '',
                          color='red')
        btn_h = 0.14 * h
        self._dialog_button('Apply',
                            (x0 + 0.30 * w, y0 + 0.05 * h, 0.18 * w, btn_h),
                            self._apply_range_dialog)
        self._dialog_button('Data range',
                            (x0 + 0.52 * w, y0 + 0.05 * h, 0.24 * w, btn_h),
                            self._apply_data_range)
        self._dialog_button('Cancel',
                            (x0 + 0.80 * w, y0 + 0.05 * h, 0.18 * w, btn_h),
                            self._close_dialog)
        self.fig.canvas.draw_idle()
        return dialog

    def _apply_range_dialog(self):
        widgets = self._dialog['widgets']
        try:
            self._parse_and_set_range(widgets['min'].text,
                                      widgets['max'].text)
        except ValueError as e:
            self._dialog['texts']['error'].set_text(str(e))
            self.fig.canvas.draw_idle()
            return
        self._close_dialog(draw=False)
        self._set_intensity_bounds()
        self.fig.canvas.draw_idle()

    def _apply_data_range(self):
        self.stack.set_range(None, None)
        self._close_dialog(draw=False)
        self._set_intensity_bounds()
        self.fig.canvas.draw_idle()

    def _set_intensity_bounds(self):
        """This method copies stack.vmin and stack.vmax to the range slider and the images."""
        stack = self.stack
        slider = self.intensity_slider
        slider.valmin = stack.vmin
        slider.valmax = stack.vmax
        slider.valfmt = self._intensity_valfmt(stack.vmin, stack.vmax)
        slider.ax.set_xlim(stack.vmin, stack.vmax)
        slider.set_val((stack.vmin, stack.vmax))

    # --- Data dict dialogs ---

    def _on_dict_button(self, i):
        data_dict = self.stack.data_dicts[i]
        if not data_dict:
            self._open_text_dialog('No data dict',
                                   'No data dict included with this image')
            return
        names = list(data_dict.keys())
        if len(names) == 1:
            self._open_text_dialog(names[0], str(data_dict[names[0]]))
            return
        self._open_dict_chooser(i)

    def _open_dict_chooser(self, i):
        names = list(self.stack.data_dicts[i].keys())
        self._open_choice_dialog(
            'Choose an entry to display', names,
            lambda name, i=i: self._open_text_dialog(
                name, str(self.stack.data_dicts[i][name]),
                back=lambda i=i: self._open_dict_chooser(i)))

    def _open_choice_dialog(self, title, labels, on_choice, sublabels=None):
        self._open_dialog('choice')
        rows = len(labels) + 1
        row_h_in = 0.42
        panel_h = min(rows * row_h_in + 0.7, 7.0)
        x0, y0, w, h = self._dialog_panel(4.6, panel_h)
        self._dialog_text('title', (x0 + 0.02 * w, y0 + h - 0.05), title,
                          fontweight='bold')
        btn_h = (h - 0.12) / rows * 0.8
        for j, label in enumerate(labels):
            text = label if sublabels is None else f"{label}: {sublabels[j]}"
            top = y0 + h - 0.10 - (j + 1) * (h - 0.12) / rows
            button = self._dialog_button(
                text, (x0 + 0.05 * w, top, 0.9 * w, btn_h),
                lambda label=label: self._dialog_choice_made(on_choice, label))
            button.label.set_fontsize(DIALOG_FONT_SIZE)
        top = y0 + h - 0.10 - rows * (h - 0.12) / rows
        self._dialog_button('Cancel', (x0 + 0.65 * w, top, 0.3 * w, btn_h),
                            self._close_dialog)
        self.fig.canvas.draw_idle()

    def _dialog_choice_made(self, on_choice, label):
        self._close_dialog(draw=False)
        on_choice(label)

    def _open_text_dialog(self, title, body, back=None):
        self._open_dialog('text')
        x0, y0, w, h = self._dialog_panel(5.6, 6.0)
        self._dialog_text('title', (x0 + 0.02 * w, y0 + h - 0.045), title,
                          fontweight='bold')
        lines = str(body).splitlines() or ['']
        pages = [lines[k:k + DIALOG_PAGE_LINES]
                 for k in range(0, len(lines), DIALOG_PAGE_LINES)]
        state = self._dialog['state']
        state['pages'] = pages
        state['page'] = 0
        body_artist = self._dialog_text(
            'body', (x0 + 0.03 * w, y0 + h - 0.09), '',
            fontsize=DIALOG_BODY_FONT_SIZE, family='monospace', va='top')
        body_artist.set_wrap(True)
        btn_w, btn_h = 0.16 * w, 0.05
        if len(pages) > 1:
            self._dialog_button('Prev',
                                (x0 + 0.03 * w, y0 + 0.02, btn_w, btn_h),
                                lambda: self._flip_text_page(-1))
            self._dialog_button('Next',
                                (x0 + 0.22 * w, y0 + 0.02, btn_w, btn_h),
                                lambda: self._flip_text_page(+1))
        if back is not None:
            self._dialog_button('Back',
                                (x0 + 0.55 * w, y0 + 0.02, btn_w, btn_h),
                                lambda: self._dialog_choice_made(
                                    lambda _label: back(), None))
        self._dialog_button('Close',
                            (x0 + 0.78 * w, y0 + 0.02, btn_w, btn_h),
                            self._close_dialog)
        self._render_text_page()
        self.fig.canvas.draw_idle()

    def _render_text_page(self):
        state = self._dialog['state']
        pages, page = state['pages'], state['page']
        text = '\n'.join(pages[page])
        if len(pages) > 1:
            text += f"\n\n[page {page + 1} of {len(pages)}]"
        self._dialog['texts']['body'].set_text(text)

    def _flip_text_page(self, step):
        state = self._dialog['state']
        state['page'] = int(np.clip(state['page'] + step, 0,
                                    len(state['pages']) - 1))
        self._render_text_page()
        self.fig.canvas.draw_idle()

    # File selection tries the native dialogs first, in this order: Qt, the macOS panel through
    # osascript, and tkinter.  If none of them runs, the viewer uses the file browser in the figure.

    def _native_choose_file(self, mode, directory, initial_file):
        """This method shows a native file dialog and returns the chosen path.

        The method returns None if the user cancels.  It returns ``_NATIVE_UNAVAILABLE`` if no
        native dialog can run.
        """
        backend = matplotlib.get_backend().lower()
        if backend in NONINTERACTIVE_BACKENDS:
            return _NATIVE_UNAVAILABLE
        if backend.startswith('qt'):
            try:
                return self._choose_file_qt(mode, directory, initial_file)
            except Exception:
                pass
        if sys.platform == 'darwin':
            try:
                return self._choose_file_osascript(mode, directory,
                                                   initial_file)
            except Exception:
                pass
        # A Tk window in this process is safe only with TkAgg.  With the event loop of another
        # toolkit, Tk can crash the process.  In that case, the viewer uses the file browser in the figure.
        if not self._in_process_tk_ok():
            return _NATIVE_UNAVAILABLE
        try:
            return self._choose_file_tkinter(mode, directory, initial_file)
        except Exception:
            return _NATIVE_UNAVAILABLE

    def _choose_file_qt(self, mode, directory, initial_file):
        from matplotlib.backends.qt_compat import QtWidgets
        parent = getattr(self.fig.canvas.manager, 'window', None)
        if mode == 'load':
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                parent, 'Load a volume', directory,
                'Volumes (*.npy *.npz *.h5 *.hdf5);;All files (*)')
        else:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                parent, 'Save volume to HDF5',
                os.path.join(directory, initial_file), 'HDF5 (*.h5)')
        return path or None

    def _choose_file_osascript(self, mode, directory, initial_file):
        import subprocess

        def quoted(text):
            return text.replace('\\', '\\\\').replace('"', '\\"')

        if mode == 'load':
            script = ('POSIX path of (choose file with prompt '
                      '"Load a volume" of type {"npy", "npz", "h5", "hdf5"} '
                      f'default location POSIX file "{quoted(directory)}")')
        else:
            script = ('POSIX path of (choose file name with prompt '
                      '"Save volume to HDF5" '
                      f'default name "{quoted(initial_file)}" '
                      f'default location POSIX file "{quoted(directory)}")')
        result = subprocess.run(['osascript', '-e', script],
                                capture_output=True, text=True)
        if result.returncode != 0:
            if 'canc' in (result.stderr or '').lower():
                return None  # The user cancelled the dialog.
            raise RuntimeError(result.stderr.strip())
        path = result.stdout.strip()
        return path or None

    def _choose_file_tkinter(self, mode, directory, initial_file):
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        try:
            if mode == 'load':
                path = filedialog.askopenfilename(
                    parent=root, initialdir=directory, title='Load a volume',
                    filetypes=[('Volumes', '*.npy *.npz *.h5 *.hdf5'),
                               ('All files', '*.*')])
            else:
                path = filedialog.asksaveasfilename(
                    parent=root, initialdir=directory,
                    initialfile=initial_file, title='Save volume to HDF5',
                    defaultextension='.h5', filetypes=[('HDF5', '*.h5')])
        finally:
            root.destroy()
        return path or None

    def _file_action(self, i, mode):
        def run():
            chosen = self._native_choose_file(mode, self._last_dir,
                                              'volume.h5')
            if chosen is _NATIVE_UNAVAILABLE:
                self._open_file_dialog(i, mode)
                return
            if chosen is None:
                return
            self._last_dir = os.path.dirname(chosen) or self._last_dir
            if mode == 'load':
                self._start_load(i, chosen)
            else:
                self._finish_save(i, chosen)

        if matplotlib.get_backend().lower() == 'tkagg':
            # Tk runs the dialog after this callback returns.  A second Tk root that
            # starts inside a callback does not work correctly.
            try:
                self.fig.canvas.manager.window.after(10, run)
                return
            except Exception:
                pass
        run()

    def _open_file_dialog(self, i, mode, directory=None, page=0,
                          filename=None, select_name=None):
        """This method opens the load or save dialog inside the figure.

        The dialog shows a directory listing and a text box for the path.
        """
        directory = os.path.abspath(directory or self._last_dir)
        self._last_dir = directory
        is_load = mode == 'load'
        self._open_dialog('file')
        x0, y0, w, h = self._dialog_panel(6.0, 6.4)
        title = ('Load a volume (.npy, .npz, .h5, .hdf5)' if is_load
                 else 'Save volume and data dict to HDF5')
        self._dialog_text('title', (x0 + 0.02 * w, y0 + h - 0.04), title,
                          fontweight='bold')
        if filename is None:
            filename = '' if is_load else 'volume.h5'
        self._dialog_textbox('path', 'Path ',
                             (x0 + 0.09 * w, y0 + h - 0.105, 0.88 * w, 0.045),
                             os.path.join(directory, filename))
        self._dialog_text('error', (x0 + 0.02 * w, y0 + 0.115), '',
                          color='red')

        state = self._dialog['state']
        state.update(volume_index=i, mode=mode, directory=directory)
        try:
            names = sorted(os.listdir(directory))
        except OSError as e:
            names = []
            self._dialog['texts']['error'].set_text(str(e))
        entries = []
        parent = os.path.dirname(directory)
        if parent != directory:
            entries.append(('[..]', parent, True))
        visible = [n for n in names if not n.startswith('.')]
        for name in visible:
            full = os.path.join(directory, name)
            if os.path.isdir(full):
                entries.append((name + os.sep, full, True))
        for name in visible:
            full = os.path.join(directory, name)
            if os.path.isfile(full) and \
                    name.lower().endswith(FILE_EXTENSIONS):
                entries.append((name, full, False))
        # When the user moves up one directory, the browser selects the previous directory.
        # The browser then shows the page that contains that directory, not the first page.
        if select_name is not None:
            for index, (label, _full, _is_dir) in enumerate(entries):
                if label.rstrip(os.sep) == select_name:
                    page = index // FILE_DIALOG_ROWS
                    break
        n_pages = max(1, -(-len(entries) // FILE_DIALOG_ROWS))
        page = int(np.clip(page, 0, n_pages - 1))
        state['page'] = page
        shown = entries[page * FILE_DIALOG_ROWS:
                        (page + 1) * FILE_DIALOG_ROWS]
        state['entries_shown'] = shown

        list_top = y0 + h - 0.135
        row_h = 0.042
        for j, (label, full, is_dir) in enumerate(shown):
            going_up = os.path.basename(directory) if label == '[..]' else None
            button = self._dialog_button(
                label, (x0 + 0.04 * w, list_top - (j + 1) * (row_h + 0.008),
                        0.92 * w, row_h),
                lambda full=full, is_dir=is_dir, going_up=going_up:
                    self._file_dialog_entry(full, is_dir, going_up),
                key=f'entry{j}')
            button.label.set_fontsize(DIALOG_BODY_FONT_SIZE)
        if n_pages > 1:
            self._dialog_text('page', (x0 + 0.04 * w, y0 + 0.075),
                              f'page {page + 1} of {n_pages}')
            self._dialog_button('Prev', (x0 + 0.24 * w, y0 + 0.06,
                                         0.13 * w, 0.045),
                                lambda: self._file_dialog_page(-1))
            self._dialog_button('Next', (x0 + 0.40 * w, y0 + 0.06,
                                         0.13 * w, 0.045),
                                lambda: self._file_dialog_page(+1))
        accept_label = 'Load' if is_load else 'Save'
        accept = (self._load_dialog_accept if is_load
                  else self._save_dialog_accept)
        self._dialog_button(accept_label,
                            (x0 + 0.58 * w, y0 + 0.015, 0.18 * w, 0.045),
                            lambda i=i: accept(i))
        self._dialog_button('Cancel',
                            (x0 + 0.79 * w, y0 + 0.015, 0.17 * w, 0.045),
                            self._close_dialog)
        self.fig.canvas.draw_idle()

    def _file_dialog_entry(self, full_path, is_dir, select_name=None):
        state = self._dialog['state']
        if is_dir:
            filename = None
            if state['mode'] == 'save':
                filename = os.path.basename(
                    self._dialog['widgets']['path'].text) or 'volume.h5'
            self._open_file_dialog(state['volume_index'], state['mode'],
                                   directory=full_path, filename=filename,
                                   select_name=select_name)
            return
        self._dialog['widgets']['path'].set_val(full_path)
        if state['mode'] == 'load':
            self._load_dialog_accept(state['volume_index'])
        else:
            self.fig.canvas.draw_idle()

    def _file_dialog_page(self, step):
        state = self._dialog['state']
        filename = None
        if state['mode'] == 'save':
            filename = os.path.basename(
                self._dialog['widgets']['path'].text) or 'volume.h5'
        self._open_file_dialog(state['volume_index'], state['mode'],
                               directory=state['directory'],
                               page=state['page'] + step, filename=filename)

    def _dialog_error(self, message):
        self._dialog['texts']['error'].set_text(message)
        self.fig.canvas.draw_idle()

    def _on_load_button(self, i):
        self._file_action(i, 'load')

    def _load_dialog_accept(self, i):
        path = os.path.expanduser(self._dialog['widgets']['path'].text.strip())
        if os.path.isdir(path):
            self._open_file_dialog(i, 'load', directory=path)
            return
        if not os.path.isfile(path):
            self._dialog_error(f"File not found: {path or '(empty)'}")
            return
        self._last_dir = os.path.dirname(path) or self._last_dir
        self._start_load(i, path)

    def _file_error(self, message):
        """This method shows an error in the open dialog, or as a message in the figure if no
        dialog is open."""
        if self._dialog is not None:
            self._dialog_error(message)
        else:
            self._show_message(True, message=f"{message}. Press Esc to exit.")

    def _start_load(self, i, path):
        """This method loads the file ``path`` into volume ``i``.

        If the file holds more than one array, the method first asks the user to choose one.
        """
        try:
            listing = VolumeStack.list_file_arrays(path)
        except Exception as e:
            self._file_error(f"Failed to load file: {e}")
            return
        if listing is None or len(listing[0]) == 1:
            self._close_dialog(draw=False)
            self._finish_load(i, path, None)
            return
        names, shapes = listing
        self._close_dialog(draw=False)
        self._open_choice_dialog(
            f"Arrays in {os.path.basename(path)}", names,
            lambda name, i=i, path=path: self._finish_load(i, path, name),
            sublabels=[f"shape={tuple(s)}" for s in shapes])

    def _finish_load(self, i, path, name):
        try:
            array, data_dict = VolumeStack.read_file_array(path, name)
            replaced = self.stack.load_array(i, array, data_dict)
        except Exception as e:
            self._show_message(
                True, message=f"Failed to load file: {e}. Press Esc to exit.")
            return
        for j in replaced:
            self._reset_view(j)
        self.refresh()
        self._update_slice_slider()
        self._reset_navigation()
        self.fig.canvas.draw_idle()

    def _on_save_button(self, i):
        self._file_action(i, 'save')

    def _save_dialog_accept(self, i):
        path = os.path.expanduser(self._dialog['widgets']['path'].text.strip())
        if not path:
            self._dialog_error('Enter a file path')
            return
        if os.path.isdir(path):
            self._dialog_error('Path is a directory; add a file name')
            return
        self._finish_save(i, path)

    def _finish_save(self, i, path):
        if not path.lower().endswith('.h5'):
            path += '.h5'
        # With TkAgg, the user can edit the data dict before it is written.  With every
        # other backend, the data dict is saved unchanged.
        if self._in_process_tk_ok():
            try:
                result = _tk_dict_editor(self.stack.data_dicts[i])
            except Exception:
                result = ('as-is',)
            if result is None:
                return  # The user cancelled the save.
            if result[0] == 'save':
                self.stack.data_dicts[i] = result[1] or None
        try:
            self.save_fn(path, self.stack.original_data[i], 'recon',
                         self.stack.data_dicts[i])
        except Exception as e:
            self._file_error(f"Failed to save: {e}")
            return
        self._last_dir = os.path.dirname(path) or self._last_dir
        self._close_dialog(draw=False)
        self._show_message(True,
                           message=f"Saved to {path}. Press Esc to dismiss.")

    # A partial redraw repaints only the changed panels or sliders.  Only the clearing
    # rectangle is marked animated, so full draws and savefig need no special case.
    # A backend without blit support uses draw_idle instead.

    def _partial_redraw(self, volume_indices=None, widgets=()):
        canvas = self.fig.canvas
        overlay_active = (self._dialog is not None
                          or self._message_artist is not None)
        if (not self.enable_blit or not self._renderer_ready
                or overlay_active
                or matplotlib.get_backend().lower() not in BLIT_BACKENDS
                or not getattr(canvas, 'supports_blit', False)):
            canvas.draw_idle()
            return
        renderer = canvas.get_renderer()
        # The region is clipped to the renderer buffer, not to the logical canvas size.
        # On a HiDPI display, the two sizes differ by the device pixel ratio.
        buffer_w = getattr(renderer, 'width', 0) or canvas.get_width_height()[0]
        buffer_h = getattr(renderer, 'height', 0) or canvas.get_width_height()[1]
        canvas_box = Bbox([[0, 0], [buffer_w, buffer_h]])
        groups = []
        indices = (range(self.stack.n_volumes) if volume_indices is None
                   else volume_indices)
        for i in indices:
            groups.append((('volume', i), [self.axes[i], self.caxes[i]]))
        widget_axes = {'slice': self._slice_slider_ax,
                       'intensity': self.intensity_slider.ax}
        for name in widgets:
            ax = widget_axes[name]
            if ax.get_visible():
                groups.append((('widget', name), [ax]))
        for key, group in groups:
            boxes = []
            for ax in group:
                try:
                    boxes.append(ax.get_tightbbox(renderer))
                except Exception:
                    boxes.append(ax.bbox)
            current = Bbox.union(boxes).padded(6)
            # The cleared region includes the previous region of the group.  In this way, text
            # that shrank or moved leaves no old pixels.
            previous = self._last_blit_regions.get(key)
            region = (current if previous is None
                      else Bbox.union([current, previous]))
            self._last_blit_regions[key] = current
            region = Bbox.intersection(region, canvas_box)
            if region is None:
                continue
            self._clear_rect.set_bounds(region.x0, region.y0,
                                        region.width, region.height)
            self._clear_rect.draw(renderer)
            for ax in group:
                ax.draw(renderer)
            canvas.blit(region)
        try:
            canvas.flush_events()
        except NotImplementedError:
            pass

    # --- Showing ---

    def show(self, block=True):
        """This method displays the viewer window.

        If ``block`` is True, the method runs the GUI event loop until every open matplotlib
        window is closed.  If ``block`` is False, the method draws the window and leaves it
        open.  The window becomes fully interactive when a later blocking show runs.  On a
        non-interactive backend such as Agg, the method issues a warning and returns.
        """
        if matplotlib.get_backend().lower() in NONINTERACTIVE_BACKENDS:
            warnings.warn(
                'slice_viewer: matplotlib backend '
                f'{matplotlib.get_backend()!r} is non-interactive; the '
                'viewer window cannot be shown.')
            return
        if block:
            plt.show()
            plt.close(self.fig)
        else:
            self.fig.show()
            self.fig.canvas.draw_idle()


# This list keeps nonblocking viewers alive if the caller discards the returned viewer.
# The next blocking call to slice_viewer closes them.
_NONBLOCKING_VIEWERS = []


def slice_viewer(*datasets, data_dicts=None, title='', vmin=None, vmax=None,
                 slice_label=None, slice_axis=None, cmap='gray',
                 show_instructions=True, block=True, save_fn=None):
    """This function opens an interactive viewer for 2D or 3D image arrays.

    The function builds a :class:`SliceViewer`, shows it, and returns it.  The viewer has one
    slice slider for all volumes, even if their depths differ.  It also shows ROI statistics
    and difference images, transposes images, and loads and saves files.

    Args:
        *datasets (ndarray or None): One or more 2D or 3D arrays to display.  A 2D array gets
            a third axis of length 1.  A None entry becomes a placeholder array of zeros.
        data_dicts (None, dict, or list of dict/None, optional): The dicts of strings that go
            with the volumes.  The user can view them in the viewer.
        title (str, optional): The figure title.  Defaults to ''.
        vmin (float, optional): The minimum display intensity.  Defaults to the minimum over
            all datasets.
        vmax (float, optional): The maximum display intensity.  Defaults to the maximum over
            all datasets.
        slice_label (str or list of str, optional): The label or labels shown before the slice
            number in each panel title.  Defaults to "Slice".
        slice_axis (int or list of int, optional): The axis to slice along (0, 1, or 2).
            Defaults to the last axis (2).
        cmap (str, optional): The colormap.  Defaults to 'gray'.
        show_instructions (bool, optional): If True, the viewer shows the "Press h for help"
            hint.  Defaults to True.
        block (bool, optional): If True, the function returns after the window is closed.  If
            False, the function returns at once.  In that case, the window becomes fully
            interactive when the next blocking call to slice_viewer runs.  That call returns
            after all open windows are closed.  Defaults to True.
        save_fn (callable, optional): A function that replaces the built-in HDF5 writer.  The
            viewer calls it as ``save_fn(file_path, array, array_name, attributes_dict)``.

    Returns:
        SliceViewer: The viewer object.  The viewer stays open even if the caller discards the
        return value.
    """
    viewer = SliceViewer(*datasets, data_dicts=data_dicts, title=title,
                         vmin=vmin, vmax=vmax, slice_label=slice_label,
                         slice_axis=slice_axis, cmap=cmap,
                         show_instructions=show_instructions, save_fn=save_fn)
    viewer.show(block=block)
    if not block:
        _NONBLOCKING_VIEWERS.append(viewer)
        return viewer
    # The blocking show has returned, so this code closes the earlier nonblocking viewers.
    # With TkAgg, this code also runs garbage collection on the main thread.
    for nonblocking_viewer in _NONBLOCKING_VIEWERS:
        plt.close(nonblocking_viewer.fig)
    _NONBLOCKING_VIEWERS.clear()
    if matplotlib.get_backend() == 'TkAgg':
        import gc
        gc.collect()
    return viewer

if __name__ == '__main__':
    slice_viewer(np.zeros((20, 20, 20)), np.zeros((20, 20, 20)))
"""This module estimates the GPU memory and the compute time of a reconstruction before it
runs.  The estimate needs only the model, not the data.  The time estimates use measured speeds
stored in ``data/gpu_speeds.json``.
"""
import dataclasses
import importlib.resources
import json
import math

import numpy as np
import torch

from .cone_beam import ConeBeamModel
from .multiaxis_parallel import MultiAxisParallelModel
from .parallel_beam import ParallelBeamModel
from .tomography_model import TomographyModel
from .translation_model import TranslationModel
from .utilities import copy_ct_model

__all__ = ['estimate_resources']

_GIB = 2 ** 30

#: The default overlap, in detector rows, that ``recon_split_sino`` uses.
_SPLIT_HALF_OVERLAP = 5

#: The default minimum number of slices in each section of a parallel beam split reconstruction.
_MIN_SLICES_PER_SECTION = 200

#: The memory, in GiB, that the GPU driver keeps.  The estimate uses this value when the speed
#: file has no value for the GPU.
_DEFAULT_DRIVER_RESERVE_GIB = 2.0

#: The time estimate is meant to be within this factor of the measured time.
_TIME_ACCURACY_FACTOR = 1.5

_STEP_NAMES = {'direct': 'direct reconstruction', 'recon': 'full reconstruction',
               'split': 'split reconstruction'}


@dataclasses.dataclass
class _StepEstimate:
    """This class is used to hold the estimate for one reconstruction step.  The estimate
    gives the GPU memory, whether the step fits, and the time.  ``fits`` is None when the GPU
    memory is unknown.  ``time_minutes`` is None when the speed file has no data for the case.
    ``note`` says why a value is missing.
    """
    gpu_memory_gib: float = None
    fits: bool = None
    time_minutes: float = None
    note: str = ''


@dataclasses.dataclass
class _ResourceEstimate:
    """This class is used to hold the estimate for one model on the named hardware.
    Printing an instance gives the report."""
    model_name: str
    sinogram_shape: tuple
    recon_shape: tuple
    gpu_model: str
    num_gpus: int
    gpu_memory_gb: float
    usable_memory_gib: float
    max_iterations: int
    host_memory_gib: dict
    direct: _StepEstimate
    recon: _StepEstimate
    split: _StepEstimate
    speed_data: str
    notes: list

    def __str__(self):
        lines = [f'Estimate for {self.model_name}: sinogram {tuple(self.sinogram_shape)}, '
                 f'recon {tuple(self.recon_shape)}']
        if self.gpu_memory_gb is None:
            lines.append(f'Hardware: {self.num_gpus} x {self.gpu_model}, memory per GPU unknown')
        else:
            lines.append(f'Hardware: {self.num_gpus} x {self.gpu_model}, {self.gpu_memory_gb:g} GB each, '
                         f'about {self.usable_memory_gib:.0f} GiB usable')
        host = self.host_memory_gib
        lines.append(f'Host memory needed: {host["total"]:.0f} GiB (sinogram {host["sinogram"]:.0f}, '
                     f'weights {host["weights"]:.0f}, reconstruction {host["recon"]:.0f})')
        lines.append('')
        lines.append(f'{"":30s}{"GPU memory needed":21s}{"fits":8s}time, {self.max_iterations} iterations')
        for key in ('direct', 'recon', 'split'):
            step = getattr(self, key)
            memory = 'not available' if step.gpu_memory_gib is None else f'{step.gpu_memory_gib:.1f} GiB'
            fits = '?' if step.fits is None else ('yes' if step.fits else 'no')
            if step.time_minutes is None:
                time = 'not available'
            elif step.time_minutes < 1.5:
                time = 'about 1 minute'
            else:
                time = f'about {step.time_minutes:.0f} minutes'
            lines.append(f'{_STEP_NAMES[key]:30s}{memory:21s}{fits:8s}{time}')
            if step.note:
                lines.append(f'  ({step.note})')
        lines.append('')
        fitting = [s.fits for s in (self.direct, self.recon, self.split) if s.fits is not None]
        if fitting and not any(fitting):
            lines.append(f'Nothing fits on {self.num_gpus} GPUs.  To fit, use more GPUs, or reduce the data with')
            lines.append('downsample_factor, subsample_view_factor, or cropping.')
        lines.extend(self.notes)
        if self.speed_data is None:
            lines.append('No times are available, because the speed file could not be read.')
        else:
            lines.append(f'Times are from {self.speed_data}, and are accurate to about a factor of '
                         f'{_TIME_ACCURACY_FACTOR:g}.')
        lines.append('GPU memory includes the 15% safety margin and assumes the reconstruction uses weights.')
        return '\n'.join(lines)


def estimate_resources(ct_model, gpu_model='H100', num_gpus=1, gpu_memory_gb=None, max_iterations=15):
    """This function estimates the GPU memory and the compute time that a
    reconstruction needs on hardware that the user names.  The function makes estimates for
    the direct reconstruction, the full reconstruction, and the split reconstruction
    (``recon_split_sino``).

    The memory estimate uses the same memory check that mbirtorch runs at the start of every
    reconstruction.  If the estimate says that a reconstruction fits, then that check passes.
    The estimate needs no data and no GPU.  The function does not change ``ct_model``.  The
    estimate assumes that the reconstruction uses weights.

    Args:
        ct_model (TomographyModel): The model to estimate.
        gpu_model (str, optional): The GPU model name, such as ``'H100'`` or ``'A100'``.
            Defaults to ``'H100'``.
        num_gpus (int, optional): The number of GPUs the job will use.  Defaults to 1.
        gpu_memory_gb (float, optional): The memory of each GPU, as printed on the card.
            Defaults to the value stored for ``gpu_model``.
        max_iterations (int, optional): The number of iterations, as in ``recon``.  Defaults
            to 15.

    Returns:
        An estimate.  Printing it gives the report.  ``estimate.direct``, ``estimate.recon``,
        and ``estimate.split`` each have these values:

        - ``gpu_memory_gib``: the memory each GPU needs, in GiB.
        - ``fits``: True if that memory fits on each GPU, False if not, and None if the GPU
          memory is unknown.
        - ``time_minutes``: the time in minutes, or None if it cannot be estimated.
        - ``note``: the reason a value is missing.

    Example:
        >>> estimate = mbirtorch.estimate_resources(ct_model, gpu_model='H100', num_gpus=4)
        >>> print(estimate)
        >>> if not estimate.recon.fits:  # the full reconstruction does not fit on these GPUs
        ...     recon, recon_dict = ct_model.recon_split_sino(sino)
    """
    num_gpus = int(num_gpus)
    if num_gpus < 1:
        raise ValueError('num_gpus must be at least 1.')
    speeds, speed_reason = _load_speed_file()
    gpu_info = (speeds or {}).get('gpus', {}).get(gpu_model, {})
    notes = []
    if gpu_memory_gb is None:
        gpu_memory_gb = gpu_info.get('card_memory_gb')
        if gpu_memory_gb is None:
            notes.append(f'No memory is stored for {gpu_model}, so whether each step fits is not judged.')
    reserve = gpu_info.get('driver_reserve_gib')
    if reserve is None:
        reserve = _DEFAULT_DRIVER_RESERVE_GIB
        if gpu_memory_gb is not None:
            notes.append(f'No driver reserve is measured for {gpu_model}; {reserve:g} GiB is assumed.')
    usable = None if gpu_memory_gb is None else float(gpu_memory_gb) - reserve

    sinogram_shape = tuple(int(n) for n in ct_model.get_params('sinogram_shape'))
    recon_shape = tuple(int(n) for n in ct_model.get_params('recon_shape'))
    sino_gib = math.prod(sinogram_shape) * 4 / _GIB
    recon_gib = math.prod(recon_shape) * 4 / _GIB
    host = {'sinogram': sino_gib, 'weights': sino_gib, 'recon': recon_gib,
            'total': 2 * sino_gib + recon_gib}

    def step(model, workload, iterations, gpus=num_gpus):
        """This function returns the _StepEstimate of one reconstruction of ``model`` on
        ``gpus`` GPUs, without splitting."""
        memory = _gpu_memory_gib(model, gpus, workload)
        fits = None if usable is None else memory <= usable
        minutes, reason = _time_minutes(model, speeds, speed_reason, gpu_model, gpus, workload, iterations)
        return _StepEstimate(memory, fits, minutes, reason or '')

    direct = step(ct_model, 'direct', 1)
    recon = step(ct_model, 'recon', max_iterations)
    split = _split_estimate(ct_model, num_gpus, usable, step, max_iterations)

    speed_data = None if speeds is None else (
        f'the nightly runs of {speeds["source"]["date"]} on {speeds["reference_gpu"]} GPUs')
    return _ResourceEstimate(type(ct_model).__name__, sinogram_shape, recon_shape, gpu_model, num_gpus,
                            gpu_memory_gb, usable, max_iterations, host, direct, recon, split,
                            speed_data, notes)


def projection_work(ct_model):
    """This function returns the projection work W of one forward or back projection.  W is
    the number of voxels times the number of views.  The speed file gives each time as a
    linear function of W.
    """
    num_views = int(ct_model.get_params('sinogram_shape')[0])
    return float(math.prod(int(n) for n in ct_model.get_params('recon_shape'))) * num_views


# ── memory ──────────────────────────────────────────────────────────────────────────────

def _gpu_projection_functions(ct_model):
    """This function returns the (forward, back) projection functions that a CUDA GPU runs
    for this geometry.  The function returns None when the geometry uses the same functions on
    every kind of device.
    """
    if isinstance(ct_model, ConeBeamModel):
        from .triton_cone import _cone_back_view_batch_triton, _cone_forward_view_batch_triton
        return _cone_forward_view_batch_triton, _cone_back_view_batch_triton
    if isinstance(ct_model, ParallelBeamModel):
        from .triton_parallel import _parallel_back_view_batch_triton, _parallel_forward_view_batch_triton
        return _parallel_forward_view_batch_triton, _parallel_back_view_batch_triton
    if isinstance(ct_model, MultiAxisParallelModel):
        from .triton_multiaxis import _multiaxis_back_view_batch_triton, _multiaxis_forward_view_batch_triton
        return _multiaxis_forward_view_batch_triton, _multiaxis_back_view_batch_triton
    return None


def _gpu_memory_gib(ct_model, num_gpus, workload, weights=True):
    """This function returns the GPU memory, in GiB, that a reconstruction of ``ct_model``
    needs on each of ``num_gpus`` GPUs.  The value includes the safety margin.
    """
    model = copy_ct_model(ct_model, no_warning=True)
    gpu_functions = _gpu_projection_functions(model)
    if gpu_functions is not None:
        # This line changes only the copy, not ct_model.
        model._view_batch_bodies = lambda: gpu_functions
    devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
    call_arrays = {}
    if weights:
        # This read-only array has the shape of the sinogram and uses almost no memory.
        shape = tuple(int(n) for n in model.get_params('sinogram_shape'))
        call_arrays['weights'] = np.broadcast_to(np.ones(1, dtype=np.float32), shape)
    ledger = model._build_memory_ledger(devices=devices, workload=workload, **call_arrays)
    return max(ledger.per_device_peaks()) * (1.0 + model.memory_preflight_margin) / _GIB


# ── split reconstruction ────────────────────────────────────────────────────────────────

def _split_estimate(ct_model, num_gpus, usable, step, max_iterations):
    """This function returns the _StepEstimate of ``recon_split_sino``.  The memory is the
    memory of the largest section.  The time is the time of the slowest group of GPUs.
    """
    devices = list(range(num_gpus))

    def fits_all(sections):
        return lambda group: all(_gpu_memory_gib(section, len(group), 'recon') <= usable for section in sections)

    if isinstance(ct_model, ConeBeamModel):
        sections, note = _cone_split_parts(ct_model)
        gpus_per_group = num_gpus
        if sections is not None and len(sections) == 2 and usable is not None:
            gpus_per_group = TomographyModel._choose_gpus_per_group(devices, fits_all(sections), max_groups=2)
    elif isinstance(ct_model, ParallelBeamModel):
        sections, note, gpus_per_group = _parallel_split_sections(ct_model, num_gpus, usable)
    else:
        return _StepEstimate(note=f'{type(ct_model).__name__} has no split reconstruction')
    if sections is None:
        return _StepEstimate(note=note)
    num_groups = num_gpus // gpus_per_group
    if num_groups > 1:
        note = f'{note}; {num_groups} groups of {gpus_per_group} GPUs side by side'
    estimates = [step(section, 'recon', max_iterations, gpus_per_group) for section in sections]
    memory = max(e.gpu_memory_gib for e in estimates)
    fits = None if usable is None else memory <= usable
    times = [e.time_minutes for e in estimates]
    minutes = None if any(t is None for t in times) else _slowest_group_minutes(times, num_groups)
    time_note = next((e.note for e in estimates if e.note), '')
    return _StepEstimate(memory, fits, minutes, '; '.join(n for n in (note, time_note) if n))


def _slowest_group_minutes(times, num_groups):
    """This function returns the time, in minutes, at which the last group of GPUs finishes.
    The function assumes that each group starts the next waiting section when it finishes its
    current section."""
    finish = [0.0] * num_groups
    for minutes in times:
        group = finish.index(min(finish))
        finish[group] += minutes
    return max(finish)


def _cone_split_parts(ct_model):
    """This function returns the two part models that ConeBeamModel.recon_split_sino
    reconstructs, and a note.  Each part model is a copy of ``ct_model`` with fewer detector
    rows and fewer slices.
    """
    if any(np.asarray(ct_model.get_params('view_params_array'))[:, 1] != 0):
        return None, 'not available for a helical scan'
    num_views, num_rows, num_cols = (int(n) for n in ct_model.get_params('sinogram_shape'))
    delta_det_row, det_row_offset = ct_model.get_params(['delta_det_row', 'det_row_offset'])
    delta_voxel, voxel_slice_aspect, voxel_row_aspect = ct_model.get_params(
        ['delta_voxel', 'voxel_slice_aspect', 'voxel_row_aspect'])
    recon_shape = tuple(int(n) for n in ct_model.get_params('recon_shape'))
    recon_slice_offset = ct_model.get_params('recon_slice_offset')
    source_iso_dist, use_ror_mask = ct_model.get_params(['source_iso_dist', 'use_ror_mask'])
    delta_voxel_slice = voxel_slice_aspect * delta_voxel
    from .vcd_utils import get_support_radius
    half_overlap = _SPLIT_HALF_OVERLAP
    ratio = delta_voxel_slice / max(delta_det_row / ct_model.get_magnification(), 1e-12)
    half_overlap_sino = int(round(half_overlap * ratio)) if ratio > 1 else half_overlap
    support_radius = get_support_radius(recon_shape, voxel_row_aspect * delta_voxel, delta_voxel,
                                        use_ror_mask=use_ror_mask)
    half_overlap_recon = int(np.ceil(half_overlap_sino * (1.0 + support_radius / float(source_iso_dist))
                                     / ratio)) + 2
    det_iso_row = int(round((num_rows - 1) / 2.0 + det_row_offset / delta_det_row))
    if not 0 < det_iso_row < num_rows:
        return None, 'not available: the central ray misses the detector'
    split_index = int(round((recon_shape[2] - 1) / 2.0 - recon_slice_offset / delta_voxel_slice))
    top_slices = split_index + 1
    if (split_index < 1 or split_index > recon_shape[2] - 2 or top_slices < half_overlap_recon
            or recon_shape[2] - top_slices < half_overlap_recon):
        return [ct_model], 'the volume is too thin to split, so it runs as one reconstruction'
    rows = [min(det_iso_row + half_overlap_sino, num_rows), num_rows - max(det_iso_row - half_overlap_sino, 0)]
    slices = [top_slices + half_overlap_recon, recon_shape[2] - top_slices + half_overlap_recon]
    parts = [_part_model(ct_model, r, (recon_shape[0], recon_shape[1], s)) for r, s in zip(rows, slices)]
    return parts, f'2 parts of {rows[0]} and {rows[1]} detector rows'


def _parallel_split_sections(ct_model, num_gpus, usable):
    """This function returns the section models that ParallelBeamModel.recon_split_sino
    reconstructs on the given GPUs, a note, and the number of GPUs in each group.
    """
    num_rows = int(ct_model.get_params('sinogram_shape')[1])
    recon_rows, recon_cols = (int(n) for n in ct_model.get_params('recon_shape')[:2])
    half_overlap = _SPLIT_HALF_OVERLAP
    max_sections = num_rows // (2 * half_overlap)
    if max_sections < 2:
        return [ct_model], 'the volume is too thin to split, so it runs as one reconstruction', num_gpus
    if usable is None:
        return (None, 'not available: the number of sections depends on the GPU memory, which is not known',
                num_gpus)

    def section(rows):
        return _part_model(ct_model, rows, (recon_rows, recon_cols, rows))

    def fits(rows, gpus):
        return _gpu_memory_gib(section(rows), gpus, 'recon') <= usable

    min_rows = min(_MIN_SLICES_PER_SECTION + 2 * half_overlap, num_rows)
    gpus_per_group = TomographyModel._choose_gpus_per_group(list(range(num_gpus)),
                                                            lambda group: fits(min_rows, len(group)))
    num_sections = max_sections
    for candidate in range(1, max_sections + 1):
        if fits(_parallel_largest_section_rows(num_rows, candidate, half_overlap), gpus_per_group):
            num_sections = candidate
            break
    num_groups = num_gpus // gpus_per_group
    if num_groups > 1:
        num_sections = max(num_sections, min(num_groups, num_rows // _MIN_SLICES_PER_SECTION, max_sections))
    if num_sections == 1:
        return [ct_model], 'it fits without splitting, so it runs as one reconstruction', num_gpus
    rows = _parallel_largest_section_rows(num_rows, num_sections, half_overlap)
    return [section(rows)] * num_sections, f'{num_sections} sections of up to {rows} detector rows', gpus_per_group


def _parallel_largest_section_rows(num_rows, num_sections, half_overlap):
    """This function returns the number of detector rows in the largest section model when
    ParallelBeamModel.recon_split_sino makes ``num_sections`` sections.
    """
    biggest = -(-num_rows // num_sections)
    if num_sections == 1:
        return biggest
    return biggest + (half_overlap if num_sections == 2 else 2 * half_overlap)


def _part_model(ct_model, num_rows, recon_shape):
    """This function returns a copy of ``ct_model`` with ``num_rows`` detector rows and
    ``recon_shape``."""
    model = copy_ct_model(ct_model, new_num_det_rows=num_rows, no_warning=True)
    model.set_params(no_warning=True, auto_regularize_flag=False, recon_shape=recon_shape)
    return model


# ── time ────────────────────────────────────────────────────────────────────────────────

def _load_speed_file():
    """This function returns the contents of the speed file and None.  If the file cannot be
    read, the function returns None and the reason."""
    try:
        text = importlib.resources.files('mbirtorch').joinpath('data/gpu_speeds.json').read_text()
        return json.loads(text), None
    except (OSError, ValueError, ModuleNotFoundError) as error:
        return None, f'the speed file could not be read ({type(error).__name__})'


def _geometry_key(ct_model):
    """This function returns the name that the speed file uses for the geometry of
    ``ct_model``."""
    for cls, key in ((ConeBeamModel, 'cone'), (ParallelBeamModel, 'parallel'),
                     (MultiAxisParallelModel, 'multiaxis'), (TranslationModel, 'translation')):
        if isinstance(ct_model, cls):
            return key
    return type(ct_model).__name__


def _time_minutes(ct_model, speeds, speed_reason, gpu_model, num_gpus, workload, iterations):
    """This function returns the time, in minutes, of one reconstruction without splitting,
    and a note.  If no time is available, the function returns None and the reason."""
    if speeds is None:
        return None, speed_reason
    gpu = speeds['gpus'].get(gpu_model)
    if gpu is None or gpu.get('time_factor') is None:
        return None, f'no speed data for {gpu_model}'
    lines = speeds['times'].get(_geometry_key(ct_model), {}).get(workload)
    if not lines:
        return None, 'no speed data for this geometry'
    counts = sorted(int(n) for n in lines)
    count = num_gpus if num_gpus in counts else max(n for n in counts if n <= num_gpus) \
        if any(n <= num_gpus for n in counts) else None
    if count is None:
        return None, f'no speed data for {num_gpus} GPUs'
    line = lines[str(count)]
    work = projection_work(ct_model)
    seconds = (line['a'] + line['b'] * work) * gpu['time_factor']
    if workload != 'direct':
        seconds *= iterations
    notes = []
    if count != num_gpus:
        notes.append(f'time measured on {count} GPUs')
    if gpu['time_factor'] != 1.0:
        notes.append(f'time scaled from {speeds["reference_gpu"]}')
    if work > line['largest_work']:
        notes.append(f'time extrapolated to {work / line["largest_work"]:.0f} times the largest measured problem')
    return seconds / 60.0, '; '.join(notes)

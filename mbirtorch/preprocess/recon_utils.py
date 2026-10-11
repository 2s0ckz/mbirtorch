import numpy as np
import torch
from mbirtorch import _sharding

__all__ = ['segment_plastic_metal', 'multi_threshold_otsu', 'apply_cylindrical_mask']

# The largest number of elements in one chunk when the histogram is computed one shard at a time.
_HISTOGRAM_CHUNK_ELEMENTS = 1 << 24


def _shard_valid_masks(valid_mask, placement, ndim):
    """This function splits a mask in host memory into one piece for each shard.

    A shard is the part of a volume that is stored on one device.  If the mask has length 1 on
    the shard axis, each piece is the whole mask.  If ``valid_mask`` is None, the function
    returns a list of None values.
    """
    if valid_mask is None:
        return [None] * placement.n_devices
    mask = np.asarray(valid_mask)
    axis = placement.axis % ndim
    pieces = []
    for _dev, (s0, s1) in placement.shard_ranges():
        if mask.shape[axis] == 1:
            pieces.append(mask)
        else:
            sel = [slice(None)] * ndim
            sel[axis] = slice(s0, s1)
            pieces.append(mask[tuple(sel)])
    return pieces


def _shard_chunks(tensor, mask_piece):
    """This function splits a tensor along its first axis and yields (chunk, mask_chunk) pairs.

    Each chunk has at most ``_HISTOGRAM_CHUNK_ELEMENTS`` elements.  Each mask_chunk is a torch
    bool tensor or None.
    """
    per_row = max(1, int(np.prod(tensor.shape[1:], dtype=np.int64)))
    step = max(1, _HISTOGRAM_CHUNK_ELEMENTS // per_row)
    m = None
    if mask_piece is not None:
        m = torch.as_tensor(np.asarray(mask_piece), dtype=torch.bool,
                            device=tensor.device)
    for i in range(0, tensor.shape[0], step):
        chunk = tensor[i:i + step]
        if m is None:
            yield chunk, None
        else:
            mc = m if m.shape[0] == 1 else m[i:i + step]
            yield chunk, mc.expand(chunk.shape)


def _sharded_masked_histogram(shards, valid_mask, num_bins):
    """This function computes the histogram of the valid entries of a sharded volume, one shard
    at a time.

    A value near a bin edge can fall in a different bin than the bin that np.histogram chooses.

    Returns:
        tuple: (hist, bin_edges).  Both are numpy arrays in host memory.  ``hist`` holds int64
        counts, and ``bin_edges`` holds the edges that np.histogram computes.

    Raises:
        ValueError: If there are no valid entries, or if all valid entries have the same value.
    """
    masks = _shard_valid_masks(valid_mask, shards.placement,
                               shards.tensors[0].ndim)

    lo, hi = np.inf, -np.inf
    for t, mp in zip(shards.tensors, masks):
        for chunk, mc in _shard_chunks(t, mp):
            vals = chunk.reshape(-1) if mc is None else chunk[mc]
            if vals.numel() == 0:
                continue          # The chunk is empty, or the mask excludes all of it.
            lo = min(lo, float(vals.min()))
            hi = max(hi, float(vals.max()))

    # A range of zero width raises an error, because numpy would widen it silently.
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(
            'The sharded volume has no valid entries to histogram: every '
            'shard was empty or entirely excluded by valid_mask.')
    if hi <= lo:
        raise ValueError(
            f'The valid entries span the degenerate range [{lo}, {hi}] '
            f'(min == max), so there are no intensity classes to separate.  '
            f'Segmentation needs a volume that takes more than one value.')

    # Each chunk is counted on its own device.  The last bin is closed, so the value hi falls in it.
    hist = np.zeros(num_bins, dtype=np.int64)
    scale = num_bins / (hi - lo)
    for t, mp in zip(shards.tensors, masks):
        for chunk, mc in _shard_chunks(t, mp):
            vals = chunk.reshape(-1) if mc is None else chunk[mc]
            vals = vals[(vals >= lo) & (vals <= hi)]
            if vals.numel() == 0:
                continue
            idx = torch.clamp(((vals - lo) * scale).to(torch.int64),
                              max=num_bins - 1)
            hist += torch.bincount(idx, minlength=num_bins).cpu().numpy()

    # np.histogram on an empty array gives the same bin edges as the unsharded case.
    edges_dtype = shards.dtype
    empty = np.empty(0, dtype=str(edges_dtype).replace('torch.', ''))
    _, bin_edges = np.histogram(empty, bins=num_bins, range=(float(lo), float(hi)))
    return hist, bin_edges


def _masked_histogram(image, valid_mask, num_bins, xp):
    """This function computes the histogram of the valid entries of ``image``.

    The histogram range runs from the smallest to the largest valid entry.
    """
    inf = xp.asarray(xp.inf, dtype=image.dtype)
    lo = xp.min(xp.where(valid_mask, image, inf))
    hi = xp.max(xp.where(valid_mask, image, -inf))
    sentinel = hi + xp.maximum(xp.abs(hi), xp.asarray(1.0, dtype=image.dtype))  # Above hi, so outside the range.
    # The range endpoints are Python floats.  float32 endpoints would shift some edges.
    return xp.histogram(xp.where(valid_mask, image, sentinel), bins=num_bins, range=(float(lo), float(hi)))


def multi_threshold_otsu(image, classes=2, num_bins=1024, valid_mask=None):
    """
    This function computes the Otsu thresholds that divide an image into ``classes`` intensity
    classes.

    The thresholds minimize the within-class variance of the histogram.  The image can be a
    numpy array, a torch tensor, or a sharded volume.

    Args:
        image (numpy.ndarray, torch.Tensor, or Shards): The image, with floating-point values.
        classes (int, optional): The number of classes.  It must be at least 2.  Defaults to 2.
        num_bins (int, optional): The number of histogram bins.  Defaults to 1024.
        valid_mask (array or None, optional): A boolean mask of the entries to include.  If
            None, the function includes all entries.  Defaults to None.

    Returns:
        list of float: The ``classes - 1`` thresholds in increasing order.

    Example:
        >>> low, high = multi_threshold_otsu(recon, classes=3)
        >>> plastic = (recon > low) & (recon <= high)
    """
    if classes < 2:
        raise ValueError("Number of classes must be at least 2")

    if num_bins < classes:
        raise ValueError("Number of bins must be at least equal to number of classes")

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.detach().cpu().numpy()

    if isinstance(image, _sharding.Shards):
        hist, bin_edges = _sharded_masked_histogram(image, valid_mask, num_bins)
    elif valid_mask is not None:
        hist, bin_edges = _masked_histogram(image, np.asarray(valid_mask), num_bins, np)
    else:
        hist, bin_edges = np.histogram(image, bins=num_bins, range=(float(np.min(image)), float(np.max(image))))

    thresholds = _otsu_thresholds_dp(hist, classes - 1)

    # Values below bin_edges[t] fall in the lower classes.
    scaled_thresholds = [bin_edges[t] for t in thresholds]

    return scaled_thresholds


def _otsu_thresholds_dp(hist, num_thresholds):
    """This function finds the multi-threshold Otsu class boundaries by dynamic programming.

    A boundary ``t`` means that bin ``t`` is the first bin of the next class.

    Args:
        hist (ndarray): The histogram counts of the image.
        num_thresholds (int): The number of thresholds to find.

    Returns:
        list of int: The boundary indices in strictly increasing order, in ``[1, len(hist) - 1]``.
    """
    if num_thresholds == 0:
        return []

    hist = np.asarray(hist, dtype=np.float64)
    num_bins = len(hist)
    # The bin coordinates are scaled to [-1, 1] so that the prefix sums are well conditioned.
    half_span = max((num_bins - 1) / 2.0, 1.0)     # The max handles a one-bin histogram.
    bin_coord = (np.arange(num_bins, dtype=np.float64) - (num_bins - 1) / 2.0) / half_span

    # Each prefix sum P starts with a zero, so the moment of bins [a, b) is P[b] - P[a].
    m0 = np.concatenate(([0.0], np.cumsum(hist)))
    m1 = np.concatenate(([0.0], np.cumsum(bin_coord * hist)))
    m2 = np.concatenate(([0.0], np.cumsum(bin_coord * bin_coord * hist)))

    int_m0 = m0[None, :] - m0[:, None]
    int_m1 = m1[None, :] - m1[:, None]
    int_m2 = m2[None, :] - m2[:, None]

    # The within-class cost of bins [a, b) is M2 - M1^2/M0.  An empty interval has cost zero.
    # Entries with a >= b are set to infinity.
    mean_sq_term = np.divide(int_m1 ** 2, int_m0, out=np.zeros_like(int_m0), where=int_m0 > 0)
    cost = np.maximum(int_m2 - mean_sq_term, 0.0)      # Rounding can make the cost slightly negative.
    invalid = ~np.triu(np.ones((num_bins + 1, num_bins + 1), dtype=bool), k=1)   # a >= b
    cost[invalid] = np.inf

    # best[b] is the smallest cost of dividing bins [0, b) into the classes placed so far.
    best = cost[0, :].copy()                           # One class covering [0, b).
    split_of = np.zeros((num_thresholds, num_bins + 1), dtype=np.int64)
    for stage in range(num_thresholds):
        total = best[:, None] + cost                   # total[s, b]
        split_of[stage] = np.argmin(total, axis=0)
        best = np.min(total, axis=0)

    boundaries = []
    b = num_bins
    for stage in range(num_thresholds - 1, -1, -1):
        b = int(split_of[stage][b])
        boundaries.append(b)
    return boundaries[::-1]


def segment_plastic_metal(recon, num_metal, radial_margin=None, top_margin=None, bottom_margin=None):
    """
    This function segments a reconstruction into a plastic mask and one mask for each metal.

    The function first applies :func:`apply_cylindrical_mask` to the volume.  It then uses Otsu
    thresholds to divide the volume into ``num_metal + 2`` classes.  The classes are the
    background, the plastic, and the metals in order of increasing value.  The scale of each mask
    is the mean of the volume over the mask.  The masks have the same type as ``recon``.

    Args:
        recon (numpy.ndarray, torch.Tensor, or Shards): The reconstruction, with shape
            (num_rows, num_cols, num_slices).
        num_metal (int): The number of metals.
        radial_margin (int or None, optional): The number of pixels subtracted from the cylinder
            radius.  If None, the function uses
            ``max(2, min(10, min(num_rows, num_cols) // 25))``.  Defaults to None.
        top_margin (int or None, optional): The number of slices masked at the top.  If None,
            the function uses ``max(2, min(10, num_slices // 25))``.  Defaults to None.
        bottom_margin (int or None, optional): The number of slices masked at the bottom.  If
            None, the function uses the same default as ``top_margin``.  Defaults to None.

    Returns:
        tuple: ``(plastic_mask, metal_masks, plastic_scale, metal_scales)``.  ``metal_masks`` is
        a list of ``num_metal`` masks.  ``plastic_scale`` is a float, and ``metal_scales`` is a
        list of floats.
    """
    if num_metal <= 0:
        raise ValueError("num_metal must be positive")

    is_shards = isinstance(recon, _sharding.Shards)
    if is_shards and recon.placement.is_trivial:
        pl = recon.placement
        plastic_mask, metal_masks, plastic_scale, metal_scales = segment_plastic_metal(
            recon.tensors[0], num_metal, radial_margin=radial_margin,
            top_margin=top_margin, bottom_margin=bottom_margin)
        return (_sharding.Shards([plastic_mask], pl),
                [_sharding.Shards([m], pl) for m in metal_masks],
                plastic_scale, metal_scales)

    shape = recon.tensors[0].shape if is_shards else recon.shape
    num_slices = recon.placement.axis_len if is_shards else recon.shape[2]
    if radial_margin is None:
        radial_margin = max(2, min(10, min(shape[0], shape[1]) // 25))
    if top_margin is None:
        top_margin = max(2, min(10, num_slices // 25))
    if bottom_margin is None:
        bottom_margin = max(2, min(10, num_slices // 25))

    recon = apply_cylindrical_mask(recon, radial_margin=radial_margin, top_margin=top_margin,
                                       bottom_margin=bottom_margin)

    thresholds = multi_threshold_otsu(recon, classes=num_metal + 2)

    is_torch = isinstance(recon, torch.Tensor)

    plastic_low_threshold = thresholds[0]
    plastic_metal_threshold = thresholds[1]

    def _tensor_class_mask(vol, lower, upper):
        return ((vol > lower) & (vol <= upper)).to(vol.dtype)

    def class_mask(lower, upper):
        if is_shards:
            return _sharding.Shards(
                [_tensor_class_mask(t, lower, upper) for t in recon.tensors],
                recon.placement)
        if is_torch:
            return _tensor_class_mask(recon, lower, upper)
        in_class = (recon > lower) & (recon <= upper)
        return np.where(in_class, np.float32(1.0), np.float32(0.0)).astype(recon.dtype)

    plastic_mask = class_mask(plastic_low_threshold, plastic_metal_threshold)
    plastic_scale = compute_scaling_factor(recon, plastic_mask)

    metal_masks = []
    metal_scales = []
    for i in range(1, num_metal + 1):
        lower = thresholds[i]
        upper = thresholds[i + 1] if i + 1 < len(thresholds) else np.inf
        metal_mask = class_mask(lower, upper)
        metal_masks.append(metal_mask)
        metal_scales.append(compute_scaling_factor(recon, metal_mask))

    return plastic_mask, metal_masks, plastic_scale, metal_scales


def apply_cylindrical_mask(recon, radial_margin=0, top_margin=0, bottom_margin=0):
    """
    This function sets to zero the voxels of a reconstruction that lie outside a centered
    cylinder.

    This mask removes bright artifacts at the edge of the volume.  The cylinder radius is half the
    larger of the row and column extents, minus ``radial_margin``.  The function does not modify
    ``recon``.

    Args:
        recon (numpy.ndarray, torch.Tensor, or Shards): The reconstruction, with shape
            (num_rows, num_cols, num_slices).
        radial_margin (int, optional): The number of pixels subtracted from the cylinder radius.
            Defaults to 0.
        top_margin (int, optional): The number of slices set to zero at the top.  Defaults to 0.
        bottom_margin (int, optional): The number of slices set to zero at the bottom.
            Defaults to 0.

    Returns:
        The masked reconstruction, with the same shape and type as ``recon``.
    """
    # The top and bottom margins count slices of the whole volume, not of one shard.
    if isinstance(recon, _sharding.Shards):
        pl = recon.placement
        num_slices = pl.axis_len
        out = []
        for t, (_dev, (s0, s1)) in zip(recon.tensors, pl.shard_ranges()):
            masked = apply_cylindrical_mask(t, radial_margin=radial_margin)
            lo = max(s0, 0)
            hi = min(s1, top_margin)
            if hi > lo:
                masked[:, :, lo - s0:hi - s0] = 0
            lo = max(s0, num_slices - bottom_margin)
            hi = min(s1, num_slices)
            if hi > lo:
                masked[:, :, lo - s0:hi - s0] = 0
            out.append(masked)
        return _sharding.Shards(out, pl)

    is_torch = isinstance(recon, torch.Tensor)

    num_recon_rows, num_recon_cols, num_slices = recon.shape
    row_center = (num_recon_rows - 1) / 2
    col_center = (num_recon_cols - 1) / 2

    base_radius = max(row_center, col_center)
    radius = base_radius - radial_margin

    if is_torch:
        row_coords, col_coords = torch.meshgrid(
            torch.arange(num_recon_rows, device=recon.device),
            torch.arange(num_recon_cols, device=recon.device), indexing='ij')
        dist_sq = (row_coords - row_center) ** 2 + (col_coords - col_center) ** 2
        circular_mask = (dist_sq <= radius ** 2).to(recon.dtype)
    else:
        row_coords, col_coords = np.meshgrid(np.arange(num_recon_rows), np.arange(num_recon_cols), indexing='ij')
        dist_sq = (row_coords - row_center) ** 2 + (col_coords - col_center) ** 2
        circular_mask = (dist_sq <= radius ** 2).astype(recon.dtype)

    # This multiplication creates a new array, so the input is not modified.
    recon = recon * circular_mask[:, :, None]

    if top_margin > 0:
        recon[:, :, :top_margin] = 0
    if bottom_margin > 0:
        recon[:, :, num_slices - bottom_margin:num_slices] = 0

    return recon


def compute_scaling_factor(target_vect, vect_to_scale) -> float:
    """
    This function returns the scalar α that approximately minimizes
    ‖target_vect – α vect_to_scale‖².

    The function computes α as
    <target_vect, vect_to_scale> / (<vect_to_scale, vect_to_scale> + epsilon), with epsilon = 1e-8.

    Args:
        target_vect (ndarray or tensor): The target array, of any shape.
        vect_to_scale (ndarray or tensor): An array with the same shape as ``target_vect``.

    Returns:
        float: The scalar α.

    Example:
        >>> v = np.array([1.0, 2.0, 3.0])
        >>> u = np.array([0.5, 1.0, 1.5])
        >>> alpha = compute_scaling_factor(v,u)
    """
    if isinstance(target_vect, _sharding.Shards):
        numerator = 0.0
        denominator = 0.0
        for t, v in zip(target_vect.tensors, vect_to_scale.tensors):
            n, d = _dot_sums(t, v)
            numerator += n
            denominator += d
        return float(numerator / (denominator + 1e-8))

    if not isinstance(target_vect, torch.Tensor):
        target_vect = torch.as_tensor(np.asarray(target_vect))
    if not isinstance(vect_to_scale, torch.Tensor):
        vect_to_scale = torch.as_tensor(np.asarray(vect_to_scale), device=target_vect.device)

    if target_vect.ndim == 0:
        numerator = float(vect_to_scale * target_vect)
        denominator = float(vect_to_scale * vect_to_scale)
    else:
        numerator, denominator = _dot_sums(target_vect, vect_to_scale)
    epsilon = 1e-8
    return float(numerator / (denominator + epsilon))


def _dot_sums(target_vect, vect_to_scale):
    """This function returns the inner products <v,t> and <v,v>."""
    numerator = 0.0
    denominator = 0.0
    per_row = max(1, int(np.prod(target_vect.shape[1:], dtype=np.int64)))
    step = max(1, (1 << 27) // per_row)
    for i in range(0, target_vect.shape[0], step):
        t = target_vect[i:i + step]
        v = vect_to_scale[i:i + step]
        numerator += float(torch.sum(v * t))
        denominator += float(torch.sum(v * v))
    return numerator, denominator

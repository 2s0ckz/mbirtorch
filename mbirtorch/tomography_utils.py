"""This module builds the filter for direct reconstruction and applies it to each detector row.
"""

import numpy as np
import torch

# The number of detector rows filtered in each batch.
ROW_FILTER_BATCH = 1024


def generate_direct_recon_filter(num_channels, filter_name="ramp"):
    """This function returns the filter for direct reconstruction in the space domain.  The
    filter has length 2*num_channels - 1."""
    supported_filters = ["ramp"]
    if filter_name not in supported_filters:
        raise ValueError(f"Unsupported filter. Supported filters are: "
                         f"{', '.join(supported_filters)}.")
    n = np.arange(-num_channels + 1, num_channels)
    recon_filter = (1 / 2) * np.sinc(n) - (1 / 4) * (np.sinc(n / 2)) ** 2
    return recon_filter.astype(np.float32)


def apply_row_filter(block, filter_arr, row_weight=None):
    """This function convolves every detector row of ``block`` with ``filter_arr``.  The
    function keeps only the 'valid' part of each convolution, which has the same length as the
    row.  If ``row_weight`` is given, the function multiplies each row by ``row_weight`` before
    filtering.

    Args:
        block: (views, rows, channels) float32 tensor.
        filter_arr: (2*channels - 1,) float32 tensor.  The caller scales the filter.
        row_weight: (rows, channels) float32 tensor or None.  This is the FDK cosine weight,
            and the function applies the same weight to every view.  None (default) means
            that no weight is applied, as in FBP.

    Returns:
        (views, rows, channels) tensor.
    """
    n_views, n_rows, n_channels = block.shape
    filt_len = filter_arr.shape[0]
    full_len = n_channels + filt_len - 1
    rows = block.reshape(-1, n_channels)
    total_rows = rows.shape[0]

    filt_f = torch.fft.rfft(filter_arr, n=full_len)
    # The valid part of a length C signal convolved with a length 2C-1 filter is
    # filtered[C-1 : C-1+C].
    start = n_channels - 1
    out = torch.empty_like(rows)
    # A range step of zero is an error even when the range is empty.
    batch = max(1, min(ROW_FILTER_BATCH, total_rows))
    for r0 in range(0, total_rows, batch):
        window = rows[r0:r0 + batch]
        if row_weight is not None:
            # Row k of the flattened array is detector row k % n_rows.
            det_rows = torch.arange(r0, r0 + window.shape[0],
                                    device=block.device) % n_rows
            window = window * row_weight[det_rows]
        win_f = torch.fft.rfft(window, n=full_len, dim=1)
        filtered = torch.fft.irfft(win_f * filt_f, n=full_len, dim=1)
        out[r0:r0 + window.shape[0]] = filtered[:, start:start + n_channels]
    return out.reshape(n_views, n_rows, n_channels)

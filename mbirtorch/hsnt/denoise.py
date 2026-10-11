"""This module compresses and restores hyperspectral neutron data with a factorization X = W H.
The function ``dehydrate`` computes the factors, and the function ``rehydrate`` multiplies them."""
import warnings

import numpy as np

from ..utilities import _to_host

# These keywords belong to the MBIRJAX version of dehydrate, which used scikit-learn.  mbirtorch rejects them.
_L2_KEYWORDS = ("safety_factor", "beta_loss", "max_iter", "tolerance", "batch_size", "random_state")


def _reject_unknown_keywords(name, kwargs):
    if not kwargs:
        return
    legacy = sorted(set(kwargs) & set(_L2_KEYWORDS))
    if legacy:
        raise TypeError(f"{name}() fits the Poisson likelihood and does not take {', '.join(legacy)}, which belong to "
                        "MBIRJAX's scikit-learn NMF dehydrate; MBIRTorch does not include that method")
    raise TypeError(f"{name}() got unexpected keyword argument(s) {', '.join(sorted(kwargs))}")


def _spatial_shape(lead):
    """This function returns the leading axes as (views, rows, cols).  It sets views to 1 when
    there are two leading axes, and it returns None when there are not two or three."""
    return tuple(lead) if len(lead) == 3 else (1,) + tuple(lead) if len(lead) == 2 else None


def _clean_transmission(T):
    """This function sets each NaN, infinite, or negative entry of a float32 transmission array to
    zero, in place.  It returns a dict with the fraction of entries that it changed."""
    info = {}
    bad = ~np.isfinite(T)
    if bad.any():
        info["nonfinite_frac"] = float(bad.mean())
        T[bad] = 0.0
    negative = T < 0
    if negative.any():
        info["negative_frac"] = float(negative.mean())
        T[negative] = 0.0
    return info


def _warn_cleaning(info):
    if info.get("nonfinite_frac"):
        warnings.warn(f"{100 * info['nonfinite_frac']:.3g}% of the entries are NaN or have an infinite transmission; "
                      "treated as zero counts")
    if info.get("negative_frac"):
        warnings.warn(f"{100 * info['negative_frac']:.3g}% of the entries have a negative transmission; clipped at "
                      "zero")


def _to_transmission(data, dataset_type):
    """This function converts the data to a float32 transmission array of shape (pixels, bins).
    It cleans the transmission array with ``_clean_transmission``.  It returns the transmission
    array and the shape of the input."""
    if dataset_type not in ("attenuation", "transmission"):
        raise ValueError("'dataset_type' must be either 'attenuation' or 'transmission'.")
    a = _to_host(data)
    if a.ndim < 2:
        raise ValueError(f"data must have at least one leading axis and a spectral last axis; got shape {a.shape}")
    shape = a.shape
    a = a.reshape(-1, shape[-1]).astype(np.float32, copy=False)
    if dataset_type == "attenuation":
        with np.errstate(over="ignore", invalid="ignore"):
            T = np.exp(-a)
    else:
        T = a.copy()
    _warn_cleaning(_clean_transmission(T))
    if not bool((T > 0).any()):
        raise ValueError("the data hold no counts: the transmission is zero everywhere")
    return np.ascontiguousarray(T, dtype=np.float32), shape


def dehydrate(data, dataset_type="attenuation", num_materials=None, *, subspace_basis=None, spectra="mle", dose=None,
              penalty="auto", max_steps=1000, rel_tol=1e-8, max_rank=6, device=None,
              compile_mode="auto", mode="auto", chunk_pixels=None, max_passes=5, verbose=1, **kwargs):
    """This function factors the attenuation of a hyperspectral dataset as X = W H by maximum
    likelihood.  The factors satisfy W >= 0 and H >= 0.

    The spectral axis of the data is the last axis.  The maps ``W`` keep the leading axes of the
    data.  This function handles zero counts correctly.  The components form a nonnegative basis
    of the data, but they are not necessarily the spectra of pure materials.

    Args:
        data (numpy.ndarray or torch.Tensor): The data.  The last axis is the spectral axis,
            of length :math:`N_k`.  This function treats a NaN or infinite transmission as zero
            counts.  It clips a negative transmission to zero.
        dataset_type (str, optional): 'attenuation' or 'transmission'. Defaults to 'attenuation'.
        num_materials (int, optional): The rank :math:`N_m`.  Defaults to None, in which case
            this function estimates the rank.
        subspace_basis (numpy.ndarray or torch.Tensor, optional): Spectra of shape
            :math:`(N_m, N_k)`.  When it is given, this function holds the spectra fixed and
            computes only the maps.  Defaults to None.
        spectra (str, optional): The method for estimating the spectra.  'mle' gives the
            maximum likelihood estimate.  'unconstrained' removes the bias that the constraint
            W >= 0 causes at low dose.  'support' selects which components are present in each
            pixel, and it requires ``dose``.  Defaults to 'mle'.
        dose (float, optional): The open beam counts per pixel and bin.  Only 'support' uses
            it.  Defaults to None.
        penalty (str or float, optional): The penalty per component that 'support' uses.  It
            is either a multiple of log(N_k) or 'auto'.  Defaults to 'auto'.
        max_steps (int, optional): The maximum number of solver iterations.  Defaults to 1000.
        rel_tol (float, optional): The solver stops when the relative change in the loss falls
            below this value.  Defaults to 1e-8.
        max_rank (int, optional): The largest rank that the rank estimate considers.  Defaults
            to 6.
        device (str, optional): The torch device.  Defaults to None, which uses CUDA when it is
            available and the CPU otherwise.
        compile_mode (str, optional): 'auto', 'on', or 'off'.  The value sets whether this
            function uses torch.compile.  Defaults to 'auto'.
        mode (str, optional): 'full', 'stream', or 'auto'.  'stream' processes the pixels in
            chunks.  Defaults to 'auto'.
        chunk_pixels (int, optional): The number of pixels in each chunk when streaming.
            Defaults to None, in which case this function sets it from the available memory.
        max_passes (int, optional): The number of passes over the chunks when streaming.
            Defaults to 5.
        verbose (int, optional): 0 prints nothing, 1 prints a summary, and 2 also prints the
            rank search.  Defaults to 1.

    Returns:
        list: The list [subspace_data, subspace_basis, dataset_type].  subspace_data is ``W``.
            Its axes are the leading axes of the data, followed by an axis of length :math:`N_m`.
            subspace_basis is ``H``, with shape :math:`(N_m, N_k)`.  Both are float32.
            dataset_type is the dataset_type of the input.
    """
    from ._device import _default_device
    from ._fit import _fit
    from .rank import _estimate_rank
    _reject_unknown_keywords("dehydrate", kwargs)
    if mode not in ("auto", "full", "stream"):
        raise ValueError(f"mode must be 'auto', 'full' or 'stream', got {mode!r}")
    if chunk_pixels is not None and int(chunk_pixels) < 1:
        raise ValueError(f"chunk_pixels must be a positive number of pixels, got {chunk_pixels}")
    if int(max_passes) < 0:
        raise ValueError(f"max_passes must be at least 0, got {max_passes}")
    T, shape = _to_transmission(data, dataset_type)
    device = _default_device(device)
    lead = shape[:-1]
    if subspace_basis is not None:
        from ._fit import _fit_fixed_basis
        H = _check_basis(subspace_basis, shape[-1], num_materials, spectra)
        W, H, rep = _fit_fixed_basis(T, H, device=device, mode=mode, chunk_pixels=chunk_pixels, max_steps=max_steps,
                                     rel_tol=rel_tol, compile_mode=compile_mode)
        if verbose >= 1:
            print("dehydrate(): ")
            print("   -Spectral dimension: ", shape[-1], " -> rank: ", H.shape[0], "(the given subspace_basis)")
            print("   -Pixels: ", T.shape[0], f"; maps for the given basis ({rep['chunks']} chunk(s))")
        return [W.reshape(*lead, H.shape[0]), H, dataset_type]
    note = f"rank {num_materials} given"
    if num_materials is None:
        num_materials, note, _ = _estimate_rank(T, spatial_shape=_spatial_shape(lead), device=device, max_rank=max_rank,
                                                verbose=max(0, verbose - 1))
    W, H, rep = _fit(T, int(num_materials), spectra=spectra, dose=dose, penalty=penalty, device=device, mode=mode, chunk_pixels=chunk_pixels, max_steps=max_steps, rel_tol=rel_tol,
                     max_passes=int(max_passes), compile_mode=compile_mode)
    subspace_data = W.reshape(*lead, int(num_materials))
    if verbose >= 1:
        solve = f"{rep['steps']} steps" if "steps" in rep else f"streamed, {rep['passes']} polish passes"
        print("dehydrate(): ")
        print("   -Spectral dimension: ", shape[-1], " -> rank: ", int(num_materials), f"({note})")
        print("   -Pixels: ", T.shape[0], f"; spectra: {spectra} ({solve})")
    return [subspace_data, H, dataset_type]


def _check_basis(subspace_basis, bins, num_materials, spectra):
    """This function checks ``subspace_basis`` against the data and returns it as a float32 array
    of shape (rank, bins)."""
    H = np.asarray(_to_host(subspace_basis), dtype=np.float32)
    if H.ndim != 2 or H.shape[1] != bins:
        raise ValueError(f"subspace_basis must have shape (rank, {bins}) for data with {bins} bins; got {H.shape}")
    if not np.isfinite(H).all() or (H < 0).any():
        raise ValueError("subspace_basis must be finite and nonnegative")
    if num_materials is not None and int(num_materials) != H.shape[0]:
        raise ValueError(f"num_materials={num_materials} disagrees with the {H.shape[0]} rows of subspace_basis")
    if spectra != "mle":
        raise ValueError(f"spectra={spectra!r} re-estimates the spectra, which a given subspace_basis holds fixed")
    return H


def hyper_denoise(data, dataset_type="attenuation", num_materials=None, **kwargs):
    """This function denoises hyperspectral data.  It calls ``dehydrate`` and then ``rehydrate``.

    Args:
        data (numpy.ndarray or torch.Tensor): The data.  The last axis is the spectral axis.
        dataset_type (str, optional): 'attenuation' or 'transmission'. Defaults to 'attenuation'.
        num_materials (int, optional): The rank.  Defaults to None, in which case
            ``dehydrate`` estimates the rank.
        **kwargs: The keyword arguments of :func:`~mbirtorch.hsnt.dehydrate`.

    Returns:
        numpy.ndarray: The denoised float32 data, with the shape and dataset_type of the input.
    """
    _reject_unknown_keywords("hyper_denoise", {k: v for k, v in kwargs.items() if k in _L2_KEYWORDS})
    return rehydrate(dehydrate(data, dataset_type=dataset_type, num_materials=num_materials, **kwargs))


def rehydrate(dehydrated_data, hyperspectral_idx=None):
    """This function restores hyperspectral data from the output of ``dehydrate``.  It multiplies
    the maps by the spectra, for all bins or for selected bins.  The method is described in
    M. S. N. Chowdhury et al., "Fast Hyperspectral Neutron Tomography," IEEE TCI, vol. 11,
    pp. 663-677, 2025, doi:10.1109/TCI.2025.3567854.

    Args:
        dehydrated_data: The list [subspace_data, subspace_basis, dataset_type].  subspace_data
            holds the maps, with the component axis last.  subspace_basis holds the spectra,
            with shape :math:`(N_s, N_k)`.  dataset_type is 'attenuation' or 'transmission'.
        hyperspectral_idx: The indices of the bins to restore.  Defaults to None, which
            restores all bins.

    Returns:
        numpy.ndarray: The restored data.  Its last axis holds the bins, and its other axes match
            subspace_data.  The values are attenuation or transmission, as dataset_type says.
    """
    [subspace_data, subspace_basis, dataset_type] = dehydrated_data

    if hyperspectral_idx is None:
        rehydrated_data = subspace_data @ subspace_basis
    else:
        rehydrated_data = subspace_data @ subspace_basis[:, hyperspectral_idx]

    if dataset_type == 'transmission':
        rehydrated_data = np.exp(-rehydrated_data)

    return rehydrated_data

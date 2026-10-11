import os
import random
import tempfile
import warnings
import numpy as np
import mbirtorch as mt
import tqdm


def _normalize_by_norm(x, eps):
    """This function returns ``x / (||x|| + eps)``."""
    return x / (np.linalg.norm(x) + eps)


def _normalize_and_project(x, ref, eps):
    """This function returns the normalized ``x`` and its inner product with ``ref`` over all axes."""
    x_normalized = x / (np.linalg.norm(x) + eps)
    return x_normalized, np.tensordot(x_normalized, ref, axes=x_normalized.ndim)


def _make_single_view_sibling(ct_model):
    """This function returns a copy of ``ct_model`` that has one view.  The copy is used to
    back project one view at a time.
    """
    required, optional, regularization = ct_model.get_all_params()
    required = dict(required)
    sino_shape = tuple(required['sinogram_shape'])
    num_views = sino_shape[0]
    required['sinogram_shape'] = (1,) + sino_shape[1:]
    for name, value in list(required.items()):
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == num_views:
            required[name] = value[0:1]
    optional = dict(optional)
    optional['recon_shape'] = tuple(ct_model.get_params('recon_shape'))
    sibling = mt.build_model(required, optional, regularization)
    # The model constructors take no device argument.  This line places the copy on the first
    # device of ``ct_model``.
    sibling.configure_devices(devices=[ct_model.torch_device])
    return sibling


def _set_sibling_view_params(sibling, view_params_row):
    """This function sets the view parameters of the one-view model copy to those of one view."""
    view_params_name = sibling.get_params('view_params_name')
    sibling.set_params(no_warning=True, **{view_params_name: np.asarray(view_params_row)})


def subsample_R_gamma(R, gamma, selected_indices):
    """
    This function returns the submatrix of R and the subvector of gamma at the selected
    indices.

    Args:
        R (ndarray): The full covariance matrix, with shape (N, N).
        gamma (ndarray): The full gamma vector, with shape (N, 1).
        selected_indices (ndarray): A 1D array of the K indices to select.

    Returns:
        tuple: (R_sub, gamma_sub), with shapes (K, K) and (K, 1).
    """
    R_sub = R[selected_indices[:, None], selected_indices]
    gamma_sub = gamma[selected_indices, :]
    return R_sub, gamma_sub


def max_abs_neighbor_diff(arr):
    padded = np.pad(arr, pad_width=1, mode='reflect')
    center = arr
    max_diff = np.zeros_like(arr)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for di, dj in directions:
        neighbor = padded[1 + di : 1 + di + arr.shape[0],
                          1 + dj : 1 + dj + arr.shape[1]]
        diff = np.abs(center - neighbor)
        np.maximum(max_diff, diff, out=max_diff)

    return max_diff



def get_opt_views(ct_model, reference_object, num_selected_views, r_1=0.002, r_2=0.5,
                  prev_selected_view_inds=np.array([], dtype=int), priority_order=False, verbose=0, seed=None, roi=None):
    """
    This function selects view angles by minimizing the View Covariance Loss (VCL).  It uses a
    stochastic greedy search.  The VCL is defined in J. Lin, A. Ziabari, S. V.
    Venkatakrishnan, O. Rahman, G. T. Buzzard, C. A. Bouman, "Tomographic Sparse View
    Selection using the View Covariance Loss", IEEE Transactions on Pattern Analysis and
    Machine Intelligence, 2025.

    Args:
        ct_model (TomographyModel): A CT model, such as ParallelBeamModel or ConeBeamModel,
            with the system geometry and the candidate angles.
        reference_object (ndarray): A 3D reference volume, such as the ground truth.
        num_selected_views (int): The number of view angles to select.
        r_1 (float, optional): The fraction of voxels sampled in the reference object.
            Defaults to 0.002.
        r_2 (float, optional): The fraction of candidate views sampled in each step of the
            stochastic search.  Defaults to 0.5.
        prev_selected_view_inds (ndarray, optional): A 1D array of view indices selected
            earlier.  Defaults to an empty NumPy array.
        priority_order (bool, optional): If True, the function orders the selected views from
            most to least important.  Defaults to False.
        verbose (int, optional): The verbosity level.  If verbose > 0, the function plots the
            covariance matrix and the gamma vector.
        seed (int, optional): The random seed for reproducible results.
        roi (ndarray, optional): A 3D mask with the same shape as reference_object.  The
            nonzero voxels mark the region used to select views.  If None, the function uses
            the whole reference_object.

    Returns:
        Tuple[ndarray, float]: The (K,) indices of the selected views in the ct_model angles,
        and the VCL value of the selected subset.

    Example:
        >>> angles = np.linspace(0, np.pi, num=180, endpoint=False)
        >>> ct_model = mt.ParallelBeamModel((180, 128, 100), angles)
        >>> ref_obj = np.random.rand(128, 128, 100)
        >>> selected_angles_idx, vcl_value = get_opt_views(ct_model, ref_obj, num_selected_views=10)
        >>> selected_angles = angles[selected_angles_idx]
    """
    if roi is not None:
        roi = np.asarray(roi)
        if roi.shape != reference_object.shape:
            raise ValueError(
                "roi.shape and reference_object.shape must match.\n"
                f" Got roi.shape = {roi.shape}, "
                f"reference_object.shape = {reference_object.shape}."
            )
        roi = roi.astype(bool)

    num_views = ct_model.get_params('sinogram_shape')[0]
    # Each model stores its view parameters under its own name.  The first component of each
    # view's parameters is its angle.
    view_params_name = ct_model.get_params('view_params_name')
    view_params = np.asarray(ct_model.get_params(view_params_name))
    angle_candidates = view_params.reshape(view_params.shape[0], -1)[:, 0]
    recon_shape = ct_model.get_params('recon_shape')
    if tuple(recon_shape) != tuple(reference_object.shape):
        raise ValueError("The recon shape from ct_model and reference_object.shape must match.\n Got ct_model recon_shape = {}, reference_shape = {}.".format(recon_shape, reference_object.shape))

    if isinstance(prev_selected_view_inds, list):
        prev_selected_view_inds = np.array(prev_selected_view_inds)
    if prev_selected_view_inds.size > 0:
        if not (np.issubdtype(prev_selected_view_inds.dtype, np.integer) and
                np.all(prev_selected_view_inds >= 0) and np.all(prev_selected_view_inds < len(angle_candidates))):
            raise ValueError("prev_selected_view_inds must contain integers in the range [0, {}).".format(num_views))

    with tempfile.TemporaryDirectory() as data_store_dir:
        gamma = compute_view_basis_functions(
            ct_model, reference_object, r_1=r_1,
            data_store_dir=data_store_dir, seed=seed, roi=roi
        )
        R = compute_cov_matrix(num_views, data_store_dir)

    if verbose > 0:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(R)
        axes[0].set_title(r'$R$ View-Covariance Matrix')
        axes[1].imshow(np.linalg.inv(R))
        axes[1].set_title(r'$B=R^{-1}$ View-Precision Matrix')
        axes[2].plot(gamma)
        axes[2].set_ylim([0, 1])
        axes[2].set_title('Gamma Vector')
        plt.tight_layout()
        plt.show()

    optimal_angle_inds, vcl_value = compute_opt_angle_subset(R, gamma, angle_candidates, num_selected_views, r_2, prev_selected_view_inds, seed=seed)

    if priority_order:
        optimal_angle_inds = reorder_by_priority(optimal_angle_inds, prev_selected_view_inds, R, gamma)

    return optimal_angle_inds, vcl_value



def compute_view_basis_functions(ct_model, ref_object, r_1, data_store_dir, seed=None, roi=None):
    """
    This function computes the view basis functions and saves them as .npy files.  It also
    computes and returns the inner product vector gamma that the VCL search uses.

    Args:
        ct_model (TomographyModel): The CT model with the system geometry.
        ref_object (ndarray): The 3D reference object, with shape (rows, cols, slices).
        r_1 (float): The fraction of the voxels of the reference object that are sampled.
        data_store_dir (str): The directory where the function saves the basis functions.
        seed (int, optional): The random seed for reproducible results.  Defaults to None.
        roi (ndarray, optional): A binary 3D mask.  It selects the reference voxels used in
            the inner products.  Defaults to None.

    Returns:
        ndarray: The gamma column vector, with shape (num_views, 1).

    Example:
        >>> gamma = compute_view_basis_functions(ct_model, ref_object, 0.001, "/tmp/recons")
    """
    eps = 1e-12

    print('Creating sinogram for reference object of shape {}'.format(ref_object.shape))
    ref_sino = ct_model.forward_project(ref_object)

    if roi is None:
        mask = mt.get_2d_ror_mask(ref_object[:, :, 0].shape)
        sparse_indices, row_col_indices = get_2d_subsampling_indices(np.asarray(mask), r_1, seed=seed)
        ref_object_flat = ref_object.reshape(ref_object.shape[0] * ref_object.shape[1], ref_object.shape[2])
        sparse_ref_object = _normalize_by_norm(np.asarray(ref_object_flat[sparse_indices, :]), eps)
    else:
        # A (row, col) location is in the ROI if any slice has an ROI voxel there.
        mask = np.any(roi, axis=2)
        sparse_indices, row_col_indices = get_2d_subsampling_indices(mask, r_1, seed=seed)
        ref_object_flat = ref_object.reshape(ref_object.shape[0] * ref_object.shape[1], ref_object.shape[2])
        sampled_ref_object = ref_object_flat[sparse_indices, :]
        roi_flat = roi.reshape(roi.shape[0] * roi.shape[1], roi.shape[2])
        sampled_roi_mask = roi_flat[sparse_indices, :]
        sparse_ref_object = _normalize_by_norm(
            np.asarray(sampled_ref_object[sampled_roi_mask]).reshape(-1, 1), eps
        )

    num_views = ct_model.get_params('sinogram_shape')[0]

    print('Creating recon bases on the {} device'.format(
        'default' if ct_model.device_layout_is_automatic else 'requested'))

    filtered_sinogram = np.asarray(ct_model.direct_filter(ref_sino))
    del ref_sino

    single_view_model = _make_single_view_sibling(ct_model)
    view_params_name = ct_model.get_params('view_params_name')
    full_view_params = np.asarray(ct_model.get_params(view_params_name))

    gamma = np.zeros((num_views, 1))
    for i in tqdm.trange(num_views, desc='Computing and storing view basis functions'):
        view_sino = filtered_sinogram[i:i + 1]
        _set_sibling_view_params(single_view_model, full_view_params[i:i + 1])
        recon_i = single_view_model.sparse_back_project(view_sino, sparse_indices)
        recon_i = recon_i.detach().cpu().numpy() if hasattr(recon_i, 'detach') else np.asarray(recon_i)
        if roi is not None:
            recon_i = recon_i[sampled_roi_mask].reshape(-1, 1)

        recon_i, gamma_i = _normalize_and_project(recon_i, sparse_ref_object, eps)

        with open(os.path.join(data_store_dir, f'view_basis_function{i}.npy'), 'wb') as f:
            np.save(f, recon_i)

        gamma[i, 0] = gamma_i
    return gamma


def compute_cov_matrix(num_views, data_store_dir, batch_size=100):
    """
    This function computes the covariance matrix of the view basis functions.  It loads the
    basis functions from the ``.npy`` files that :func:`compute_view_basis_functions` saves.

    Args:
        num_views (int): The number of view basis functions.
        data_store_dir (str): The directory that contains the ``.npy`` files.
        batch_size (int, optional): The number of views loaded at a time.  Defaults to 100.

    Returns:
        ndarray: The symmetric covariance matrix, with shape ``(num_views, num_views)``.

    Example:
        >>> cov_matrix = compute_cov_matrix(180, "/tmp/recons")
    """

    cov_matrix = np.zeros((num_views, num_views))
    num_batches = int(np.ceil(num_views / batch_size))
    batches = np.array_split(np.arange(num_views), num_batches)
    recon0 = np.load(os.path.join(data_store_dir, f'view_basis_function{0}.npy'))
    recon_size = recon0.size

    for batch_index, batch in enumerate(tqdm.tqdm(batches, desc='Computing covariance matrix')):
        recons_batch = np.zeros((len(batch), recon_size))
        for i in batch:
            recons_batch[i - batch[0]] = np.load(os.path.join(data_store_dir, f'view_basis_function{i}.npy')).flatten()
        batch_start, batch_stop = batch[0], batch[0] + len(batch)
        dot_products = recons_batch @ recons_batch.T
        cov_matrix[batch_start:batch_stop, batch_start:batch_stop] = dot_products

        for batch2_index, batch2 in enumerate(batches[batch_index+1:]):
            recons_batch2 = np.zeros((len(batch2), recon_size))
            for j in batch2:
                recons_batch2[j - batch2[0]] = np.load(
                    os.path.join(data_store_dir, f'view_basis_function{j}.npy')).flatten()
            batch2_start, batch2_stop = batch2[0], batch2[0] + len(batch2)
            dot_products = recons_batch @ recons_batch2.T

            # The matrix is symmetric, so the block and its transpose are both written.
            cov_matrix[batch_start:batch_stop, batch2_start:batch2_stop] = dot_products
            cov_matrix[batch2_start:batch2_stop, batch_start:batch_stop] = dot_products.T


    return cov_matrix


def compute_vcl(sub_R, sub_gamma):
    """
    This function computes the View Covariance Loss, VCL = 1 - γᵀ R⁻¹ γ, for a subset of
    views.  A lower VCL means that the subset is more informative and less redundant.

    Args:
        sub_R (ndarray): The (K, K) covariance matrix of a subset of K views.
        sub_gamma (ndarray): The (K, 1) inner products between the view basis functions and
            the reference object.

    Returns:
        float: The VCL value for the subset of views.
    """
    loss_value = 1 - sub_gamma.T @ np.linalg.solve(sub_R, sub_gamma)
    return loss_value.item()


def compute_opt_angle_subset(R, gamma, candidate_angles, K, r_2, prev_selected_view_inds, search_min=30, max_iterations = 100, seed=None):
    """
    This function selects the subset of view angles that minimizes the VCL of
    :func:`compute_vcl`.  It uses a stochastic greedy search.  The search replaces one selected
    angle at a time and keeps each replacement that lowers the VCL.

    Args:
        R (ndarray): The covariance matrix, with shape (num_views, num_views).
        gamma (ndarray): The column vector of inner products between the view basis functions
            and the reference object, with shape (num_views, 1).
        candidate_angles (ndarray): The (num_views,) view angles that correspond to R and gamma.
        K (int): The number of view angles to select.
        r_2 (float): The fraction of unselected candidates that the search tries for each
            selected view in each iteration.
        prev_selected_view_inds (ndarray): The indices in ``candidate_angles`` of the view
            angles selected earlier.
        search_min (int, optional): The minimum number of candidates tried for each selected
            view in each iteration.  Defaults to 30.
        max_iterations (int, optional): The maximum number of iterations.  Defaults to 100.
        seed (int, optional): The random seed for reproducible results.  Defaults to None.

    Returns:
        Tuple[ndarray, float]: The sorted (K,) indices in ``candidate_angles`` of the selected
        views, and the VCL of the subset.

    Example:
        >>> selected = compute_opt_angle_subset(R, gamma, candidate_angles, K=10, r_2=0.01)

    Raises:
        ValueError: If K <= 0.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    candidate_angles_inds = np.arange(len(candidate_angles))
    avail_angle_inds = np.setdiff1d(candidate_angles_inds, prev_selected_view_inds, assume_unique=False)

    num_avail_angle_inds = len(avail_angle_inds)
    num_unselected_angles = num_avail_angle_inds - K

    if K <= 0:
        raise ValueError("K must be positive. Received K={}".format(K))

    if num_unselected_angles <= 0:
        warnings.warn(f"Requested {K} views, but only {num_avail_angle_inds} available. Returning all candidates.")
        sorted_angle_inds = avail_angle_inds
        return sorted_angle_inds, float(compute_vcl(*subsample_R_gamma(R, gamma, sorted_angle_inds)))

    num_search_candidates = np.minimum(np.maximum(int(r_2 * num_unselected_angles), search_min), num_unselected_angles)

    # The search starts from uniformly spaced angles.
    pos = np.linspace(0, len(avail_angle_inds), K, endpoint=False).astype(int)
    selected_angle_inds = avail_angle_inds[pos].astype(int)

    combined_selected_angle_inds = np.concatenate((prev_selected_view_inds, selected_angle_inds))
    R_chosen, gamma_chosen = subsample_R_gamma(R, gamma, combined_selected_angle_inds)
    vcl_current_best = compute_vcl(R_chosen, gamma_chosen)

    for i in range(max_iterations):
        prev_selected_angle_inds = np.copy(selected_angle_inds)
        for j in range(K):
            candidate_indices = np.setdiff1d(avail_angle_inds, selected_angle_inds, assume_unique=True).tolist()
            random.shuffle(candidate_indices)
            candidate_indices = candidate_indices[:num_search_candidates]

            for k in candidate_indices:
                selected_angle_inds_tmp = np.copy(selected_angle_inds)
                selected_angle_inds_tmp[j] = k
                combined_selected_angle_inds_tmp = np.concatenate((prev_selected_view_inds, selected_angle_inds_tmp))
                R_temp, gamma_temp = subsample_R_gamma(R, gamma, combined_selected_angle_inds_tmp)
                vcl_temp = compute_vcl(R_temp, gamma_temp)

                if vcl_temp < vcl_current_best:
                    vcl_current_best = np.copy(vcl_temp)
                    selected_angle_inds = np.copy(selected_angle_inds_tmp)

        if np.array_equal(selected_angle_inds, prev_selected_angle_inds):
            print(f'Early stopping at iteration {i}, no change in indices')
            break

    best_view_angle_inds = np.sort(selected_angle_inds)
    return best_view_angle_inds, float(vcl_current_best)


def get_2d_subsampling_indices(mask, r_1, seed=None, blue_noise=False):
    """
    This function samples voxel indices within a 2D mask.  It samples uniformly at random or
    with a stored blue noise pattern.  The blue noise pattern does not suit a ramp-filtered
    signal.

    Args:
        mask (ndarray): A 2D binary array that marks the region of interest (ROI).
        r_1 (float): The fraction of voxels to sample from the ROI.  It must be in (0, 1].
        seed (int, optional): The random seed for reproducible results with uniform sampling.
        blue_noise (bool, optional): If False, the function uses uniform random sampling.
            If True, it uses the blue noise pattern.  Defaults to False.

    Returns:
        Tuple:
            random_indices_2d (ndarray): The flattened 1D indices of the selected voxels.
            (row_inds, col_inds) (Tuple[ndarray, ndarray]): The row and column indices of the
                selected voxels.
    """
    import math

    if r_1 <= 0 or r_1 > 1:
        raise ValueError("r_1 must be in the range (0, 1].")

    num_rows, num_cols = mask.shape
    mask_flat = mask.ravel()
    num_total = np.sum(mask_flat)
    num_samples = min(int(num_total * r_1), int(num_total))

    if not blue_noise:
        if seed is not None:
            np.random.seed(seed)
        eligible_indices = np.where(mask_flat > 0)[0]
        flat_indices = np.random.choice(eligible_indices, size=num_samples, replace=False)
    else:
        bn_pattern = mt.bn256
        tile_rows = math.ceil(num_rows / bn_pattern.shape[0])
        tile_cols = math.ceil(num_cols / bn_pattern.shape[1])
        tiled_pattern = np.tile(bn_pattern, (tile_rows, tile_cols))
        tiled_pattern = tiled_pattern[:num_rows, :num_cols]

        # Values outside the mask are set to infinity.  The num_samples smallest values are sampled.
        masked_values = np.where(mask, tiled_pattern, np.inf)
        flat_indices = np.argsort(masked_values.ravel())[:num_samples]

    row_inds, col_inds = np.unravel_index(flat_indices, (num_rows, num_cols))
    random_indices_2d = row_inds * num_cols + col_inds

    return random_indices_2d, (row_inds, col_inds)


def show_image_with_projection_rays(
    image: np.ndarray,
    *,
    rotation_angles_deg: np.ndarray = None,
    rotation_angles_rad: np.ndarray = None,
    title: str = None
) -> None:
    """
    This function displays an image with one arrow for each rotation angle.  Each arrow points
    along the projection from the source to the detector.  With the detector at the top, the
    arrow for 0 degrees points from the bottom to the top.  The object rotates clockwise as the
    angle increases.  The caller must give exactly one of ``rotation_angles_deg`` and
    ``rotation_angles_rad``.

    Args:
        image (np.ndarray): The 2D image.
        rotation_angles_deg (np.ndarray, optional): A 1D array of angles in degrees.
        rotation_angles_rad (np.ndarray, optional): A 1D array of angles in radians.
        title (str, optional): The title to display above the plot.

    Returns:
        None
    """
    if image.ndim != 2:
        raise ValueError("Image must be a 2D array")
    if (rotation_angles_deg is None and rotation_angles_rad is None) or (rotation_angles_deg is not None and rotation_angles_rad is not None):
        raise ValueError("Exactly one of rotation_angles_deg or rotation_angles_rad must be None, and the other must be an array of floats")

    if rotation_angles_rad is None:
        rotation_angles_rad = np.deg2rad(rotation_angles_deg)

    rotation_angles_rad = np.pi / 2 + rotation_angles_rad

    rows, cols = image.shape
    center_x, center_y = cols / 2, rows / 2
    radius = min(rows, cols) / 2

    import matplotlib.pyplot as plt
    plt.imshow(image, cmap='gray', origin='upper', extent=[0, cols, rows, 0])
    plt.gca().set_aspect('equal')

    colors = plt.cm.tab10(np.arange(len(rotation_angles_rad)) % 10)

    for i, theta in enumerate(rotation_angles_rad):
        dx = 0.95*radius * np.cos(theta)
        dy = -0.95*radius * np.sin(theta)

        plt.arrow(center_x, center_y, dx, dy, color=colors[i], linewidth=1.75,
                  head_width=min(rows, cols) * 0.02, length_includes_head=True)

    plt.title(title or "Image with Overlaid Angles")
    plt.axis('off')
    plt.show()


def reorder_by_priority(optimal_angle_inds, prev_selected_view_inds, R, gamma):
    """
    This function reorders the selected angle indices from most to least important.  At each
    step it removes the angle whose removal gives the lowest VCL.  The angles removed last are
    the most important.

    Args:
        optimal_angle_inds (np.ndarray): The selected angle indices to reorder.
        prev_selected_view_inds (np.ndarray): The angle indices selected earlier.  These
            indices are kept in every subset.
        R (np.ndarray): The full covariance matrix.
        gamma (np.ndarray): The full gamma vector.

    Returns:
        np.ndarray: The reordered array of angle indices.
    """
    drop_inds_list = []
    while len(optimal_angle_inds) > 1:
        vcl_list = []
        for i in range(len(optimal_angle_inds)):
            temp_inds = np.delete(optimal_angle_inds, i)
            combined_selected_inds = np.concatenate((prev_selected_view_inds, temp_inds))
            R_temp, gamma_temp = subsample_R_gamma(R, gamma, combined_selected_inds)
            vcl_temp = compute_vcl(R_temp, gamma_temp)
            vcl_list.append(vcl_temp)

        idx_to_drop = np.argmin(vcl_list)
        drop_inds_list.append(optimal_angle_inds[idx_to_drop])
        optimal_angle_inds = np.delete(optimal_angle_inds, idx_to_drop)

    drop_inds_list.append(optimal_angle_inds[0])
    # The reversed removal order ranks the angles from most to least important.
    optimal_angle_inds = np.array(drop_inds_list)[::-1]

    return optimal_angle_inds

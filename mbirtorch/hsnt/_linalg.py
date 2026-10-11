import math

import torch


def _randomized_svd(X, n_components, n_oversamples=10, n_iter=4, seed=0):
    """This function computes a truncated SVD of ``X`` by a randomized method.  The random
    numbers come from a generator seeded with ``seed``, so the result is repeatable."""
    n_rows, n_cols = X.shape
    rank = min(n_components + n_oversamples, n_rows, n_cols)
    generator = torch.Generator(device=X.device).manual_seed(seed)
    Q, _ = torch.linalg.qr(X @ torch.randn(n_cols, rank, generator=generator,
                                           dtype=X.dtype, device=X.device))
    for _ in range(n_iter):                      # power iterations improve the estimate of the range
        Q, _ = torch.linalg.qr(X.T @ Q)
        Q, _ = torch.linalg.qr(X @ Q)
    U, singular_values, Vh = torch.linalg.svd(Q.T @ X, full_matrices=False)
    return Q @ U, singular_values, Vh


def nndsvda(X, n_components):
    """This function computes the NNDSVD initial values of ``W`` and ``H`` for the
    factorization X ~= W @ H.

    The NNDSVD factors contain zero entries.  This function replaces each zero entry with
    sqrt(mean X), which is the typical size of a factor entry.

    Args:
        X: A nonnegative array of shape (n_samples, n_features).
        n_components: The rank of the factorization.

    Returns:
        W: An array of shape (n_samples, n_components).
        H: An array of shape (n_components, n_features).
    """
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if n_components + 10 < min(X.shape):
        U, singular_values, Vh = _randomized_svd(X, n_components)
    else:
        U, singular_values, Vh = torch.linalg.svd(X, full_matrices=False)

    n_components = min(
        n_components,
        U.shape[1],
        Vh.shape[0],
    )

    W = torch.zeros((X.shape[0], n_components), dtype=X.dtype, device=X.device)
    H = torch.zeros((n_components, X.shape[1]), dtype=X.dtype, device=X.device)

    # First singular triplet.
    scale = torch.sqrt(singular_values[0])
    W[:, 0] = scale * torch.abs(U[:, 0])
    H[0, :] = scale * torch.abs(Vh[0, :])

    # Remaining components.
    for component in range(1, n_components):
        u = U[:, component]
        v = Vh[component, :]

        u_pos = torch.clip(u, min=0)
        u_neg = torch.clip(-u, min=0)
        v_pos = torch.clip(v, min=0)
        v_neg = torch.clip(-v, min=0)

        positive_strength = (
            torch.linalg.norm(u_pos) * torch.linalg.norm(v_pos)
        )
        negative_strength = (
            torch.linalg.norm(u_neg) * torch.linalg.norm(v_neg)
        )

        use_positive = positive_strength > negative_strength

        selected_u = torch.where(use_positive, u_pos, u_neg)
        selected_v = torch.where(use_positive, v_pos, v_neg)

        selected_u /= torch.linalg.norm(selected_u) + torch.finfo(X.dtype).eps
        selected_v /= torch.linalg.norm(selected_v) + torch.finfo(X.dtype).eps

        scale = torch.sqrt(singular_values[component])
        W[:, component] = scale * selected_u
        H[component, :] = scale * selected_v

    fill_value = torch.sqrt(torch.mean(X).clamp_min(0))
    W = torch.where(W == 0, fill_value, W)
    H = torch.where(H == 0, fill_value, H)

    return W, H


def _batched_spd_solve(M, g, jitter_rel=1e-9):
    """This function solves M[b] d[b] = g[b] for a batch of small symmetric positive definite
    matrices.  It adds a small multiple of each diagonal entry to that entry before it solves."""
    rank = M.shape[-1]
    eye = torch.eye(rank, dtype=M.dtype, device=M.device)
    s = torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(torch.finfo(M.dtype).tiny).sqrt()
    A = M / (s[:, :, None] * s[:, None, :]) + jitter_rel * eye
    L, info = torch.linalg.cholesky_ex(A)
    failed = (info > 0)[:, None, None]
    L = torch.where(failed, eye.expand_as(L), L)
    gs = g / s
    d = torch.cholesky_solve(gs.unsqueeze(-1), L).squeeze(-1)
    d = torch.where(failed[:, :, 0], gs / torch.diagonal(A, dim1=-2, dim2=-1), d) / s
    # A row with a zero diagonal entry makes the scaling overflow, so its step is set to zero.
    return torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def _joint_dot(a, b, c, d):
    return (a * b).sum() + (c * d).sum()


def _joint_blocks(flat, rows, cols, rank, free, jitter):
    """This function builds a batch of symmetric positive definite matrices of shape (B, R, R)
    from their upper triangles of shape (B, Q), and returns their Cholesky factors."""
    M = flat.new_zeros(flat.shape[0], rank, rank)
    M[:, rows, cols] = flat
    M[:, cols, rows] = flat
    eye = torch.eye(rank, dtype=M.dtype, device=M.device)
    M = torch.where(free[:, :, None] & free[:, None, :], M, eye.expand_as(M))
    # Each diagonal entry is increased by ``jitter`` times itself, so the components are scaled separately.
    M = M + torch.diag_embed(jitter * torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(torch.finfo(M.dtype).tiny))
    L, info = torch.linalg.cholesky_ex(M)
    return torch.where((info > 0)[:, None, None], eye.expand_as(L), L)


def _reseed_dead(W, H, T=None, seed=0):
    """This function replaces each component whose map or spectrum is exactly zero with small
    random values in both ``W`` and ``H``.  It returns ``(W, H, n)``, where ``n`` is the number
    of components replaced."""
    dead = (W.amax(0) <= 0) | (H.amax(1) <= 0)
    n_dead = int(dead.sum())
    if n_dead == 0:
        return W, H, 0
    live = ~dead
    W = W.clone()
    H = H.clone()
    scale = _attenuation_scale(T, W.shape[1]) if T is not None else 0.0
    if not scale > 0:
        scale = (float((W[:, live].mean(0).double() * H[live].mean(1).double()).sqrt().median())
                 if bool(live.any()) else 1.0)
    g = torch.Generator(device=W.device).manual_seed(int(seed))
    W[:, dead] = 1e-2 * scale * torch.rand(W.shape[0], n_dead, generator=g, dtype=W.dtype, device=W.device)
    H[dead] = 1e-2 * scale * torch.rand(n_dead, H.shape[1], generator=g, dtype=H.dtype, device=H.device)
    return W, H, n_dead


def _attenuation_scale(T, rank, chunk=2 ** 23):
    """This function returns sqrt(-log(mean T) / rank), which is the typical size of one factor
    entry.  It returns 0 when the mean transmission is not between 0 and 1."""
    rows = max(1, chunk // max(T.shape[1], 1))
    total = torch.zeros((), dtype=torch.float64, device=T.device)
    for i in range(0, T.shape[0], rows):
        total = total + T[i:i + rows].sum(dtype=torch.float64)
    mean_T = float(total) / max(T.numel(), 1)
    if not 0 < mean_T < 1:
        return 0.0
    return (-math.log(mean_T) / max(rank, 1)) ** 0.5


def _attenuation_for_start(T):
    """This function returns -log T.  Before it takes the log, it sets each transmission below
    1e-12 to half of the smallest transmission above 1e-12."""
    real = T > 1e-12
    if bool(real.any()) and not bool(real.all()):
        T = torch.where(real, T, 0.5 * T[real].min())
    else:
        T = T.clamp_min(torch.finfo(T.dtype).tiny)
    return -torch.log(T)


def _nonneg_least_squares_start(A, H, ridge=1e-6):
    """This function returns a nonnegative ``W`` that approximately minimizes ||A - W H||.  It
    solves the ridge regularized normal equations by Cholesky factorization, and then it clips
    negative entries to zero."""
    Hd = H.double()
    G = Hd @ Hd.T
    diag = torch.diagonal(G)
    floor = 1e-12 * diag.mean().clamp_min(torch.finfo(torch.float64).tiny)     # keeps G definite if a row of H is zero
    G = G + ridge * torch.diag_embed(torch.maximum(diag, floor))
    W = torch.cholesky_solve((A.double() @ Hd.T).T, torch.linalg.cholesky(G)).T
    return W.clamp_(min=0).to(A.dtype)

"""From-scratch implicit-feedback ALS (Hu, Koren & Volinsky, 2008).

Every user gets a factor vector x_u, every item y_i. Observed positives get
preference p_ui = 1 (everything else p_ui = 0), each pair weighted by a
confidence c_ui = 1 + alpha * r_ui (r_ui = 0 when unobserved, so unobserved
pairs still participate at low confidence c_ui = 1).

Naive per-user solves cost O(n_items * f^2); this uses the standard
factorization trick instead: C^u = I + (C^u - I), and (C^u - I) is non-zero
only on the items the user actually touched, so

    Y^T C^u Y = Y^T Y (shared by all users, computed once per sweep)
                + Y^T (C^u - I) Y (only over the n_u touched items)

giving O(f^2 * N + f^3 * U) per sweep for N total interactions, U users. Each
f-by-f system is symmetric positive definite for lambda > 0, solved via
Cholesky.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve


def _solve_factors(
    fixed: np.ndarray, confidence_mat: sp.csr_matrix, reg: float,
) -> np.ndarray:
    """Solve for one side's factors given the other side fixed.

    fixed: (n_fixed, f) factor matrix (e.g. item factors Y when solving for users).
    confidence_mat: (n_solve, n_fixed) sparse matrix where confidence_mat[u, i] = c_ui
        for observed (u, i) pairs and 0 for unobserved pairs.
    """
    n_solve, n_fixed = confidence_mat.shape
    f = fixed.shape[1]
    FtF = fixed.T @ fixed  # shared across all rows this sweep
    reg_I = reg * np.eye(f)

    out = np.zeros((n_solve, f))
    csr = confidence_mat.tocsr()
    for u in range(n_solve):
        start, end = csr.indptr[u], csr.indptr[u + 1]
        if start == end:
            # No observed interactions: confidence is 1 everywhere, factor stays at
            # the regularised zero solution (no personalised signal to fit).
            out[u] = 0.0
            continue
        idx = csr.indices[start:end]
        c = csr.data[start:end]  # confidence values c_ui for this user's touched items
        Yi = fixed[idx]                      # (n_u, f)
        A = FtF + Yi.T @ ((c - 1.0)[:, None] * Yi) + reg_I
        b = Yi.T @ c
        chol = cho_factor(A, lower=True, check_finite=False)
        out[u] = cho_solve(chol, b, check_finite=False)
    return out


class ImplicitALS:
    def __init__(
        self, factors: int = 32, reg: float = 0.1, alpha: float = 40.0,
        iterations: int = 15, random_state: int = 42,
    ):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state
        self.X: np.ndarray | None = None  # user factors
        self.Y: np.ndarray | None = None  # item factors

    def fit(self, interactions: sp.csr_matrix) -> "ImplicitALS":
        """interactions: sparse (n_users x n_items) matrix of raw ratings at
        observed positive interactions (0 elsewhere)."""
        n_users, n_items = interactions.shape
        rng = np.random.default_rng(self.random_state)
        self.X = rng.normal(scale=0.01, size=(n_users, self.factors))
        self.Y = rng.normal(scale=0.01, size=(n_items, self.factors))

        confidence = interactions.copy().astype(np.float64)
        confidence.data = 1.0 + self.alpha * confidence.data
        confidence_t = confidence.T.tocsr()

        for _ in range(self.iterations):
            self.X = _solve_factors(self.Y, confidence, self.reg)
            self.Y = _solve_factors(self.X, confidence_t, self.reg)
        return self

    def score(self, user_idx: int) -> np.ndarray:
        return self.X[user_idx] @ self.Y.T

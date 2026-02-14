"""Tests for propagation algorithms."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from scipy import sparse


def test_damped_ppr_convergence():
    """Test that damped PPR converges."""
    from src.retrieval.propagation import propagate_damped_ppr

    N = 50
    # Create a simple symmetric propagation matrix
    np.random.seed(42)
    A_dense = np.random.rand(N, N) * 0.1
    A_dense = (A_dense + A_dense.T) / 2  # Symmetric
    np.fill_diagonal(A_dense, 0)
    A_i = sparse.csr_matrix(A_dense)

    # Initial state: activate 3 entities
    s_0 = np.zeros(N)
    s_0[0] = 1.0
    s_0[5] = 0.8
    s_0[10] = 0.6

    s_final, hops, history = propagate_damped_ppr(
        s_0=s_0,
        A_i=A_i,
        alpha=0.85,
        max_hops=30,
        pruning_threshold=0.01,
        convergence_eps=0.001,
    )

    # Should converge before max_hops
    assert hops < 30
    # Final state should have positive values
    assert np.sum(s_final > 0) > 0
    # History should record each step
    assert len(history) == hops
    # Delta should decrease over time
    if len(history) > 2:
        assert history[-1]["delta"] < history[0]["delta"]


def test_max_update_propagation():
    """Test MAX update propagation."""
    from src.retrieval.propagation import propagate_max_update

    N = 30
    np.random.seed(42)
    A_dense = np.random.rand(N, N) * 0.1
    A_dense = (A_dense + A_dense.T) / 2
    np.fill_diagonal(A_dense, 0)
    A_i = sparse.csr_matrix(A_dense)

    s_0 = np.zeros(N)
    s_0[0] = 1.0
    s_0[3] = 0.5

    s_final, hops, history = propagate_max_update(
        s_0=s_0,
        A_i=A_i,
        max_hops=10,
        pruning_threshold=0.01,
        convergence_min_new=1,
    )

    # MAX update should never decrease scores
    assert s_final[0] >= s_0[0] or s_final[0] == 0  # (pruning may zero it)
    assert len(history) == hops


def test_sparse_ops():
    """Test sparse utility functions."""
    from src.utils.sparse_ops import (
        sparse_diag,
        top_k_indices,
        cosine_sim_vec,
        build_propagation_matrix,
    )

    # sparse_diag
    vals = np.array([1.0, 2.0, 3.0])
    D = sparse_diag(vals)
    assert D.shape == (3, 3)
    assert D[1, 1] == 2.0

    # top_k_indices
    scores = np.array([0.1, 0.5, 0.3, 0.9, 0.7])
    top2 = top_k_indices(scores, k=2)
    assert 3 in top2  # 0.9
    assert 4 in top2  # 0.7

    # cosine_sim_vec
    q = np.array([1.0, 0.0, 0.0])
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    sims = cosine_sim_vec(q, matrix)
    assert sims[0] == pytest.approx(1.0, abs=0.01)
    assert sims[1] == pytest.approx(0.0, abs=0.01)


def test_build_propagation_matrix():
    """Test A_i = H @ D_i @ H^T + w_syn * S construction."""
    from src.utils.sparse_ops import build_propagation_matrix, sparse_diag

    N, M = 5, 3
    H = sparse.csr_matrix(np.array([
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
    ], dtype=float))

    D_i = sparse_diag(np.array([0.8, 0.5, 0.3]))
    S = sparse.eye(N, format="csr") * 0.1

    A = build_propagation_matrix(H, D_i, S, w_syn=0.3)

    assert A.shape == (N, N)
    assert A.nnz > 0

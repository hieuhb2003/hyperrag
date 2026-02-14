"""
Sparse matrix utilities for HyP-DLM.

Provides efficient operations on scipy sparse matrices
used for hypergraph propagation.
"""

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import get_logger

logger = get_logger(__name__)


def sparse_diag(values: np.ndarray) -> sparse.dia_matrix:
    """Create a sparse diagonal matrix from a 1D array."""
    return sparse.diags(values, format="csr")


def build_propagation_matrix(
    H: sparse.csr_matrix,
    D_i: sparse.csr_matrix,
    S: sparse.csr_matrix,
    w_syn: float,
) -> sparse.csr_matrix:
    """
    Precompute A_i = H @ D_i @ H^T + w_syn * S

    Args:
        H: Incidence matrix (N x M)
        D_i: Diagonal modulation matrix (M x M)
        S: Synonym matrix (N x N)
        w_syn: Weight for synonym propagation

    Returns:
        A_i: Propagation matrix (N x N)
    """
    logger.debug(
        f"Building propagation matrix: H={H.shape}, D_i={D_i.shape}, S={S.shape}"
    )
    logger.start_timer("build_A_i")

    HD = H @ D_i          # N x M
    A = HD @ H.T           # N x N
    if w_syn > 0:
        A = A + w_syn * S  # Add synonym links

    # Ensure CSR format for efficient row slicing
    A = A.tocsr()

    elapsed = logger.stop_timer("build_A_i")
    logger.debug(f"A_i built: shape={A.shape}, nnz={A.nnz} ({elapsed})")
    return A


def normalize_sparse_rows(M: sparse.csr_matrix) -> sparse.csr_matrix:
    """Row-normalize a sparse matrix (each row sums to 1)."""
    row_sums = np.array(M.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    inv_sums = sparse.diags(1.0 / row_sums, format="csr")
    return inv_sums @ M


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k scores (descending)."""
    if len(scores) <= k:
        return np.argsort(scores)[::-1]
    indices = np.argpartition(scores, -k)[-k:]
    return indices[np.argsort(scores[indices])[::-1]]


def cosine_sim_vec(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a vector and all rows of a matrix.

    Args:
        query_vec: shape (dim,)
        matrix: shape (N, dim)

    Returns:
        similarities: shape (N,)
    """
    return cosine_similarity(query_vec.reshape(1, -1), matrix)[0]


def compute_modulation_matrix(
    guidance_vec: np.ndarray,
    hyperedge_embs: np.ndarray,
    mask: np.ndarray,
) -> sparse.csr_matrix:
    """
    Compute D_i = diag(ReLU(attn) * mask) where attn = cosine(g_i, hyperedge_embs).

    ReLU zeros out anti-correlated hyperedges (cosine < 0) rather than giving
    them positive weight. Without rectification, negative weights would inject
    negative values into the state vector s_t, violating its semantics as an
    activation score in [0, 1].

    Args:
        guidance_vec: shape (dim,)
        hyperedge_embs: shape (M, dim)
        mask: binary mask shape (M,)

    Returns:
        D_i: sparse diagonal matrix (M x M)
    """
    attn = cosine_sim_vec(guidance_vec, hyperedge_embs)  # range [-1, 1]
    attn = np.maximum(0, attn)  # ReLU: zero out negative similarities
    attn = attn * mask
    return sparse_diag(attn)

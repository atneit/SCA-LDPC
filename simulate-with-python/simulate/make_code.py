from random import randint
import numpy as np
from scipy.linalg import circulant
from . import utils


def fixed_weight_vec(size, samplings: int):
    curr_weight = 0
    # initialize with zeroes
    a = np.zeros(size, dtype=int)

    # Add random ones
    while curr_weight < samplings:
        i = randint(0, size - 1)
        if a[i] == 0:
            a[i] = 1
            curr_weight += 1
    return a


def flatten_matrix_parts(parts: np.ndarray):
    return np.concatenate(parts, axis=1)


def make_qc_parity_check_matrix(block_len: int, column_weight: int, num_blocks: int):
    """Constructs a parity check matrix H=[H_0 + H_i + I] where i is `num_blocks` and I is the identity matrix"""

    # construct the cyclic blocks
    parts = [
        circulant(fixed_weight_vec(block_len, column_weight))
        for _ in range(0, num_blocks)
    ]

    # Add identity block
    parts.append(np.identity(block_len, dtype=int))

    # Flatten into one big matrix
    return flatten_matrix_parts(parts)


def make_regular_ldpc_parity_check_matrix(k, r, column_weight, row_weight, seed=None):
    """
    Constructs a regular ldpc parity check matrix.

    The shape is [H_(r*k)|I_(r*r)] where k is the number of variable
    nodes and r is the number of check nodes.
    """
    assert k * column_weight == r * row_weight

    rng = utils.check_random_state(seed)

    n = k + r

    n_equations = (n * row_weight) // column_weight

    block = np.zeros((n_equations // row_weight, n), dtype=int)
    H = np.empty((n_equations, n))
    block_size = n_equations // row_weight

    # Filling the first block with consecutive ones in each row of the block

    for i in range(block_size):
        for j in range(i * column_weight, (i + 1) * column_weight):
            block[i, j] = 1
    H[:block_size] = block

    # reate remaining blocks by permutations of the first block's columns:
    for i in range(1, row_weight):
        H[i * block_size : (i + 1) * block_size] = rng.permutation(block.T).T
    H = H.astype(int)
    return H

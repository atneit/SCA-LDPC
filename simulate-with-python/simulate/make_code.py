from random import randint
import numpy as np
from scipy.linalg import circulant


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

from random import randint
import sys
import numpy as np
from scipy.linalg import circulant
from . import utils
from logzero import logger


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
    with np.printoptions(threshold=sys.maxsize):
        for part in parts:
            logger.debug("part: \n" + str(part))
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

    The shape is [H_(r*k)] where k is the number of variable
    nodes and r is the number of check nodes.

    Code is shamelessly copied (and adapted) from:
    https://hichamjanati.github.io/pyldpc/_modules/pyldpc/code.html
    """
    rng = utils.check_random_state(seed)

    if (
        column_weight <= 1
    ):  # d_v (number of parity check equations, per bit) column weight
        raise ValueError("""column_weight must be at least 2.""")

    if (
        row_weight < column_weight
    ):  # d_c (number of bits in the same parity check equation) row weight
        raise ValueError("""row_weight must be greater than or equal column_weight.""")

    if k % row_weight:
        raise ValueError("""row_weight must divide n for a regular LDPC matrix H.""")

    if r != (k * column_weight) // row_weight:
        raise ValueError(
            """r must follow (k * column_weight) // row_weight for the parity check matrix to be regular"""
        )

    with np.printoptions(threshold=sys.maxsize):

        H0_n = k

        block = np.zeros((r // column_weight, H0_n), dtype=int)
        logger.debug("block shape: \n" + str(block))
        H0 = np.zeros((r, H0_n))
        block_size = r // column_weight
        logger.debug("block_size: \n" + str(block_size))

        # Filling the first block with consecutive ones in each row of the block
        for i in range(block_size):
            for j in range(i * row_weight, (i + 1) * row_weight):
                block[i, j] = 1
        logger.debug("block 0: \n" + str(block))
        H0[:block_size] = block

        # create remaining blocks by permutations of the first block's columns:
        for i in range(1, column_weight):
            block_i = rng.permutation(block.T).T
            logger.debug(f"block {i}: \n" + str(block_i))
            H0[i * block_size : (i + 1) * block_size] = block_i
        H0 = H0.astype(int)

        return H0


def make_regular_ldpc_parity_check_matrix_identity(
    k, r, column_weight, row_weight, seed=None
):
    """
    Constructs a regular ldpc parity check matrix.

    The shape is [H_(r*k)|I_(r*r)] where k is the number of variable
    nodes and r is the number of check nodes.

    We add identity matrix at the end, so the resulting row wight will
    be one higher than specified.
    """

    return flatten_matrix_parts(
        [
            make_regular_ldpc_parity_check_matrix(
                k, r, column_weight, row_weight, seed
            ),
            np.identity(r, dtype=int),
        ]
    )

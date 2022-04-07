import numpy as np
from ldpc import bp_decoder
import logging

logger = logging.getLogger(__name__)


def simulate_frame_error_rate(
    H: np.ndarray, error_rate: float, runs: int, rng: np.random.RandomState
):
    """
    Simulates the frame error rate (FER) of the provided code.

    Example usage:
    >>> from ldpc.codes import rep_code
    >>> from . import utils
    >>> rng = utils.make_random_state(0)
    >>> n = 13
    >>> error_rate = 0.05
    >>> runs = 100
    >>> H = rep_code(n)
    >>> simulate_frame_error_rate(H, error_rate, runs, rng)
    100
    """
    n = H.shape[1]
    # BP decoder class. Make sure this is defined outside the loop
    bpd = bp_decoder(H, error_rate=error_rate, max_iter=n, bp_method="product_sum")
    error = np.zeros(n).astype(int)  # error vector

    successes = 0
    for _ in range(runs):
        for i in range(n):
            if rng.rand() < error_rate:
                error[i] = 1
            else:
                error[i] = 0
        syndrome = H @ error % 2  # calculates the error syndrome
        logger.debug(f"Error: \n{error}")
        logger.debug(f"Syndrome: \n{syndrome}")
        decoding = bpd.decode(syndrome)
        logger.debug(f"Decoding: \n{decoding}")
        cmp = decoding == error
        cmp = cmp.all()
        successes += int(cmp)

    return successes


def simulate_frame_error_rate_rust(
    H: np.ndarray, error_rate: float, runs: int, rng: np.random.RandomState
):
    from simulate_rs import bp_decode

    n = H.shape[1]
    error = np.zeros(n).astype(int)  # error vector

    successes = 0
    for _ in range(runs):
        for i in range(n):
            if rng.rand() < error_rate:
                error[i] = 1
            else:
                error[i] = 0
        syndrome = H @ error % 2  # calculates the error syndrome
        logger.debug(f"Error: \n{error}")
        logger.debug(f"Syndrome: \n{syndrome}")
        decoding = bp_decode(H, np.array([error_rate] * n), n, syndrome)
        logger.debug(f"Decoding: \n{decoding}")
        cmp = decoding == error
        cmp = cmp.all()
        successes += int(cmp)

    return successes

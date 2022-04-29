import numpy as np
from ldpc import bp_decoder
import logging
import itertools

logger = logging.getLogger(__name__)


class ErrorsProvider:
    """
    Class for generating errors according to specified distribution.

    If file is not provided, each position is 1 with probability error_rate, otherwise 0.
    File allows to manage other distributions. Each line corresponds to separate position,
    where each line should contain n probabilities, divided by spaces and/or commas, that
    add up to 1, where n is assumed to be odd. If n = 1, binary case is assumed, otherwise
    generated errors lies in [-n/2, ..., n/2].
    """

    def __init__(self, error_rate, error_file, rng):
        self.error_rate = error_rate
        self.error_distribution = None
        self.rng = rng
        import re

        if error_file is not None:
            error_distribution = []
            with open(error_file, "rt") as f:
                for line in f:
                    line = line.strip()
                    pr = re.split("[, ]+", line)
                    pr = list(float(x) for x in pr)
                    error_distribution.append(pr)
            self.error_distribution = error_distribution

    def __get_binary_error(self, threshold):
        if self.rng.rand() < threshold:
            return 1
        else:
            return 0

    def get_error(self, pos):
        """
        Get error based on the distribution. Position pos is taken modulo length of file
        (if it was provided, otherwise pos is ignored).

        Usage of the same error rate:
        >>> from . import utils
        >>> EPS = 0.009
        >>> N = 10000
        >>> rng = utils.make_random_state(0)
        >>> error_rate = 0.05
        >>> errors_provider = ErrorsProvider(error_rate, None, rng)
        >>> s = 0
        >>> for i in range(N):
        ...     s += errors_provider.get_error(0)
        >>> abs(s/N - error_rate) < EPS
        True

        Usage of binary distributions:
        >>> errors_provider = ErrorsProvider(0.05, 'simulate-with-python/binary_distr.txt', rng)
        >>> expected = [0.1, 0.3, 0.05, 0.14]
        >>> all_correct = True
        >>> for i, expect in enumerate(expected):
        ...     s = 0
        ...     for _ in range(N):
        ...         s += errors_provider.get_error(i)
        ...     all_correct = all_correct and (abs(s/N - expected[i]) < EPS)
        >>> all_correct
        True

        Usage of q-ary distributions:
        >>> errors_provider = ErrorsProvider(0.05, 'simulate-with-python/qary_distr.txt', rng)
        >>> expected = [{-1: 0.2, 0: 0.5, 1: 0.3}, {-1: 0.1, 0: 0.6, 1: 0.3}]
        >>> all_correct = True
        >>> from collections import defaultdict
        >>> for i, expect in enumerate(expected):
        ...     s = defaultdict(int)
        ...     for _ in range(N):
        ...         e = errors_provider.get_error(i)
        ...         s[e] += 1
        ...     for val, prob in expect.items():
        ...         cond = abs(s[val]/N - prob) < EPS
        ...         all_correct = all_correct and cond
        >>> all_correct
        True
        """
        if self.error_distribution is None:
            return self.__get_binary_error(self.error_rate)
        l = len(self.error_distribution)
        pr = self.error_distribution[pos % l]
        pr_len = len(pr)
        if pr_len == 1:
            return self.__get_binary_error(pr[0])
        else:
            rand = self.rng.rand()
            res = -(pr_len // 2)
            threshold = 0
            for p in pr:
                threshold += p
                if threshold > rand:
                    return res
                res += 1

    def get_error_rate(self):
        if self.error_distribution is None:
            return self.error_rate
        return None

    def get_binary_channel_probs(self, n=None):
        """
        Get error distribution for each position. If parameter n is not provided,
        function return array of distributions from the file. If n is present, function
        return array of length n, where array is obtained by concatenating distribution
        array from the file with itself until length n is reached
        """
        if self.error_distribution is None:
            return [None]
        if len(self.error_distribution[0]) != 1:
            raise ValueError("Distribution from the file isn't binary")
        if n is None:
            return list(x[0] for x in self.error_distribution)
        pr = [0] * n
        iter = itertools.cycle(self.error_distribution)
        for i in range(n):
            pr[i] = next(iter)[0]
        return pr


def simulate_frame_error_rate(
    H: np.ndarray,
    errors_provider: ErrorsProvider,
    runs: int,
    rng: np.random.RandomState,
):
    """
    Simulates the frame error rate (FER) of the provided code.

    Example usage:
    >>> from ldpc.codes import rep_code
    >>> from . import utils
    >>> rng = utils.make_random_state(0)
    >>> n = 13
    >>> error_rate = 0.05
    >>> errors_provider = ErrorsProvider(error_rate, None, rng)
    >>> runs = 100
    >>> H = rep_code(n)
    >>> simulate_frame_error_rate(H, errors_provider, runs, rng)
    100
    """
    n = H.shape[1]
    # BP decoder class. Make sure this is defined outside the loop
    error_rate = errors_provider.get_error_rate()
    channel_probs = errors_provider.get_binary_channel_probs(n)
    bpd = bp_decoder(
        H,
        error_rate=error_rate,
        max_iter=n,
        bp_method="product_sum",
        channel_probs=channel_probs,
    )
    error = np.zeros(n).astype(int)  # error vector

    successes = 0
    for _ in range(runs):
        for i in range(n):
            error[i] = errors_provider.get_error(i)
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
    from simulate_rs import DecoderN450R150V7C3GF16

    n = H.shape[1]
    q = 16
    channel_output = np.zeros((n, q)).astype(float)  # error vector

    decoder = DecoderN450R150V7C3GF16(H.astype("uint8"))

    p_linear = 1 / q
    channel_prob = [p_linear for x in range(0, q)]
    good_variable = np.array(channel_prob)
    bad_variable = np.array(channel_prob)
    good_variable[[0, 1]] = np.array([2 * p_linear, 0])
    bad_variable[[-1, -2]] = np.array([2 * p_linear, 0])

    logger.debug(f"Good variable: \n{good_variable}")
    logger.debug(f"Bad variable: \n{bad_variable}")

    successes = 0
    run = 0
    max_errs_success = 0
    min_errs_fail = 99999999
    while run < runs:
        errs = 0
        for i in range(n):
            if rng.rand() < error_rate:
                channel_output[i, :] = bad_variable
                errs += 1
            else:
                channel_output[i, :] = good_variable
        if not errs:
            continue
        logger.debug(f"Channel output: \n{channel_output}")
        decoding = decoder.min_sum(channel_output)
        logger.debug(f"Decoding: \n{decoding}")
        if decoding == [0] * n:
            max_errs_success = max(errs, max_errs_success)
            successes += 1
        else:
            min_errs_fail = min(errs, min_errs_fail)
        run += 1

    logger.info(f"maximum symbols corrected: {max_errs_success}")
    logger.info(f"minimum symbols failed to correct: {min_errs_fail}")

    return successes

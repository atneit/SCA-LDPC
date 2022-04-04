import numpy as np
from ldpc import bp_decoder
from logzero import logger


def simulate_frame_error_rate(
    H: np.ndarray, error_rate: float, runs: int, rng: np.random.RandomState
):
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

class ErrorsProvider:
    """
    Class for generating errors according to specified distribution.

    If file is not provided, each position is 1 with probability error_rate, otherwise 0.
    File allows to manage other distributions. Each line corresponds to separate position,
    where each line should contain n probabilities, divided by spaces and/or commas, that
    add up to 1, where n is assumed to be odd. If n = 1, binary case is assumed, otherwise
    generated errors lies in [-n/2, ..., n/2].
    """

    def __init__(self, error_rate, error_file, rnd):
        self.error_rate = error_rate
        self.error_distribution = None
        self.rnd = rnd
        import re
        if error_file is not None:
            error_distribution = []
            with open(error_file, 'rt') as f:
                for line if f:
                    line = line.strip()
                    pr = re.split('[, ]+', line)
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
        """
        if self.error_distribution is None:
            return __get_binary_error(self.error_rate)
        l = len(self.error_distribution)
        pr = self.error_distribution[pos % l]
        pr_len = len(pr)
        if pr_len == 1:
            return __get_binary_error(pr[0])
        else:
            rand = self.rng.rand()
            res = -(pr_len // 2)
            threshold = 0
            for p in pr:
                threshold += p
                if threshold > rand:
                    return res
                res += 1


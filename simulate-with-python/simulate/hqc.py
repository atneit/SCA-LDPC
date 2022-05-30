#!/bin/env python

# dirty fix for running file as executable
import sys
sys.path.append("../simulate")

import itertools
import numpy as np
from .make_code import make_random_ldpc_parity_check_matrix_with_identity
from .utils import make_random_state
from simulate_rs import Hqc128
import logging

logger = logging.getLogger(__name__)


def search_distinguishable_plaintext(HQC, rng: np.random.RandomState):
    ptlen = len(HQC.new_plaintext())
    logger.debug(f"plaintext length: {ptlen}")
    logger.info(
        "Starting search for plaintext that results in 3 additional seed expansions"
    )
    distr = {0: 0, 1: 0, 2: 0, 3: 0}
    for attempt in itertools.count():
        pt = rng.bytes(ptlen)
        rejects = HQC.num_rejections(pt)
        additionalseedexpansions = (rejects // 1000) - 3
        distr[additionalseedexpansions] += 1
        if additionalseedexpansions >= 3:
            logger.debug(f"Seedexpansion distribution: {distr}")
            logger.info(
                f"Found plaintext with good timing properties in attempt nmbr {attempt}"
            )
            return pt


def simulate_hqc_idealized_oracle(rng: np.random.RandomState):
    logger.info("Selecting Hqc128 as current HQC version!")
    HQC = Hqc128()
    N = HQC.params("N")

    # construct/load key-pair to crack
    logger.info(
        "Creating random HQC keypair (randomness does not depend on provided seed)!"
    )
    (pub, priv) = HQC.keypair()  # Randomness does not depend on rng
    (x, y) = HQC.secrets_from_key(priv)

    # generate / select plain text message
    pt = search_distinguishable_plaintext(HQC, rng)

    # Generate a parity check matrix (H)
    logger.info(f"Create random (L/M)DPC parity check of size {N}!")
    H = make_random_ldpc_parity_check_matrix_with_identity(N, 3, rng)

    # encapsulate with r_2 and e to all-zero. Set r_1 to h_0
    h0_len = len(H[0])//2
    h0 = H[0][0:h0_len]
    r1_sparse = [i for (i, x) in enumerate(h0) if x != 0]
    logger.debug(f"r1_sparse: {r1_sparse}")
    (ct , ss) = HQC.encaps_with_plaintext_and_r1(pub, pt, r1_sparse)
    logger.info(f"Encaps result: {len(ss)} // {len(ct)}")

    # Select outer blocks that: if flipped back results in decoding success.

    # for each such block find bits that: if flipped back, results in decoding success

    # miss-classification probabilities are derrived from "idealized-oracle" calls in CHES2022 paper.


def shift_and_add_mod_2_sparse(y, j, n):
    """
    This accepts a sparse vector (indexes of set bits) and adds 
    itself with itself but bit-shifted right with j steps, modulo 2.
    The bitshift is a rotation over a vector of size n.

    e.g. return y + ((y >> j) % n)

    >>> shift_and_add_mod_2_sparse([1, 5, 8, 12], 3, 15)
    [0, 1, 4, 5, 11, 12]
    """
    y_shift_j = [(yi + j)%n for yi in y]
    yyj = y + y_shift_j
    del y
    del y_shift_j
    yyj.sort()
    ret = []
    discarded = False
    for (y1, y2) in zip(yyj[:], yyj[1:]+[n+1]):
        if discarded:
            discarded = False
        elif y1 == y2:
            discarded = True
        else:
            ret.append(y1)
            discarded = False
    return ret

def test_hqc_encaps_with_plaintext_and_r1():
    """
    This is a unit test. 
    
    We use doctest for test discovery:
    >>> test_hqc_encaps_with_plaintext_and_r1()
    True
    """
    HQC = Hqc128()
    N = HQC.params("N")
    (pub, priv) = HQC.keypair()  # Randomness does not depend on rng
    rng = make_random_state(0)
    pt = search_distinguishable_plaintext(HQC, rng)
    for j in rng.choice(N, 100, replace=False):
        (_, y) = HQC.secrets_from_key(priv)
        yyj = shift_and_add_mod_2_sparse(y, j, N)
        (ct, _) = HQC.encaps_with_plaintext_and_r1(pub, pt, [0, j])
        eprime = HQC.eprime(ct, priv, pt)
        bits = np.unpackbits(eprime, bitorder='little')
        indices = [i for (i, x) in enumerate(bits) if x]

        for (yi, ei) in zip(yyj, indices):
            if yi != ei:
                return False

    return True

if __name__ == "__main__":
    test_hqc_encaps_with_plaintext_and_r1()
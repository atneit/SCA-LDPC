import itertools
import numpy as np
from .make_code import make_random_ldpc_parity_check_matrix_with_identity
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

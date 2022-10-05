import logging

logger = logging.getLogger(__name__)

from typing import Union
import numpy as np
from simulate_rs import Hqc128
from .hqc import (
    read_or_generate_keypair,
    search_distinguishable_plaintext,
    toggle_bits_in_v,
)


def modify_beyond_correction_limit(
    HQC,
    rng: np.random.RandomState,
    ct: Union[bytes, bytearray],
):
    BLOCK_SIZE = 384
    BLOCK_NUM = 46
    NOISE_PER_BLOCK = 288
    BLOCKS_WITH_NOISE = 16
    N = HQC.params("N")
    for block in rng.choice(BLOCK_NUM, BLOCKS_WITH_NOISE, replace=False):
        logger.debug(f"Flipping outer block {block}")
        block_bits = [
            bit + BLOCK_SIZE * block
            for bit in rng.choice(BLOCK_SIZE, NOISE_PER_BLOCK, replace=False)
        ]
        ct = toggle_bits_in_v(ct, block_bits, N)
    return ct

def oracle(HQC, ctymod, priv, measure):
    measurments = HQC.decode_oracle(ctymod, priv, measure)
    min_1p = sorted(measurments)[measure//100]
    return min_1p

def hqc_eval_oracle(rng: np.random.RandomState, keyfile=None):
    # alg selection
    HQC = Hqc128()
    NUM_PROFILING = 2**14
    NUM_TRIALS = 100
    MEASUREMENTS = [2**x for x in range(16)]

    # get keypair
    (pub, priv) = read_or_generate_keypair(HQC, keyfile)

    # generate / select plain text message
    pt = search_distinguishable_plaintext(HQC, rng)

    # Craft invalid ciphertext with zero noise
    (ctnmod, _) = HQC.encaps_with_plaintext_and_r1(pub, pt, [])
    ctymod = modify_beyond_correction_limit(HQC, rng, ctnmod)

    profiling_diff = 0
    
    logger.info(
        f"Doing {NUM_PROFILING} decapsulations for warmup"
    )
    oracle(HQC, ctnmod, priv, NUM_PROFILING)
    
    while profiling_diff <= 0:
        profile_time_nmod = None
        profile_time_ymod = None
        # 1st profile phase
        while not profile_time_nmod:
            logger.info(
                f"Doing {NUM_PROFILING} decapsulations for profiling step. Ciphertext modified: no"
            )
            profile_time_nmod = oracle(HQC, ctnmod, priv, NUM_PROFILING)
        logger.info(f"Profiling result (nmod): {profile_time_nmod}")

        # 2nd profile phase
        while not profile_time_ymod:
            logger.info(
                f"Doing {NUM_PROFILING} decapsulations for profiling step. Ciphertext modified: yes"
            )
            profile_time_ymod = oracle(HQC, ctymod, priv, NUM_PROFILING)
        logger.info(f"Profiling result (ymod): {profile_time_ymod}")
        profiling_diff = profile_time_nmod - profile_time_ymod
        logger.info(f"Profiling diff nmod - ymod: {profiling_diff}")

    profiling_threshold = profile_time_nmod - profiling_diff / 2
    logger.info(f"Profiling threshold: {profiling_threshold}")

    # measurement phase
    results = {}
    for measure in MEASUREMENTS:
        results[measure] = []
        sum = 0
        for trial in range(NUM_TRIALS):
            # add some random noise, enough to cause decoding failure
            ctymod = modify_beyond_correction_limit(HQC, rng, ctnmod)

            new = oracle(HQC, ctymod, priv, measure)
            if new:
                decision = new >= profiling_threshold
                expected = False
                correct = float(int(decision == expected))
                sum += correct
                results[measure].append(correct)
                logger.info(f"Oracle attempt {trial} with {measure} measurements outputs {correct}, cumulative: {sum/len(results[measure])}")


    

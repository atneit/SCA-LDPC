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

def profiling_oracle(HQC, rng, pub, priv, why, mod, num_profiles=2**2):
    MEASUREMENTS = 2**16
    # 1st profile phase
    while True:
        logger.info(
            f"Doing {MEASUREMENTS * num_profiles} decapsulations for {why}. Ciphertext modified: {mod}"
        )
        
        profs = []
        for _ in range(num_profiles):
            pt = search_distinguishable_plaintext(HQC, rng, target_additional_seedexpansions=2 if mod else 3)
            (ct, _) = HQC.encaps_with_plaintext_and_r1(pub, pt, [])
            t = oracle(HQC, ct, priv, MEASUREMENTS)
            if t:
                profs.append(t)
        
        if profs:
            return sum(profs)/len(profs)

def hqc_eval_oracle(rng: np.random.RandomState, keyfile=None):
    # alg selection
    HQC = Hqc128()
    NUM_TRIALS = 1000
    MEASUREMENTS = [2**x for x in range(18)]

    # get keypair
    (pub, priv) = read_or_generate_keypair(HQC, keyfile)


    profiling_diff = 0
    
    profiling_oracle(HQC, rng, pub, priv, "warmup", False, num_profiles=1)
    
    while profiling_diff <= 0:
        # 1st profile phase
        profile_time_nmod = profiling_oracle(HQC, rng, pub, priv, "profile phase 1", False)
        logger.info(f"Profiling result (nmod): {profile_time_nmod}")

        # 2nd profile phase
        profile_time_ymod = profiling_oracle(HQC, rng, pub, priv, "profile phase 2", True)
        logger.info(f"Profiling result (ymod): {profile_time_ymod}")

        profiling_diff = profile_time_nmod - profile_time_ymod
        logger.info(f"Profiling diff nmod - ymod: {profiling_diff}")

    profiling_threshold = profile_time_nmod - profiling_diff / 2
    logger.info(f"Profiling threshold: {profiling_threshold}")
    
    # generate / select plain text message
    pt = search_distinguishable_plaintext(HQC, rng)

    # Craft valid/invalid ciphertext with zero noise
    (ctnmod, _) = HQC.encaps_with_plaintext_and_r1(pub, pt, [])
    ctymod = modify_beyond_correction_limit(HQC, rng, ctnmod)

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
                logger.debug(f"Oracle attempt {trial} with {measure} measurements outputs {correct}, cumulative: {sum/len(results[measure])}")

        logger.info(f"Oracle with {measure} measurements outputs cumulative: {sum/len(results[measure])}")

    

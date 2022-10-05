#!/bin/env python

# Ugly hack to allow import from the current package
# if running current module as main. Please forgive the heresy.
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "simulate"

from collections import Counter
from math import isnan, prod
from random import random
from typing import Tuple, Union
import pickle
import itertools
import numpy as np
from .make_code import make_random_ldpc_parity_check_matrix
from .utils import make_random_state
from simulate_rs import Hqc128
from enum import Enum
import logging
from ldpc import bp_decoder
import pandas as pd

logger = logging.getLogger(__name__)

# Flipped or not
class FlipStatus(Enum):
    UNFLIPPED = 0
    FLIPPED = 1

    def toggled(self):
        raise Exception("don't use this!")
        if self == FlipStatus.UNFLIPPED:
            return FlipStatus.FLIPPED
        elif self == FlipStatus.FLIPPED:
            return FlipStatus.UNFLIPPED
        else:
            raise ValueError(f"Bad value for enum FlipStatus: {self}")


# code element status
class IfFlipResult(Enum):
    UNKNOWN = 0
    NOCHANGE = 1
    SUCCESS = 2
    FAILURE = 3


class NoMoreUntestedRmBlocks(Exception):
    pass


class SingletonAssertDecodingFailure(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SingletonAssertDecodingFailure, cls).__new__(cls)
            cls.instance.raise_exception = True
        return cls.instance

    def assert_decoding_success(self, expect, *args, **kwargs):
        kwargs["debug"] = True
        result = wrapped_hqc_decoding_oracle(
            *args, **kwargs, require_false=0.9999, require_true=0.9999
        )
        if self.raise_exception:
            assert result == expect
        elif result != expect:
            logger.warning(f'Failed assertion "decoding success = {expect}"')


def read_or_generate_keypair(HQC, filename=None):
    if filename:
        try:
            with open(filename, "rb") as file:
                key = pickle.load(file)
                logger.info(f"Loaded existing key from {filename}")
        except Exception as e:
            with open(filename, "wb") as file:
                logger.info(
                    f"Creating random HQC keypair in {filename} (randomness does not depend on provided seed)!"
                )
                key = HQC.keypair()  # Randomness does not depend on rng
                pickle.dump(key, file)
                logger.debug(f"new key saved in {filename}")
    else:
        logger.info(
            "Generating random HQC keypair (randomness does not depend on provided seed)!"
        )
        key = HQC.keypair()  # Randomness does not depend on rng
    return key


def search_distinguishable_plaintext(HQC, rng: np.random.RandomState):
    """
    Finds plaintexts by random search that matches the special property of requirering
    the full 6 seed expansions (3 extra over minimum) that is the rejection sampling based timing
    side-channel of HQC.

    >>> HQC = Hqc128
    >>> pt = search_distinguishable_plaintext(HQC, make_random_state(0))
    >>> rejections = HQC.num_rejections(pt)
    >>> rejections // 1000
    6
    """
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


def toggle_bits_in_v(ciphertext: Union[bytes, bytearray], bits: list, u_size: int):
    """
    This flips the specified bits in the v part (by skipping over the u part) of
    a ciphertext.

    >>> toggle_bits_in_v(bytearray((0xff, 0xff, 0xff)), [0, 6], 9).hex()
    'ffffbe'
    """
    ct = bytearray(ciphertext)
    startat = u_size + (8 - (u_size % 8))
    for bit in bits:
        byte = (bit + startat) // 8
        mod = bit % 8
        shifted = 1 << mod
        ctbyte = ct[byte]
        ctbyte = shifted ^ ctbyte
        ct[byte] = ctbyte
    return ct


def toggle_outer_block(
    ciphertext: Union[bytes, bytearray], block: int, u_size: int, block_size: int
):
    """
    This flips the entirety of block 'block' of size 'block_size' in vector v (by
    skipping over the u vector) of ciphertext

    >>> toggle_outer_block(bytearray((0xff, 0xff, 0xfe, 0xfe)), 0, 9, 10).hex()
    'ffff01fd'
    """
    bits = [x for x in range(block_size * block, block_size * (block + 1))]
    return toggle_bits_in_v(ciphertext, bits, u_size)


class HqcSimulationParams:
    def __init__(
        self,
        HQC,
        OUTER_DECODING_LIMIT: int,
        EPSILON: Tuple[int, int],
        DECODE_EVERY: int,
        WEIGHT: int,
        N_OVERRIDE: int = None,
    ):
        self.HQC = HQC
        self.N = N_OVERRIDE if N_OVERRIDE else HQC.params("N")
        self.N1 = None if N_OVERRIDE else HQC.params("N1")
        self.N2 = None if N_OVERRIDE else HQC.params("N2")
        self.OUTER_DECODING_LIMIT = OUTER_DECODING_LIMIT
        self.EPSILON = EPSILON
        self.DECODE_EVERY = DECODE_EVERY
        self.WEIGHT = WEIGHT

    def __repr__(self) -> str:
        return (
            f"N: {self.N}"
            f", N1: {self.N1}"
            f", N2: {self.N2}"
            f", OUTER_DECODING_LIMIT: {self.OUTER_DECODING_LIMIT}"
            f", EPSILON: {self.EPSILON}"
            f", WEIGHT: {self.WEIGHT}"
        )


class HqcSimulationTracking:
    def __init__(self, params: HqcSimulationParams):
        self.num_oracle_calls = 0
        self.params = params
        self.decoder_stats = []

    def reset_block_status(self):
        self.current_block_nr = None
        self.block_status = [
            {
                "status": FlipStatus.UNFLIPPED,
                "result": IfFlipResult.UNKNOWN,
            }
            for _ in range(self.params.N1)
        ]

    def set_current_block_nr(self, current_block_nr):
        self.current_block_nr = current_block_nr
        if "bits" not in self.block_status[self.current_block_nr]:
            bit_status = [
                {"status": FlipStatus.UNFLIPPED, "result": IfFlipResult.UNKNOWN}
                for _ in range(self.params.N2)
            ]
            self.block_status[self.current_block_nr]["bits"] = bit_status

    def current_block(self):
        return self.block_status[self.current_block_nr]

    def current_bits_status(self):
        return self.current_block()["bits"]

    def add_decoder_stats(
        self,
        checks,
        unsatisfied,
        good_flips,
        bad_flips,
        found_bad_satisfied_checks,
        found_bad_unsatisfied_checks,
        success,
    ):
        self.decoder_stats.append(
            {
                "checks": checks,
                "oracle_calls": self.num_oracle_calls,
                "unsatisfied": unsatisfied,
                "good_flips": good_flips,
                "bad_flips": bad_flips,
                "found_bad_satisfied_checks": found_bad_satisfied_checks,
                "found_bad_unsatisfied_checks": found_bad_unsatisfied_checks,
                "success": success,
            }
        )

    def decoder_stats_data_frame(self, label="", write_header=True):
        static_columns = ["label", "alg", "weight", "epsilon0", "epsilon1"]
        ep0 = self.params.EPSILON[0]
        ep1 = self.params.EPSILON[1]
        # if ep0 == 1.0 and ep1 == 1.0: # commented out: we want
        #     ep0 = "miss-use"
        #     ep1 = "miss-use"
        static_values = [
            label,
            self.params.HQC.name(),
            self.params.WEIGHT,
            ep0,
            ep1,
        ]
        df = pd.DataFrame.from_dict(self.decoder_stats)
        dynamic_columns = list(df.columns)
        df[static_columns] = static_values
        # Rearrange the columns
        column_order = static_columns + dynamic_columns
        return df[column_order]


def next_failure_block(
    params: HqcSimulationParams,
    tracking: HqcSimulationTracking,
    rng: np.random.RandomState,
    priv,
    pt,
    ct,
):
    """
    Finds next block that if flipped, will result in decoding failure.

    Assumes input ciphertext 'ct' decodes successfully.

    The returned ciphertext will have multiple blocks flipped but not block 'block'.
    The ciphertext will successfully decode to the same plaintext.
    """
    outer_decoding_limit = 15

    # Verify starting assumption
    SingletonAssertDecodingFailure().assert_decoding_success(
        True, params, tracking, ct, priv, pt, rng, debug=True
    )

    # First we flip up to 15 blocks that we have already evaluated
    evaluated_blocks = [
        i
        for i in range(params.N1)
        if tracking.block_status[i]["status"] == FlipStatus.UNFLIPPED
        and tracking.block_status[i]["result"] != IfFlipResult.UNKNOWN
    ]
    to_sample = min(len(evaluated_blocks), outer_decoding_limit)
    blocks = 0
    for block in rng.choice(evaluated_blocks, to_sample, replace=False):
        blocks += 1
        logger.info(f"Flipping outer block {block} (evaluated)")
        # flip block 'block' of v in ct
        ct = toggle_outer_block(ct, block, params.N, params.N2)
        tracking.block_status[block]["status"] = FlipStatus.FLIPPED

    # Then we flip some unknown blocks so that the total flipped
    # blocks goes up to outer_decoding_limit
    unknown_blocks = [
        i
        for i in range(params.N1)
        if tracking.block_status[i]["status"] == FlipStatus.UNFLIPPED
        and tracking.block_status[i]["result"] == IfFlipResult.UNKNOWN
    ]
    for block in rng.choice(unknown_blocks, len(unknown_blocks), replace=False):
        blocks += 1
        logger.info(f"Flipping outer block {block} (unknown)")
        # flip block 'block' of v in ct
        ct = toggle_outer_block(ct, block, params.N, params.N2)
        tracking.block_status[block]["status"] = FlipStatus.FLIPPED
        if blocks == outer_decoding_limit:
            # Verify still decoding success
            SingletonAssertDecodingFailure().assert_decoding_success(
                True, params, tracking, ct, priv, pt, rng, debug=True
            )
        elif blocks == outer_decoding_limit + 1:
            # Verify decoding failure
            SingletonAssertDecodingFailure().assert_decoding_success(
                False, params, tracking, ct, priv, pt, rng, debug=True
            )

            # undo the flip
            ct = toggle_outer_block(ct, block, params.N, params.N2)
            tracking.block_status[block]["status"] = FlipStatus.UNFLIPPED
            tracking.block_status[block]["result"] = IfFlipResult.FAILURE
            logger.info(f"Decoding Failure by flipping block {block}")
            return (block, ct)

    # No more unknown blocks to evaluate
    return None


def reset_full_block_flips(
    params: HqcSimulationParams, tracking: HqcSimulationTracking, ct
):
    """
    Resets all fully flipped blocks where 'result' is IfFlipResult.UNKNOWN
    """
    for (block, bs) in enumerate(tracking.block_status):
        if bs["status"] == FlipStatus.FLIPPED:
            logger.debug(f"Unflipping block {block}")
            ct = toggle_outer_block(ct, block, params.N, params.N2)
            bs["status"] = FlipStatus.UNFLIPPED
    return ct


def reset_current_block(
    params: HqcSimulationParams, tracking: HqcSimulationTracking, ct
):
    """
    Resets all bits in the specified RM block where FlipStatus == FLIPPED
    """
    available_bits = [
        i
        for (i, b) in (enumerate(tracking.current_bits_status()))
        if b["status"] == FlipStatus.FLIPPED
    ]
    for bit in available_bits:
        ct = flip_single_bit(ct, tracking.current_block_nr, bit, params.N, params.N2)

    return ct


def flip_single_bit(ct, block, bit, N, N2):
    """
    Correctly calculates the position of a single bit in v part of ct. And flips it.
    """
    return toggle_bits_in_v(ct, [block * N2 + bit], N)


def next_success_bit(
    rng: np.random.RandomState, HQC, priv, pt, ct, bit_status, block, epsilon
):
    """
    Finds next bit that if toggled, will result in decoding success.

    Assumes input ciphertext 'ct' fails to decodes.

    Returns None if ct still fails to decode after all available bits have been exhausted.
    Returns tuple of (bit nr., bit status, ciphertext)
    """
    N = HQC.params("N")
    N1 = HQC.params("N1")
    N2 = HQC.params("N2")
    if True:
        # Verify starting assumption
        SingletonAssertDecodingFailure().assert_decoding_success(
            False, HQC, ct, priv, pt, rng, epsilon=epsilon, debug=True
        )

    available_bits = [
        i
        for (i, b) in (enumerate(bit_status))
        if b["result"] == IfFlipResult.UNKNOWN and b["status"] == FlipStatus.UNFLIPPED
    ]
    totry = min(3, len(available_bits))
    for bit in rng.choice(available_bits, totry, replace=False):
        ct = flip_single_bit(ct, block, bit, N, N2)
        bit_status[bit]["status"] = bit_status[bit]["status"].toggled()
        if wrapped_hqc_decoding_oracle(
            HQC,
            ct,
            priv,
            pt,
            rng,
            epsilon=epsilon,
            result_meta=bit_status[bit],
            require_true=0.999,
        ):
            bit_status[bit]["result"] = IfFlipResult.SUCCESS
            logger.info(
                f"Decoding success by flipping bit {bit} in block {block} check = 1"
            )
            # hqc_decoding_oracle(HQC, ct, priv, pt, rng, debug=True, epsilon=epsilon) # debug log the oracle output
            return (bit, bit_status[bit]["status"], ct)
        else:
            logger.debug(f"Flipped bit {bit} of outer block {block}")

    available_bits = [
        i
        for (i, b) in (enumerate(bit_status))
        if b["result"] == IfFlipResult.UNKNOWN and b["status"] == FlipStatus.FLIPPED
    ]
    for bit in rng.choice(available_bits, len(available_bits), replace=False):
        ct = flip_single_bit(ct, block, bit, N, N2)
        bit_status[bit]["status"] = bit_status[bit]["status"].toggled()
        if wrapped_hqc_decoding_oracle(
            HQC,
            ct,
            priv,
            pt,
            rng,
            epsilon=epsilon,
            result_meta=bit_status[bit],
            require_true=0.999,
        ):
            bit_status[bit]["result"] = IfFlipResult.SUCCESS
            logger.info(
                f"Decoding success by unflipping bit {bit} in block {block} check = 0"
            )
            # hqc_decoding_oracle(HQC, ct, priv, pt, rng, debug=True, epsilon=epsilon) # debug log the oracle output
            return (bit, bit_status[bit]["status"], ct)
        else:
            logger.debug(f"Unflipped bit {bit} of outer block {block}")

    return None


def next_failure_bit(
    params: HqcSimulationParams,
    tracking: HqcSimulationTracking,
    rng: np.random.RandomState,
    priv,
    pt,
    ct,
):
    """
    Finds next bit that if toggled, will result in decoding failure.

    Assumes input ciphertext 'ct' succeeds to decodes.

    Returns None if ct still succeed to decode after all available bits have been exhausted.
    Returns tuple of (bit nr., bit status, ciphertext)
    """

    # Verify starting assumption
    SingletonAssertDecodingFailure().assert_decoding_success(
        True, params, tracking, ct, priv, pt, rng, debug=True
    )

    # available_bits = [
    #     i
    #     for (i, b) in (enumerate(bit_status))
    #     if b["result"] == IfFlipResult.UNKNOWN and b["status"] == FlipStatus.FLIPPED
    # ]
    # totry = min(3, len(available_bits))
    # for bit in rng.choice(available_bits, totry, replace=False):
    #     ct = flip_single_bit(ct, block, bit, N, N2)
    #     bit_status[bit]["status"] = bit_status[bit]["status"].toggled()
    #     if not hqc_decoding_oracle(HQC, ct, priv, pt, rng, epsilon=epsilon, result_meta=bit_status[bit]):
    #         bit_status[bit]["result"] = IfFlipResult.FAILURE
    #         logger.info(
    #             f"Decoding failure by unflipping bit {bit} in block {block} check = 1"
    #         )
    #         #hqc_decoding_oracle(HQC, ct, priv, pt, rng, debug=True, epsilon=epsilon) # debug log the oracle output
    #         return (bit, bit_status[bit]["status"], ct)
    #     else:
    #         logger.debug(f"Unflipped bit {bit} of outer block {block}")

    available_bits = [
        i
        for (i, b) in (enumerate(tracking.current_bits_status()))
        if b["result"] == IfFlipResult.UNKNOWN and b["status"] == FlipStatus.UNFLIPPED
    ]
    for bit in rng.choice(available_bits, len(available_bits), replace=False):
        ct = flip_single_bit(ct, tracking.current_block_nr, bit, params.N, params.N2)
        tracking.current_bits_status()[bit]["status"] = FlipStatus.FLIPPED
        if not wrapped_hqc_decoding_oracle(
            params,
            tracking,
            ct,
            priv,
            pt,
            rng,
            result_meta=tracking.current_bits_status()[bit],
            require_false=0.99999,
        ):
            tracking.current_bits_status()[bit]["result"] = IfFlipResult.FAILURE
            logger.info(
                f"Decoding failure by flipping bit {bit} in block {tracking.current_block_nr} check = 0"
            )
            # confirm our result
            SingletonAssertDecodingFailure().assert_decoding_success(
                False, params, tracking, ct, priv, pt, rng, debug=True
            )  # debug log the oracle output
            return (bit, tracking.current_bits_status()[bit]["status"], ct)
        else:
            logger.debug(
                f"Flipped bit {bit} of outer block {tracking.current_block_nr}"
            )

    return None


def find_minimal_failure_flips(
    params: HqcSimulationParams,
    tracking: HqcSimulationTracking,
    rng,
    priv,
    pt,
    ct,
    save_results=False,
):
    """
    Returns (successes, ct) where successes is an array of bits that if
    unflipped would result in a decoding success. ct is the resulting
    ciphertext with minimal number of flips that still results in a
    decoding failure in the current block
    """
    nrflipped = len(
        [x for x in tracking.current_bits_status() if x["status"] == FlipStatus.FLIPPED]
    )
    logger.debug(f"Enter find_minimal_failure_flips with {nrflipped} flipped bits")

    # Verify starting assumption "fails to decode"
    SingletonAssertDecodingFailure().assert_decoding_success(
        False, params, tracking, ct, priv, pt, rng, debug=True
    )

    available_bits = [
        i
        for (i, b) in (enumerate(tracking.current_bits_status()))
        if b["result"] == IfFlipResult.UNKNOWN and b["status"] == FlipStatus.FLIPPED
    ]
    successes = []
    for bit in available_bits:
        ctmod = flip_single_bit(ct, tracking.current_block_nr, bit, params.N, params.N2)
        if wrapped_hqc_decoding_oracle(
            params,
            tracking,
            ctmod,
            priv,
            pt,
            rng,
            result_meta=tracking.current_bits_status()[bit],
            require_false=0.9999,
            require_true=0.99,
        ):
            if save_results:
                tracking.current_bits_status()[bit]["result"] = IfFlipResult.SUCCESS
                logger.info(
                    f"Decoding success if unflipping bit {bit} in block {tracking.current_block_nr} check = 0"
                )
                wrapped_hqc_decoding_oracle(
                    params, tracking, ctmod, priv, pt, rng, debug=True
                )
                # Add bit to return
                successes.append(
                    (bit, tracking.current_bits_status()[bit]["certainty"])
                )
            else:
                logger.debug(
                    f"Discarded unflipped bit {bit} of outer block {tracking.current_block_nr}"
                )
        else:
            tracking.current_bits_status()[bit]["status"] = FlipStatus.UNFLIPPED
            logger.debug(
                f"Unflipped bit {bit} of outer block {tracking.current_block_nr}"
            )
            # set new ciphertext with fewer flipped bits but still decoding failure
            ct = ctmod
    nrflipped = len(
        [x for x in tracking.current_bits_status() if x["status"] == FlipStatus.FLIPPED]
    )
    logger.debug(f"Exiting find_minimal_failure_flips with {nrflipped} flipped bits")
    return (successes, ct)


def find_successes_by_flipping(
    params: HqcSimulationParams, tracking: HqcSimulationTracking, rng, priv, pt, ct
):
    """
    Returns array of bits that if flipped would result in a decoding success
    """

    # Verify starting assumption "fails to decode"
    SingletonAssertDecodingFailure().assert_decoding_success(
        False, params, tracking, ct, priv, pt, rng, debug=True
    )

    available_bits = [
        i
        for (i, b) in (enumerate(tracking.current_bits_status()))
        if b["result"] == IfFlipResult.UNKNOWN and b["status"] == FlipStatus.UNFLIPPED
    ]
    successes = []
    failures = []
    for bit in available_bits:
        ctmod = flip_single_bit(ct, tracking.current_block_nr, bit, params.N, params.N2)
        # tracking.current_bits_status()[bit]["status"] = tracking.current_bits_status()[bit]["status"].toggled()
        if wrapped_hqc_decoding_oracle(
            params,
            tracking,
            ctmod,
            priv,
            pt,
            rng,
            result_meta=tracking.current_bits_status()[bit],
            require_false=0.99,
            require_true=0.999,
        ):
            tracking.current_bits_status()[bit]["result"] = IfFlipResult.SUCCESS
            logger.info(
                f"Decoding success if flipping bit {bit} in block {tracking.current_block_nr} check = 1"
            )
            # debug
            wrapped_hqc_decoding_oracle(
                params, tracking, ctmod, priv, pt, rng, debug=True
            )
            # Return the unflipped ct
            successes.append((bit, tracking.current_bits_status()[bit]["certainty"]))
        else:
            tracking.current_bits_status()[bit]["result"] = IfFlipResult.FAILURE
            logger.debug(
                f"Still decoding failure if flipped bit {bit} of outer block {tracking.current_block_nr}"
            )
            failures.append((bit, tracking.current_bits_status()[bit]["certainty"]))

    return (successes, failures)


def decode(
    params: HqcSimulationParams,
    tracking: HqcSimulationTracking,
    Hin: np.array,
    checks: list,
    y_sparse: list,
):
    """
    Uses parity check matrix 'H' to decodes the message
    [0_0, 0_1, 0_2, ..., 0_N | c_0, c_1, ..., c_R],
    where N is the length of the decoded message and
    R is the number of check equations given by 'H'.
    checks provides the parity check measurements, but as a tuple of value and certainty (probability of being correct):
    checks = [c_0, c_1, ..., c_R].
    'y' (sparsely) provides the actual message to compare the
    decoding against.
    """
    R = Hin.shape[0]

    H = np.concatenate((Hin, np.identity(R, dtype=int)), axis=1, dtype=int)
    logger.debug(f"decode, N: {params.N}, R: {R}, Hin: {Hin.shape}, H: {H.shape}")

    # Create empty "message" to be decoded
    msg_weight = len(y_sparse)
    prob_for_one = msg_weight / params.N
    assumed_zero = np.full(params.N, prob_for_one, dtype=np.float64)

    # failure rate for individual check bits, depending on its value
    check_part = np.array([1 - p for (_, p) in checks], dtype=np.float64)
    channel_probs = np.concatenate((assumed_zero, check_part))
    logger.debug(f"channel_probs: {channel_probs}")

    # Creating the decoder
    bpd = bp_decoder(
        H,
        max_iter=100,
        bp_method="product_sum",
        channel_probs=channel_probs,
    )

    logger.info(f"Attempting decode with {R} checks.")

    msg = np.concatenate(
        (np.zeros(params.N, dtype=int), np.array([c for (c, _) in checks], dtype=int))
    )
    logger.debug(f"msg: {msg}")

    decoded = bpd.decode(msg)
    decoded_msg_sparse = []
    decoded_checks_sparse = []
    unsatisfied = 0
    good_flips = 0
    bad_flips = 0
    found_bad_satisfied_checks = 0
    found_bad_unsatisfied_checks = 0

    for (i, x) in enumerate(decoded[: params.N]):
        if x:
            s = str(i)
            if i in y_sparse:
                good_flips += 1
                s += "*"
            else:
                bad_flips += 1
            decoded_msg_sparse.append(s)
    for (i, (x, (c, _))) in enumerate(zip(decoded[params.N :], checks)):
        if c:
            unsatisfied += 1
            if not x:
                decoded_checks_sparse.append(params.N + i)
                found_bad_unsatisfied_checks += 1
        elif x:
            decoded_checks_sparse.append(params.N + i)
            found_bad_satisfied_checks += 1

    logger.info(
        f"Decoded with {R} checks, made {len(decoded_msg_sparse)} "
        f"flips to recover y: {decoded_msg_sparse} and found "
        f"{len(decoded_checks_sparse)} measurement errors: {decoded_checks_sparse}"
    )

    unequal = 0
    for (i, yip) in enumerate(decoded[: params.N]):
        yi = i in y_sparse
        if yi or yip:
            logger.debug(f"y'[{i}]: {yip}, y[{i}]: {yi}")
            unequal |= yi != yip

    success = not bool(unequal)
    tracking.add_decoder_stats(
        R,
        unsatisfied,
        good_flips,
        bad_flips,
        found_bad_satisfied_checks,
        found_bad_unsatisfied_checks,
        success,
    )
    return success


saved_rm_enc = None
saved_input_decoder = None


def bytes_compare(array, compare_to, delimit=False, pad=True):
    hex = ""
    for x in range(0, len(array)):
        if delimit and x % delimit == 0:
            if pad:
                hex += "|"
        if array[x] == compare_to[x]:
            if pad:
                hex += "__"
        else:
            hex += f"{array[x]:02x}"
    if delimit:
        hex += "|"
    return hex


def wrapped_hqc_decoding_oracle(*args, require_false=0.5, require_true=0.5, **kwargs):

    if "require" in kwargs:
        raise Exception(
            "obsolete argument require, use require_true and require_false instead"
        )

    result_meta = kwargs.pop("result_meta", dict(certainty=0.0))

    require = (require_false, require_true)
    results = ([], [])
    tries = 0
    while True:
        tries += 1
        new_meta = dict()
        result = inner_hqc_decoding_oracle(*args, **kwargs, result_meta=new_meta)
        results[result].append(new_meta["certainty"])

        certainty = 1.0 - prod([1.0 - p for p in results[result]])
        if certainty >= require[result]:
            logger.debug(
                f'Wrapped oracle made decision "{result}" after {tries} tries. Certainty: {certainty} >= {require[result]}'
            )
            result_meta["certainty"] = certainty
            return result


def inner_hqc_decoding_oracle(
    params: HqcSimulationParams,
    tracking: HqcSimulationTracking,
    ct,
    priv,
    pt,
    rng: np.random.RandomState,
    debug=False,
    save_checkpoint_now=False,
    result_meta=None,
):
    """
    Returns true if ct results in decoding success, by checking result against pt. This is a cheating oracle,
    which simulates a real oracle by providing error rates (epsilon argument) for false negatives and false positives as a tuple
    """
    (pt_prime, rs_enc, rm_dec, input_decoder, _u, _v) = params.HQC.decode_intermediates(
        ct, priv
    )

    result = pt == pt_prime

    failure_rate = params.EPSILON[int(result)]
    invert_result = rng.rand() > failure_rate

    if debug:
        global saved_rm_enc, saved_input_decoder
        hex_rs_enc = ""
        hex_rm_dec = ""
        # hex_input_decoder = ""

        pt_prime_num_rejections = params.HQC.num_rejections(bytes(pt_prime))

        if saved_rm_enc:
            hex_rs_enc = bytes_compare(rs_enc, saved_rm_enc)
            hex_rm_dec = bytes_compare(rm_dec, saved_rm_enc)
        hex_pt_prime = bytes_compare(pt_prime, pt)
        # if saved_input_decoder:
        #     hex_input_decoder = bytes_compare(
        #         input_decoder, saved_input_decoder, delimit=HQC.params("N2") // 8
        #     )

        logger.debug(f"Decapsulation called!")
        # logger.debug(f"decode in: '{hex_input_decoder}'")
        if invert_result:
            logger.debug(f"Epsilon argument ignored (would have inverted the result)")
        logger.debug(f"RM decode: '{hex_rm_dec}'")
        logger.debug(f"Plaintext: '{hex_pt_prime}'")
        logger.debug(f"Number of rejections for plaintext: {pt_prime_num_rejections}")
        logger.debug(f"RS encode: '{hex_rs_enc}'")
    else:
        tracking.num_oracle_calls += 1
        if invert_result:
            logger.info(
                f"Inverting oracle decision (originally: {result}) due to specified epsilon value: {failure_rate}"
            )
            result = not result

    if save_checkpoint_now:
        saved_rm_enc = rs_enc
        saved_input_decoder = input_decoder
    if isinstance(result_meta, dict):
        result_meta["certainty"] = failure_rate
    return result


def cheating_sum_checks(row, r1_y_sparse):
    sum = 0
    sumpos = []
    for (pos, value) in enumerate(row):
        if value:
            if pos in r1_y_sparse:
                sumpos.append(pos)
                sum += 1
    return (sum, sumpos)


def add_check(H, Hgen, r1_y_sparse, bit_n, checks, check, certainty):
    row = Hgen[bit_n]

    add = True
    if r1_y_sparse:
        bit_set = bit_n in r1_y_sparse
        logger.debug(f"check: {check}, bit_set: {bit_set}")

        if check != bit_set:
            # add = False
            if certainty == 1.0:
                logger.error(
                    f"Certainty {certainty} but still added false result for bit {bit_n}, check: {check}, bit value: {bit_set}!"
                )
            else:
                logger.warning(
                    f"Certainty {certainty} resulted in a false result for bit {bit_n}, check: {check}, bit value: {bit_set}!"
                )

    if add:
        logger.info(f"Adding to H the check={check} corresponding to bit {bit_n}.")
        H = np.vstack([H, row]) if H is not None else Hgen[bit_n]
        checks.append((check, certainty))
    return H


debug_bytearray_global = None


def debug_bytearray(arr, store=False):
    global debug_bytearray_global
    if store:
        debug_bytearray_global = arr
    if debug_bytearray_global:
        return bytes_compare(arr, debug_bytearray_global, pad=False)
    else:
        return ""


def sparse_times_sparse(A, B, N, mod=2):
    """
    Multiplies a sparse vector with another sparse vector

    >>> A = [3, 5, 9]
    >>> B = [0, 2]
    >>> sparse_times_sparse(A, B, N=10, mod=None)
    [1, 3, 5, 5, 7, 9]
    >>> sparse_times_sparse(A, B, N=10, mod=2)
    [1, 3, 7, 9]
    """
    a_times_b = []

    for b in B:
        A_shifted_b = [(a + b) % N for a in A]
        a_times_b += A_shifted_b

    if mod:
        # Reduce duplicates by specified modulus
        counts = Counter(a_times_b)
        a_times_b = []
        for (k, v) in counts.items():
            a_times_b.extend([k for _ in range(0, v % mod)])

    a_times_b.sort()

    return a_times_b


def add_checks(
    params: HqcSimulationParams,
    tracking: HqcSimulationTracking,
    check_value,
    bits,
    H,
    Hgen,
    checks: list,
    y_sparse,
    y_times_r1,
):
    previous_decoding = 0
    for (b, certainty) in bits:
        bit_n = tracking.current_block_nr * params.N2 + b
        # Add row from Hgen to H, to correspond to the parity check equation
        # given by the current bit
        H = add_check(H, Hgen, y_times_r1, bit_n, checks, check_value, certainty)
        R = len(checks)

        if R % params.DECODE_EVERY == 0 and R != 0 and previous_decoding != R:
            # this prevents decoding again if R doesn't change
            previous_decoding = R
            unsatisfied = sum(c for (c, p) in checks)
            logger.info(
                f"{tracking.num_oracle_calls} decapsulation calls performed so-far, "
                f"{unsatisfied} unsatisfied checks, out of total {len(checks)}."
            )
            if decode(params, tracking, H, checks, y_sparse):
                logger.info(f"Successfully decoded y")
                return True

    return (H, checks)


def simulate_hqc_idealized_oracle(
    rng: np.random.RandomState,
    decode_every: int,
    weight: int,
    keyfile=None,
    error_rate=0.0,
):
    """
    Main function for simulating HQC attack
    """
    logger.info("Selecting Hqc128 as current HQC version!")
    if error_rate > 0.0:
        SingletonAssertDecodingFailure().raise_exception = False
    noise_level = 1.0 - error_rate
    if isnan(error_rate):
        epsilon = (1.0, 1.0)
    else:
        epsilon = (
            0.9942 * noise_level,
            1.0 * noise_level,
        ),  # HQC ideal decoding oracle multiplied with measurement noise level
    params = HqcSimulationParams(
        HQC=Hqc128,
        OUTER_DECODING_LIMIT=15,
        EPSILON=epsilon,
        DECODE_EVERY=decode_every,
        WEIGHT=weight,
    )
    logger.info(f"Params {params}")

    tracking = HqcSimulationTracking(params)

    H = None
    checks = []

    # construct/load key-pair to crack
    (pub, priv) = read_or_generate_keypair(params.HQC, keyfile)
    (_, y_sparse) = params.HQC.secrets_from_key(priv)

    y_sparse = sorted(y_sparse)
    logger.info(f"y weight: {len(y_sparse)}, y bits: {y_sparse}")

    # Keep trying new plain texts, until we manage to decode and find y
    while True:
        # generate / select plain text message
        pt = search_distinguishable_plaintext(params.HQC, rng)

        # Generate a parity check matrix (H)
        logger.info(f"Create random (L/M)DPC parity check of size {params.N}!")
        Hgen = make_random_ldpc_parity_check_matrix(params.N, weight, rng)

        # encapsulate with r_2 and e to all-zero. Set r_1 to h_0
        r1_sparse = [i for (i, x) in enumerate(Hgen[:, 0]) if x != 0]
        assert weight == len(r1_sparse)
        logger.debug(f"r1_sparse: {r1_sparse}")
        (ct, ss) = params.HQC.encaps_with_plaintext_and_r1(pub, pt, r1_sparse)
        debug_bytearray(ct, store=True)
        logger.debug(f"Encaps result: {len(ss)} // {len(ct)}")

        # Print debug output
        y_times_r1 = sparse_times_sparse(y_sparse, r1_sparse, params.N)
        logger.debug(f"y_times_r1: {y_times_r1}")
        wrapped_hqc_decoding_oracle(
            params,
            tracking,
            ct,
            priv,
            pt,
            rng,
            debug=True,
            save_checkpoint_now=True,
        )

        tracking.reset_block_status()
        try:
            while True:
                # Select outer blocks that: if flipped back results in decoding success.
                ret = next_failure_block(params, tracking, rng, priv, pt, ct)
                if ret is None:
                    # restart with an new plaintext
                    raise NoMoreUntestedRmBlocks
                (current_block, ct) = ret
                tracking.set_current_block_nr(current_block)

                ret = next_failure_bit(params, tracking, rng, priv, pt, ct)
                if ret is None:
                    # No more bits to flip in this block
                    tracking.current_block()["status"] = FlipStatus.UNFLIPPED
                    # Verify assumption of decoding success after bit flip
                    SingletonAssertDecodingFailure().assert_decoding_success(
                        True, params, tracking, ct, priv, pt, rng, debug=True
                    )
                    break

                (_, _, ct) = ret

                # Find minimal bit pattern for decoding failure in this inner RM block
                (successes, ct) = find_minimal_failure_flips(
                    params,
                    tracking,
                    rng,
                    priv,
                    pt,
                    ct,
                    save_results=True,
                )

                # Add the bits that we have decided
                ret = add_checks(
                    params,
                    tracking,
                    0,
                    successes,
                    H,
                    Hgen,
                    checks,
                    y_sparse,
                    y_times_r1,
                )
                # first check for abort condition
                if isinstance(ret, bool):
                    return (ret, tracking)
                # else extract return values
                (H, checks) = ret

                # Use the minimal failure pattern to decide which bits are '1' in r1*y
                (successes, failures) = find_successes_by_flipping(
                    params, tracking, rng, priv, pt, ct
                )

                # add these bits
                ret = add_checks(
                    params,
                    tracking,
                    1,
                    successes,
                    H,
                    Hgen,
                    checks,
                    y_sparse,
                    y_times_r1,
                )

                # first check for abort condition
                if isinstance(ret, bool):
                    return (ret, tracking)
                # else extract return values
                (H, checks) = ret

                ## Find maximum successfull pattern, start from minimal
                ## failure pattern, with one bit flipped which we know is a success

                ## flip bits in pattern to recover '1's

                ct = reset_current_block(params, tracking, ct)

                # Reset all full-block flips that provide no information
                ct = reset_full_block_flips(params, tracking, ct)
        except NoMoreUntestedRmBlocks:
            # Continue with another ciphertext
            continue

    # miss-classification probabilities are derived from "idealized-oracle" calls in CHES2022 paper.


def shift_and_add_mod_2_sparse(y, j, n):
    """
    This accepts a sparse vector (indexes of set bits) and adds
    itself with itself but bit-shifted right with j steps, modulo 2.
    The bitshift is a rotation over a vector of size n.

    e.g. return y + ((y >> j) % n)

    >>> shift_and_add_mod_2_sparse([1, 5, 8, 12], 3, 15)
    [0, 1, 4, 5, 11, 12]
    """
    y_shift_j = [(yi + j) % n for yi in y]
    yyj = y + y_shift_j
    del y
    del y_shift_j
    yyj.sort()
    ret = []
    discarded = False
    for (y1, y2) in zip(yyj[:], yyj[1:] + [n + 1]):
        if discarded:
            discarded = False
        elif y1 == y2:
            discarded = True
        else:
            ret.append(y1)
            discarded = False
    return ret


def test_hqc_encaps_with_plaintext_and_r1(seed):
    """
    This is a unit test.

    We use doctest for test discovery:
    >>> test_hqc_encaps_with_plaintext_and_r1(0)
    True
    """
    HQC = Hqc128()
    N = HQC.params("N")
    (pub, priv) = read_or_generate_keypair(HQC, "test-hqc.key")
    rng = make_random_state(seed)
    pt = search_distinguishable_plaintext(HQC, rng)
    for j in rng.choice(N, 100, replace=False):
        (_, y) = HQC.secrets_from_key(priv)
        yyj = shift_and_add_mod_2_sparse(y, j, N)
        (ct, _) = HQC.encaps_with_plaintext_and_r1(pub, pt, [0, j])
        eprime = HQC.eprime(ct, priv, pt)
        bits = np.unpackbits(eprime, bitorder="little")
        indices = [i for (i, x) in enumerate(bits) if x]

        for (yi, ei) in zip(yyj, indices):
            if yi != ei:
                return False

    return True


def test_hqc_decode_toy_example(seed):
    """
    This is a unit test.

    We use doctest for test discovery:
    >>> test_hqc_decode_toy_example(0)
    True
    """
    params = HqcSimulationParams(Hqc128, None, None, None, WEIGHT=3, N_OVERRIDE=20)
    tracking = HqcSimulationTracking(params)

    rng = make_random_state(seed)
    logger.debug(f"weight: {params.WEIGHT}")
    logger.debug(f"N: {params.N}")
    y_sparse = [4, 5, 7, 9]  # TODO
    logger.debug(f"y_sparse: {y_sparse}")

    Hgen = make_random_ldpc_parity_check_matrix(params.N, params.WEIGHT, rng)
    logger.debug(f"Hgen: \n{Hgen}")

    r1_sparse = [i for (i, x) in enumerate(Hgen[:, 0]) if x != 0]
    assert params.WEIGHT == len(r1_sparse)
    logger.debug(f"r1_sparse: {r1_sparse}")

    y_times_r1 = sparse_times_sparse(y_sparse, r1_sparse, params.N)
    logger.debug(f"y_times_r1: {y_times_r1}")

    checks = []
    check_values = [(i, i in y_times_r1) for i in range(params.N)]  # TODO
    logger.debug(f"check_values: {check_values}")

    H = None
    for (bit_n, check_value) in check_values:
        if check_value or True:
            H = add_check(
                H, Hgen, y_times_r1, bit_n, checks, check_value, certainty=1.0
            )

    from ldpc.code_util import get_code_parameters

    code_params = get_code_parameters(Hgen)
    logger.debug(f"code_params: \n{code_params}")

    logger.debug(f"checks: {checks}")
    logger.debug(f"H: \n{H}")
    return decode(params, tracking, H, checks, y_sparse)


def test_hqc_decode_full_example(seed):
    """
    This is a unit test.

    We use doctest for test discovery:
    >>> test_hqc_decode_full_example(0)
    True
    """
    params = HqcSimulationParams(Hqc128, None, None, None, WEIGHT=3)
    tracking = HqcSimulationTracking(params)

    rng = make_random_state(seed)
    OMEGA = params.HQC.params("OMEGA")

    y_sparse = rng.choice(params.N, OMEGA, replace=False)

    Hgen = make_random_ldpc_parity_check_matrix(params.N, params.WEIGHT, rng)

    r1_sparse = [i for (i, x) in enumerate(Hgen[:, 0]) if x != 0]
    assert params.WEIGHT == len(r1_sparse)

    y_times_r1 = sparse_times_sparse(y_sparse, r1_sparse, params.N)

    checks = []
    check_values = [(i, i in y_times_r1) for i in range(params.N)]

    H = None
    for (bit_n, check_value) in check_values:
        if check_value:
            H = add_check(
                H, Hgen, y_times_r1, bit_n, checks, check_value, certainty=1.0
            )

    logger.debug(f"checks: {len(checks)}")
    return decode(params, tracking, H, checks, y_sparse)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(level="DEBUG")
    with np.printoptions(linewidth=99999):
        for seed in range(0, 10):
            print(
                f"test_hqc_encaps_with_plaintext_and_r1({seed}): {'Success' if test_hqc_encaps_with_plaintext_and_r1(seed) else 'Failure'}!"
            )
            print(
                f"test_hqc_decode_toy_example({seed}): {'Success' if test_hqc_decode_toy_example(seed) else 'Failure'}!"
            )
            print(
                f"test_hqc_decode_full_example({seed}): {'Success' if test_hqc_decode_full_example(seed) else 'Failure'}!"
            )

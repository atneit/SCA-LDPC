#!/bin/env python

# Ugly hack to allow import from the current package
# if running current module as main. Please forgive the heresy.
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "simulate"

import types
from typing import Union

import itertools
import numpy as np
from .make_code import make_random_ldpc_parity_check_matrix
from .utils import make_random_state
from simulate_rs import Hqc128
from enum import Enum
import logging
from ldpc import bp_decoder

logger = logging.getLogger(__name__)

# Flipped or not
class FlipStatus(Enum):
    UNFLIPPED = 0
    FLIPPED = 1

    def toggled(self):
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


def next_failure_block(rng: np.random.RandomState, HQC, priv, pt, ct, block_status):
    """
    Finds next block that if flipped, will result in decoding failure.

    Assumes input ciphertext 'ct' decodes successfully.
    """
    N = HQC.params("N")
    N1 = HQC.params("N1")
    N2 = HQC.params("N2")
    if True:
        # Verify starting assumption
        assert hqc_decoding_oracle(HQC, ct, priv, pt)
    for block in rng.choice(N1, N1, replace=False):
        if block_status[block]["status"] == FlipStatus.UNFLIPPED:
            logger.info(f"Flipping outer block {block}")
            # flip block 'block' of v in ct
            ct = toggle_outer_block(ct, block, N, N2)
            block_status[block]["status"] = FlipStatus.FLIPPED
            # Determine decryption failure or not
            if not hqc_decoding_oracle(HQC, ct, priv, pt):
                block_status[block]["result"] = IfFlipResult.FAILURE
                logger.info(f"Decoding Failure by flipping block {block}")
                return (block, ct)

    return None


def reset_full_block_flips(HQC, ct, block_status):
    """
    Resets all fully flipped blocks where 'result' is IfFlipResult.UNKNOWN
    """
    N = HQC.params("N")
    N2 = HQC.params("N2")
    for (block, bs) in enumerate(block_status):
        if bs["status"] == FlipStatus.FLIPPED and bs["result"] == IfFlipResult.UNKNOWN:
            logger.debug(f"Unflipping block {block}")
            ct = toggle_outer_block(ct, block, N, N2)
            bs["status"] = FlipStatus.UNFLIPPED
    return ct


def flip_single_bit(ct, block, bit, N, N2):
    """
    Correctly calculates the position of a single bit in v part of ct. And flips it.
    """
    return toggle_bits_in_v(ct, [block * N2 + bit], N)


def next_success_bit(rng: np.random.RandomState, HQC, priv, pt, ct, bit_status, block):
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
        assert not hqc_decoding_oracle(HQC, ct, priv, pt)

    available_bits = [
        i
        for (i, b) in (enumerate(bit_status))
        if b["result"] == IfFlipResult.UNKNOWN and b["status"] == FlipStatus.UNFLIPPED
    ]
    frac = 10
    totry = (len(available_bits)+frac-1)//frac
    for bit in rng.choice(available_bits, totry, replace=False):
        ct = flip_single_bit(ct, block, bit, N, N2)
        bit_status[bit]["status"] = bit_status[bit]["status"].toggled()
        if hqc_decoding_oracle(HQC, ct, priv, pt):
            bit_status[bit]["result"] = IfFlipResult.SUCCESS
            logger.info(
                f"Decoding success by flipping bit {bit} in block {block} check = 1"
            )
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
        if hqc_decoding_oracle(HQC, ct, priv, pt):
            bit_status[bit]["result"] = IfFlipResult.SUCCESS
            logger.info(
                f"Decoding success by unflipping bit {bit} in block {block} check = 0"
            )
            return (bit, bit_status[bit]["status"], ct)
        else:
            logger.debug(f"Unflipped bit {bit} of outer block {block}")

    return None


def next_failure_bit(rng: np.random.RandomState, HQC, priv, pt, ct, bit_status, block):
    """
    Finds next bit that if toggled, will result in decoding failure.

    Assumes input ciphertext 'ct' succeeds to decodes.

    Returns None if ct still succeed to decode after all available bits have been exhausted.
    Returns tuple of (bit nr., bit status, ciphertext)
    """
    N = HQC.params("N")
    N2 = HQC.params("N2")
    if True:
        # Verify starting assumption
        assert hqc_decoding_oracle(HQC, ct, priv, pt)

    available_bits = [
        i
        for (i, b) in (enumerate(bit_status))
        if b["result"] == IfFlipResult.UNKNOWN and b["status"] == FlipStatus.FLIPPED
    ]
    frac = 10
    totry = (len(available_bits)+frac-1)//frac
    for bit in rng.choice(available_bits, totry, replace=False):
        ct = flip_single_bit(ct, block, bit, N, N2)
        bit_status[bit]["status"] = bit_status[bit]["status"].toggled()
        if not hqc_decoding_oracle(HQC, ct, priv, pt):
            bit_status[bit]["result"] = IfFlipResult.FAILURE
            logger.info(
                f"Decoding failure by unflipping bit {bit} in block {block} check = 1"
            )
            return (bit, bit_status[bit]["status"], ct)
        else:
            logger.debug(f"Unflipped bit {bit} of outer block {block}")

    available_bits = [
        i
        for (i, b) in (enumerate(bit_status))
        if b["result"] == IfFlipResult.UNKNOWN and b["status"] == FlipStatus.UNFLIPPED
    ]
    for bit in rng.choice(available_bits, len(available_bits), replace=False):
        ct = flip_single_bit(ct, block, bit, N, N2)
        bit_status[bit]["status"] = bit_status[bit]["status"].toggled()
        if not hqc_decoding_oracle(HQC, ct, priv, pt):
            bit_status[bit]["result"] = IfFlipResult.FAILURE
            logger.info(
                f"Decoding failure by flipping bit {bit} in block {block} check = 0"
            )
            return (bit, bit_status[bit]["status"], ct)
        else:
            logger.debug(f"Flipped bit {bit} of outer block {block}")

    return None


def decode(Hin: np.array, N: int, checks: list, y_sparse):
    """
    Uses parity check matrix 'H' to decodes the message
    [0_0, 0_1, 0_2, ..., 0_N | 1_0, 1_1, ..., 1_R],
    where N is the length of the decoded message and
    R is the number of check equations given by 'H'.
    'y' provides the actual message to compare the
    decoding against.
    """
    R = Hin.shape[0]

    H = np.concatenate((Hin, np.identity(R)), axis=1)
    logger.debug(f"decode, N: {N}, R: {R}, Hin: {Hin.shape}, H: {H.shape}")

    # Create empty "message" to be decoded
    assumed_zero = np.full(N, 0.00371, dtype=np.float64)
    # failed_checks = np.full(R, 0.9942, dtype=np.float64)
    failed_checks = np.full(R, 0.0, dtype=np.float64)
    channel_probs = np.concatenate((assumed_zero, failed_checks))

    # Creating the decoder
    bpd = bp_decoder(
        H,
        max_iter=N,
        bp_method="product_sum",
        channel_probs=channel_probs,
    )

    logger.info(f"Attempting decode with {R} checks.")

    msg = np.concatenate((np.zeros(N, dtype=int), np.array(checks, dtype=int)))

    decoded = bpd.decode(msg)
    decoded_msg_sparse = []
    decoded_checks_sparse = []

    for (i, x) in enumerate(decoded[:N]):
        if x:
            s = str(i)
            if i in y_sparse:
                s += "*"
            decoded_msg_sparse.append(s)
    for (i, x) in enumerate(decoded[N:]):
        if x != checks[i]:
            s = str(N + i)
            decoded_checks_sparse.append(s)

    logger.info(
        f"Decoded with {R} checks, made {len(decoded_msg_sparse)} "
        f"flips to recover y: {decoded_msg_sparse} and found "
        f"{len(decoded_checks_sparse)} measurement errors: {decoded_checks_sparse}"
    )

    unequal = 0
    yiter = iter(y_sparse)
    yii = next(yiter)
    for (i, yip) in enumerate(decoded):
        yi = yip in y_sparse
        if yi or yip:
            logger.debug(f"y'[{i}]: {yip}, y[{i}]: {yi}")
            unequal |= yi != yip
    return not bool(unequal)


num_decodes = 0


def hqc_decoding_oracle(HQC, ct, priv, pt):
    """
    Returns true if ct results in decoding success, by checking result against pt

    This is dynamically attached as a method on the HQC object
    """
    global num_decodes
    num_decodes += 1
    (pt_prime, _, _, _, _) = HQC.decode_intermediates(ct, priv)
    return pt == pt_prime


def simulate_hqc_idealized_oracle(rng: np.random.RandomState, decode_every: int):
    """
    Main function for simulating HQC attack
    """
    logger.info("Selecting Hqc128 as current HQC version!")
    HQC = Hqc128()

    # Attach a decoding oracle to the HQC object
    global num_decodes
    num_decodes = 0

    N = HQC.params("N")
    N1 = HQC.params("N1")
    N2 = HQC.params("N2")
    weight = 20
    logger.info(f"Params N: {N}, N1: {N1}, N2: {N2}, weight: {weight}")

    # construct/load key-pair to crack
    logger.info(
        "Creating random HQC keypair (randomness does not depend on provided seed)!"
    )
    (pub, priv) = HQC.keypair()  # Randomness does not depend on 'rng'
    (_, y) = HQC.secrets_from_key(priv)

    logger.info(f"y: {sorted(y)}")

    #Keep trying new plain texts, until we manage to decode and find y
    while True:
        # generate / select plain text message
        pt = search_distinguishable_plaintext(HQC, rng)

        # Generate a parity check matrix (H)
        logger.info(f"Create random (L/M)DPC parity check of size {N}!")
        Hgen = make_random_ldpc_parity_check_matrix(N, weight, rng)
        H = None

        # encapsulate with r_2 and e to all-zero. Set r_1 to h_0
        r1_sparse = [i for (i, x) in enumerate(Hgen[0][0:N]) if x != 0]
        assert weight == len(r1_sparse)
        logger.debug(f"r1_sparse: {r1_sparse}")
        (ct, ss) = HQC.encaps_with_plaintext_and_r1(pub, pt, r1_sparse)
        logger.debug(f"Encaps result: {len(ss)} // {len(ct)}")

        block_status = [
            {"status": FlipStatus.UNFLIPPED, "result": IfFlipResult.UNKNOWN}
            for _ in range(N1)
        ]
        R = 0
        checks = []
        while True:
            # Select outer blocks that: if flipped back results in decoding success.
            ret = next_failure_block(rng, HQC, priv, pt, ct, block_status)
            if ret is None:
                #restart with an new plaintext
                continue
            (current_block, ct) = ret
            current_block_status = block_status[current_block]

            # for each such block find bits that: if flipped back, results in decoding success
            bit_status = [
                {"status": FlipStatus.FLIPPED, "result": IfFlipResult.UNKNOWN}
                for _ in range(N2)
            ]
            current_block_status["bits"] = bit_status
            current_bit = 0
            while True:
                ret = next_success_bit(rng, HQC, priv, pt, ct, bit_status, current_block)
                if ret is None:
                    # No more bits to flip in this block
                    # Make sure this block can be decoded
                    ct = flip_single_bit(ct, current_block, current_bit, N, N2)
                    current_block_status["status"] = FlipStatus.UNFLIPPED
                    bit_status[current_bit]["status"] = FlipStatus.UNFLIPPED
                    # Verify assumption of decoding success after bit flip
                    assert hqc_decoding_oracle(HQC, ct, priv, pt)
                    break

                (current_bit, current_bit_status, ct) = ret
                bit_n = current_block * N2 + current_bit
                # Add row from Hgen to H, to correspond to the parity check equation
                # given by the current bit
                H = np.vstack([H, Hgen[bit_n]]) if H is not None else Hgen[bit_n]
                R += 1
                checks.append(0 if current_bit_status == FlipStatus.UNFLIPPED else 1)

                if R % decode_every == 0 and R != 0:
                    unsatisfied = checks.count(1)
                    logger.info(
                        f"{num_decodes} decapsulation calls performed so-far, "
                        f"{unsatisfied} unsatisfied checks, out of total {len(checks)}."
                    )
                    if decode(H, N, checks, y):
                        logger.info(f"Successfully decoded y")
                        break

                ret = next_failure_bit(rng, HQC, priv, pt, ct, bit_status, current_block)
                if ret is None:
                    # No more bits to flip in this block
                    current_block_status["status"] = FlipStatus.UNFLIPPED
                    # Verify assumption of decoding success after bit flip
                    assert hqc_decoding_oracle(HQC, ct, priv, pt)
                    break

                (current_bit, current_bit_status, ct) = ret
                bit_n = current_block * N2 + current_bit
                # Add row from Hgen to H, to correspond to the parity check equation
                # given by the current bit
                H = np.vstack([H, Hgen[bit_n]]) if H is not None else Hgen[bit_n]
                R += 1
                checks.append(0 if current_bit_status == FlipStatus.FLIPPED else 1)

                if R % decode_every == 0 and R != 0:
                    unsatisfied = checks.count(1)
                    logger.info(
                        f"{num_decodes} decapsulation calls performed so-far, "
                        f"{unsatisfied} unsatisfied checks, out of total {len(checks)}."
                    )
                    if decode(H, N, checks, y):
                        logger.info(f"Successfully decoded y")
                        break

            # Reset all full-block flips that provide no information
            ct = reset_full_block_flips(HQC, ct, block_status)

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
        bits = np.unpackbits(eprime, bitorder="little")
        indices = [i for (i, x) in enumerate(bits) if x]

        for (yi, ei) in zip(yyj, indices):
            if yi != ei:
                return False

    return True


if __name__ == "__main__":
    while test_hqc_encaps_with_plaintext_and_r1():
        pass
    print("Failure")

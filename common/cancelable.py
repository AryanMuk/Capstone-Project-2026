from __future__ import annotations

import math

import numpy as np


def binarize(x: np.ndarray, signs: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """x: (..., d); signs broadcastable to x; proj: (k, d). Returns int8 +/-1 of shape (..., k)."""
    z = (x * signs) @ proj.T
    return np.where(z >= 0, 1, -1).astype(np.int8)


def pack_code(code: np.ndarray) -> bytes:
    return np.packbits(code > 0).tobytes()


def unpack_code(data: bytes, bits: int) -> np.ndarray:
    arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:bits]
    if len(arr) != bits:
        raise ValueError(f"expected {bits} bits, got {len(arr)}")
    return np.where(arr == 1, 1, -1).astype(np.int8)


def score_from_agreement(mean_product: np.ndarray | float) -> np.ndarray | float:
    """mean_product = mean(code_a * code_b) in [-1, 1] -> cosine-scale score cos(pi * HD/k)."""
    hd = (1.0 - np.asarray(mean_product, dtype=np.float64)) / 2.0
    return np.cos(math.pi * hd)


def code_score(code_a: np.ndarray, code_b: np.ndarray) -> float:
    return float(score_from_agreement(np.mean(code_a.astype(np.float32) * code_b)))


def reconstruct(code: np.ndarray, signs: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """Best linear guess of the embedding from a code, available ONLY to someone holding the
    key material (signs + projection). Used by the key-compromise simulation to measure how
    much of the embedding the code leaks."""
    x_hat = signs.astype(np.float32) * (proj.T @ code.astype(np.float32))
    n = np.linalg.norm(x_hat)
    return x_hat / n if n > 0 else x_hat

from __future__ import annotations

import numpy as np

from common import cancelable
from common.config import SCHEMES
from common.crypto import (KeyStore, TemplateIntegrityError, aes_gcm_decrypt, aes_gcm_encrypt,
                           ecies_decrypt, ecies_encrypt)

FLOAT_SCHEMES = ("plaintext", "aes", "ecies")


def aad_for(user_id: str, scheme: str, version: int) -> bytes:
    """Binds a ciphertext to its owner, scheme and key version."""
    return f"{user_id}|{scheme}|v{int(version)}".encode()


def protect(scheme: str, ks: KeyStore, user_id: str, version: int,
            vec: np.ndarray, bits: int) -> bytes:
    """Turn a unit embedding into the blob stored in the database."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    vec = np.asarray(vec, dtype="<f4")
    aad = aad_for(user_id, scheme, version)
    if scheme == "plaintext":
        return vec.tobytes()
    if scheme == "aes":
        return aes_gcm_encrypt(ks.aes_key(user_id, version), vec.tobytes(), aad)
    if scheme == "ecies":
        return ecies_encrypt(ks.ecies_public_bytes(user_id, version), vec.tobytes(), aad)
    signs = ks.cancelable_signs(user_id, version, vec.shape[-1])
    code = cancelable.binarize(vec, signs, ks.cancelable_projection(bits, vec.shape[-1]))
    return aes_gcm_encrypt(ks.cancelable_wrap_key(user_id, version),
                           cancelable.pack_code(code), aad)


def open_blob(scheme: str, ks: KeyStore, user_id: str, version: int,
              blob: bytes, dim: int, bits: int) -> np.ndarray:
    """Decrypt a stored blob. Raises TemplateIntegrityError if it was tampered with or moved."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    aad = aad_for(user_id, scheme, version)
    if scheme == "cancelable":
        packed = aes_gcm_decrypt(ks.cancelable_wrap_key(user_id, version), blob, aad)
        try:
            return cancelable.unpack_code(packed, bits)
        except ValueError as exc:
            raise TemplateIntegrityError(str(exc)) from exc
    if scheme == "plaintext":
        raw = blob
    elif scheme == "aes":
        raw = aes_gcm_decrypt(ks.aes_key(user_id, version), blob, aad)
    else:
        raw = ecies_decrypt(ks.ecies_private(user_id, version), blob, aad)
    if len(raw) != dim * 4:
        raise TemplateIntegrityError(f"expected {dim * 4} bytes, got {len(raw)}")
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)

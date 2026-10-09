from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from common.config import Paths, get_paths

NONCE_LEN = 12
TAG_LEN = 16
X25519_LEN = 32
MASTER_ENV = "CAPSTONE_MASTER_KEY_HEX"


class TemplateIntegrityError(Exception):
    """A stored template failed authentication/parsing (tampered, wrong key, or wrong owner)."""


def hkdf_bytes(secret: bytes, info: bytes, length: int = 32) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info).derive(secret)


def _raw_public(pub: X25519PublicKey) -> bytes:
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


# ----------------------------------------------------------------------------- AES-GCM
def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(NONCE_LEN)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def aes_gcm_decrypt(key: bytes, blob: bytes, aad: bytes) -> bytes:
    if len(blob) < NONCE_LEN + TAG_LEN:
        raise TemplateIntegrityError("blob too short")
    try:
        return AESGCM(key).decrypt(blob[:NONCE_LEN], blob[NONCE_LEN:], aad)
    except InvalidTag as exc:
        raise TemplateIntegrityError("authentication tag mismatch") from exc


# ----------------------------------------------------------------------------- ECIES
def _ecies_key(shared: bytes, eph_pub: bytes, recipient_pub: bytes) -> bytes:
    return hkdf_bytes(shared, b"capstone|v1|ecies-kdf|" + eph_pub + recipient_pub, 32)


def ecies_encrypt(recipient_pub: bytes, plaintext: bytes, aad: bytes) -> bytes:
    """blob = ephemeral_public(32) || nonce(12) || ciphertext || tag(16)."""
    eph = X25519PrivateKey.generate()
    eph_pub = _raw_public(eph.public_key())
    shared = eph.exchange(X25519PublicKey.from_public_bytes(recipient_pub))
    key = _ecies_key(shared, eph_pub, recipient_pub)
    return eph_pub + aes_gcm_encrypt(key, plaintext, aad)


def ecies_decrypt(private: X25519PrivateKey, blob: bytes, aad: bytes) -> bytes:
    if len(blob) < X25519_LEN + NONCE_LEN + TAG_LEN:
        raise TemplateIntegrityError("blob too short")
    eph_pub, rest = blob[:X25519_LEN], blob[X25519_LEN:]
    try:
        shared = private.exchange(X25519PublicKey.from_public_bytes(eph_pub))
    except ValueError as exc:
        raise TemplateIntegrityError("invalid ephemeral public key") from exc
    key = _ecies_key(shared, eph_pub, _raw_public(private.public_key()))
    return aes_gcm_decrypt(key, rest, aad)


# ----------------------------------------------------------------------------- KeyStore
def _info(kind: str, user_id: str, version: int) -> bytes:
    return f"capstone|v1|{kind}|{user_id}|{int(version)}".encode()


class KeyStore:
    """All keys are derived on demand from `master`; the database never holds key material."""

    def __init__(self, master: bytes):
        if len(master) != 32:
            raise ValueError("master key must be 32 bytes")
        self._master = master
        self._proj_cache: Dict[Tuple[int, int], np.ndarray] = {}

    # -- construction -------------------------------------------------------------------
    @classmethod
    def load_or_create(cls, paths: Optional[Paths] = None, create: bool = True) -> "KeyStore":
        """Master key from $CAPSTONE_MASTER_KEY_HEX, else <keys>/master.key (created if absent)."""
        env = os.environ.get(MASTER_ENV)
        if env:
            return cls(bytes.fromhex(env.strip()))
        paths = paths or get_paths()
        key_file = paths.keys / "master.key"
        if key_file.exists():
            return cls(bytes.fromhex(key_file.read_text().strip()))
        if not create:
            raise FileNotFoundError(
                f"No master key: set {MASTER_ENV} or copy master.key to {key_file}")
        paths.keys.mkdir(parents=True, exist_ok=True)
        master = os.urandom(32)
        key_file.write_text(master.hex())
        try:
            os.chmod(key_file, 0o600)  
        except OSError:
            pass
        return cls(master)

    def fingerprint(self) -> str:
        """Short non-reversible id to compare keys across machines without revealing them."""
        return hashlib.sha256(b"fp|" + self._master).hexdigest()[:12]

    # -- per-user symmetric / asymmetric keys -------------------------------------------
    def aes_key(self, user_id: str, version: int) -> bytes:
        return hkdf_bytes(self._master, _info("aes", user_id, version), 32)

    def ecies_private(self, user_id: str, version: int) -> X25519PrivateKey:
        seed = hkdf_bytes(self._master, _info("ecies", user_id, version), 32)
        return X25519PrivateKey.from_private_bytes(seed)

    def ecies_public_bytes(self, user_id: str, version: int) -> bytes:
        return _raw_public(self.ecies_private(user_id, version).public_key())

    # -- cancelable-template key material -----------------------------------------------
    def cancelable_wrap_key(self, user_id: str, version: int) -> bytes:
        return hkdf_bytes(self._master, _info("cancel-wrap", user_id, version), 32)

    def cancelable_signs(self, user_id: str, version: int, dim: int) -> np.ndarray:
        """Per-user +/-1 vector of length `dim` (int8). Changing `version` revokes the template."""
        raw = hkdf_bytes(self._master, _info("cancel-signs", user_id, version), dim // 8)
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:dim]
        return np.where(bits == 1, 1, -1).astype(np.int8)

    def cancelable_projection(self, bits: int, dim: int) -> np.ndarray:
        """Shared secret +/-1 projection matrix (bits x dim, float32), derived via SHAKE-256."""
        cache_key = (bits, dim)
        if cache_key not in self._proj_cache:
            seed = hkdf_bytes(self._master, f"capstone|v1|cancel-proj|{bits}x{dim}".encode(), 32)
            stream = hashlib.shake_256(seed).digest(bits * dim // 8)
            raw = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))[: bits * dim]
            self._proj_cache[cache_key] = np.where(raw == 1, 1.0, -1.0).astype(
                np.float32).reshape(bits, dim)
        return self._proj_cache[cache_key]

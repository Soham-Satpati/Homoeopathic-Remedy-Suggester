"""Encryption primitives for patient data at rest.

Design (envelope encryption):

* One random 256-bit **data key** (DEK) encrypts every patient record with AES-256-GCM.
* The DEK is never stored in the clear. It is stored *wrapped* (AES-GCM encrypted) once per
  user, under a key derived from that user's password with scrypt, and once under the
  clinic's **recovery key** (high-entropy, shown once at setup).
* Logging in = unwrapping the DEK. A wrong password fails GCM authentication, so there is
  no separate password hash to attack offline beyond the scrypt-hardened wrap.
* Each record's ciphertext is bound to its table and row id (AES-GCM associated data), so
  sealed blobs cannot be swapped between rows or tables.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

KEY_LEN = 32
NONCE_LEN = 12
BLOB_VERSION = 1

# scrypt cost: ~64 MiB, ~0.1–0.3 s on a laptop. Stored with each wrap so it can be raised later.
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 16, 8, 1

_RECOVERY_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O/1/I confusion


class WrongSecret(Exception):
    """Password or recovery key does not unlock the data key."""


def new_data_key() -> bytes:
    return AESGCM.generate_key(bit_length=256)


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


# ── wrapping the data key ────────────────────────────────────────────────────
def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return Scrypt(salt=salt, length=KEY_LEN, n=n, r=r, p=p).derive(password.encode("utf-8"))


def wrap_with_password(dek: bytes, password: str) -> dict:
    salt = os.urandom(16)
    kek = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    nonce = os.urandom(NONCE_LEN)
    ct = AESGCM(kek).encrypt(nonce, dek, b"horus-dek-v1")
    return {"kdf": "scrypt", "n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P,
            "salt": _b64(salt), "nonce": _b64(nonce), "ct": _b64(ct)}


def unwrap_with_password(wrap: dict, password: str) -> bytes:
    kek = _scrypt(password, _unb64(wrap["salt"]), wrap["n"], wrap["r"], wrap["p"])
    try:
        return AESGCM(kek).decrypt(_unb64(wrap["nonce"]), _unb64(wrap["ct"]), b"horus-dek-v1")
    except InvalidTag as e:
        raise WrongSecret("wrong password") from e


def new_recovery_key() -> str:
    """32 characters from a 32-letter alphabet = 160 bits, shown as 8 groups of 4."""
    raw = "".join(secrets.choice(_RECOVERY_ALPHABET) for _ in range(32))
    return "-".join(raw[i:i + 4] for i in range(0, 32, 4))


def normalise_recovery_key(text: str) -> str:
    """Ignore case, spaces and dashes (the alphabet has no 0/O/1/I to confuse)."""
    return "".join(c for c in (text or "").upper() if c.isalnum())


def _recovery_kek(recovery_key: str, salt: bytes) -> bytes:
    material = normalise_recovery_key(recovery_key).encode("ascii")
    return HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=salt,
                info=b"horus-recovery-v1").derive(material)


def wrap_with_recovery(dek: bytes, recovery_key: str) -> dict:
    salt = os.urandom(16)
    nonce = os.urandom(NONCE_LEN)
    ct = AESGCM(_recovery_kek(recovery_key, salt)).encrypt(nonce, dek, b"horus-dek-v1")
    return {"kdf": "hkdf", "salt": _b64(salt), "nonce": _b64(nonce), "ct": _b64(ct)}


def unwrap_with_recovery(wrap: dict, recovery_key: str) -> bytes:
    try:
        return AESGCM(_recovery_kek(recovery_key, _unb64(wrap["salt"]))).decrypt(
            _unb64(wrap["nonce"]), _unb64(wrap["ct"]), b"horus-dek-v1")
    except InvalidTag as e:
        raise WrongSecret("wrong recovery key") from e


# ── sealing records ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Sealer:
    """Encrypts/decrypts JSON-able values with the data key."""

    dek: bytes

    def seal(self, value, context: str) -> bytes:
        nonce = os.urandom(NONCE_LEN)
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return bytes([BLOB_VERSION]) + nonce + AESGCM(self.dek).encrypt(nonce, data, context.encode())

    def open(self, blob: bytes, context: str):
        if not blob or blob[0] != BLOB_VERSION:
            raise ValueError("unknown record format")
        nonce, ct = blob[1:1 + NONCE_LEN], blob[1 + NONCE_LEN:]
        try:
            return json.loads(AESGCM(self.dek).decrypt(nonce, ct, context.encode()).decode("utf-8"))
        except InvalidTag as e:
            raise ValueError(f"record {context} failed integrity check") from e

    def subkey(self, purpose: str) -> bytes:
        return HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=None,
                    info=f"horus-{purpose}-v1".encode()).derive(self.dek)

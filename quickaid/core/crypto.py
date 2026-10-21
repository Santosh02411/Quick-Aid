"""Encryption at rest for sensitive health text (chat messages, interview answers,
full stored analyses).

Values are encrypted with Fernet (AES-128-CBC + HMAC-SHA256, authenticated) before
they reach SQLite, so a copied database file or a backup does not expose people's
health conversations.

Key source, in order:
  1. CHAT_ENCRYPTION_KEY  - recommended for production; any long random string.
  2. SECRET_KEY           - derived with SHA-256 (what a default setup uses).
  3. data/.data_key       - generated and stored with owner-only permissions, so
                            development still works if neither is set.

Rotating the key (for example after a leak) without losing data:
  1. Put the OLD key material in CHAT_ENCRYPTION_KEY_PREVIOUS (comma-separated if several), and the NEW one in CHAT_ENCRYPTION_KEY.
  2. Restart. New writes use the new key; old rows still decrypt because every previous key is tried.
  3. Run `python -m quickaid.core.rekey` to re-encrypt every stored value under the new key (it backs up first).
  4. Remove CHAT_ENCRYPTION_KEY_PREVIOUS. The old key can no longer read anything.
If a key is changed WITHOUT doing this, old encrypted rows can no longer be read
(they show as unavailable rather than crashing the app).
"""

import base64
import hashlib
import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from quickaid import PROJECT_ROOT
from quickaid.core.logging_config import get_logger

logger = get_logger('crypto')

PREFIX = 'enc:v1:'
UNAVAILABLE = '[unavailable - encryption key changed]'
_PLACEHOLDER_PREFIX = 'change_this'


def _derive(material: str) -> bytes:
    digest = hashlib.sha256(b'quickaid-data-v1:' + material.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(digest)


def _key_file() -> str:
    default_dir = os.path.dirname(os.getenv('DATABASE_PATH') or str(PROJECT_ROOT / 'data' / 'quickaid.db'))
    return os.path.join(default_dir or str(PROJECT_ROOT / 'data'), '.data_key')


def _material() -> str:
    explicit = os.getenv('CHAT_ENCRYPTION_KEY', '').strip()
    if explicit:
        return explicit
    secret = os.getenv('SECRET_KEY', '').strip()
    if secret and not secret.startswith(_PLACEHOLDER_PREFIX):
        return secret
    path = _key_file()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            existing = f.read().strip()
            if existing:
                return existing
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    generated = base64.urlsafe_b64encode(os.urandom(32)).decode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(generated)
    logger.warning("No CHAT_ENCRYPTION_KEY or SECRET_KEY set; generated a local data key at %s. "
                   "Set CHAT_ENCRYPTION_KEY in production.", path)
    return generated


def _previous_materials() -> tuple:
    """Older key material that must still decrypt (CHAT_ENCRYPTION_KEY_PREVIOUS, comma-separated)."""
    raw = os.getenv('CHAT_ENCRYPTION_KEY_PREVIOUS', '')
    return tuple(m.strip() for m in raw.split(',') if m.strip())


@lru_cache(maxsize=16)
def _multi_fernet(materials: tuple) -> MultiFernet:
    """Encrypts with the FIRST key; decrypts with any of them."""
    return MultiFernet([Fernet(_derive(m)) for m in materials])


def _ring() -> MultiFernet:
    primary = _material()
    return _multi_fernet((primary,) + tuple(m for m in _previous_materials() if m != primary))


def encrypt(plaintext: str) -> str:
    """Encrypt text for storage. Empty/None stays empty."""
    if plaintext is None or plaintext == '':
        return plaintext or ''
    return PREFIX + _ring().encrypt(plaintext.encode('utf-8')).decode('ascii')


def decrypt(stored: str) -> str:
    """Decrypt a stored value. Legacy plaintext passes through unchanged."""
    if not stored:
        return stored or ''
    if not stored.startswith(PREFIX):
        return stored
    try:
        return _ring().decrypt(stored[len(PREFIX):].encode('ascii')).decode('utf-8')
    except InvalidToken:
        logger.error("Could not decrypt a stored value (key changed or data corrupted)")
        return UNAVAILABLE


def decrypts_with_primary(stored: str) -> bool:
    """True when `stored` is already under the current primary key (used by the re-key command to skip finished rows)."""
    try:
        Fernet(_derive(_material())).decrypt(stored[len(PREFIX):].encode('ascii'))
        return True
    except (InvalidToken, ValueError):
        return False

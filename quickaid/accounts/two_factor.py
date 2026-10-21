"""
TOTP (RFC 6238) two-factor authentication for Quick Aid.

Self-contained (no external 2FA service) - works with any standard
authenticator app (Google Authenticator, Authy, 1Password, etc).
"""

import base64
import io

import pyotp
import qrcode

ISSUER = "Quick Aid"


def generate_secret() -> str:
    """A fresh base32 TOTP secret, to be confirmed before it's enabled."""
    return pyotp.random_base32()


def provisioning_uri(secret: str, account_email: str) -> str:
    """otpauth:// URI that authenticator apps understand, encoded into the QR code."""
    return pyotp.TOTP(secret).provisioning_uri(name=account_email, issuer_name=ISSUER)


def verify_code(secret: str, code: str) -> bool:
    """Check a 6-digit code against the secret, allowing 1 step of clock drift."""
    if not secret or not code:
        return False
    code = code.strip().replace(' ', '')
    if not code.isdigit():
        return False
    try:
        return pyotp.TOTP(secret).verify(code, valid_window=1)
    except Exception:
        return False


def verify_code_step(secret: str, code: str, now: float = None):
    """Like verify_code, but returns the matching 30-second time-step (or None), so a code can be accepted only ONCE."""
    import hmac
    import time
    if not secret or not code:
        return None
    code = code.strip().replace(' ', '')
    if not code.isdigit() or len(code) != 6:
        return None
    totp = pyotp.TOTP(secret)
    current = int((now if now is not None else time.time()) // totp.interval)
    for step in (current, current - 1, current + 1):                  # this window first, then one step of clock drift either way
        if hmac.compare_digest(totp.at(step * totp.interval), code):
            return step
    return None


def qr_code_data_uri(uri: str) -> str:
    """Render the provisioning URI as a PNG QR code, inlined as a data: URI."""
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'

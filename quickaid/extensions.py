"""Flask extensions, created once and attached to each app by `init_app` (the standard factory pattern)."""

import os

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import CSRFProtect

from quickaid.accounts.auth import login_manager  # noqa: F401  (re-exported so everything imports extensions from one place)

csrf = CSRFProtect()


def user_or_ip() -> str:
    """Rate-limit key for signed-in endpoints: the PERSON, not the address. One user cannot multiply their allowance by changing network, and
    several people behind one office or university NAT do not eat each other's allowance. Anonymous requests fall back to the address."""
    from flask_login import current_user
    if current_user and current_user.is_authenticated:
        return f'user:{current_user.id}'
    return get_remote_address()

RATELIMIT_STORAGE_URI = os.getenv('RATELIMIT_STORAGE_URI', 'memory://')

# Protects the AI-backed endpoints from abuse/quota burn and login/register from brute force.
# In-memory storage is per-process: for several workers/instances use Redis (RATELIMIT_STORAGE_URI=redis://...).
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"], storage_uri=RATELIMIT_STORAGE_URI)

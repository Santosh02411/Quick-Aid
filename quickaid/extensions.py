"""Flask extensions, created once and attached to each app by `init_app` (the standard factory pattern)."""

import os

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import CSRFProtect

from quickaid.accounts.auth import login_manager  # noqa: F401  (re-exported so everything imports extensions from one place)

csrf = CSRFProtect()

RATELIMIT_STORAGE_URI = os.getenv('RATELIMIT_STORAGE_URI', 'memory://')

# Protects the AI-backed endpoints from abuse/quota burn and login/register from brute force.
# In-memory storage is per-process: for several workers/instances use Redis (RATELIMIT_STORAGE_URI=redis://...).
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"], storage_uri=RATELIMIT_STORAGE_URI)

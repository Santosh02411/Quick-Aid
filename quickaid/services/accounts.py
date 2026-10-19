"""Account helpers shared by the auth views (kept free of request handling where possible)."""

import re
import secrets

from flask import request

from quickaid import config
from quickaid.core import database as db


def is_valid_email(email) -> bool:
    return bool(email) and '@' in email and '.' in email.split('@')[-1]


def build_absolute_url(path: str) -> str:
    """A link to put in an email. Uses APP_BASE_URL when set (right behind a proxy/CDN), else the request's own host."""
    return f"{config.app_base_url() or request.host_url.rstrip('/')}{path}"


def generate_username_from_email(email: str) -> str:
    """A valid, available username for a first-time OAuth sign-in, e.g. 'jane.doe+test@x.com' -> 'jane_doe'."""
    base = re.sub(r'[^A-Za-z0-9_]', '_', email.split('@')[0]).strip('_')[:24] or 'user'
    if len(base) < 3:
        base = (base + '_user')[:24]
    candidate = base
    while db.username_taken(candidate):
        candidate = f"{base}_{secrets.token_hex(3)}"[:30]
    return candidate

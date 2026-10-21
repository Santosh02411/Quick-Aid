"""Request-bound session handling: starting a session safely, expiring it, and describing the client without storing who they are."""

import ipaddress
import time
from typing import Dict

from flask import request, session
from flask_login import current_user, login_user, logout_user

from quickaid import config
from quickaid.accounts.auth import User
from quickaid.core import database as db
from quickaid.core.logging_config import get_logger

logger = get_logger('sessions')


def client_id() -> str:
    """The full client address, used ONLY as input to a one-way hash for lockout counters (never stored or logged as-is)."""
    return request.remote_addr or ''


def network() -> str:
    """The client's address, truncated so it identifies a network, not a person (IPv4 /24, IPv6 /48)."""
    raw = request.remote_addr or ''
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return ''
    prefix = 24 if ip.version == 4 else 48
    return str(ipaddress.ip_network(f'{raw}/{prefix}', strict=False).network_address) + f'/{prefix}'


def event(user_id, name: str, detail: str = '') -> None:
    """Record a security-relevant event for the account owner to see (network is truncated; no user agent, no body, no secrets)."""
    try:
        db.add_security_event(user_id, name, detail, network())
    except Exception:
        logger.warning("Could not record a security event", exc_info=True)


def start_session(user_row: Dict, method: str) -> None:
    """Sign a user in. The old session is discarded first (protects against session fixation) and stamped with the account's
    current session version, so a later password change or "sign out everywhere" invalidates it."""
    session.clear()
    now = time.time()
    session['sv'] = user_row.get('session_version', 0)
    session['auth_at'] = now
    session['seen_at'] = now
    session.permanent = True
    login_user(User.from_row(user_row))
    event(user_row['id'], 'login', method)


def end_session() -> None:
    logout_user()
    session.clear()


def refresh_after_credential_change(user_id: int) -> None:
    """After the person changes their own password or 2FA, every OTHER device is signed out (the version was bumped), but this one stays in."""
    row = db.get_user_by_id(user_id)
    now = time.time()
    session['sv'] = row['session_version']
    session['seen_at'] = now


def enforce_limits() -> None:
    """Idle and absolute session lifetime. Called before every request."""
    if not current_user.is_authenticated:
        return
    now = time.time()
    seen, started = session.get('seen_at'), session.get('auth_at')
    if (seen and now - seen > config.session_idle_minutes() * 60) or (started and now - started > config.session_absolute_hours() * 3600):
        uid = int(current_user.id)
        end_session()
        event(uid, 'session_expired')
        return
    session['seen_at'] = now
    session.setdefault('auth_at', now)

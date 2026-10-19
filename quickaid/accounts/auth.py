"""
Flask-Login wiring for Quick Aid.

A thin User wrapper around the `users` table row (see database.py) plus the
LoginManager that Flask-Login needs to reload a user from the session cookie
on each request.
"""

from flask_login import LoginManager, UserMixin
from quickaid.core import database as db
from quickaid.core.logging_config import get_logger

logger = get_logger('auth')

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to use Quick Aid.'
login_manager.login_message_category = 'info'


class User(UserMixin):
    @property
    def is_admin(self) -> bool:
        """Admins are listed in ADMIN_USERNAMES (comma-separated, case-insensitive). Checked on every request."""
        import os
        names = {n.strip().lower() for n in os.getenv('ADMIN_USERNAMES', '').split(',') if n.strip()}
        return bool(self.username) and self.username.lower() in names

    def __init__(self, id, username, email, email_verified=False, has_password=True,
                 totp_enabled=False, oauth_provider=None, region=None):
        # Flask-Login expects get_id() to return a string.
        self.id = str(id)
        self.username = username
        self.email = email
        self.email_verified = bool(email_verified)
        self.has_password = bool(has_password)
        self.totp_enabled = bool(totp_enabled)
        self.oauth_provider = oauth_provider
        self.region = region

    @staticmethod
    def from_row(row):
        if not row:
            return None
        return User(
            row['id'], row['username'], row['email'],
            email_verified=row.get('email_verified', 0) if isinstance(row, dict) else row['email_verified'],
            has_password=row.get('has_password', 1) if isinstance(row, dict) else row['has_password'],
            totp_enabled=row.get('totp_enabled', 0) if isinstance(row, dict) else row['totp_enabled'],
            oauth_provider=row.get('oauth_provider') if isinstance(row, dict) else row['oauth_provider'],
            region=(row.get('region') if isinstance(row, dict) else row['region']) if ('region' in (row if isinstance(row, dict) else row.keys())) else None,
        )


@login_manager.user_loader
def load_user(user_id: str):
    try:
        row = db.get_user_by_id(int(user_id))
    except (TypeError, ValueError):
        return None
    return User.from_row(row)


@login_manager.unauthorized_handler
def unauthorized():
    from flask import redirect, request, url_for
    from quickaid.api.errors import wants_json
    from quickaid.api.responses import fail
    logger.info("Unauthorized access attempt: %s %s", request.method, request.path)
    # API/JSON endpoints get a 401 they can handle programmatically; regular page loads are redirected to the login page.
    if wants_json():
        return fail('Authentication required. Please log in.', 401, 'unauthenticated')
    return redirect(url_for('auth.login', next=request.path))

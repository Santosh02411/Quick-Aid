"""
Flask-Login wiring for Quick Aid.

A thin User wrapper around the `users` table row (see database.py) plus the
LoginManager that Flask-Login needs to reload a user from the session cookie
on each request.
"""

from flask_login import LoginManager, UserMixin
import database as db
from logging_config import get_logger

logger = get_logger('auth')

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to use Quick Aid.'
login_manager.login_message_category = 'info'


class User(UserMixin):
    def __init__(self, id, username, email, email_verified=False, has_password=True,
                 totp_enabled=False, oauth_provider=None):
        # Flask-Login expects get_id() to return a string.
        self.id = str(id)
        self.username = username
        self.email = email
        self.email_verified = bool(email_verified)
        self.has_password = bool(has_password)
        self.totp_enabled = bool(totp_enabled)
        self.oauth_provider = oauth_provider

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
    from flask import request, jsonify, redirect, url_for
    logger.info("Unauthorized access attempt: %s %s", request.method, request.path)
    # API/JSON endpoints get a 401 they can handle programmatically;
    # regular page loads get redirected to the login page.
    if request.path.startswith('/api/') or request.path in ('/upload', '/analyze_symptoms'):
        return jsonify({'error': 'Authentication required. Please log in.'}), 401
    return redirect(url_for('login', next=request.path))

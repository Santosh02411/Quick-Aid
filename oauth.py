"""
Optional Google OAuth / SSO login for Quick Aid.

Like the Gemini API key, this degrades gracefully: if GOOGLE_CLIENT_ID and
GOOGLE_CLIENT_SECRET aren't set, Google sign-in is simply not registered
and the "Continue with Google" button is hidden in the templates - the
rest of the app (username/password auth) works exactly the same either
way. To enable it, create an OAuth 2.0 Client ID at
https://console.cloud.google.com/apis/credentials with an authorized
redirect URI of <your-domain>/login/google/callback, then set:

    GOOGLE_CLIENT_ID=...
    GOOGLE_CLIENT_SECRET=...
"""

import os

from authlib.integrations.flask_client import OAuth

from logging_config import get_logger

logger = get_logger('oauth')

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')

oauth = OAuth()


def is_google_oauth_configured() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


def init_oauth(app):
    oauth.init_app(app)
    if is_google_oauth_configured():
        oauth.register(
            name='google',
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
            client_kwargs={'scope': 'openid email profile'},
        )
        logger.info("Google OAuth configured and registered.")
    else:
        logger.info("Google OAuth not configured (GOOGLE_CLIENT_ID/SECRET unset) - Google sign-in disabled.")
    return oauth

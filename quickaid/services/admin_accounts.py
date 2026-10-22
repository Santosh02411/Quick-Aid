"""Admin operations on OTHER people's accounts - shared by the terminal tool (quickaid.core.accounts_cli) and the
web admin dashboard (blueprints/admin.py), so the two can never behave differently and every action is logged and
notified exactly once, however it was triggered.

Deliberately narrow: nothing here reads or returns a person's symptoms, photos, chat messages or notes. An admin
can see that an account exists and manage its sign-in state, never the health information inside it.
"""

from typing import Dict, List, Optional

from quickaid.accounts.mailer import send_email
from quickaid.core import database as db
from quickaid.services import security


class AdminActionError(Exception):
    """A requested action could not be carried out (account not found, confirmation mismatch, nothing to do)."""


def find_account(identifier: str) -> Optional[Dict]:
    return db.get_user_by_username(identifier) or db.get_user_by_email((identifier or '').lower())


def require_account(identifier: str) -> Dict:
    user = find_account(identifier)
    if not user:
        raise AdminActionError(f"No account found for {identifier!r} (checked both username and email).")
    return user


def list_accounts(query: Optional[str] = None) -> List[Dict]:
    return db.list_all_users(query)


def account_detail(identifier: str) -> Dict:
    user = require_account(identifier)
    detail = {k: user[k] for k in ('id', 'username', 'email', 'email_verified', 'totp_enabled', 'has_password', 'created_at')}
    detail['recovery_codes_remaining'] = db.recovery_codes_remaining(user['id']) if user['totp_enabled'] else None
    detail['recent_security_events'] = db.list_security_events(user['id'], limit=15)
    return detail


def _log(user_id: int, event: str, by: str) -> None:
    db.add_security_event(user_id, event, f'by an administrator ({by})' if by else 'by an administrator')


def reset_two_factor(identifier: str, by: str = '') -> Dict:
    """Disables 2FA and clears recovery codes, signs out every device. The primary recovery path for a locked-out
    account - the person must set 2FA up again from scratch afterwards if they still want it."""
    user = require_account(identifier)
    if not user['totp_enabled']:
        raise AdminActionError(f"{user['username']} does not have two-factor authentication enabled - nothing to reset.")
    db.disable_totp(user['id'])
    _log(user['id'], '2fa_disabled', by)
    security.notify(user, 'Two-factor authentication was reset on your Quick Aid account',
                    'An administrator reset two-factor authentication on your account. If you did not request '
                    'this, contact whoever administers this Quick Aid deployment immediately.')
    return {'username': user['username']}


def force_logout(identifier: str, by: str = '') -> Dict:
    user = require_account(identifier)
    db.bump_session_version(user['id'])
    _log(user['id'], 'signed_out_everywhere', by)
    return {'username': user['username']}


def start_password_reset(identifier: str, build_link, by: str = '') -> Dict:
    """Creates a reset token and emails the link (built by the caller, which knows the real site address - this
    module never guesses or constructs a URL itself). Returns the raw token too, for a CLI operator who is not
    sending email and just needs the path to hand to the person directly."""
    user = require_account(identifier)
    token = db.create_token(user['id'], 'password_reset', 3600)
    link = build_link(token)
    send_email(user['email'], 'Reset your Quick Aid password',
              f"Hi {user['username']},\n\nAn administrator started a password reset for your Quick Aid account. "
              f"If this was you, click the link below (valid for 1 hour, works once):\n\n{link}\n\n"
              "If you did not expect this, contact whoever administers this Quick Aid deployment.")
    _log(user['id'], 'password_reset_requested', by)
    return {'username': user['username'], 'token': token, 'link': link}


def delete_account(identifier: str, confirm: Optional[str], by: str = '') -> Dict:
    """Permanently deletes the account and everything it owns, via the same secure-deletion path a person deleting
    their own account uses (physical overwrite, deletion tombstone). `confirm` must repeat `identifier` exactly -
    a deliberate typo guard, since this cannot be undone."""
    user = require_account(identifier)
    if confirm != identifier:
        raise AdminActionError(f"Confirmation did not match. Pass {identifier!r} again, exactly, to actually delete this account.")
    username = user['username']
    db.delete_user(user['id'])
    return {'username': username}

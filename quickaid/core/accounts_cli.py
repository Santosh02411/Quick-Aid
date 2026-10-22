"""Account administration from the terminal - for an operator with shell/database access, not exposed on any page
a signed-up user could reach. This is the RIGHT place to recover a locked-out account (lost authenticator, no
saved recovery codes): it needs someone who can already run commands on the server, which is a much stronger bar
than "knows a username and password", so it cannot become a new way to take over someone else's account.

Command line:
    python -m quickaid.core.accounts_cli list [--admins-only] [--query TEXT]
    python -m quickaid.core.accounts_cli show USERNAME_OR_EMAIL
    python -m quickaid.core.accounts_cli reset-2fa USERNAME_OR_EMAIL
    python -m quickaid.core.accounts_cli force-logout USERNAME_OR_EMAIL
    python -m quickaid.core.accounts_cli send-password-reset USERNAME_OR_EMAIL [--site https://your-domain.example]
    python -m quickaid.core.accounts_cli delete USERNAME_OR_EMAIL --confirm USERNAME_OR_EMAIL

Every action here (and the identical one from the web admin dashboard - both call the same code, see
quickaid/services/admin_accounts.py) is logged to that user's own "recent security activity" (visible to them on
/account) and, where there is somewhere to send it, emails the account owner - so an admin action is never
invisible to the person it affects.

The web dashboard (/admin/users, for an account listed in ADMIN_USERNAMES) wraps the same operations for day-to-day
use without shell access; see docs/OPERATIONS.md. Use this CLI when you only have a terminal, or when the account
doing the fixing is itself the one that is locked out.
"""

import argparse
import sys

from quickaid.services import admin_accounts as svc


def _by() -> str:
    import getpass
    try:
        return getpass.getuser()
    except Exception:
        return 'cli'


def list_users(admins_only: bool = False, query: str = None) -> None:
    import os
    admin_names = {n.strip().lower() for n in os.getenv('ADMIN_USERNAMES', '').split(',') if n.strip()}
    rows = svc.list_accounts(query)
    print(f"{'username':24s} {'email':32s} {'verified':9s} {'2fa':5s} {'admin':6s} created")
    for u in rows:
        if admins_only and u['username'].lower() not in admin_names:
            continue
        print(f"{u['username']:24s} {u['email']:32s} {('yes' if u['email_verified'] else 'no'):9s} "
              f"{('on' if u['totp_enabled'] else 'off'):5s} {('yes' if u['username'].lower() in admin_names else ''):6s} {u['created_at'][:10]}")
    if not admin_names:
        print("\nnote: ADMIN_USERNAMES is not set, so no account currently has web admin access - see docs/OPERATIONS.md.")


def show_user(identifier: str) -> None:
    d = svc.account_detail(identifier)
    print(f"id:              {d['id']}")
    print(f"username:        {d['username']}")
    print(f"email:           {d['email']}  (verified: {'yes' if d['email_verified'] else 'no'})")
    print(f"2FA enabled:     {'yes' if d['totp_enabled'] else 'no'}")
    if d['totp_enabled']:
        print(f"recovery codes:  {d['recovery_codes_remaining']} remaining")
    print(f"created:         {d['created_at']}")
    print("\nrecent security activity:")
    for e in d['recent_security_events']:
        print(f"  {e['created_at'][:16].replace('T', ' ')}  {e['event']:28s} {e.get('detail', '')}")


def reset_2fa(identifier: str) -> None:
    r = svc.reset_two_factor(identifier, by=_by())
    print(f"2FA disabled and all sessions signed out for {r['username']}. They can sign in with just their password now,")
    print("and can set up 2FA again from Account > Two-factor authentication once they are back in.")


def force_logout(identifier: str) -> None:
    r = svc.force_logout(identifier, by=_by())
    print(f"every signed-in session for {r['username']} has been invalidated.")


def send_password_reset(identifier: str, site: str = None) -> None:
    if not site:
        from quickaid.core import database as db
        user = svc.require_account(identifier)
        token = db.create_token(user['id'], 'password_reset', 3600)
        print(f"Password reset token created for {user['username']} (valid 1 hour, single use):")
        print(f"  /reset-password/{token}")
        print("Prepend your deployment's real address, or pass --site https://your-domain.example to also email it.")
        return
    r = svc.start_password_reset(identifier, lambda token: f"{site.rstrip('/')}/reset-password/{token}", by=_by())
    print(f"emailed a password-reset link to {r['username']}.")


def delete_account(identifier: str, confirm: str) -> None:
    r = svc.delete_account(identifier, confirm, by=_by())
    print(f"account {r['username']!r} and everything it owned has been permanently deleted.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    lp = sub.add_parser('list', help='list accounts')
    lp.add_argument('--admins-only', action='store_true')
    lp.add_argument('--query', help='filter by a substring of username or email')

    sp = sub.add_parser('show', help="show one account's details and recent security activity")
    sp.add_argument('identifier')

    rp = sub.add_parser('reset-2fa', help='disable 2FA on a locked-out account so the person can sign back in with just their password')
    rp.add_argument('identifier')

    fp = sub.add_parser('force-logout', help='sign the account out of every device')
    fp.add_argument('identifier')

    pp = sub.add_parser('send-password-reset', help='create a password-reset link, optionally email it')
    pp.add_argument('identifier')
    pp.add_argument('--site', help='your real site address (e.g. https://your-domain.example) - emails the link if given, otherwise just prints the path')

    dp = sub.add_parser('delete', help='permanently delete an account and everything it owns')
    dp.add_argument('identifier')
    dp.add_argument('--confirm', help='must exactly repeat the identifier argument')

    args = ap.parse_args(argv)
    try:
        if args.cmd == 'list':
            list_users(args.admins_only, args.query)
        elif args.cmd == 'show':
            show_user(args.identifier)
        elif args.cmd == 'reset-2fa':
            reset_2fa(args.identifier)
        elif args.cmd == 'force-logout':
            force_logout(args.identifier)
        elif args.cmd == 'send-password-reset':
            send_password_reset(args.identifier, args.site)
        elif args.cmd == 'delete':
            delete_account(args.identifier, args.confirm)
    except svc.AdminActionError as e:
        print('error:', e, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

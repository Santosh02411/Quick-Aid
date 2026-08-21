"""
Minimal email sending for Quick Aid (password reset + email verification).

No SMTP is configured out of the box - rather than silently discarding
mail or crashing, send_email() logs the message when SMTP_* isn't set,
so the reset/verification flows are fully exercisable in dev and tests
without a real mail server. Set SMTP_HOST/SMTP_USERNAME/SMTP_PASSWORD in
.env to actually deliver mail in production.
"""

import os
import smtplib
import ssl
from email.message import EmailMessage

from logging_config import get_logger

logger = get_logger('mailer')

SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', 'True').lower() in ('1', 'true', 'yes')
MAIL_FROM = os.getenv('MAIL_FROM', 'Quick Aid <no-reply@quickaid.local>')


def is_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD)


def send_email(to: str, subject: str, body: str) -> bool:
    """
    Send a plain-text email. Returns True if it was handed off to the SMTP
    server successfully. If SMTP isn't configured, logs the content instead
    (at WARNING level, since a real deployment should have this set up) and
    returns False - callers should NOT treat a False return as a hard
    failure to show the user, since the account action itself (token
    creation, etc.) already succeeded.
    """
    if not is_configured():
        logger.warning(
            "SMTP not configured - email NOT sent. To=%s Subject=%s\n%s",
            to, subject, body
        )
        return False

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = MAIL_FROM
    msg['To'] = to
    msg.set_content(body)

    try:
        if SMTP_USE_TLS:
            context = ssl.create_default_context()
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.starttls(context=context)
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
        else:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=10) as server:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
        logger.info("Email sent to=%s subject=%s", to, subject)
        return True
    except Exception:
        logger.error("Failed to send email to=%s subject=%s", to, subject, exc_info=True)
        return False

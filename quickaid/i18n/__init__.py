"""Languages: English, Hindi, Marathi, Kannada.

* The current language is a per-request context variable (default English), set by the web layer from, in order: an explicit choice
  (?lang= or the language menu), the signed-in person's saved preference, a cookie, the browser's Accept-Language.
* `t(key)` looks a string up in the catalog and falls back to English, never to an error, so a missing translation can only ever show English.
* Safety-critical emergency content lives in `i18n/emergency.py` and is wired into the triage engine.

TRANSLATION STATUS: written by an AI assistant and not yet reviewed by native-speaking clinicians. The app says so on screen in every
non-English language (see `notice()`), and docs/TRANSLATION_REVIEW.md lists everything a reviewer must check.
"""

import contextvars
import re
from typing import Dict, Optional

SUPPORTED = ('en', 'hi', 'mr', 'kn')
NATIVE_NAMES = {'en': 'English', 'hi': 'हिन्दी', 'mr': 'मराठी', 'kn': 'ಕನ್ನಡ'}
ENGLISH_NAMES = {'en': 'English', 'hi': 'Hindi', 'mr': 'Marathi', 'kn': 'Kannada'}
COOKIE_NAME = 'qa_lang'

_current: contextvars.ContextVar = contextvars.ContextVar('qa_language', default='en')


def current() -> str:
    return _current.get()


def set_language(code: str):
    return _current.set(code if code in SUPPORTED else 'en')


def reset(token) -> None:
    _current.reset(token)


def normalize_code(value: Optional[str]) -> Optional[str]:
    """'hi', 'HI', 'hi-IN', 'kn_IN' -> 'hi' / 'kn'. Anything unsupported -> None."""
    if not value or not isinstance(value, str):
        return None
    primary = re.split(r'[-_]', value.strip().lower())[0]
    return primary if primary in SUPPORTED else None


def from_accept_language(header: Optional[str]) -> Optional[str]:
    """The best supported language in an Accept-Language header, honouring q-values."""
    if not header:
        return None
    ranked = []
    for part in header.split(',')[:20]:
        bits = part.strip().split(';')
        code = normalize_code(bits[0])
        q = 1.0
        for b in bits[1:]:
            b = b.strip()
            if b.startswith('q='):
                try:
                    q = float(b[2:])
                except ValueError:
                    q = 0.0
        if code and q > 0:
            ranked.append((q, code))
    return max(ranked, key=lambda x: x[0])[1] if ranked else None


def negotiate(explicit: Optional[str] = None, user_pref: Optional[str] = None, cookie: Optional[str] = None, accept: Optional[str] = None) -> str:
    for candidate in (explicit, user_pref, cookie):
        code = normalize_code(candidate)
        if code:
            return code
    return from_accept_language(accept) or 'en'


def t(key: str, lang: Optional[str] = None, **fmt) -> str:
    from quickaid.i18n.catalog import STRINGS
    entry = STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(lang or current()) or entry['en']
    return text.format(**fmt) if fmt else text


def js_strings(lang: Optional[str] = None) -> Dict[str, str]:
    """The `js.*` strings (result headings, buttons) the browser script needs, in the current language."""
    from quickaid.i18n.catalog import STRINGS
    lang = lang or current()
    return {k[3:]: (v.get(lang) or v['en']) for k, v in STRINGS.items() if k.startswith('js.')}


def needs_review_notice(lang: Optional[str] = None) -> bool:
    return (lang or current()) != 'en'

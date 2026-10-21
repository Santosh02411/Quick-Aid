"""Tell the AI model which language to write in. Safety-relevant values stay in English so the rules that check the answer still work."""

from quickaid import i18n

_NOTE = (
    "\n\nLANGUAGE: Write every sentence the person will read (summary, explanations, advice, warning signs, questions and answer options) in {name}, "
    "using simple everyday words and the medical terms people actually use in {name}. Keep these EXACTLY as specified, untranslated: JSON keys, enum values "
    "(for example urgency levels), emergency phone numbers, and drug names. If you are unsure of a medical term in {name}, give the common English term in brackets. "
    "Never make advice less cautious because of the language."
)


def note(lang=None) -> str:
    lang = lang or i18n.current()
    return '' if lang == 'en' else _NOTE.format(name=i18n.ENGLISH_NAMES[lang])

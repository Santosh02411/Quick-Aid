"""Versioning for every prompt the app sends to a model.

A prompt change can improve or silently break safety, so each prompt gets a short content
hash. Runs, telemetry and the dashboard all record these hashes: if a prompt is edited, its
hash changes automatically (even if nobody remembers to bump a label), and results can be
compared "before vs after" the edit. Set PROMPT_VERSION_LABEL in .env for a human-readable tag.
"""

import hashlib
import os
from functools import lru_cache
from typing import Dict


def _h(text: str) -> str:
    return hashlib.sha256((text or '').encode('utf-8')).hexdigest()[:8]


def prompt_texts() -> Dict[str, str]:
    """The current text of every prompt (imported lazily to avoid import cycles)."""
    from quickaid.ai import conversation, interview, medical_analyzer, symptom_checker
    return {
        'symptom': symptom_checker.SYSTEM_INSTRUCTION,
        'image': medical_analyzer.SYSTEM_INSTRUCTION,
        'interview_plan': interview.PLAN_INSTRUCTION,
        'interview_assess': interview.ASSESS_INSTRUCTION,
        'chat': conversation.CHAT_SYSTEM_INSTRUCTION,
    }


def prompt_version(name: str) -> str:
    """Short hash of ONE prompt (changes whenever its text changes)."""
    return _h(prompt_texts().get(name, ''))


@lru_cache(maxsize=1)
def _cached_label() -> str:
    return os.getenv('PROMPT_VERSION_LABEL', '').strip()


def fingerprint() -> Dict:
    """{'fingerprint': hash of all prompts, 'parts': {name: hash}, 'label': optional human tag}."""
    parts = {name: _h(text) for name, text in prompt_texts().items()}
    combined = _h('|'.join(f'{k}:{v}' for k, v in sorted(parts.items())))
    return {'fingerprint': combined, 'parts': parts, 'label': os.getenv('PROMPT_VERSION_LABEL', '').strip()}


def diff(a: Dict, b: Dict) -> Dict:
    """Which prompts differ between two fingerprints (as stored on two runs)."""
    pa, pb = (a or {}).get('parts', {}), (b or {}).get('parts', {})
    names = sorted(set(pa) | set(pb))
    return {n: {'a': pa.get(n), 'b': pb.get(n)} for n in names if pa.get(n) != pb.get(n)}

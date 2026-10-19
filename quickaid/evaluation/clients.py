"""Model clients for evaluation. They all expose `.models.generate_content(**kwargs)`, like the real Gemini client.

  MockClient       synthetic answers, NO network. Modes: 'weak' (always "low urgency"), 'flaky' (random API errors and
                   invalid JSON around a rule-based answer), 'fixture' (exact answers you supply). For automated tests.
  RecordingClient  wraps a REAL client and appends every request/response (text, token usage, latency, errors) to a JSONL file.
  ReplayClient     answers from such a JSONL file with no network. Lets you re-score REAL model outputs after changing the
                   safety rules, schemas or metrics, reproducibly and for free. Recorded latency and tokens are replayed too.

Mock answers are labelled synthetic everywhere they surface. They test the plumbing and the safety net, never the model.
"""

import hashlib
import json
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


class FakeResponse:
    """Looks like a Gemini response for everything the app reads."""

    def __init__(self, text: str, input_tokens: int = 0, output_tokens: int = 0, recorded_latency_ms: Optional[float] = None):
        self.text = text
        self.parsed = None
        self.usage_metadata = SimpleNamespace(prompt_token_count=input_tokens, candidates_token_count=output_tokens, thoughts_token_count=0)
        self.recorded_latency_ms = recorded_latency_ms


def _schema_name(kwargs: Dict) -> str:
    cfg = kwargs.get('config')
    schema = getattr(cfg, 'response_schema', None)
    return getattr(schema, '__name__', '') or ''


def _flatten(contents: Any) -> str:
    """Request contents -> stable text (strings, chat Content objects, PIL images by pixel hash)."""
    if isinstance(contents, str):
        return contents
    parts: List[str] = []
    for item in (contents if isinstance(contents, (list, tuple)) else [contents]):
        if isinstance(item, str):
            parts.append(item)
        elif hasattr(item, 'parts'):
            parts.append(f"{getattr(item, 'role', '')}:" + ''.join(getattr(p, 'text', '') or '' for p in item.parts))
        elif hasattr(item, 'tobytes'):
            parts.append('image:' + hashlib.sha256(item.tobytes()).hexdigest()[:16])
        else:
            parts.append(str(item))
    return '\n'.join(parts)


def request_key(kwargs: Dict) -> str:
    cfg = kwargs.get('config')
    system = getattr(cfg, 'system_instruction', '') or ''
    blob = json.dumps({'model': kwargs.get('model'), 'system': hashlib.sha256(str(system).encode()).hexdigest()[:12],
                       'schema': _schema_name(kwargs), 'contents': _flatten(kwargs.get('contents'))}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

def _user_text(contents: Any) -> str:
    m = re.search(r'"""\n(.*?)\n"""', _flatten(contents), re.S)
    return m.group(1) if m else _flatten(contents)


def rules_answer(text: str) -> Dict:
    """A synthetic 'model' answer built from the app's own keyword rules (so mock runs have realistic structure)."""
    from quickaid.ai.symptom_checker import SymptomChecker
    ck = SymptomChecker()
    ck.use_gemini = False
    syms = ck._extract_symptoms(text.lower())
    analysis = ck._analyze_symptom_combination(syms)
    expl = [{'name': c, 'likelihood': 'more likely' if i == 0 else 'possible', 'why': '', 'supported_by': syms}
            for i, c in enumerate(analysis['conditions'][:3])]
    return {'summary': 'Synthetic mock answer.', 'reported_symptoms': syms, 'possible_explanations': expl,
            'urgency_level': analysis['urgency'], 'emergency_alert': False, 'recommendations': ['Rest', 'See a doctor if worse'],
            'warning_signs': ['Trouble breathing'], 'information_gaps': ['Age and duration were not given']}


class _MockModels:
    def __init__(self, mode: str, seed: int, fixtures: Optional[Dict[str, Any]], error_rate: float, invalid_rate: float):
        self.mode, self.rng, self.fixtures = mode, random.Random(seed), fixtures or {}
        self.error_rate, self.invalid_rate = error_rate, invalid_rate
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        schema = _schema_name(kwargs)
        text_in = _user_text(kwargs.get('contents'))
        if self.mode == 'fixture':
            for needle, answer in self.fixtures.items():
                if needle.lower() in text_in.lower():
                    out = answer if isinstance(answer, str) else json.dumps(answer)
                    return FakeResponse(out, _tokens(text_in), _tokens(out))
            raise KeyError('no fixture matched this request')
        if self.mode == 'flaky':
            r = self.rng.random()
            if r < self.error_rate:
                raise RuntimeError('503 Service Unavailable (mock)')
            if r < self.error_rate + self.invalid_rate:
                return FakeResponse('Sorry, I cannot answer that {not valid json', _tokens(text_in), 12)
        if schema == 'ImageAssessment':
            data = {'urgency': 'low', 'confidence': 'medium', 'image_quality': 'good', 'recommendations': ['Keep the area clean']}
        elif schema == 'InterviewPlan':
            data = {'detected_symptoms': [], 'clarifying': [], 'associated': {'text': 'Anything else?', 'type': 'multi', 'options': [
                {'label': 'Fever', 'flag': 'none'}, {'label': 'Cough', 'flag': 'none'}, {'label': 'Rash', 'flag': 'none'}]}}
        elif self.mode == 'weak':
            data = {'summary': 'This is probably nothing serious.', 'urgency_level': 'low', 'emergency_alert': False,
                    'possible_explanations': [{'name': 'common cold', 'likelihood': 'more likely', 'why': '', 'supported_by': []}],
                    'recommendations': ['Rest and drink fluids']}
        elif schema:
            data = rules_answer(text_in)
        else:   # chat (free text)
            return FakeResponse('This is a synthetic mock reply. Please see a doctor if you are worried.', _tokens(text_in), 20)
        out = json.dumps(data)
        return FakeResponse(out, _tokens(text_in) + 400, _tokens(out))


class MockClient:
    """`mode`: 'weak' | 'flaky' | 'fixture' | 'rules'."""

    def __init__(self, mode: str = 'rules', seed: int = 7, fixtures: Optional[Dict[str, Any]] = None,
                 error_rate: float = 0.15, invalid_rate: float = 0.10):
        self.mode = mode
        self.models = _MockModels(mode, seed, fixtures, error_rate, invalid_rate)


# ---------------------------------------------------------------------------
# Record / replay of REAL model outputs
# ---------------------------------------------------------------------------

class _RecordingModels:
    def __init__(self, real, path: Path):
        self.real, self.path = real, path

    def generate_content(self, **kwargs):
        key, t0 = request_key(kwargs), time.perf_counter()
        try:
            resp = self.real.models.generate_content(**kwargs)
        except Exception as e:
            self._write({'key': key, 'error': type(e).__name__, 'latency_ms': round((time.perf_counter() - t0) * 1000, 1)})
            raise
        meta = getattr(resp, 'usage_metadata', None)
        num = lambda v: int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0  # noqa: E731
        text = getattr(resp, 'text', None)
        parsed = getattr(resp, 'parsed', None)
        if text is None and hasattr(parsed, 'model_dump_json'):
            text = parsed.model_dump_json()
        self._write({'key': key, 'model': kwargs.get('model'), 'text': text, 'latency_ms': round((time.perf_counter() - t0) * 1000, 1),
                     'input_tokens': num(getattr(meta, 'prompt_token_count', 0)),
                     'output_tokens': num(getattr(meta, 'candidates_token_count', 0)) + num(getattr(meta, 'thoughts_token_count', 0))})
        return resp

    def _write(self, rec: Dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


class RecordingClient:
    def __init__(self, real_client, path):
        self.models = _RecordingModels(real_client, Path(path))


class ReplayMiss(RuntimeError):
    """The recording has no answer for this request (counts as an API failure in the run)."""


class _ReplayModels:
    def __init__(self, path: Path):
        self.records: Dict[str, Dict] = {}
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                rec = json.loads(line)
                self.records.setdefault(rec['key'], rec)     # first recording wins (stable)

    def generate_content(self, **kwargs):
        rec = self.records.get(request_key(kwargs))
        if rec is None:
            raise ReplayMiss('no recorded response for this request')
        if rec.get('error'):
            raise RuntimeError(f"recorded API error: {rec['error']}")
        return FakeResponse(rec.get('text') or '', rec.get('input_tokens', 0), rec.get('output_tokens', 0), rec.get('latency_ms'))


class ReplayClient:
    def __init__(self, path):
        self.models = _ReplayModels(Path(path))
        self.path = Path(path)

    @property
    def recorded_models(self):
        return sorted({r.get('model') for r in self.models.records.values() if r.get('model')})

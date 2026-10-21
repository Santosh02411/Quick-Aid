"""Telemetry (latency, tokens, validity, failures, cost) and prompt versioning. Mocked clients only: no network."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from quickaid.ai import prompt_registry, telemetry
from quickaid.ai.symptom_checker import SymptomChecker
from quickaid.evaluation.clients import FakeResponse, MockClient


def usage(i=1000, o=500):
    return SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=i, candidates_token_count=o, thoughts_token_count=0))


@pytest.fixture(autouse=True)
def no_prices(monkeypatch):
    monkeypatch.delenv('GEMINI_PRICE_INPUT_PER_M', raising=False)
    monkeypatch.delenv('GEMINI_PRICE_OUTPUT_PER_M', raising=False)


def checker_with(client):
    c = SymptomChecker()
    c.use_gemini, c.client = True, client
    return c


class TestCallRecords:
    def test_successful_call_records_latency_tokens_and_validity(self):
        with telemetry.Collector() as col:
            with telemetry.call('symptom', 'm1', 'symptom') as c:
                c.api_done(usage(1000, 500))
                c.valid(True)
        r = col.calls[0]
        assert r['ok'] and r['valid'] is True and r['input_tokens'] == 1000 and r['output_tokens'] == 500
        assert r['latency_ms'] >= 0 and r['model'] == 'm1' and r['error_type'] is None and r['prompt_version']

    def test_api_failure_is_recorded_and_the_exception_still_propagates(self):
        with telemetry.Collector() as col:
            with pytest.raises(TimeoutError):
                with telemetry.call('symptom', 'm1'):
                    raise TimeoutError('slow')
        r = col.calls[0]
        assert r['ok'] is False and r['error_type'] == 'TimeoutError' and r['valid'] is None

    def test_invalid_output_is_not_an_api_failure(self):
        with telemetry.Collector() as col:
            with telemetry.call('symptom', 'm1') as c:
                c.api_done(usage())
                c.valid(False)
        r = col.calls[0]
        assert r['ok'] is True and r['valid'] is False and r['error_type'] == 'invalid_output'

    def test_exception_after_the_api_answered_counts_as_invalid_not_api_failure(self):
        with telemetry.Collector() as col:
            with pytest.raises(ValueError):
                with telemetry.call('symptom', 'm1') as c:
                    c.api_done(usage())
                    raise ValueError('bad schema')
        assert col.calls[0]['ok'] is True and col.calls[0]['valid'] is False

    def test_mock_objects_never_pollute_token_counts(self):
        with telemetry.Collector() as col:
            with telemetry.call('chat', 'm1') as c:
                c.api_done(MagicMock())           # MagicMock attributes are not numbers
        assert col.calls[0]['input_tokens'] == 0 and col.calls[0]['output_tokens'] == 0

    def test_replayed_responses_report_their_recorded_latency(self):
        with telemetry.Collector() as col:
            with telemetry.call('symptom', 'm1') as c:
                c.api_done(FakeResponse('{}', 10, 5, recorded_latency_ms=1234.5))
        assert col.calls[0]['latency_ms'] == 1234.5

    def test_no_user_text_is_ever_in_a_record(self):
        c = checker_with(MockClient('rules'))
        with telemetry.Collector() as col:
            c.analyze_symptoms('my very private rash on my left shoulder')
        blob = json.dumps(col.calls + col.analyses)
        assert 'private' not in blob and 'shoulder' not in blob


class TestCost:
    def test_cost_is_computed_only_when_prices_are_configured(self, monkeypatch):
        assert telemetry.cost_usd(1000, 500) is None
        monkeypatch.setenv('GEMINI_PRICE_INPUT_PER_M', '1.0')
        monkeypatch.setenv('GEMINI_PRICE_OUTPUT_PER_M', '2.0')
        assert telemetry.cost_usd(1_000_000, 500_000) == pytest.approx(2.0)
        with telemetry.Collector() as col:
            with telemetry.call('symptom', 'm1') as c:
                c.api_done(usage(1000, 500))
        assert col.calls[0]['cost_usd'] == pytest.approx(0.002)

    @pytest.mark.parametrize('a,b', [('x', '1'), ('1', ''), ('', ''), ('-', '-')])
    def test_bad_or_partial_prices_mean_no_cost_not_a_crash(self, monkeypatch, a, b):
        monkeypatch.setenv('GEMINI_PRICE_INPUT_PER_M', a)
        monkeypatch.setenv('GEMINI_PRICE_OUTPUT_PER_M', b)
        assert telemetry.pricing() is None and telemetry.cost_usd(5, 5) is None


class TestAnalysisScope:
    def test_one_analysis_groups_its_calls_and_totals_tokens(self, monkeypatch):
        monkeypatch.setenv('GEMINI_PRICE_INPUT_PER_M', '1')
        monkeypatch.setenv('GEMINI_PRICE_OUTPUT_PER_M', '1')

        @telemetry.analysis_scope('symptom')
        def work():
            for _ in range(2):
                with telemetry.call('symptom', 'm1') as c:
                    c.api_done(usage(1000, 100))
                    c.valid(True)
            return {'analysis_mode': 'ai', 'urgency_level': 'low', 'emergency_alert': {'alert': False}}

        with telemetry.Collector() as col:
            work()
        a = col.analyses[0]
        assert a['calls'] == 2 and a['input_tokens'] == 2000 and a['output_tokens'] == 200 and a['mode'] == 'ai' and a['ai_attempted']
        assert {c['analysis_id'] for c in col.calls} == {a['id']}
        assert a['cost_usd'] == pytest.approx(0.0022)

    def test_fallback_is_flagged_when_the_model_failed_and_basic_mode_answered(self, monkeypatch):
        import quickaid.ai.symptom_checker as sc
        monkeypatch.setattr(sc.time, 'sleep', lambda *_: None)
        broken = MagicMock()
        broken.models.generate_content.side_effect = RuntimeError('503')
        with telemetry.Collector() as col:
            out = checker_with(broken).analyze_symptoms('fever and cough')
        assert out['analysis_mode'] == 'basic'
        assert len(col.calls) == 2 and not any(c['ok'] for c in col.calls)         # both attempts recorded
        assert col.analyses[0]['fallback'] is True and col.analyses[0]['mode'] == 'basic'

    def test_invalid_json_marks_validity_false_and_falls_back(self, monkeypatch):
        import quickaid.ai.symptom_checker as sc
        monkeypatch.setattr(sc.time, 'sleep', lambda *_: None)
        client = MagicMock()
        client.models.generate_content.return_value = FakeResponse('not json {', 50, 5)
        with telemetry.Collector() as col:
            checker_with(client).analyze_symptoms('fever and cough')
        assert all(c['ok'] and c['valid'] is False for c in col.calls)
        assert col.analyses[0]['fallback'] is True

    def test_rules_only_analyses_have_no_calls_and_no_fallback(self):
        c = SymptomChecker()
        c.use_gemini = False
        with telemetry.Collector() as col:
            c.analyze_symptoms('fever and cough')
        a = col.analyses[0]
        assert a['calls'] == 0 and a['ai_attempted'] is False and a['fallback'] is False and a['cost_usd'] is None

    def test_failed_function_still_records_an_error_analysis_and_reraises(self):
        @telemetry.analysis_scope('symptom')
        def boom():
            raise RuntimeError('x')
        with telemetry.Collector() as col:
            with pytest.raises(RuntimeError):
                boom()
        assert col.analyses[0]['error'] is True

    def test_emergency_chat_reply_is_recorded_as_rules_without_calls(self):
        from quickaid.ai.conversation import ConversationService
        s = ConversationService()
        s.use_gemini, s.client = True, MagicMock()
        with telemetry.Collector() as col:
            s.reply(None, '{}', [], 'my father has crushing chest pain', 'IN')
        a = col.analyses[0]
        assert a['kind'] == 'chat' and a['mode'] == 'rules' and a['emergency'] and a['calls'] == 0
        s.client.models.generate_content.assert_not_called()

    def test_image_and_interview_calls_are_instrumented(self, tmp_path, monkeypatch):
        monkeypatch.setenv('INTERVIEW_AI_PLAN', 'true')
        import numpy as np
        from PIL import Image
        from quickaid.ai.interview import InterviewEngine
        from quickaid.ai.medical_analyzer import MedicalAnalyzer
        rng = np.random.default_rng(1)
        path = tmp_path / 'p.png'
        Image.fromarray(np.clip(rng.integers(-25, 25, (300, 300, 1)) + np.array([215, 165, 145]), 0, 255).astype('uint8')).save(path)
        a = MedicalAnalyzer()
        a.use_gemini, a.client = True, MockClient('rules')
        ck = checker_with(MockClient('rules'))
        with telemetry.Collector() as col:
            a.analyze_image(str(path))
            InterviewEngine(ck).start('cough for a few days', 'IN')
        kinds = {c['kind'] for c in col.calls}
        assert {'image', 'interview_plan'} <= kinds and {x['kind'] for x in col.analyses} >= {'image', 'interview_plan'}


class TestSinks:
    def test_a_failing_sink_never_breaks_the_analysis(self):
        class Bad:
            def on_call(self, rec): raise RuntimeError('db down')
            def on_analysis(self, rec): raise RuntimeError('db down')
        bad = Bad()
        telemetry.add_sink(bad)
        try:
            c = SymptomChecker()
            c.use_gemini = False
            assert c.analyze_symptoms('fever and cough')['urgency_level']
        finally:
            telemetry.remove_sink(bad)

    def test_isolated_collector_hides_traffic_from_other_sinks(self):
        seen = telemetry.Collector()
        telemetry.add_sink(seen)
        try:
            with telemetry.Collector(isolate=True) as mine:
                checker_with(MockClient('rules')).analyze_symptoms('fever and cough')
            assert mine.calls and not seen.calls          # evaluation traffic did not reach the "production" sink
            checker_with(MockClient('rules')).analyze_symptoms('fever and cough')
            assert seen.calls                              # and normal operation resumed afterwards
        finally:
            telemetry.remove_sink(seen)

    def test_database_sink_persists_and_retention_purges(self, db_module):
        db_module.init_db()
        sink = telemetry.DbSink()
        telemetry.add_sink(sink)
        try:
            checker_with(MockClient('rules')).analyze_symptoms('fever and cough')
        finally:
            telemetry.remove_sink(sink)
        calls = db_module.fetch_ai_calls('2000-01-01')
        analyses = db_module.fetch_ai_analyses('2000-01-01')
        assert len(calls) == 1 and len(analyses) == 1 and calls[0]['analysis_id'] == analyses[0]['id']
        assert db_module.purge_telemetry(0) == 2 and db_module.fetch_ai_calls('2000-01-01') == []


class TestPromptRegistry:
    def test_every_prompt_has_a_stable_hash(self):
        a, b = prompt_registry.fingerprint(), prompt_registry.fingerprint()
        assert a == b and set(a['parts']) == {'symptom', 'image', 'interview_plan', 'interview_assess', 'chat'}
        assert all(len(h) == 8 for h in a['parts'].values())

    def test_editing_a_prompt_changes_its_hash_and_the_fingerprint(self, monkeypatch):
        import quickaid.ai.symptom_checker as sc
        before = prompt_registry.fingerprint()
        monkeypatch.setattr(sc, 'SYSTEM_INSTRUCTION', sc.SYSTEM_INSTRUCTION + '\nAlways be extra careful.')
        after = prompt_registry.fingerprint()
        assert after['fingerprint'] != before['fingerprint']
        assert prompt_registry.diff(before, after).keys() == {'symptom'}

    def test_label_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv('PROMPT_VERSION_LABEL', 'v2-stricter')
        assert prompt_registry.fingerprint()['label'] == 'v2-stricter'

    def test_calls_record_the_prompt_version_in_force(self):
        with telemetry.Collector() as col:
            checker_with(MockClient('rules')).analyze_symptoms('fever and cough')
        assert col.calls[0]['prompt_version'] == prompt_registry.prompt_version('symptom')

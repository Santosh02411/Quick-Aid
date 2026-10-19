"""Evaluation platform: stats, dataset versioning, versioned runs + comparison, mock/record/replay clients, run builder."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from quickaid.ai import telemetry
from quickaid.ai.symptom_checker import SymptomChecker
from quickaid.evaluation import clients, datasets, runner, runs, stats


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """A private copy of the datasets and an empty runs folder, so tests can edit and save freely."""
    ds = tmp_path / 'datasets'
    shutil.copytree(datasets.DATASET_DIR, ds)
    monkeypatch.setattr(datasets, 'DATASET_DIR', ds)
    monkeypatch.setattr(datasets, 'MANIFEST_PATH', ds / 'MANIFEST.json')
    monkeypatch.setattr(runs, 'RUNS_DIR', tmp_path / 'runs')
    monkeypatch.setattr(runner, 'DATA', ds)
    return tmp_path


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def call(**kw):
    base = {'ts': '2026-10-16T10:00:00', 'kind': 'symptom', 'model': 'm', 'prompt_version': 'aa', 'ok': True, 'valid': True, 'latency_ms': 1000,
            'input_tokens': 100, 'output_tokens': 50, 'cost_usd': 0.001, 'error_type': None}
    base.update(kw)
    return base


class TestStats:
    def test_percentiles(self):
        assert stats.percentile([], 50) is None and stats.percentile([5], 95) == 5
        assert stats.percentile([1, 2, 3, 4, 5], 50) == 3 and stats.percentile([None, 10, 20], 50) == 15

    def test_rates_and_definitions(self):
        calls = [call(), call(valid=False, error_type='invalid_output'), call(ok=False, valid=None, error_type='TimeoutError', latency_ms=10)]
        s = stats.summarize(calls, [])
        assert s['api_failure_rate'] == pytest.approx(1 / 3, abs=1e-3)
        assert s['structured_output_validity'] == 0.5            # only calls that answered AND were checked count
        assert s['invalid_outputs'] == 1 and s['errors'] == {'invalid_output': 1, 'TimeoutError': 1}

    def test_latency_ignores_failed_calls(self):
        s = stats.summarize([call(latency_ms=100), call(latency_ms=300), call(ok=False, valid=None, latency_ms=99999)], [])
        assert s['latency_ms']['p50'] == 200 and s['latency_ms']['max'] == 300

    def test_cost_per_analysis_uses_completed_analyses_only(self):
        an = [{'kind': 'symptom'}, {'kind': 'image'}, {'kind': 'interview_plan'}]
        s = stats.summarize([call(cost_usd=0.004)], an)
        assert s['completed_analyses'] == 2 and s['cost_per_analysis_usd'] == 0.002        # the plan stage is not a finished analysis

    def test_unconfigured_prices_give_none_not_zero(self):
        s = stats.summarize([call(cost_usd=None)], [{'kind': 'symptom'}])
        assert s['cost_per_analysis_usd'] is None and s['cost_configured'] is False

    def test_empty_inputs_are_safe(self):
        s = stats.summarize([], [])
        assert s['calls'] == 0 and s['api_failure_rate'] is None and s['structured_output_validity'] is None and s['daily'] == []

    def test_breakdowns_by_model_and_prompt_version(self):
        s = stats.summarize([call(model='a', prompt_version='p1'), call(model='b', prompt_version='p2', ok=False, valid=None)], [])
        assert set(s['by_model']) == {'a', 'b'} and s['by_prompt_version']['p2']['api_failure_rate'] == 1.0

    def test_fallback_and_safety_rates(self):
        an = [{'kind': 'symptom', 'ai_attempted': True, 'fallback': True, 'safety_raised': True},
              {'kind': 'symptom', 'ai_attempted': True, 'fallback': False}, {'kind': 'symptom', 'ai_attempted': False}]
        s = stats.summarize([], an)
        assert s['fallback_rate'] == 0.5 and s['safety_intervention_rate'] == pytest.approx(1 / 3, abs=1e-3)


# ---------------------------------------------------------------------------
# dataset versioning
# ---------------------------------------------------------------------------

class TestDatasetVersioning:
    def test_shipped_manifest_matches_the_files(self):
        assert datasets.verify() == []

    def test_every_dataset_declares_role_provenance_and_review_status(self):
        for name, meta in datasets.load_manifest()['datasets'].items():
            assert meta['version'].count('.') == 2 and meta['role'] and meta['labelled_by'] and meta['changelog'], name
            assert meta['clinician_reviewed'] in (True, False)

    def test_editing_a_dataset_without_bumping_is_detected(self, sandbox):
        f = datasets.DATASET_DIR / 'symptom_cases.json'
        data = json.loads(f.read_text())
        data['cases'][0]['text'] += ' (edited)'
        f.write_text(json.dumps(data))
        assert any('changed since version' in p for p in datasets.verify())
        assert datasets.describe()[0]['status'] == 'changed-without-bump'

    def test_bump_records_version_hash_count_and_changelog(self, sandbox):
        f = datasets.DATASET_DIR / 'symptom_cases.json'
        data = json.loads(f.read_text())
        data['cases'].append(dict(data['cases'][0], id='ZZ1'))
        f.write_text(json.dumps(data))
        meta = datasets.bump('symptom_cases', 'added one case', 'minor')
        assert meta['version'] == '1.1.0' and meta['cases'] == len(data['cases']) and meta['changelog'][-1]['note'] == 'added one case'
        assert datasets.verify() == []

    def test_version_parts(self, sandbox):
        assert datasets.bump('model_output_cases', 'x', 'patch')['version'] == '1.0.1'
        assert datasets.bump('model_output_cases', 'x', 'major')['version'] == '2.0.0'

    def test_frozen_test_sets_refuse_to_be_bumped(self, sandbox):
        with pytest.raises(PermissionError):
            datasets.bump('symptom_holdout2', 'tuning to the test set')
        assert datasets.bump('symptom_holdout2', 'deliberate', force=True)['version'] == '1.1.0'

    def test_an_unlisted_dataset_file_is_flagged(self, sandbox):
        (datasets.DATASET_DIR / 'sneaky.json').write_text('{"cases": []}')
        assert any('sneaky.json' in p and 'not listed' in p for p in datasets.verify())

    def test_versions_include_hash_and_integrity_flag(self):
        v = datasets.versions(['symptom_cases'])['symptom_cases']
        assert v['version'] and len(v['sha256']) == 64 and v['matches_manifest'] is True

    def test_the_held_out_sets_are_marked_frozen(self):
        ds = datasets.load_manifest()['datasets']
        assert ds['symptom_holdout2']['frozen'] and ds['symptom_holdout']['frozen'] and not ds['symptom_cases']['frozen']


# ---------------------------------------------------------------------------
# runs + comparison
# ---------------------------------------------------------------------------

def rec(run_id='20261016-100000-offline-aaaa', **kw):
    base = {'run_id': run_id, 'created': '2026-10-16T10:00:00', 'mode': 'offline', 'label': '', 'model': {'name': 'rules', 'synthetic': False},
            'prompts': {'fingerprint': 'f1', 'parts': {'symptom': 'a1', 'chat': 'c1'}}, 'engine_version': 2, 'sets': ['symptom_cases'],
            'datasets': {'symptom_cases': {'version': '1.0.0', 'sha256': 'h'}}, 'metrics': {'emergency_recall': 1.0, 'under_triage': 0.0},
            'outcomes': [], 'threshold_failures': [], 'gated': True}
    base.update(kw)
    return base


class TestRuns:
    def test_save_load_list_roundtrip(self, sandbox):
        runs.save_run(rec())
        assert runs.load_run('20261016-100000-offline-aaaa')['metrics']['emergency_recall'] == 1.0
        assert [r['run_id'] for r in runs.list_runs()] == ['20261016-100000-offline-aaaa']

    @pytest.mark.parametrize('bad', ['../../etc/passwd', '..', 'a/b', 'a\\b', '', 'x', '.hidden', 'a' * 200, None, 123, '%2e%2e%2fsecret'])
    def test_run_ids_from_urls_cannot_escape_the_runs_folder(self, sandbox, bad):
        (sandbox / 'secret.json').write_text('{"run_id": "secret"}')
        assert runs.load_run(bad) is None

    def test_saving_rejects_unsafe_ids(self, sandbox):
        with pytest.raises(ValueError):
            runs.save_run(rec(run_id='../escape'))

    def test_list_is_newest_first_and_skips_corrupt_files(self, sandbox):
        runs.save_run(rec('20261016-100000-offline-aaaa', created='2026-10-16T10:00:00'))
        runs.save_run(rec('20261017-100000-offline-bbbb', created='2026-10-17T10:00:00'))
        (runs.RUNS_DIR / 'broken.json').write_text('{not json')
        assert [r['run_id'] for r in runs.list_runs()] == ['20261017-100000-offline-bbbb', '20261016-100000-offline-aaaa']

    def test_run_ids_are_unique_and_valid(self):
        ids = {runs.new_run_id('mock-flaky') for _ in range(50)}
        assert len(ids) == 50 and all(runs.RUN_ID_RE.match(i) for i in ids)


class TestCompare:
    def test_verdicts_follow_the_direction_of_each_metric(self):
        a = rec(metrics={'emergency_recall': 0.9, 'false_positive_rate': 0.05, 'latency_p50_ms': 1000, 'abstention': 0.1})
        b = rec(run_id='20261016-110000-offline-bbbb', metrics={'emergency_recall': 1.0, 'false_positive_rate': 0.10, 'latency_p50_ms': 800, 'abstention': 0.2})
        rows = {r['metric']: r for r in runs.compare(a, b)['metrics']}
        assert rows['emergency_recall']['verdict'] == 'better'          # higher is better
        assert rows['false_positive_rate']['verdict'] == 'worse'        # lower is better
        assert rows['latency_p50_ms']['verdict'] == 'better'
        assert rows['abstention']['verdict'] == 'info'                  # no good direction

    def test_missing_metrics_are_not_compared(self):
        rows = {r['metric']: r for r in runs.compare(rec(metrics={'emergency_recall': 1.0}), rec(metrics={'emergency_recall': 1.0, 'api_failure_rate': 0.1}))['metrics']}
        assert rows['api_failure_rate']['verdict'] == 'n/a'

    def test_different_dataset_versions_make_the_comparison_unfair(self):
        b = rec(datasets={'symptom_cases': {'version': '1.1.0', 'sha256': 'h2'}})
        c = runs.compare(rec(), b)
        assert c['comparable'] is False and c['dataset_diff'] == {'symptom_cases': {'a': '1.0.0', 'b': '1.1.0'}}
        assert any('different dataset versions' in w for w in c['warnings'])

    def test_same_dataset_is_comparable(self):
        assert runs.compare(rec(), rec())['comparable'] is True

    def test_prompt_changes_are_pinpointed(self):
        b = rec(prompts={'fingerprint': 'f2', 'parts': {'symptom': 'a2', 'chat': 'c1'}})
        assert runs.compare(rec(), b)['prompt_diff'] == {'symptom': {'a': 'a1', 'b': 'a2'}}

    def test_changing_model_and_prompt_together_is_called_out(self):
        b = rec(model={'name': 'other', 'synthetic': False}, prompts={'fingerprint': 'f2', 'parts': {'symptom': 'a2'}})
        assert any('cannot tell which one caused' in w for w in runs.compare(rec(), b)['warnings'])

    def test_synthetic_models_are_flagged(self):
        b = rec(model={'name': 'mock-weak', 'synthetic': True})
        assert any('synthetic' in w for w in runs.compare(rec(), b)['warnings'])

    def test_case_level_regressions_and_improvements(self):
        ok = {'id': 'E01', 'level': 'high', 'alert': True, 'problems': [], 'text': 't', 'category': 'emergency'}
        bad = dict(ok, alert=False, problems=['MISSED_EMERGENCY'])
        a = rec(outcomes=[ok, dict(ok, id='E02', problems=['MISSED_EMERGENCY'], alert=False)])
        b = rec(outcomes=[bad, dict(ok, id='E02')])
        c = runs.compare(a, b)
        assert [r['id'] for r in c['regressions']] == ['E01'] and [r['id'] for r in c['improvements']] == ['E02']


# ---------------------------------------------------------------------------
# clients
# ---------------------------------------------------------------------------

def checker_with(client):
    c = SymptomChecker()
    c.use_gemini, c.client = True, client
    return c


class TestMockClients:
    def test_weak_model_always_says_low(self):
        out = checker_with(clients.MockClient('weak'))._parse_gemini_symptom_response(
            clients.MockClient('weak').models.generate_content(contents='"""\nchest pain\n"""', config=None, model='m'))
        assert out['urgency_level'] == 'low'

    def test_flaky_model_produces_both_errors_and_invalid_json_deterministically(self):
        m = clients.MockClient('flaky', seed=1, error_rate=0.3, invalid_rate=0.3).models
        outcomes = []
        for _ in range(60):
            try:
                outcomes.append('bad' if 'not valid json' in m.generate_content(contents='"""\nfever\n"""', config=None, model='m').text else 'ok')
            except RuntimeError:
                outcomes.append('err')
        assert {'ok', 'bad', 'err'} <= set(outcomes)
        m2 = clients.MockClient('flaky', seed=1, error_rate=0.3, invalid_rate=0.3).models
        again = []
        for _ in range(60):
            try:
                again.append('bad' if 'not valid json' in m2.generate_content(contents='"""\nfever\n"""', config=None, model='m').text else 'ok')
            except RuntimeError:
                again.append('err')
        assert outcomes == again

    def test_fixture_mode_returns_exactly_what_you_supply(self):
        m = clients.MockClient('fixture', fixtures={'chest': {'urgency_level': 'high', 'emergency_alert': True}}).models
        assert json.loads(m.generate_content(contents='"""\nmy chest hurts\n"""', config=None, model='m').text)['urgency_level'] == 'high'
        with pytest.raises(KeyError):
            m.generate_content(contents='"""\nunrelated\n"""', config=None, model='m')


class TestRecordReplay:
    def test_replay_reproduces_a_recorded_run_exactly(self, tmp_path):
        rec_file = tmp_path / 'rec.jsonl'
        c1 = checker_with(clients.RecordingClient(clients.MockClient('rules'), rec_file))
        r1 = c1.analyze_symptoms('fever and cough for two days')
        c2 = checker_with(clients.ReplayClient(rec_file))
        r2 = c2.analyze_symptoms('fever and cough for two days')
        for key in ('urgency_level', 'possible_conditions', 'care_level', 'recommendations'):
            assert r1[key] == r2[key]

    def test_replay_restores_latency_and_tokens(self, tmp_path):
        f = tmp_path / 'rec.jsonl'
        checker_with(clients.RecordingClient(clients.MockClient('rules'), f)).analyze_symptoms('fever and cough')
        line = json.loads(f.read_text().splitlines()[0])
        line['latency_ms'], line['input_tokens'] = 4321.0, 777
        f.write_text(json.dumps(line) + '\n')
        with telemetry.Collector(isolate=True) as col:
            checker_with(clients.ReplayClient(f)).analyze_symptoms('fever and cough')
        assert col.calls[0]['latency_ms'] == 4321.0 and col.calls[0]['input_tokens'] == 777

    def test_a_request_that_was_never_recorded_counts_as_an_api_failure(self, tmp_path, monkeypatch):
        import quickaid.ai.symptom_checker as sc
        monkeypatch.setattr(sc.time, 'sleep', lambda *_: None)
        f = tmp_path / 'empty.jsonl'
        f.write_text('')
        with telemetry.Collector(isolate=True) as col:
            out = checker_with(clients.ReplayClient(f)).analyze_symptoms('fever and cough')
        assert out['analysis_mode'] == 'basic' and col.calls and not any(c['ok'] for c in col.calls)
        assert col.calls[0]['error_type'] == 'ReplayMiss'

    def test_recorded_api_errors_are_replayed_as_errors(self, tmp_path):
        f = tmp_path / 'rec.jsonl'
        flaky = clients.MockClient('flaky', error_rate=1.0)
        rc = clients.RecordingClient(flaky, f)
        with pytest.raises(RuntimeError):
            rc.models.generate_content(contents='"""\nx\n"""', config=None, model='m')
        with pytest.raises(RuntimeError):
            clients.ReplayClient(f).models.generate_content(contents='"""\nx\n"""', config=None, model='m')

    def test_recordings_never_contain_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setenv('GEMINI_API_KEY', 'AIza-SECRET-KEY-VALUE')
        f = tmp_path / 'rec.jsonl'
        checker_with(clients.RecordingClient(clients.MockClient('rules'), f)).analyze_symptoms('fever and cough')
        assert 'AIza-SECRET' not in f.read_text()

    def test_replay_knows_which_model_was_recorded(self, tmp_path):
        f = tmp_path / 'rec.jsonl'
        checker_with(clients.RecordingClient(clients.MockClient('rules'), f)).analyze_symptoms('fever and cough')
        assert clients.ReplayClient(f).recorded_models == [SymptomChecker.MODEL_NAME]


# ---------------------------------------------------------------------------
# run builder + metric definitions
# ---------------------------------------------------------------------------

class TestMetricDefinitions:
    def outcome(self, **kw):
        base = {'id': 'x', 'category': 'c', 'text': '', 'level': 'low', 'alert': False, 'abstained': False, 'before': None, 'raised': False,
                'reliability': None, 'problems': [], 'known_limitation': False, 'expected_min': None, 'expected_max': None,
                'emergency': None, 'forbid_emergency': False, 'abstain_ok': False}
        base.update(kw)
        return base

    def test_recall_false_negative_false_positive_and_precision(self):
        outs = [self.outcome(id='e1', emergency=True, alert=True), self.outcome(id='e2', emergency=True, alert=True),
                self.outcome(id='e3', emergency=True, alert=False, problems=['MISSED_EMERGENCY']),
                self.outcome(id='n1', forbid_emergency=True, alert=True, problems=['FALSE_ALARM']),
                self.outcome(id='n2', forbid_emergency=True), self.outcome(id='n3', forbid_emergency=True), self.outcome(id='n4', forbid_emergency=True)]
        m = runner.aggregate(outs)
        assert m['emergency_recall'] == pytest.approx(2 / 3, abs=1e-3) and m['false_negative_rate'] == pytest.approx(1 / 3, abs=1e-3)
        assert m['false_positive_rate'] == 0.25 and m['precision'] == pytest.approx(2 / 3, abs=1e-3)

    def test_known_limitations_are_excluded_from_the_rates(self):
        outs = [self.outcome(emergency=True, alert=True), self.outcome(id='k', emergency=True, alert=False, problems=['MISSED_EMERGENCY'], known_limitation=True)]
        assert runner.aggregate(outs)['emergency_recall'] == 1.0

    def test_no_emergency_cases_means_recall_is_undefined_not_perfect(self):
        m = runner.aggregate([self.outcome(forbid_emergency=True)])
        assert m['emergency_recall'] is None and m['false_negative_rate'] is None


class TestBuildRun:
    def test_offline_run_is_gated_and_records_everything_needed_to_reproduce_it(self, sandbox):
        r = runner.build_run('offline', sets=['symptom_cases'])
        assert r['gated'] is True and r['mode'] == 'offline' and r['threshold_failures'] == []
        assert r['prompts']['fingerprint'] and r['engine_version'] and r['datasets']['symptom_cases']['sha256']
        assert r['datasets']['symptom_cases']['matches_manifest'] and r['cases_evaluated'] > 80 and r['outcomes']
        assert runs.load_run(r['run_id'])['metrics'] == r['metrics']

    def test_flaky_model_shows_api_failures_and_invalid_output_without_touching_the_network(self, sandbox):
        r = runner.build_run('mock-flaky', sets=['symptom_cases'])
        m = r['metrics']
        assert m['api_failure_rate'] > 0 and m['structured_output_validity'] < 1 and m['fallback_rate'] > 0
        assert r['model']['synthetic'] is True and r['gated'] is False and 'Synthetic' in r['notes']

    def test_unsafe_model_is_visibly_worse_than_rules_alone(self, sandbox):
        good = runner.build_run('offline', sets=['symptom_cases'], save=False)['metrics']
        weak = runner.build_run('mock-weak', sets=['symptom_cases'], save=False)['metrics']
        assert weak['under_triage'] > good['under_triage'] and weak['structured_output_validity'] == 1.0
        c = runs.compare(runner.build_run('offline', save=False), runner.build_run('mock-weak', save=False))
        assert c['regressions'] and {r['verdict'] for r in c['metrics'] if r['metric'] == 'under_triage'} == {'worse'}

    def test_evaluation_traffic_never_reaches_the_live_telemetry_sinks(self, sandbox):
        live = telemetry.Collector()
        telemetry.add_sink(live)
        try:
            runner.build_run('mock-rules', sets=['symptom_cases'], save=False)
        finally:
            telemetry.remove_sink(live)
        assert live.calls == [] and live.analyses == []

    def test_unknown_dataset_and_unsafe_specs_are_refused(self, sandbox, monkeypatch):
        with pytest.raises(ValueError):
            runner.build_run('offline', sets=['nope'], save=False)
        monkeypatch.delenv('GEMINI_API_KEY', raising=False)
        with pytest.raises(RuntimeError):
            runner.checker_for('live')
        with pytest.raises(ValueError):
            runner.checker_for('replay')
        with pytest.raises(ValueError):
            runner.checker_for('banana')

    def test_replay_run_rescored_from_a_recording(self, sandbox, tmp_path):
        f = tmp_path / 'rec.jsonl'
        mock = clients.RecordingClient(clients.MockClient('rules'), f)
        rec_checker = checker_with(mock)
        rec_checker.use_gemini = True
        cases = [c for c in runner.load('symptom_cases.json')['cases'] if c['mode'] == 'oneshot'][:25]
        for c in cases:
            rec_checker.analyze_symptoms(c['text'])
        data = json.loads((datasets.DATASET_DIR / 'symptom_cases.json').read_text())
        small = {**data, 'cases': cases}
        (datasets.DATASET_DIR / 'symptom_cases.json').write_text(json.dumps(small))
        datasets.bump('symptom_cases', 'trimmed for the replay test')
        r = runner.build_run('replay', sets=['symptom_cases'], replay_from=str(f), save=False)
        assert r['mode'] == 'replay' and r['metrics']['api_failure_rate'] == 0 and r['model']['recording'] == 'rec.jsonl'

    def test_thresholds_apply_to_gated_modes_only(self, sandbox):
        assert runner.build_run('mock-weak', save=False)['gated'] is False
        assert runner.GATED_SPECS == ('offline', 'live', 'replay')


class TestImageReference:
    def make_set(self, tmp_path, items):
        rng = np.random.default_rng(9)
        for i, it in enumerate(items):
            arr = np.clip(rng.integers(-25, 25, (320, 320, 1)) + np.array([215, 165, 145]), 0, 255).astype('uint8')
            if it.pop('_dark', False):
                arr = (arr * 0.05).astype('uint8')
            Image.fromarray(arr).save(tmp_path / it['file'])
        (tmp_path / 'manifest.json').write_text(json.dumps({'images': items}))
        return tmp_path

    base = dict(split='test', source='own photo, consent given', license='CC0', labelled_by='Dr. Example (clinician)', clinician_reviewed=True)

    def test_manifest_requires_provenance(self, tmp_path):
        d = self.make_set(tmp_path, [dict(file='a.png', split='dev', source='x', license='y', labelled_by='z')])
        bad = json.loads((d / 'manifest.json').read_text())
        del bad['images'][0]['labelled_by']
        (d / 'manifest.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='labelled_by'):
            runner.load_image_reference(d)

    def test_manifest_rejects_bad_splits_missing_files_and_missing_manifest(self, tmp_path):
        d = self.make_set(tmp_path, [dict(file='a.png', **{**self.base, 'split': 'train'})])
        with pytest.raises(ValueError, match='split'):
            runner.load_image_reference(d)
        (d / 'manifest.json').write_text(json.dumps({'images': [dict(file='missing.png', **self.base)]}))
        with pytest.raises(ValueError, match='not found'):
            runner.load_image_reference(d)
        with pytest.raises(ValueError, match='no manifest'):
            runner.load_image_reference(tmp_path / 'nowhere')

    def test_scores_real_model_output_against_independent_labels(self, tmp_path):
        d = self.make_set(tmp_path, [dict(file='ok.png', expected_max_urgency='low', **self.base),
                                     dict(file='dark.png', must_be_unusable=True, _dark=True, **self.base),
                                     dict(file='needs.png', expected_min_urgency='high', **self.base)])
        checker, _ = runner.checker_for('mock-rules')
        res = runner.evaluate_image_reference(d, runner.analyzer_for(checker))
        assert res['images'] == 3 and res['unusable_photo_recall'] == 1.0 and res['usable_photos_wrongly_rejected'] == 0
        assert res['severe_under_triage'] == 1.0                     # the mock says "low" where the label demands "high"
        assert res['failures'] == [{'file': 'needs.png', 'problems': ['SEVERE_UNDER_TRIAGE']}]

    def test_caveats_are_stated_when_the_set_is_tiny_or_unreviewed(self, tmp_path):
        d = self.make_set(tmp_path, [dict(file='a.png', **{**self.base, 'clinician_reviewed': False})])
        res = runner.evaluate_image_reference(d, runner.analyzer_for(runner.checker_for('mock-rules')[0]))
        assert any('Fewer than 30' in c for c in res['caveats']) and any('clinician' in c for c in res['caveats'])

    def test_split_filter(self, tmp_path):
        d = self.make_set(tmp_path, [dict(file='a.png', **{**self.base, 'split': 'dev'}), dict(file='b.png', **self.base)])
        a = runner.analyzer_for(runner.checker_for('mock-rules')[0])
        assert runner.evaluate_image_reference(d, a, split='test')['images'] == 1


# ---------------------------------------------------------------------------
# command-line tools
# ---------------------------------------------------------------------------

class TestCommandLine:
    @pytest.fixture()
    def manage(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('manage_datasets', Path(datasets.EVAL_DIR) / 'manage_datasets.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_check_passes_on_the_shipped_datasets(self, manage, capsys):
        assert manage.main(['check']) == 0 and 'datasets OK' in capsys.readouterr().out

    def test_check_fails_when_a_dataset_was_edited_without_a_bump(self, manage, sandbox, capsys):
        f = datasets.DATASET_DIR / 'model_output_cases.json'
        f.write_text(f.read_text().replace('Dose in the advice', 'Dose in the advice (edited)'))
        assert manage.main(['check']) == 1 and 'changed since version' in capsys.readouterr().out

    def test_bump_then_check_passes_and_frozen_sets_are_refused(self, manage, sandbox, capsys):
        f = datasets.DATASET_DIR / 'model_output_cases.json'
        f.write_text(f.read_text().replace('Dose in the advice', 'Dose in the advice (edited)'))
        assert manage.main(['bump', 'model_output_cases', '--note', 'reworded G05', '--part', 'patch']) == 0
        assert manage.main(['check']) == 0
        assert manage.main(['bump', 'symptom_holdout2', '--note', 'tuning']) == 1 and 'frozen' in capsys.readouterr().out

    def test_list_shows_versions_and_roles(self, manage, capsys):
        manage.main(['list'])
        out = capsys.readouterr().out
        assert 'symptom_cases' in out and 'v1.0.0' in out and '[frozen]' in out

    def test_list_runs_and_compare_commands(self, sandbox, capsys):
        a = runner.build_run('offline', label='first')
        b = runner.build_run('mock-weak', label='second')
        assert runner.main(['--list-runs']) == 0 and 'first' in capsys.readouterr().out
        assert runner.main(['--compare', a['run_id'], b['run_id']]) == 0
        out = capsys.readouterr().out
        assert 'WARNING' in out and 'Under-triage' in out and 'regressions:' in out
        assert runner.main(['--compare', a['run_id'], 'nope-nope-nope']) == 2

    def test_a_model_run_exits_nonzero_only_when_a_gated_threshold_is_missed(self, sandbox, monkeypatch):
        assert runner.main(['--model', 'offline', '--no-write']) == 0
        assert runner.main(['--model', 'mock-weak', '--no-write']) == 0            # mocks are informational
        monkeypatch.setitem(runner.THRESHOLDS, 'emergency_recall', ('>=', 1.01))
        assert runner.main(['--model', 'offline', '--no-write']) == 1

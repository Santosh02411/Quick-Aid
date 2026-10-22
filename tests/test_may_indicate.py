"""'May indicate': a brief, hedged mention of what an emergency warning sign can point to (e.g. "Chest pain or
pressure" -> "a heart attack"), shown alongside the existing warning-sign label. Requested directly: the result
only ever showed the symptom-pattern label, never named anything concrete, which read as vaguer than it needed to.
Deliberately hedged throughout ("can be a sign of", never "this is" or "you have") and deliberately does not cover
every label - only ones with a recognisable condition worth naming.
"""

import pytest

from quickaid.ai import triage_engine as te


class TestTheMapping:
    def test_known_labels_map_to_a_named_condition(self):
        assert te.may_indicate_for(['Chest pain or pressure']) == ['a heart attack']
        assert te.may_indicate_for(['Possible stroke signs']) == ['a stroke']
        assert te.may_indicate_for(['Severe allergic reaction']) == ['anaphylaxis']

    def test_an_unmapped_label_returns_nothing_rather_than_guessing(self):
        # these are already self-explanatory (the label names the thing itself) or too generic to narrow safely
        assert te.may_indicate_for(['Poisoning or overdose']) == []
        assert te.may_indicate_for(['Snake or scorpion bite']) == []
        assert te.may_indicate_for(['Serious injury or accident']) == []
        assert te.may_indicate_for(['unknown label not in the patterns at all']) == []

    def test_several_labels_are_combined_without_duplicates(self):
        out = te.may_indicate_for(['Chest pain or pressure', 'Severe trouble breathing', 'Chest pain or pressure'])
        assert out.count('a heart attack') == 1
        assert 'a severe asthma attack' in out

    def test_the_wording_is_always_hedged_never_a_diagnosis(self):
        for label, names in te.MAY_INDICATE.items():
            for name in names:
                assert not name.lower().startswith(('you have', 'this is', 'you are having')), (label, name)


class TestPropagationThroughEveryEmergencyCodePath:
    """All three places that build an emergency result must carry this through - found and fixed one (the
    review_symptom path) that called build_emergency() but never read the new field back out of it."""

    def test_immediate_emergency(self):
        r = te.immediate_emergency('severe crushing chest pain and sweating', 'IN', 'summary')
        assert r['emergency_alert']['may_indicate'] == ['a heart attack']

    def test_emergency_analysis_directly(self):
        r = te.emergency_analysis(['Possible stroke signs'], 'IN', crisis=False, summary='s')
        assert r['emergency_alert']['may_indicate'] == ['a stroke']

    def test_review_symptom_basic_mode(self, app):
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        r = c.analyze_symptoms('severe crushing chest pain and sweating, started 20 minutes ago')
        assert r['emergency_alert']['alert'] is True
        assert 'a heart attack' in r['emergency_alert']['may_indicate']

    def test_review_symptom_does_not_invent_one_for_an_unmapped_warning_sign(self, app):
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        r = c.analyze_symptoms("she drank weedkiller by accident")
        assert r['emergency_alert']['alert'] is True
        assert r['emergency_alert'].get('may_indicate', []) == []   # poisoning is already explicit; nothing invented

    def test_crisis_only_results_get_no_may_indicate(self, app):
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        r = c.analyze_symptoms("I want to kill myself")
        assert r['emergency_alert']['alert'] is True
        assert r['emergency_alert'].get('may_indicate', []) == []


class TestEnglishOnlyForNow:
    """Like the other wording rewrites in triage_engine.py (soften()), translating specific disease names needs
    the same native-speaker review the rest of the translated content is still waiting for - so this stays
    English-only rather than guessing a translation, consistent with that existing, documented limitation."""

    def test_non_english_languages_get_no_may_indicate(self):
        info = te.build_emergency(['Chest pain or pressure'], 'IN', lang='hi')
        assert info.get('may_indicate', []) == []

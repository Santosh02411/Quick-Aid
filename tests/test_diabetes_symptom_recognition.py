"""Basic-mode (no AI) symptom recognition for the classic diabetes presentation - found missing entirely: a
real description with 7 symptoms (increased thirst, frequent urination, increased hunger, fatigue, blurred
vision, slow wound healing, unexplained weight loss) only recognised "fatigue" and suggested viral infection /
poor sleep / anaemia / stress, never diabetes, because the other six were not in the symptom database at all.
"""

import pytest


@pytest.fixture()
def checker(app):
    from quickaid.ai.symptom_checker import SymptomChecker
    return SymptomChecker()


class TestTheReportedCase:
    def test_the_exact_reported_text_now_surfaces_diabetes_first(self, checker):
        text = ("increased thirst, frequent urination, increased hunger, fatigue, blurred vision, "
               "slow wound healing, unexplained weight loss")
        r = checker.analyze_symptoms(text)
        assert len(r['detected_symptoms']) == 7
        assert r['possible_conditions'][0] == 'diabetes'
        top = r['possible_explanations'][0]
        assert top['name'] == 'diabetes' and top['likelihood'] == 'more likely'
        assert len(top['supported_by']) >= 6   # supported by most of the reported symptoms, not just one


class TestEachSymptomIsRecognisedOnItsOwn:
    """Checks the symptom KEY (via _extract_symptoms) is recognised, not the exact display text: detected_symptoms
    now shows the person's own words (see _extract_symptoms_detailed), which legitimately vary by phrasing -
    'I'm always thirsty' is correctly shown as "always thirsty", not the category name "increased thirst"."""

    @pytest.mark.parametrize('text,key', [
        ("I'm always thirsty and can't stop drinking water", 'increased_thirst'),
        ("I've been urinating a lot more than usual", 'frequent_urination'),
        ("I'm constantly hungry even after eating", 'increased_hunger'),
        ("my vision has been blurry for a few days", 'blurred_vision'),
        ("this cut on my hand just won't heal", 'slow_wound_healing'),
        ("I've lost weight without trying or dieting", 'unexplained_weight_loss'),
    ])
    def test_each_symptom_is_recognised_in_isolation(self, checker, text, key):
        assert key in checker._extract_symptoms(text)
        r = checker.analyze_symptoms(text)
        assert 'diabetes' in r['possible_conditions']

    def test_detected_symptoms_shows_the_persons_own_words_not_a_category_name(self, checker):
        r = checker.analyze_symptoms("I'm always thirsty and can't stop drinking water")
        assert 'increased thirst' not in r['detected_symptoms']   # the category name must not appear verbatim
        assert any('thirsty' in s or 'drinking' in s for s in r['detected_symptoms'])


class TestDiabetesOnlyRisesWithMultipleSymptoms:
    @pytest.mark.parametrize('text,common_cause', [
        ("I've been really hungry lately", 'low blood sugar'),
        ("I've been really thirsty today", 'dehydration'),
        ("my vision is a bit blurry after staring at my screen all day", 'eye strain'),
    ])
    def test_a_single_symptom_with_an_obviously_more_common_cause_does_not_lead_with_diabetes(self, checker, text, common_cause):
        # these three each have one far more common, usually-benign explanation for an ISOLATED, brief mention -
        # diabetes stays in the list (so it still wins once several symptoms converge, tested above), but must not
        # crowd out the much more likely everyday explanation from a single vague symptom on its own.
        r = checker.analyze_symptoms(text)
        assert r['possible_conditions'][0] == common_cause, r['possible_conditions']
        assert 'diabetes' in r['possible_conditions']

    def test_unrelated_benign_fatigue_is_unaffected(self, checker):
        r = checker.analyze_symptoms("I'm just a bit tired, nothing else going on")
        assert 'diabetes' not in r['possible_conditions']
        assert r['possible_conditions'][:1] == ['viral infection']


class TestChronicNonHealingWoundIsNotTreatedAsAFreshInjury:
    """'A cut that won't heal' describes a chronic, concerning sign - it should reach the symptom checker's
    diabetes-aware differential, not the first-aid module's acute 'clean and dress a fresh cut' guide."""

    def test_a_fresh_cut_still_gets_the_first_aid_guide(self, checker):
        from quickaid.ai import first_aid
        assert first_aid.match("I just cut my finger on a knife")

    @pytest.mark.parametrize('text', [
        "this cut on my hand just won't heal",
        "I have a cut that isn't healing",
    ])
    def test_a_cut_that_wont_heal_does_not_get_the_fresh_injury_guide(self, text):
        from quickaid.ai import first_aid
        assert not first_aid.match(text)

    def test_a_cut_that_wont_heal_reaches_the_symptom_checker_instead(self, checker):
        text = "I have a cut that isn't healing and my vision has been blurry"
        keys = checker._extract_symptoms(text)
        assert 'slow_wound_healing' in keys and 'blurred_vision' in keys
        r = checker.analyze_symptoms(text)
        assert 'diabetes' in r['possible_conditions']


class TestNegationCrossesAndBoundary:
    """'And' is now a clause break for negation (triage_engine._CLAUSE_BREAKS): found via this exact diabetes
    scenario - "a cut that isn't healing and my vision has been blurry" was losing BOTH symptoms, because 'isn't'
    (meant only for 'healing') was bleeding across 'and' to also wrongly negate 'blurry' in the unrelated clause
    that followed it.
    """

    def test_a_negation_in_the_first_clause_does_not_suppress_a_real_symptom_in_the_second(self, checker):
        r = checker.analyze_symptoms("my headache isn't too bad and I have a high fever")
        assert 'fever' in r['detected_symptoms']

    def test_a_negation_repeated_on_each_side_of_and_still_negates_both(self):
        from quickaid.ai.triage_engine import negated
        t = 'no fever and no cough'
        assert negated(t, t.index('fever')) and negated(t, t.index('cough'))

    def test_a_single_negation_does_not_elliptically_carry_across_and(self):
        from quickaid.ai.triage_engine import negated
        t = 'no fever and cough'
        assert negated(t, t.index('fever')) and not negated(t, t.index('cough'))

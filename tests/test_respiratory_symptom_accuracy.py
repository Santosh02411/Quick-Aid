"""Root-cause fix for a reported bug: inputting wheezing, shortness of breath, chest TIGHTNESS, coughing,
difficulty breathing, rapid breathing, and nighttime coughing incorrectly reported "chest pain" as a confirmed,
reported symptom (the person never said "pain"), and led the differential straight to heart attack/angina instead
of recognising the classic respiratory/asthma presentation this actually is.

Root causes found and fixed:
1. symptom_checker.py's 'chest_pain' key absorbed "tightness/pressure/discomfort/hurts" as synonyms, then every
   caller built user-facing text with `key.replace('_', ' ')` - so "chest tightness" was shown back as "chest
   pain", a different, more specific symptom never reported. Fixed by splitting into chest_pain (actual pain) and
   chest_tightness (tightness/pressure/discomfort), and by capturing the ACTUAL MATCHED TEXT for every symptom
   (_extract_symptoms_detailed) instead of ever deriving display text from the internal key name.
2. "wheezing", "rapid breathing" and "nighttime coughing" were not recognised at all - silently dropped, with no
   indication they had been ignored, and no chance to contribute to an asthma differential.
3. A second-order bug the refactor reintroduced and this file also guards against: context_floors() checks like
   'fever' in detected_symptoms stopped working once detected_symptoms showed matched text instead of the key name
   - fixed via a separate ctx['detected'] channel carrying the stable keys, independent of display wording.
"""

import pytest


@pytest.fixture()
def checker(app):
    from quickaid.ai.symptom_checker import SymptomChecker
    return SymptomChecker()


RESPIRATORY_TEXT = ("wheezing, shortness of breath, chest tightness, coughing, difficulty breathing, "
                    "rapid breathing, and nighttime coughing")


class TestTheExactReportedCase:
    def test_chest_pain_is_never_reported_as_a_confirmed_finding(self, checker):
        r = checker.analyze_symptoms(RESPIRATORY_TEXT)
        full_text = (r['summary'] + ' ' + ' '.join(r['detected_symptoms']) + ' ' +
                    ' '.join(r['reported'].get('symptoms', [])) + ' ' + (r['emergency_alert'].get('reason') or '')).lower()
        assert 'chest pain' not in full_text, full_text

    def test_all_seven_reported_symptoms_are_recognised(self, checker):
        keys = checker._extract_symptoms(RESPIRATORY_TEXT)
        for expected_key in ('wheezing', 'shortness_of_breath', 'chest_tightness', 'cough', 'rapid_breathing', 'nighttime_cough'):
            assert expected_key in keys, (expected_key, keys)

    def test_asthma_is_the_leading_possible_explanation(self, checker):
        r = checker.analyze_symptoms(RESPIRATORY_TEXT)
        assert r['possible_conditions'][0] == 'asthma', r['possible_conditions']
        top = r['possible_explanations'][0]
        assert top['name'] == 'asthma' and len(top['supported_by']) >= 4

    def test_this_is_presented_as_a_possibility_not_a_confirmed_diagnosis(self, checker):
        r = checker.analyze_symptoms(RESPIRATORY_TEXT)
        top = r['possible_explanations'][0]
        assert top['likelihood'] in ('more likely', 'possible')   # never a bare assertion of fact
        assert 'diagnos' not in r['summary'].lower() or 'not a diagnos' in r['disclaimer'].lower()

    def test_the_summary_only_names_words_the_person_actually_used(self, checker):
        r = checker.analyze_symptoms(RESPIRATORY_TEXT)
        # whatever is quoted in the summary's parenthetical must be real substrings of the original text
        import re
        m = re.search(r'\(([^)]+)\)', r['summary'])
        assert m, r['summary']
        for phrase in m.group(1).split(', '):
            assert phrase.strip() in RESPIRATORY_TEXT.lower(), (phrase, RESPIRATORY_TEXT)


class TestChestPainAndChestTightnessAreDistinct:
    def test_actual_pain_is_reported_as_pain(self, checker):
        r = checker.analyze_symptoms("my chest really hurts, sharp pain")
        assert 'chest_pain' in checker._extract_symptoms("my chest really hurts, sharp pain")
        assert any('hurt' in s or 'pain' in s for s in r['detected_symptoms'])

    def test_tightness_is_reported_as_tightness_not_pain(self, checker):
        r = checker.analyze_symptoms("I have chest tightness and I'm wheezing")
        assert not any('pain' in s for s in r['detected_symptoms']), r['detected_symptoms']
        assert any('tight' in s for s in r['detected_symptoms'])

    def test_both_still_reach_high_urgency(self, checker):
        for text in ("my chest really hurts badly", "I have severe chest tightness"):
            r = checker.analyze_symptoms(text)
            assert r['urgency_level'] == 'high', (text, r['urgency_level'])


class TestNoFabricatedFindingsAcrossARangeOfRespiratoryPhrasing:
    @pytest.mark.parametrize('text,must_not_contain', [
        ("I have a wheeze and feel breathless", 'chest pain'),
        ("coughing a lot at night, can't sleep", 'chest pain'),
        ("breathing really fast and feel wheezy", 'chest pain'),
        ("my chest feels tight when I breathe", 'pain'),
    ])
    def test_a_symptom_never_appears_unless_the_matching_text_was_actually_there(self, checker, text, must_not_contain):
        r = checker.analyze_symptoms(text)
        joined = ' '.join(r['detected_symptoms']).lower()
        assert must_not_contain not in joined or must_not_contain in text.lower()


class TestContextAwareFloorsStillWorkWithMatchedTextDisplay:
    """Regression guard: context_floors() checks like 'fever' in detected_symptoms would silently stop firing once
    detected_symptoms started showing matched text ("had a temperature") instead of the key name ("fever") -
    fixed via ctx['detected'] carrying the stable key separately. Covers the exact case this broke."""

    def test_fever_in_an_infant_still_raises_urgency_when_worded_indirectly(self, checker):
        r = checker.analyze_symptoms("my 3-month-old has had a temperature of 38.5 since this morning")
        assert r['urgency_level'] == 'high', r
        assert any('fever' in reason.lower() for reason in r.get('urgency_reasons', []))

    def test_fever_worded_directly_also_still_works(self, checker):
        r = checker.analyze_symptoms("my 2 month old baby has a fever")
        assert r['urgency_level'] == 'high'


class TestAllSectionsAgreeWithTheSameValidatedAssessment:
    """Every result goes through exactly one validation gate (triage_engine.review_symptom), which is what makes
    this guaranteed rather than coincidental: emergency_alert.alert, urgency_level, care_level and red_flags_present
    are all derived from the SAME decision, never computed independently and then shown side by side."""

    @pytest.mark.parametrize('text', [
        "wheezing, shortness of breath, chest tightness, coughing, difficulty breathing, rapid breathing, and nighttime coughing",
        "my chest really hurts and I'm sweating",
        "mild headache since this morning",
        "fever and cough for 2 days",
        "severe crushing chest pain radiating to my arm",
    ])
    def test_alert_urgency_and_care_level_never_disagree(self, checker, text):
        r = checker.analyze_symptoms(text)
        alert = r.get('emergency_alert', {}).get('alert', False)
        if alert:
            assert r['urgency_level'] == 'high' and r['care_level'] == 'emergency', r
        else:
            assert r['care_level'] != 'emergency', r

    def test_a_red_flag_being_present_always_means_the_alert_fired(self, checker, text=None):
        for text in ("severe crushing chest pain radiating to my arm", "mild headache since this morning"):
            r = checker.analyze_symptoms(text)
            if r.get('red_flags_present'):
                assert r['emergency_alert']['alert'] is True, (text, r['red_flags_present'])

"""Hindi / Marathi / Kannada: catalog quality, emergency DETECTION, emergency INSTRUCTIONS, and the language plumbing.

These tests cannot judge whether a translation is clinically right: that needs native-speaking clinicians (docs/TRANSLATION_REVIEW.md).
What they CAN do, and do, is stop the mechanical failures that make safety text dangerous: a lost or altered emergency number, a step that
silently disappears, text left in the wrong language, broken placeholders, terms used inconsistently, or an alarm that does not fire.
"""

import html as htmllib
import re

import pytest

from quickaid import i18n
from quickaid.ai import triage_engine as te
from quickaid.i18n import emergency as em
from quickaid.i18n.catalog import STRINGS

NON_EN = ('hi', 'mr', 'kn')
SCRIPT = {'hi': r'[\u0900-\u097F]', 'mr': r'[\u0900-\u097F]', 'kn': r'[\u0C80-\u0CFF]'}
ALLOWED_LATIN = {'Google', 'Gemini', 'Quick', 'Aid', 'SOS'}


def words_latin(text):
    return {w for w in re.findall(r'[A-Za-z]+', re.sub(r'\{[A-Za-z_]+\}', '', text))} - ALLOWED_LATIN


class TestCatalog:
    def test_every_string_exists_in_every_language(self):
        for key, entry in STRINGS.items():
            assert set(entry) == {'en', 'hi', 'mr', 'kn'} and all(entry[l].strip() for l in entry), key

    def test_placeholders_are_identical_in_every_language(self):
        for key, entry in STRINGS.items():
            want = sorted(re.findall(r'\{[a-z]+\}', entry['en']))
            for lang in NON_EN:
                assert sorted(re.findall(r'\{[a-z]+\}', entry[lang])) == want, (key, lang)

    def test_every_string_that_mentions_an_emergency_number_keeps_the_placeholder(self):
        for key, entry in STRINGS.items():
            if re.search(r'\{(number|n)\}', entry['en']):
                for lang in NON_EN:
                    assert re.search(r'\{(number|n)\}', entry[lang]), (key, lang)

    @pytest.mark.parametrize('lang', NON_EN)
    def test_text_is_written_in_the_right_script_not_left_in_english(self, lang):
        for key, entry in STRINGS.items():
            text = entry[lang]
            assert re.search(SCRIPT[lang], text), (key, lang, 'no native-script characters')
            assert not words_latin(text), (key, lang, words_latin(text))

    def test_helpline_and_other_numbers_use_ordinary_digits(self):
        for key, entry in STRINGS.items():
            for lang in NON_EN:
                assert not re.search(r'[\u0966-\u096F\u0CE6-\u0CEF]', entry[lang]), (key, lang, 'native numerals can be dialled wrongly')

    def test_the_helpline_number_survives_translation(self):
        for lang in NON_EN:
            assert '14416' in STRINGS['em.i10'][lang] and '14416' in em.CRISIS_SUPPORT_IN[lang]

    def test_missing_keys_and_languages_fall_back_to_english_never_to_an_error(self):
        assert i18n.t('no.such.key') == 'no.such.key'
        assert i18n.t('nav.history', lang='fr') == 'History'
        assert i18n.t('dash.call', lang='hi', number='112') == '112 पर कॉल करें'

    def test_the_browser_gets_only_the_js_strings(self):
        js = i18n.js_strings('hi')
        assert js['high_title'] == 'अभी चिकित्सा सहायता लें' and 'nav.history' not in js and all(not k.startswith('js.') for k in js)
        assert len(js) >= 25

    def test_every_language_string_has_a_review_status_entry(self):
        import json
        from pathlib import Path
        status = json.loads((Path(i18n.__file__).parent / 'translation_status.json').read_text(encoding='utf-8'))
        assert status['reviewed_by_clinician'] == {'hi': False, 'mr': False, 'kn': False}       # honest: nobody has signed these off yet
        assert 'safety_critical_files' in status


class TestEmergencyContent:
    def test_every_english_warning_sign_has_a_translation_with_the_same_number_of_steps(self):
        for lang in NON_EN:
            for label, steps in te.GUIDANCE.items():
                assert len(em.GUIDANCE[lang][label]) == len(steps), (lang, label)
            assert len(em.DEFAULT_STEPS[lang]) == len(te.DEFAULT_STEPS) and len(em.CRISIS_STEPS[lang]) == 3

    def test_no_translated_guidance_for_a_sign_the_engine_does_not_know(self):
        for lang in NON_EN:
            assert set(em.GUIDANCE[lang]) == set(te.GUIDANCE) and set(em.LABELS[lang]) == set(te.GUIDANCE) | {'Thoughts of self-harm'}

    @pytest.mark.parametrize('lang', NON_EN)
    def test_every_step_is_native_script_and_every_call_step_carries_the_number(self, lang):
        steps = [s for lst in em.GUIDANCE[lang].values() for s in lst] + em.DEFAULT_STEPS[lang] + em.CRISIS_STEPS[lang]
        for s in steps:
            assert re.search(SCRIPT[lang], s) and not words_latin(s), s
        for label, lst in em.GUIDANCE[lang].items():
            assert '{EMERGENCY_NUMBER}' in lst[0], (lang, label, 'the first step of every warning sign must tell people to call')
        assert all('{EMERGENCY_NUMBER}' in lst[0] for lst in (em.DEFAULT_STEPS[lang], em.CRISIS_STEPS[lang]))

    # the dangerous "never"s must survive translation: each is checked by a word the translation would have to contain
    DONTS = {
        ('Loss of consciousness or seizure', 1): {'hi': 'कभी कुछ न डालें', 'mr': 'कधीही काहीही घालू नका', 'kn': 'ಎಂದಿಗೂ ಏನನ್ನೂ ಹಾಕಬೇಡಿ'},      # nothing in the mouth
        ('Poisoning or overdose', 2): {'hi': 'उल्टी करवाने की कोशिश न करें', 'mr': 'उलटी करवण्याचा प्रयत्न करू नका', 'kn': 'ವಾಂತಿ ಮಾಡಿಸಲು ಪ್ರಯತ್ನಿಸಬೇಡಿ'},  # do not induce vomiting
        ('Possible stroke signs', 2): {'hi': 'न', 'mr': 'नका', 'kn': 'ಬೇಡಿ'},                                                                      # no food, drink or medicine
        ('Snake or scorpion bite', 2): {'hi': 'न काटें', 'mr': 'कापू नका', 'kn': 'ಕತ್ತರಿಸಬೇಡಿ'},                                                  # do not cut, suck or tie tight
        ('Chest pain or pressure', 3): {'hi': 'न चलाएं', 'mr': 'चालवू नका', 'kn': 'ಓಡಿಸಬೇಡಿ'},                                                      # do not drive yourself
    }

    @pytest.mark.parametrize('lang', NON_EN)
    def test_the_important_dont_instructions_are_still_there(self, lang):
        for (label, idx), words in self.DONTS.items():
            assert words[lang] in em.GUIDANCE[lang][label][idx], (lang, label, idx)

    @pytest.mark.parametrize('lang', NON_EN)
    def test_the_same_medical_term_is_used_consistently(self, lang):
        """A glossary of the terms safety text depends on. If a term appears in the label it must appear in the matching steps."""
        glossary = {
            'hi': {'Severe bleeding': 'खून', 'Possible stroke signs': 'स्ट्रोक', 'Snake or scorpion bite': 'काट', 'Poisoning or overdose': 'ज़हर'},
            'mr': {'Severe bleeding': 'रक्त', 'Possible stroke signs': 'स्ट्रोक', 'Snake or scorpion bite': 'चाव', 'Poisoning or overdose': 'विष'},
            'kn': {'Severe bleeding': 'ರಕ್ತ', 'Possible stroke signs': 'ಸ್ಟ್ರೋಕ್', 'Snake or scorpion bite': 'ಕಚ್ಚಿ', 'Poisoning or overdose': 'ವಿಷ'},
        }[lang]
        for label, term in glossary.items():
            assert term in em.LABELS[lang][label] + ' '.join(em.GUIDANCE[lang][label]), (lang, label, term)

    def test_the_fixed_result_texts_exist(self):
        for lang in NON_EN:
            assert set(em.TEXT[lang]) == {'headline', 'message', 'action', 'summary', 'timeframe', 'avoid', 'disclaimer'}
            assert '{EMERGENCY_NUMBER}' in em.TEXT[lang]['action'] and len(em.TEXT[lang]['avoid']) == 2


# phrase -> label that MUST be raised. (>=3 per label in each language; several in Latin script, the way many people type.)
POSITIVES = {
    'hi': [('मुझे सीने में तेज दर्द हो रहा है', 'Chest pain or pressure'), ('मेरे पिता को हार्ट अटैक आया है', 'Chest pain or pressure'), ('छाती में दबाव महसूस हो रहा है', 'Chest pain or pressure'),
           ('सांस लेने में बहुत तकलीफ हो रही है', 'Severe trouble breathing'), ('दम घुट रहा है', 'Severe trouble breathing'), ('बच्चे के होंठ नीले पड़ गए हैं', 'Severe trouble breathing'),
           ('चेहरा टेढ़ा हो गया है और बोलने में दिक्कत है', 'Possible stroke signs'), ('उन्हें लकवा मार गया', 'Possible stroke signs'),
           ('मेरी माँ बेहोश हो गई हैं', 'Loss of consciousness or seizure'), ('उसे मिर्गी का दौरा पड़ा है', 'Loss of consciousness or seizure'),
           ('बहुत खून बह रहा है', 'Severe bleeding'), ('खून की उल्टी हो रही है', 'Severe bleeding'),
           ('होंठ और जीभ पर सूजन आ गई है', 'Severe allergic reaction'), ('बच्चे ने जहर पी लिया', 'Poisoning or overdose'), ('उसने ज्यादा गोलियां खा ली हैं', 'Poisoning or overdose'),
           ('सांप ने काट लिया है', 'Snake or scorpion bite'), ('बिच्छू ने डंक मारा', 'Snake or scorpion bite'),
           ('बुखार है और गर्दन में अकड़न है', 'Stiff neck with fever'), ('बुखार के साथ बड़बड़ा रहा है', 'Fever with confusion'),
           ('छत से गिर गया', 'Serious injury or accident'), ('करंट लग गया', 'Serious injury or accident')],
    'mr': [('माझ्या छातीत खूप दुखतंय', 'Chest pain or pressure'), ('छातीवर दाब येतोय', 'Chest pain or pressure'), ('हृदयविकाराचा झटका आला असावा', 'Chest pain or pressure'),
           ('श्वास घेण्यास खूप त्रास होतोय', 'Severe trouble breathing'), ('दम लागत आहे', 'Severe trouble breathing'),
           ('चेहरा वाकडा झाला आहे', 'Possible stroke signs'), ('बोलताना अडखळतोय', 'Possible stroke signs'), ('आजोबा बेशुद्ध पडले', 'Loss of consciousness or seizure'), ('फिट आली आहे', 'Loss of consciousness or seizure'),
           ('खूप रक्तस्त्राव होतोय', 'Severe bleeding'), ('रक्ताची उलटी झाली', 'Severe bleeding'), ('ओठ सुजले आहेत', 'Severe allergic reaction'),
           ('मुलाने विष प्यायले', 'Poisoning or overdose'), ('साप चावला आहे', 'Snake or scorpion bite'), ('विंचू चावला', 'Snake or scorpion bite'),
           ('मान अवघडली आहे आणि ताप आहे', 'Stiff neck with fever'), ('अपघात झाला आहे', 'Serious injury or accident')],
    'kn': [('ನನಗೆ ಎದೆ ನೋವು ಇದೆ', 'Chest pain or pressure'), ('ಹೃದಯಾಘಾತ ಆಗಿದೆ', 'Chest pain or pressure'), ('ಎದೆಯಲ್ಲಿ ತೀವ್ರ ಒತ್ತಡ ಇದೆ', 'Chest pain or pressure'),
           ('ಉಸಿರಾಟದ ತೊಂದರೆ ಇದೆ', 'Severe trouble breathing'), ('ಉಸಿರಾಡಲು ಆಗುತ್ತಿಲ್ಲ', 'Severe trouble breathing'),
           ('ಮುಖ ಓರೆಯಾಗಿದೆ', 'Possible stroke signs'), ('ಮಾತನಾಡಲು ಕಷ್ಟವಾಗುತ್ತಿದೆ', 'Possible stroke signs'), ('ಅವರಿಗೆ ಪ್ರಜ್ಞೆ ತಪ್ಪಿದೆ', 'Loss of consciousness or seizure'), ('ಸೆಳೆತ ಬರುತ್ತಿದೆ', 'Loss of consciousness or seizure'),
           ('ತುಂಬಾ ರಕ್ತ ಹೋಗುತ್ತಿದೆ', 'Severe bleeding'), ('ರಕ್ತ ವಾಂತಿ ಆಗುತ್ತಿದೆ', 'Severe bleeding'), ('ತುಟಿ ಊದಿಕೊಂಡಿದೆ', 'Severe allergic reaction'),
           ('ಮಗು ವಿಷ ಕುಡಿದಿದೆ', 'Poisoning or overdose'), ('ಹಾವು ಕಚ್ಚಿದೆ', 'Snake or scorpion bite'), ('ಚೇಳು ಕುಟುಕಿದೆ', 'Snake or scorpion bite'),
           ('ಜ್ವರದ ಜೊತೆ ಕುತ್ತಿಗೆ ಬಿಗಿತ ಇದೆ', 'Stiff neck with fever'), ('ಅಪಘಾತ ಆಗಿದೆ', 'Serious injury or accident')],
    'latin': [('seene mein dard hai', 'Chest pain or pressure'), ('papa ke seene me tej dard ho raha hai', 'Chest pain or pressure'), ('chhatit dukhtay', 'Chest pain or pressure'), ('ede novu ide', 'Chest pain or pressure'),
              ('saans lene mein takleef ho rahi hai', 'Severe trouble breathing'), ('saans nahi aa rahi', 'Severe trouble breathing'), ('dam ghut raha hai', 'Severe trouble breathing'),
              ('wo behosh ho gaya', 'Loss of consciousness or seizure'), ('daura pad gaya', 'Loss of consciousness or seizure'), ('bahut khoon beh raha hai', 'Severe bleeding'), ('khoon ki ulti ho rahi hai', 'Severe bleeding'),
              ('saanp ne kaata', 'Snake or scorpion bite'), ('havu kachchide', 'Snake or scorpion bite'), ('bachche ne zeher pi liya', 'Poisoning or overdose'), ('chehra tedha ho gaya', 'Possible stroke signs')],
}
SELF_HARM = ['मुझे आत्महत्या के विचार आ रहे हैं', 'मला मरावेसे वाटते', 'ನನಗೆ ಆತ್ಮಹತ್ಯೆಯ ಆಲೋಚನೆ ಬರುತ್ತಿದೆ', 'mujhe jeena nahi hai', 'main atmahatya karna chahta hoon']
BENIGN = ['मुझे हल्का सिरदर्द और जुकाम है', 'थोड़ी खांसी है और गला खराब है', 'पेट में हल्का दर्द है', 'पैर में थोड़ी खुजली है', 'आज थोड़ा थकान महसूस हो रही है',
          'मला थोडी सर्दी आणि खोकला आहे', 'पोट थोडे दुखतंय', 'डोके थोडे दुखतंय', 'पायाला थोडी खाज आहे',
          'ನನಗೆ ಸ್ವಲ್ಪ ಶೀತ ಮತ್ತು ಕೆಮ್ಮು ಇದೆ', 'ಸ್ವಲ್ಪ ತಲೆನೋವು ಇದೆ', 'ಹೊಟ್ಟೆಯಲ್ಲಿ ಸ್ವಲ್ಪ ನೋವು ಇದೆ', 'ಕಾಲಿನಲ್ಲಿ ಸ್ವಲ್ಪ ತುರಿಕೆ ಇದೆ',
          'mujhe halka sirdard hai', 'thodi khansi aur zukam hai', 'pet mein halka dard hai', 'naak band hai']
NEGATED = ['सीने में दर्द नहीं है', 'सांस लेने में तकलीफ नहीं है', 'बेहोशी नहीं हुई', 'खून नहीं बह रहा', 'छातीत दुखत नाही', 'श्वास घेण्यास त्रास नाही', 'ಎದೆ ನೋವು ಇಲ್ಲ', 'ಉಸಿರಾಟದ ತೊಂದರೆ ಇಲ್ಲ', 'seene mein dard nahi hai', 'saans ki takleef nahi hai']


class TestEmergencyDetection:
    @pytest.mark.parametrize('lang', list(POSITIVES))
    def test_warning_signs_are_recognised(self, lang):
        misses = [(text, label) for text, label in POSITIVES[lang] if label not in te.scan(text).emergency]
        assert misses == []

    def test_each_language_has_a_real_number_of_examples(self):
        assert {k: len(v) for k, v in POSITIVES.items()}['hi'] >= 20 and sum(len(v) for v in POSITIVES.values()) >= 70

    def test_self_harm_wording_is_recognised(self):
        assert [t for t in SELF_HARM if not te.scan(t).crisis] == []

    def test_everyday_complaints_do_not_raise_an_alarm(self):
        false_alarms = [(t, te.scan(t).emergency) for t in BENIGN if te.scan(t).any_emergency]
        assert false_alarms == []

    def test_a_negated_symptom_is_not_an_alarm(self):
        assert [(t, te.scan(t).emergency) for t in NEGATED if te.scan(t).any_emergency] == []

    def test_negation_in_one_clause_does_not_hide_a_symptom_in_the_next(self):
        assert te.scan('बुखार नहीं है लेकिन सीने में तेज दर्द है').emergency == ['Chest pain or pressure']
        assert te.scan('ಜ್ವರ ಇಲ್ಲ ಆದರೆ ಎದೆ ನೋವು ಇದೆ').emergency == ['Chest pain or pressure']

    def test_spelling_variants_match(self):
        assert 'Severe trouble breathing' in te.scan('साँस लेने में तकलीफ़ है').emergency          # chandrabindu + nukta
        assert 'Poisoning or overdose' in te.scan('ज़हर खा लिया').emergency

    def test_mixed_language_text_still_works(self):
        assert 'Chest pain or pressure' in te.scan('my dad has सीने में दर्द since morning').emergency

    def test_english_detection_is_unchanged(self):
        assert te.scan('crushing chest pain').emergency == ['Chest pain or pressure']
        assert te.scan('no chest pain, just a mild cough').emergency == [] and te.scan('I have a mild headache').emergency == []

    @pytest.mark.parametrize('hostile', ['', ' ', 'क' * 5000, '\u200d' * 100, 'ಎ' * 3000, '😀' * 500, 'a\u0900' * 800, None])
    def test_hostile_input_never_crashes_or_hangs(self, hostile):
        te.scan(hostile)
        em.detect(hostile or '')


@pytest.fixture()
def lang_of(request):
    return request.param


class TestTranslatedEmergencyResults:
    @pytest.mark.parametrize('lang,text', [('hi', 'मेरे पिता को सीने में तेज दर्द है'), ('mr', 'माझ्या वडिलांना छातीत दुखतंय'), ('kn', 'ನನ್ನ ತಂದೆಗೆ ಎದೆ ನೋವು ಇದೆ')])
    def test_the_whole_emergency_result_comes_back_in_the_language_with_the_number(self, lang, text):
        tok = i18n.set_language(lang)
        try:
            r = te.immediate_emergency(text, 'IN', 'x')
        finally:
            i18n.reset(tok)
        assert r['care_level'] == 'emergency' and r['urgency_level'] == 'high'
        blob = ' '.join(r['recommendations'] + [r['summary'], r['emergency_alert']['message'], r['emergency_alert']['action'], r['timeframe'], r['disclaimer']] + r['avoid'] + r['red_flags_present'])
        assert '112' in r['emergency_alert']['action'] and '112' in r['recommendations'][0] and '{EMERGENCY_NUMBER}' not in blob
        assert not words_latin(blob), words_latin(blob)
        assert r['red_flags_present'] == [em.LABELS[lang]['Chest pain or pressure']]

    @pytest.mark.parametrize('region,number', [('IN', '112'), ('US', '911'), ('UK', '999'), ('AU', '000')])
    def test_the_number_follows_the_region_in_every_language(self, region, number):
        for lang in NON_EN:
            tok = i18n.set_language(lang)
            try:
                r = te.immediate_emergency('seene mein dard hai', region, 'x')
            finally:
                i18n.reset(tok)
            assert number in r['emergency_alert']['action'] and number in r['recommendations'][0]

    def test_several_warning_signs_give_one_call_line_not_three(self):
        tok = i18n.set_language('hi')
        try:
            r = te.immediate_emergency('सीने में दर्द है, सांस नहीं आ रही और वह बेहोश है', 'IN', 'x')
        finally:
            i18n.reset(tok)
        assert sum('112' in s and 'कॉल' in s for s in r['recommendations']) == 1

    def test_self_harm_support_is_translated_and_includes_the_helpline(self):
        for lang in NON_EN:
            tok = i18n.set_language(lang)
            try:
                r = te.immediate_emergency(SELF_HARM[NON_EN.index(lang)], 'IN', 'x')
            finally:
                i18n.reset(tok)
            assert '14416' in r['recommendations'][0] and '112' in ' '.join(r['recommendations'])

    def test_english_output_is_byte_for_byte_what_it_was(self):
        r = te.immediate_emergency('crushing chest pain', 'IN', 'summary text')
        assert r['summary'] == 'summary text' and r['emergency_alert']['action'] == 'Call 112 or go to the nearest emergency room immediately'
        assert r['recommendations'][0] == 'Call 112 now and say you have chest pain.' and r['timeframe'] == 'Right now'

    def test_an_unknown_language_falls_back_to_english(self):
        tok = i18n.set_language('fr')
        try:
            assert te.immediate_emergency('crushing chest pain', 'IN', 'x')['recommendations'][0].startswith('Call 112')
        finally:
            i18n.reset(tok)


class TestLanguagePlumbing:
    def test_negotiation_order_and_normalisation(self):
        assert i18n.negotiate(None, 'kn', 'hi', 'mr') == 'kn'              # saved account choice wins
        assert i18n.negotiate(None, None, 'hi', 'mr') == 'hi'              # then the cookie
        assert i18n.negotiate(None, None, None, 'mr-IN,mr;q=0.9,en;q=0.5') == 'mr'
        assert i18n.negotiate(None, 'xx', 'yy', 'fr-FR') == 'en'
        assert i18n.normalize_code('HI-IN') == 'hi' and i18n.normalize_code('kn_IN') == 'kn' and i18n.normalize_code('de') is None and i18n.normalize_code(None) is None

    def test_accept_language_honours_quality_values_and_garbage(self):
        assert i18n.from_accept_language('en;q=0.3, hi;q=0.9, kn;q=0.5') == 'hi'
        assert i18n.from_accept_language('hi;q=0, en') == 'en'
        assert i18n.from_accept_language('hi;q=abc, mr') == 'mr' and i18n.from_accept_language(';;;,,,') is None

    def test_the_page_follows_the_browser_language(self, app):
        html = app.test_client().get('/emergency', headers={'Accept-Language': 'hi-IN,hi;q=0.9'}).get_data(as_text=True)
        assert '<html lang="hi"' in html and 'आपातकालीन सहायता' in html and 'tel:112' in html

    def test_every_page_declares_its_language_and_shows_the_review_notice_when_translated(self, app):
        for lang, snippet in (('hi', 'चिकित्सकों ने जांच नहीं की'), ('mr', 'डॉक्टरांनी तपासणी केलेली नाही'), ('kn', 'ವೈದ್ಯರು ಇನ್ನೂ ಪರಿಶೀಲಿಸಿಲ್ಲ')):
            html = app.test_client().get('/emergency', headers={'Accept-Language': lang}).get_data(as_text=True)
            assert f'<html lang="{lang}"' in html and snippet in html and 'class="lang-notice"' in html
        english = app.test_client().get('/emergency').get_data(as_text=True)
        assert '<html lang="en"' in english and 'lang-notice' not in english.split('</header>')[0].split('<body')[-1]

    def test_the_emergency_page_lists_every_warning_sign_in_each_language_with_the_number(self, app):
        for lang in NON_EN:
            html = htmllib.unescape(app.test_client().get('/emergency', headers={'Accept-Language': lang}).get_data(as_text=True))
            for i in range(1, 11):
                assert STRINGS[f'em.i{i}'][lang] in html, (lang, i)
            assert html.count('tel:112') >= 1 and STRINGS['em.guides_english'][lang] in html

    def test_switching_language_by_link_sets_a_cookie_and_redirects_safely(self, app):
        c = app.test_client()
        r = c.get('/lang/kn?next=/emergency')
        assert r.status_code == 302 and r.headers['Location'].endswith('/emergency') and 'qa_lang=kn' in r.headers['Set-Cookie']
        assert '<html lang="kn"' in c.get('/emergency').get_data(as_text=True)

    @pytest.mark.parametrize('bad', ['//evil.example', 'https://evil.example', '/\\evil.example', 'javascript:alert(1)', '/ok\nSet-Cookie: x=1', ''])
    def test_the_language_link_cannot_be_used_as_an_open_redirect(self, app, bad):
        r = app.test_client().get('/lang/hi', query_string={'next': bad})
        assert r.status_code == 302 and r.headers['Location'] in ('/', 'http://localhost/')

    @pytest.mark.parametrize('code', ['fr', 'xx', 'hi-IN', '../etc', 'en%00'])
    def test_unknown_or_malformed_language_codes_are_a_404(self, app, code):
        assert app.test_client().get(f'/lang/{code}').status_code == 404

    def test_a_signed_in_persons_choice_is_saved_on_the_account_and_follows_them(self, app, client, registered_user):
        client.put('/api/language', json={'language': 'mr'})
        other = app.test_client()
        other.post('/login', data={'username': registered_user['username'], 'password': registered_user['password']})
        assert '<html lang="mr"' in other.get('/').get_data(as_text=True)          # a different browser, same account

    def test_language_api_validation(self, app):
        c = app.test_client()
        assert c.put('/api/language', json={'language': 'fr'}).status_code == 400
        assert c.put('/api/language', json={}).status_code == 400
        assert c.put('/api/language', json={'language': 'hi'}).get_json()['language'] == 'hi'

    def test_the_language_does_not_leak_between_requests(self, app):
        hi, en = app.test_client(), app.test_client()
        hi.get('/lang/hi')
        for _ in range(3):
            assert '<html lang="hi"' in hi.get('/emergency').get_data(as_text=True)
            assert '<html lang="en"' in en.get('/emergency').get_data(as_text=True)
        assert i18n.current() == 'en'

    def test_the_switcher_is_on_every_page_and_marks_the_current_language(self, app):
        html = app.test_client().get('/emergency', headers={'Accept-Language': 'kn'}).get_data(as_text=True)
        assert html.count('class="lang-switch"') >= 1 and 'lang="hi" hreflang="hi"' in html and 'hreflang="kn" aria-current="true"' in html and 'हिन्दी' in html and 'ಕನ್ನಡ' in html

    def test_the_dashboard_is_translated(self, app, client, registered_user):
        client.put('/api/language', json={'language': 'hi'})
        html = htmllib.unescape(client.get('/').get_data(as_text=True))
        for needle in ('डैशबोर्ड', 'आपातकाल में पहले 112 पर कॉल करें।', 'चोट या त्वचा की समस्या जांचें', 'की गई जांचें'):
            assert needle in html, needle

    def test_the_browser_script_receives_the_translations_and_the_language(self, app):
        html = app.test_client().get('/emergency', headers={'Accept-Language': 'kn'}).get_data(as_text=True)
        assert 'window.QA_LANG = "kn"' in html and 'window.QA_I18N =' in html
        assert re.search(r'\\u0c[0-9a-f]{2}', html), 'the Kannada strings are embedded (JSON-escaped)'

    def test_the_result_script_translates_through_one_helper_with_english_fallbacks(self, client):
        js = client.get('/static/js/app.js').data.decode()
        for key in ('high_title', 'high_call', 'basic_notice', 'review_notice', 'what_to_do', 'warn_urgent', 'insufficient_title', 'followup'):
            assert f"T('{key}'" in js, key
        assert 'appendCallLine' in js and "'Get medical help now'" in js            # English text is still the fallback

    def test_the_review_notice_is_never_hidden_in_a_translated_result(self, client):
        js = client.get('/static/js/app.js').data.decode()
        assert "if (LANG !== 'en') nodes.push" in js and 'review_notice' in js


class TestAiLanguage:
    def test_the_model_is_told_which_language_to_write_and_what_never_to_translate(self):
        from quickaid.i18n import prompts
        assert prompts.note('en') == ''
        for lang, name in (('hi', 'Hindi'), ('mr', 'Marathi'), ('kn', 'Kannada')):
            note = prompts.note(lang)
            assert name in note and 'enum values' in note and 'emergency phone numbers' in note and 'JSON keys' in note and 'less cautious' in note

    def test_every_model_call_carries_the_instruction_in_the_persons_language(self, app, monkeypatch):
        import io
        import numpy as np
        from PIL import Image
        from tests.test_ai_consent import Recorder, photo
        monkeypatch.setenv('REQUIRE_AI_CONSENT', 'false')
        monkeypatch.setenv('INTERVIEW_AI_PLAN', 'true')
        svc = app.extensions['quickaid_services']
        rec = Recorder()
        for obj in (svc.symptom_checker, svc.medical_analyzer, svc.chat_ai):
            obj.use_gemini, obj.client = True, rec
        c = app.test_client()
        c.post('/register', data={'username': 'ann', 'email': 'a@example.com', 'password': 'orange-kettle-42', 'confirm_password': 'orange-kettle-42'})
        c.put('/api/language', json={'language': 'kn'})
        c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea for two days'})
        c.post('/upload', data={'file': (io.BytesIO(photo()), 'a.png')}, content_type='multipart/form-data')
        cid = c.post('/api/chats', json={}).get_json()['conversation']['id']
        c.post(f'/api/chats/{cid}/messages', json={'message': 'is it serious?'})
        c.post('/api/interview/start', json={'symptoms': 'cough for a few days'})
        systems = [str(getattr(k['config'], 'system_instruction', '')) for k in rec.calls]
        assert len(systems) >= 4 and all('Kannada' in s for s in systems)

    def test_english_requests_are_unchanged(self, app, monkeypatch):
        from tests.test_ai_consent import Recorder
        monkeypatch.setenv('REQUIRE_AI_CONSENT', 'false')
        svc = app.extensions['quickaid_services']
        rec = Recorder()
        svc.symptom_checker.use_gemini, svc.symptom_checker.client = True, rec
        c = app.test_client()
        c.post('/register', data={'username': 'ann', 'email': 'a@example.com', 'password': 'orange-kettle-42', 'confirm_password': 'orange-kettle-42'})
        c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea for two days'})
        assert 'LANGUAGE:' not in str(rec.calls[0]['config'].system_instruction)

    def test_prompt_fingerprints_do_not_change_with_the_users_language(self):
        from quickaid.ai import prompt_registry
        before = prompt_registry.fingerprint()
        tok = i18n.set_language('hi')
        try:
            assert prompt_registry.fingerprint() == before
        finally:
            i18n.reset(tok)

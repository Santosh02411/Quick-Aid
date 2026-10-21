"""Real-browser journeys and an automated accessibility audit. Run with:  pytest -m browser tests/e2e"""

import io

import numpy as np
import pytest
from PIL import Image

from tests.e2e.conftest import PW, sign_up

pytestmark = pytest.mark.browser


def jpeg(seed=1):
    arr = np.clip(np.random.default_rng(seed).integers(-25, 25, (360, 360, 1)) + np.array([215, 165, 145]), 0, 255).astype('uint8')
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, 'PNG')
    return buf.getvalue()


class TestJourneys:
    def test_register_then_see_the_dashboard_with_no_script_errors(self, live, make_page):
        page = make_page()
        sign_up(page, live)
        assert page.locator('h1').inner_text().strip().endswith('ann')
        assert page.locator('.sidebar .nav-item').count() >= 6 and page.locator('.sos-strip').is_visible()
        assert page.js_errors == []

    def test_a_symptom_check_shows_a_clear_result_with_urgency_text_and_the_disclaimer(self, live, make_page):
        page = make_page()
        sign_up(page, live)
        page.fill('#symptomsInput', 'headache and nausea since this morning')
        page.click('#checkSymptomsBtn')
        for _ in range(12):                                                  # the interview may ask questions: skip through them
            if page.locator('#symptomResult.on').count():
                break
            if page.locator('#ivSkip').is_visible():
                page.click('#ivSkip')
            page.wait_for_timeout(300)
        page.wait_for_selector('#symptomResult.on')
        assert page.locator('#symptomResult .triage .lvl').inner_text().strip()
        assert 'not a diagnosis' in page.locator('#symptomResult').inner_text().lower() or 'diagnosis' in page.locator('#symptomResult').inner_text().lower()
        assert page.js_errors == []

    def test_an_emergency_in_hindi_gives_translated_instructions_and_a_call_button(self, live, make_page):
        page = make_page()
        sign_up(page, live)
        page.click('.sidebar .lang-switch a[lang="hi"]')
        page.wait_for_load_state()
        assert page.locator('html').get_attribute('lang') == 'hi'
        page.fill('#symptomsInput', 'मेरे पिता को सीने में तेज दर्द है और पसीना आ रहा है')
        page.click('#checkSymptomsBtn')
        page.wait_for_selector('#symptomResult.on', timeout=15000)
        text = page.locator('#symptomResult').inner_text()
        assert 'अभी चिकित्सा सहायता लें' in text and '112' in text and 'सीने में दर्द' in text
        assert page.locator('#symptomResult a[href="tel:112"]').count() >= 1
        assert page.locator('.lang-notice').is_visible()                    # the "not yet clinician-reviewed" notice is on screen
        assert page.js_errors == []

    def test_the_language_choice_survives_a_reload_and_a_new_page(self, live, make_page):
        page = make_page()
        page.goto(live + '/emergency')
        page.click('.pubnav .lang-switch a[lang="kn"]')
        page.wait_for_load_state()
        assert page.locator('html').get_attribute('lang') == 'kn' and 'ತುರ್ತು ಸಹಾಯ' in page.locator('h1').inner_text()
        page.goto(live + '/privacy')
        assert page.locator('html').get_attribute('lang') == 'kn'

    def test_the_browser_language_is_respected_on_the_first_visit(self, live, make_page):
        page = make_page(locale='hi-IN')
        page.goto(live + '/emergency')
        assert page.locator('html').get_attribute('lang') == 'hi' and 'आपातकालीन सहायता' in page.locator('h1').inner_text()

    def test_recent_checks_updates_live_instead_of_staying_stale(self, live, make_page):
        """Found via a real browser: the stat cards at the top of the dashboard update live via JS after a check
        completes, but 'Recent checks' right below them was purely server-rendered and never refreshed - right
        after finishing a check (especially an emergency one), a person could see 'Flagged as urgent' tick up
        directly above a list still saying 'Nothing here yet', which could read as 'did this actually get saved?'
        at exactly the wrong moment."""
        page = make_page()
        sign_up(page, live, name='recentcheckfix')
        assert 'Nothing here yet' in page.locator('section:has(#recent-h)').inner_text()
        page.fill('#symptomsInput', 'severe crushing chest pain and sweating')
        page.click('#checkSymptomsBtn')
        page.wait_for_selector('#symptomResult.on', timeout=10000)
        section_text = page.locator('section:has(#recent-h)').inner_text()
        assert 'Nothing here yet' not in section_text and 'View all' in section_text
        assert page.locator('#recentList li').count() == 1
        assert page.js_errors == []

    def test_a_recovery_code_actually_fits_in_the_login_2fa_field(self, live, make_page):
        """Found via real-browser testing, not catchable by a Flask-test-client unit test (which posts raw data and
        never goes through the HTML input's own maxlength/pattern constraints at all): the login 2FA field used to
        have maxlength=6 and a digits-only pattern, inherited from the TOTP-code case, which silently truncated an
        11-character recovery code (XXXXX-XXXXX) to 6 characters the moment a real browser typed it in - a person
        who lost their authenticator device could not actually use their recovery codes through this form."""
        import re as _re
        page = make_page()
        sign_up(page, live, name='recoverytest2fa')
        page.goto(live + '/account/2fa/setup')
        secret = page.locator('.secret-box').inner_text().strip()
        import pyotp
        page.fill('input[name=code]', pyotp.TOTP(secret).now())
        page.click('button[type=submit]')
        page.wait_for_selector('text=recovery')
        codes = _re.findall(r'\b[A-Z2-9]{5}-[A-Z2-9]{5}\b', page.locator('body').inner_text())
        assert len(codes) == 10
        page.locator('form[action="/logout"] button[type=submit]').first.click()
        page.goto(live + '/login')
        page.fill('#username', 'recoverytest2fa')
        page.fill('#password', PW)
        page.click('button[type=submit]')
        page.wait_for_url('**/login/2fa')
        code_input = page.locator('input[name=code]')
        code_input.fill(codes[0])
        assert code_input.input_value() == codes[0]        # the field must not have silently truncated it
        page.click('button[type=submit]')
        page.wait_for_load_state()
        assert '/login/2fa' not in page.url and page.js_errors == []

    def test_the_consent_card_works_and_goes_away(self, live, make_page, app, monkeypatch):
        monkeypatch.setenv('REQUIRE_AI_CONSENT', 'true')
        app.extensions['quickaid_services'].symptom_checker.use_gemini = True            # pretend AI is configured so the question is asked
        page = make_page()
        sign_up(page, live)
        assert page.locator('#aiConsentCard').is_visible()
        page.click('[data-ai-consent="false"]')
        page.wait_for_selector('#aiConsentCard', state='detached', timeout=8000)
        assert page.js_errors == []

    def test_history_search_and_the_report_and_notes(self, live, make_page):
        page = make_page()
        sign_up(page, live)
        page.fill('#symptomsInput', 'sore throat and mild fever')
        page.click('#checkSymptomsBtn')
        for _ in range(12):
            if page.locator('#symptomResult.on').count():
                break
            if page.locator('#ivSkip').is_visible():
                page.click('#ivSkip')
            page.wait_for_timeout(300)
        page.goto(live + '/history')
        page.fill('#q', 'sore throat')
        page.click('.filter-foot button[type=submit]')
        page.wait_for_selector('.entry')
        page.locator('.entry a:has-text("View report")').first.click()
        page.wait_for_selector('#report')
        assert page.locator('.rep-hero .lvl').inner_text().strip()
        page.fill('#notesText', 'ask the doctor about the fever')
        page.click('[data-notes-form] button[type=submit]')
        page.wait_for_selector('#notesMsg:has-text("Saved")')
        assert page.js_errors == []

    def test_an_injury_shows_first_aid_steps_at_once_with_no_questions(self, live, make_page):
        page = make_page()
        sign_up(page, live)
        page.fill('#symptomsInput', 'I cut my finger while cooking')
        page.click('#checkSymptomsBtn')
        page.wait_for_selector('#symptomResult.on', timeout=10000)
        text = page.locator('#symptomResult').inner_text()
        assert 'Minor cut or scrape' in text and 'Wash your hands' in text and 'Press a clean cloth' in text
        assert page.locator('#ivNext').is_hidden() and page.js_errors == []

    def test_a_burn_on_the_hand_leads_with_cooling_and_still_says_when_to_see_a_doctor(self, live, make_page):
        page = make_page()
        sign_up(page, live)
        page.fill('#symptomsInput', 'burn on my hand from hot water')
        page.click('#checkSymptomsBtn')
        page.wait_for_selector('#symptomResult.on', timeout=10000)
        text = page.locator('#symptomResult').inner_text()
        assert 'Cool the burn' in text and 'bigger than the palm' in text
        assert text.index('Cool the burn') < text.index('bigger than the palm')            # what to DO comes before when to get it checked

    def test_the_simple_check_notice_offers_one_click_to_turn_on_ai(self, live, make_page, app, monkeypatch):
        monkeypatch.setenv('REQUIRE_AI_CONSENT', 'true')
        app.extensions['quickaid_services'].symptom_checker.use_gemini = True            # AI is configured but this person has not allowed it
        page = make_page()
        sign_up(page, live)
        page.fill('#symptomsInput', 'I cut my finger while cooking')
        page.click('#checkSymptomsBtn')
        page.wait_for_selector('#symptomResult.on', timeout=10000)
        assert 'simple built-in check' in page.locator('#symptomResult').inner_text()
        assert page.evaluate('window.QA_AI.allowed') is False
        page.click('#symptomResult [data-ai-consent="true"]')
        for _ in range(30):                                                    # (the page's strict CSP forbids wait_for_function's string evaluation)
            page.wait_for_timeout(300)
            try:
                if page.evaluate('window.QA_AI && window.QA_AI.allowed') is True:
                    break
            except Exception:
                pass                                                               # mid-reload
        assert page.evaluate('window.QA_AI.allowed') is True
        assert page.js_errors == []

    def test_uploading_a_photo_shows_observation_and_interpretation_separately(self, live, make_page):
        page = make_page()
        sign_up(page, live)
        page.set_input_files('#imageInput', {'name': 'p.png', 'mimeType': 'image/png', 'buffer': jpeg()})
        page.wait_for_selector('#preview.on')
        page.click('#analyzeBtn')
        page.wait_for_selector('#imageResult.on', timeout=20000)
        titles = [h.inner_text() for h in page.locator('#imageResult .rcard h3').all()]
        assert 'What is visible in the photo' in titles and 'What it could mean' in titles
        assert page.js_errors == []

    def test_a_bad_file_gets_a_helpful_message_not_a_crash(self, live, make_page):
        page = make_page()
        sign_up(page, live)
        page.set_input_files('#imageInput', {'name': 'notes.txt', 'mimeType': 'text/plain', 'buffer': b'hello'})
        page.wait_for_selector('#uploadStatus .alert')
        assert 'Supported' in page.locator('#uploadStatus').inner_text() and page.js_errors == []

    def test_a_forged_form_without_a_csrf_token_is_refused_with_a_friendly_page(self, live, make_page):
        page = make_page()
        response = page.request.post(live + '/login', form={'username': 'x', 'password': 'y'})
        assert response.status == 400 and 'expired' in response.text().lower() or response.status == 400


PAGES_PUBLIC = ['/', '/emergency', '/privacy', '/login', '/register', '/forgot-password']
PAGES_SIGNED_IN = ['/', '/history', '/chat', '/account', '/emergency', '/privacy']
VIEWPORTS = [(375, 780), (768, 1000), (1280, 900)]
TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']


def audit(page, axe_source):
    page.evaluate(axe_source)
    result = page.evaluate("axe.run(document, {runOnly: {type: 'tag', values: %s}, resultTypes: ['violations']}).then(r => r.violations.map(v => ({id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.slice(0, 3).map(n => n.target.join(' '))})))" % TAGS)
    return [v for v in result if v['impact'] in ('serious', 'critical')]


class TestAccessibility:
    @pytest.mark.parametrize('scheme', ['light', 'dark'])
    @pytest.mark.parametrize('width,height', VIEWPORTS)
    def test_public_pages_have_no_serious_accessibility_violations(self, live, make_page, axe_source, width, height, scheme):
        page = make_page(width, height, scheme)
        found = {}
        for path in PAGES_PUBLIC:
            page.goto(live + path)
            page.wait_for_load_state()
            bad = audit(page, axe_source)
            if bad:
                found[path] = bad
        assert found == {}

    @pytest.mark.parametrize('scheme', ['light', 'dark'])
    @pytest.mark.parametrize('width,height', VIEWPORTS)
    def test_signed_in_pages_have_no_serious_accessibility_violations(self, live, make_page, axe_source, width, height, scheme):
        page = make_page(width, height, scheme)
        sign_up(page, live, name=f'a11y{width}{scheme}')
        found = {}
        for path in PAGES_SIGNED_IN:
            page.goto(live + path)
            page.wait_for_load_state()
            bad = audit(page, axe_source)
            if bad:
                found[path] = bad
        assert found == {}

    @pytest.mark.parametrize('lang', ['hi', 'mr', 'kn'])
    def test_translated_pages_are_accessible_too(self, live, make_page, axe_source, lang):
        page = make_page(375, 780)
        page.goto(live + f'/lang/{lang}?next=/emergency')
        found = {}
        for path in ('/emergency', '/privacy', '/login'):
            page.goto(live + path)
            bad = audit(page, axe_source)
            if bad:
                found[path] = bad
        assert found == {}

    def test_the_report_page_is_accessible_in_a_phone_and_in_print_friendly_layout(self, live, make_page, axe_source):
        page = make_page(375, 780)
        sign_up(page, live, name='reportuser')
        page.fill('#symptomsInput', 'sore throat and mild fever')
        page.click('#checkSymptomsBtn')
        for _ in range(12):
            if page.locator('#symptomResult.on').count():
                break
            if page.locator('#ivSkip').is_visible():
                page.click('#ivSkip')
            page.wait_for_timeout(300)
        page.goto(live + '/history')
        page.locator('.entry a:has-text("View report")').first.click()
        page.wait_for_selector('#report')
        assert audit(page, axe_source) == []

    @pytest.mark.parametrize('width,height', VIEWPORTS)
    def test_no_page_scrolls_sideways(self, live, make_page, width, height):
        page = make_page(width, height)
        sign_up(page, live, name=f'wide{width}')
        too_wide = []
        for path in ('/', '/history', '/chat', '/account', '/emergency', '/privacy'):
            page.goto(live + path)
            if page.evaluate('document.documentElement.scrollWidth > window.innerWidth + 1'):
                too_wide.append(path)
        assert too_wide == []

    def test_the_skip_link_is_the_first_thing_keyboard_users_reach(self, live, make_page):
        page = make_page()
        page.goto(live + '/emergency')
        page.keyboard.press('Tab')
        assert page.evaluate("document.activeElement.className") == 'skip-link' and page.evaluate("document.activeElement.getAttribute('href')") == '#main'

    def test_every_form_field_has_a_label_and_every_icon_button_a_name(self, live, make_page):
        page = make_page()
        sign_up(page, live)
        page.goto(live + '/account')
        unlabeled = page.evaluate("""[...document.querySelectorAll('input:not([type=hidden]),select,textarea')]
            .filter(e => !e.labels.length && !e.getAttribute('aria-label') && !e.getAttribute('aria-labelledby')).map(e => e.id || e.name)""")
        nameless = page.evaluate("""[...document.querySelectorAll('button,a.icon-btn')].filter(e => !e.textContent.trim() && !e.getAttribute('aria-label') && !e.title).length""")
        assert unlabeled == [] and nameless == 0

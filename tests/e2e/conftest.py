"""A real server on a real port, a real Chromium, and (optionally) axe-core for automated accessibility scans."""

import os
import threading

import pytest

playwright = pytest.importorskip('playwright.sync_api', reason='playwright is not installed')
from werkzeug.serving import make_server  # noqa: E402

PW = 'orange-kettle-42'
AXE_PATH = os.getenv('AXE_PATH', '/tmp/axe/node_modules/axe-core/axe.min.js')


@pytest.fixture()
def live(app):
    """The app served on a free port. Unlike the unit tests, CSRF protection is ON, because the browser sends real tokens."""
    app.config['WTF_CSRF_ENABLED'] = True
    server = make_server('127.0.0.1', 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()


@pytest.fixture(scope='session')
def browser_instance():
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:          # chromium not installed
            pytest.skip(f'Chromium is not available: {e}')
        yield browser
        browser.close()


@pytest.fixture()
def make_page(browser_instance):
    contexts = []

    def factory(width=1280, height=900, scheme='light', locale='en-US', permissions=None):
        ctx = browser_instance.new_context(viewport={'width': width, 'height': height}, color_scheme=scheme, locale=locale, permissions=permissions or [])
        contexts.append(ctx)
        page = ctx.new_page()
        page.js_errors = []
        page.on('pageerror', lambda e: page.js_errors.append(str(e)))
        page.on('console', lambda m: page.js_errors.append(m.text) if m.type == 'error' and 'ERR_NAME_NOT_RESOLVED' not in m.text and 'ERR_INTERNET' not in m.text and 'Failed to load resource' not in m.text else None)
        return page
    yield factory
    for c in contexts:
        c.close()


def sign_up(page, base, name='ann', email=None, pw=PW):
    page.goto(base + '/register')
    page.fill('#username', name)
    page.fill('#email', email or f'{name}@example.com')
    page.fill('#password', pw)
    page.fill('#confirm_password', pw)
    page.click('button[type=submit]')
    page.wait_for_load_state()


@pytest.fixture()
def axe_source():
    if not os.path.exists(AXE_PATH):
        pytest.skip('axe-core not found (npm install axe-core, then set AXE_PATH)')
    return open(AXE_PATH, encoding='utf-8').read()

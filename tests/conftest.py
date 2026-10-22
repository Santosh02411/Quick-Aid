"""
Shared pytest fixtures for Quick Aid.

Each test gets its own throwaway SQLite file (set via DATABASE_PATH before
any app/database import happens) so tests never touch the real quickaid.db
and can't interfere with each other or a running dev server.
"""

import os

# Most tests exercise analysis with mocked AI and no signed-in consent. The consent behaviour has its own tests, which turn this back on.
os.environ.setdefault('REQUIRE_AI_CONSENT', 'false')
import sys
import tempfile
import pytest

# Make the project root importable when running `pytest` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def temp_db_path(monkeypatch):
    """Point DATABASE_PATH at a fresh temp file before database.py is used."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    os.remove(path)  # start with no file - init_db() will create it
    monkeypatch.setenv('DATABASE_PATH', path)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture()
def db_module(temp_db_path):
    """
    A freshly-imported database module pointed at the temp DB.
    Reimported per test so DB_PATH (read at import time) picks up the
    temp path from temp_db_path rather than a module cached from an
    earlier test or a real run.
    """
    import importlib
    from quickaid.core import database
    importlib.reload(database)
    database.init_db()
    return database


@pytest.fixture()
def app(temp_db_path, monkeypatch):
    """A configured Flask app instance, isolated to a temp DB, for testing."""
    monkeypatch.setenv('SECRET_KEY', 'test-secret-key-for-pytest')
    monkeypatch.setenv('GEMINI_API_KEY', 'your_gemini_api_key_here')  # force basic-mode analyzers
    monkeypatch.setenv('DATABASE_PATH', temp_db_path)

    import importlib
    from quickaid.core import database
    importlib.reload(database)

    from quickaid import app as app_module
    importlib.reload(app_module)

    app_module.app.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
    })
    # The rate limiter's default storage is in-memory and PROCESS-wide (see extensions.py), not per-app-instance, so
    # without this, requests made by an earlier test in the same pytest run count against a later, unrelated test's
    # limit - occasional, hard-to-reproduce failures that depend on test order and how many tests ran first.
    from quickaid.extensions import limiter
    with app_module.app.app_context():
        limiter.reset()
    yield app_module.app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def registered_user(client):
    """Register a user and return their credentials; client stays logged in."""
    creds = {'username': 'testuser', 'email': 'testuser@example.com', 'password': 'supersecret123'}
    client.post('/register', data={**creds, 'confirm_password': creds['password']})
    return creds

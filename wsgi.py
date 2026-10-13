"""WSGI entry point for production servers.

    gunicorn -w 2 -b 0.0.0.0:5000 wsgi:app
"""

from quickaid.app import app

__all__ = ['app']

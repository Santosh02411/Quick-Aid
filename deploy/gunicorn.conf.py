"""gunicorn settings for production. Everything is overridable with environment variables."""

import os
import shutil

bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = int(os.getenv('WEB_CONCURRENCY', '2'))
threads = int(os.getenv('GUNICORN_THREADS', '2'))
worker_class = 'gthread'                       # threads let one worker wait on the AI model without blocking others
timeout = int(os.getenv('GUNICORN_TIMEOUT', '90'))          # the AI call can legitimately take a while
graceful_timeout = 30
keepalive = 5
max_requests = 1000                            # recycle workers so a slow leak can never grow without bound
max_requests_jitter = 100
worker_tmp_dir = '/dev/shm' if os.path.isdir('/dev/shm') else None

# Abuse limits at the HTTP layer (before the app sees anything)
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# The app writes its own privacy-safe access log (route PATTERNS, no tokens, no query strings). gunicorn's would record the raw path,
# which for /reset-password/<token> contains a live secret, so it is switched off.
accesslog = None
errorlog = '-'
loglevel = os.getenv('LOG_LEVEL', 'info').lower()

# Aggregate Prometheus metrics across worker processes.
os.environ.setdefault('PROMETHEUS_MULTIPROC_DIR', '/tmp/prometheus')
_PROM = os.environ['PROMETHEUS_MULTIPROC_DIR']


def on_starting(server):
    shutil.rmtree(_PROM, ignore_errors=True)
    os.makedirs(_PROM, exist_ok=True)


def child_exit(server, worker):
    try:
        from prometheus_client import multiprocess
        multiprocess.mark_process_dead(worker.pid)
    except Exception:
        pass

# Quick Aid - production container image
#
# Runs behind gunicorn (never the Flask dev server) and as a non-root user.
# The SQLite database and logs live under /app/data and /app/logs, which
# should be mounted as volumes (see docker-compose.yml) so they survive
# container restarts/rebuilds.

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr,
# which keeps container logs flowing to `docker logs` in real time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached across code-only
# changes - only requirements.txt changes force a reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code.
COPY . .

# Runtime directories: uploads (transient, cleared per-request by the app),
# the SQLite DB, and rotating log files. Created here so they exist with
# the right ownership even before a volume is mounted over them.
# The application code stays root-owned and read-only to the app user; only the data, log and upload folders are writable by it.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin quickaid \
    && mkdir -p /app/uploads /app/logs /app/data /tmp/prometheus \
    && chown -R quickaid:quickaid /app/uploads /app/logs /app/data /tmp/prometheus

USER quickaid

ENV DATABASE_PATH=/app/data/quickaid.db \
    LOG_DIR=/app/logs \
    FLASK_DEBUG=False \
    PORT=5000

EXPOSE 5000

# Liveness hits /health. Orchestrators / load balancers should use /ready for traffic decisions (database writable, schema current, disk space).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT','5000'), timeout=3).status == 200 else 1)"

# gunicorn (config in deploy/gunicorn.conf.py), never the Flask dev server. Exec form, so signals reach it and shutdown is graceful.
CMD ["gunicorn", "-c", "deploy/gunicorn.conf.py", "wsgi:app"]

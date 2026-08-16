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
    PIP_NO_CACHE_DIR=1

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
RUN mkdir -p /app/uploads /app/logs /app/data \
    && useradd --create-home --uid 1000 quickaid \
    && chown -R quickaid:quickaid /app

USER quickaid

ENV DATABASE_PATH=/app/data/quickaid.db \
    LOG_DIR=/app/logs \
    FLASK_DEBUG=False \
    PORT=5000

EXPOSE 5000

# Basic container-level health check hitting the app's own /health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).status == 200 else 1)"

# gunicorn, not the Flask dev server - matches the Procfile used for
# platform (Heroku-style) deployments.
CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:${PORT} --access-logfile - --error-logfile - app:app"]

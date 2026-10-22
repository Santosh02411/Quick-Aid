# Operating Quick Aid: migrations, backups, retention

## Database migrations

The schema is versioned (`schema_migrations` table, `quickaid/core/migrations.py`). The app upgrades the database
automatically at startup; you can also do it by hand:

```bash
python -m quickaid.core.migrations status     # current / latest version, what is pending
python -m quickaid.core.migrations upgrade    # apply pending migrations
python -m quickaid.core.migrations check      # exit 1 if anything is pending or an applied migration was edited (CI)
```

Guarantees:

* **Each migration is one transaction.** If it fails, it is rolled back completely and the database stays at the previous version.
* **A backup is taken first** (`pre-migration-vX-to-vY-...` in the backups folder) whenever the database already holds data, including
  databases created by older releases that predate the version table. These safety copies are never auto-deleted.
* **Never edit a migration that has shipped.** Add a new one at the end of `MIGRATIONS` (next number) with a test. Editing an applied
  migration is detected and logged.

## Backups

SQLite's online backup API makes a **consistent snapshot while the app is running** (copying the `.db` file with `cp` can capture a half-written
state). Each backup is integrity-checked before it is accepted, compressed, written `0600` into a `0700` folder, and described by a manifest
(SHA-256, schema version, row counts).

```bash
python -m quickaid.core.backup create            # BACKUP_DIR (default: ./data/backups), keeps the newest BACKUP_KEEP (14)
python -m quickaid.core.backup list
python -m quickaid.core.backup verify data/backups/quickaid-20261016-020000-000.db.gz
python -m quickaid.core.backup prune --keep 7
```

**Schedule it.** Either cron:

```
15 2 * * *  cd /srv/quickaid && /srv/quickaid/venv/bin/python -m quickaid.core.backup create >> logs/backup.log 2>&1
```

or Docker Compose: `docker compose --profile backup up -d` (daily, into the `quickaid-backups` volume).

**Two things a backup does not contain, and why that matters**

1. **The encryption key.** Health text is encrypted in the database; the key (`CHAT_ENCRYPTION_KEY`, or the one derived from `SECRET_KEY`) is
   deliberately *not* in the backup, so a stolen backup is not readable. Store the key in a password manager or secrets store, **separately**.
   Without it a restored database loads but its chats and analyses show as unavailable.
2. **Off-site copies.** A backup on the same disk does not survive the disk. Copy the folder to another machine or bucket (encrypt it in transit).

**Test your restores.** A backup you have never restored is a hope, not a backup. Try one on a copy every few months.

### Restoring

1. **Stop the app** (the restore refuses to run while another process holds the database).
2. `python -m quickaid.core.backup verify <file>`: confirms the checksum and SQLite integrity.
3. `python -m quickaid.core.backup restore <file>`: verifies again, saves the *current* database as `pre-restore-...` (so a restore can be undone),
   swaps the file atomically, and removes stale WAL files.
4. Start the app. It upgrades the restored database if it is older than the code.

## Data retention

| Data | Kept |
|---|---|
| Uploaded photos | deleted immediately after analysis; leftovers older than `UPLOAD_MAX_AGE_MINUTES` (30) are swept |
| Interview sessions | 24 hours |
| Sign-in / reset tokens | until they expire |
| Saved checks and chats | the person's choice (30 / 90 / 365 days / until deleted); `DEFAULT_RETENTION_DAYS` (365) until they choose; `HISTORY_MAX_RETENTION_DAYS` is a hard ceiling |
| Telemetry (no health text) | `TELEMETRY_RETENTION_DAYS` (90) |
| Database backups | newest `BACKUP_KEEP` (14) scheduled backups; **a deleted entry can remain inside older backups until they rotate out** |

A sweep enforces this about every `MAINTENANCE_INTERVAL_HOURS` (24). It is triggered by ordinary traffic and coordinated through the database, so it runs
once per interval however many workers there are. To run it from cron instead: `python -m quickaid.services.retention`.

People can always delete one entry, all history, or their account immediately, and export everything first. The plain-language version is the `/privacy` page,
generated from the live settings so it cannot drift from what the code does.

## Request tracing

Every response carries `X-Request-ID`, and error bodies include the same `request_id`. A valid incoming `X-Request-ID` (8-64 chars of `A-Za-z0-9._-`) is
reused, so a reverse proxy can propagate its own ids. Grep the log for the id to find the failing request. The user-facing 500 page shows it so a report can be traced.


## Health checks and monitoring

| Endpoint | Purpose | Who can call it |
|---|---|---|
| `GET /health` | **Liveness**: the process answers and the database is reachable. Restart the container if this fails. | anyone |
| `GET /ready` | **Readiness**: database writable, schema fully migrated, photos can be written, enough free disk. Send traffic only when this is 200. | anyone |
| `GET /metrics` | Prometheus metrics: requests and latency by *route pattern*, AI calls by outcome, analyses and emergencies, unhandled errors, schema version, uptime, database size. No user data. | `Authorization: Bearer $METRICS_TOKEN`, or an admin session. Everyone else gets 404. |

Suggested alerts: `/ready` failing for more than 2 minutes; `rate(quickaid_unhandled_errors_total[5m]) > 0`; `quickaid_ai_calls_total{outcome="failed"}` above ~5% of calls;
p95 of `quickaid_http_request_duration_seconds` above your target; `quickaid_last_maintenance_age_seconds` above 2x `MAINTENANCE_INTERVAL_HOURS`; and the dashboard's "fell back to basic mode" rate.
Every error response carries an `X-Request-ID`; the same id is in the server log, so a user's report can be traced to one request.

With several gunicorn workers, metrics are aggregated through `PROMETHEUS_MULTIPROC_DIR` (set by `deploy/gunicorn.conf.py`).

## Production deployment checklist

1. `cp .env.example .env`, fill in real values, **never commit it** (`python scripts/secret_scan.py --ignored` shows what your local `.env` holds).
2. Set `SESSION_COOKIE_SECURE=True`, `BEHIND_PROXY=True` (behind nginx/TLS), a long `SECRET_KEY` **and** an explicit `CHAT_ENCRYPTION_KEY`, `ADMIN_USERNAMES` (register that account first), and `RATELIMIT_STORAGE_URI` (Redis) if you run more than one instance.
3. `docker compose up --build -d`; optionally `docker compose --profile backup up -d` for daily backups. The web container is read-only with no capabilities; only the data and log volumes are writable.
4. Point your load balancer at `/ready` and your uptime monitor at `/health`.
5. Set up `SENTRY_DSN` (optional) and Prometheus scraping with `METRICS_TOKEN`.
6. Copy backups off the machine, store the encryption key separately, and test a restore.
7. Read the **Known limitations** in [SECURITY.md](SECURITY.md) before real patients use it, and get the translations reviewed ([TRANSLATION_REVIEW.md](TRANSLATION_REVIEW.md)).


## Account recovery: locked out of 2FA, or any account-management task

**For yourself, or when someone reports they cannot sign in**, and recovery codes are not an option (lost, never
saved, or already used up): reset their 2FA from the terminal or the web dashboard. Neither needs the person's
password, and both work even if the admin account itself is the one that is locked out (the CLI only needs shell
access on the server).

```bash
python -m quickaid.core.accounts_cli show USERNAME_OR_EMAIL        # confirm it is the right account first
python -m quickaid.core.accounts_cli reset-2fa USERNAME_OR_EMAIL   # disables 2FA, signs out everywhere, emails them
```

Or, signed in as an account listed in `ADMIN_USERNAMES`: **Users** in the sidebar (`/admin/users`) → find the
account → **Reset 2FA**. The web dashboard and the CLI call the same code
(`quickaid/services/admin_accounts.py`), so they always behave identically. Either way the action is logged to the
affected person's own security history and, if mail is configured, emailed to them - this is never invisible to
them. Full account listing, force sign-out-everywhere, emailing a password-reset link, and permanent deletion (with
a typed-confirmation guard) are also available from both; see the README's "Admin: managing users" section for the
full command list.

Deliberately **not** provided: an email-based one-time-code fallback for 2FA. That would let a compromised email
account (which is also the password-reset channel) bypass two-factor authentication entirely - a security
downgrade dressed up as a convenience. Recovery codes are the self-service path; admin reset is the correct last
resort, the same as how most real services actually handle this (verify identity out of band, then reset).

**Creating the first admin account**: register the account normally through the app, then add its username to
`ADMIN_USERNAMES` in `.env` and restart - never through the sign-up page or any page a logged-in user could reach.


## Results are slow, or say "taking too long"

1. **See what the model is doing.** `/metrics` (`quickaid_ai_call_duration_seconds`, `quickaid_ai_calls_total{outcome="failed"}`) and the admin dashboard show latency and the share of checks that fell back to the built-in mode.
2. **The limits** (see `.env.example`, section 9): every model request stops after `GEMINI_TIMEOUT_SECONDS` (25 s; photos get 10 s more); a request that *timed out* is not retried; the whole check stops waiting after
   `GEMINI_TOTAL_BUDGET_SECONDS` (40 s). So the longest anyone waits is bounded, and what they get then is the built-in check with first-aid steps, never an empty "try again".
3. **Thinking is off** for Flash models. If you switch `GEMINI_MODEL` to a Pro model, thinking cannot be disabled and answers will be slower; consider keeping Flash.
4. **The first question is instant** (built-in bank). `INTERVIEW_AI_PLAN=true` re-adds a model call before it.
5. **If everything looks like the "simple built-in check":** AI analysis is opt-in per person (see SECURITY.md). Results say so and offer a one-click "Turn on AI analysis".
6. A *consistently* slow provider is usually network egress or a rate limit on the API key: check `docker logs` for `Gemini ... attempt 1 failed`.

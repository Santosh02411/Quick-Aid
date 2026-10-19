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

# Database: SQLite now, PostgreSQL if and when it is justified

## Why SQLite is the right choice today

One file, no service to run, transactional, and fast for this workload (one app, mostly reads, small writes). It is configured for safety:
WAL journal (readers do not block the writer), `busy_timeout` (writers wait instead of failing), `foreign_keys=ON` on every connection,
`synchronous=NORMAL` (safe with WAL), explicit `BEGIN IMMEDIATE` transactions for multi-statement changes, versioned migrations, and verified online backups.
The test suite includes concurrent-writer and rollback tests.

## When to move to PostgreSQL

Move when a *measured* need appears, not before:

* more than one **application host** must write (SQLite is a single file on a single machine; network filesystems are unsafe for it);
* sustained **concurrent writes** where SQLite's single writer becomes the bottleneck (watch for `database is locked` in logs);
* you need **high availability, point-in-time recovery or read replicas**, which managed PostgreSQL gives you;
* the database outgrows one disk, or compliance asks for managed encryption/audit.

None of these is true of a single-instance deployment. Running several gunicorn *workers* on one host is fine with SQLite (WAL handles it).

## What would change (this is a plan: PostgreSQL is **not** implemented)

| Area | Today (SQLite) | For PostgreSQL |
|---|---|---|
| Driver | `sqlite3` in `core/database.py` | `psycopg` (or SQLAlchemy Core) behind the same functions |
| Placeholders | `?` | `%s` (or named parameters) |
| Keys | `INTEGER PRIMARY KEY AUTOINCREMENT` | `GENERATED ALWAYS AS IDENTITY` |
| Upserts | `INSERT OR REPLACE` (telemetry) | `INSERT ... ON CONFLICT DO UPDATE` |
| Booleans | integers 0/1 | `BOOLEAN` |
| Timestamps | ISO-8601 text (UTC) | `TIMESTAMPTZ` (text compares correctly today, so the move is mechanical) |
| Transactions | `BEGIN IMMEDIATE` | `BEGIN` + `SELECT ... FOR UPDATE` where needed |
| Pragmas, WAL | `PRAGMA ...` | none |
| Migrations | `core/migrations.py` | Alembic (the version numbers and names carry over) |
| Backups | online backup API | `pg_dump` / managed snapshots; the retention and key-handling advice is unchanged |
| Locking for the daily sweep | compare-and-set row in `maintenance_state` | the same row works; or `pg_try_advisory_lock` |

**How the code is already prepared.** SQL lives in three modules (`core/database.py`, `core/history_repo.py`, `core/migrations.py`); services and views never
contain SQL, and tests enforce that boundary (`tests/test_api_layer.py::TestArchitecture`). A port therefore means replacing those modules behind the same function
signatures, then re-running the whole suite against a PostgreSQL fixture. Application-level encryption (`core/crypto.py`) is database-independent.

**One real difference to plan for.** Free-text search decrypts a user's own rows in memory (bounded by `MAX_SCAN`) because the text is encrypted at rest.
That works the same on PostgreSQL. If a deployment ever needs server-side full-text search it must trade away at-rest encryption of that text, which is a decision to make deliberately.

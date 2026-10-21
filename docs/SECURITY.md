# Security

Quick Aid handles health information, so security and privacy are part of the product, not a layer on top. This page says what is protected and
how, what to do if a secret leaks, and, just as important, what is **not** protected.

## If a secret has leaked (a runbook)

A secret (an API key, `SECRET_KEY`, an SMTP or OAuth credential) is **leaked** the moment it is committed to a repository, pasted into a chat, or
put in an archive someone else can open, even if you delete it a minute later. Do these in order:

1. **Revoke and replace the credential at its source.** For the Gemini key: Google AI Studio / Cloud console, delete the key, create a new one and restrict it.
   Look at its usage for activity you do not recognise. *Rewriting git history does not un-leak a key that was ever public.*
2. **Rotate `SECRET_KEY` without losing data.** Encrypted data is keyed from `CHAT_ENCRYPTION_KEY` (or from `SECRET_KEY` if that is unset), so change both safely:
   ```bash
   # .env: keep the OLD value as the previous key, put the NEW ones in place
   CHAT_ENCRYPTION_KEY_PREVIOUS=<the old SECRET_KEY, or the old CHAT_ENCRYPTION_KEY>
   SECRET_KEY=<new random value>                    # python -c "import secrets; print(secrets.token_hex(32))"
   CHAT_ENCRYPTION_KEY=<new random value>
   # restart, then:
   python -m quickaid.core.rekey --dry-run          # what would change
   python -m quickaid.core.rekey                    # backs up first, re-encrypts everything in one transaction
   # finally remove CHAT_ENCRYPTION_KEY_PREVIOUS
   ```
   Changing `SECRET_KEY` also signs every user out (their session cookies were signed with the old key). That is intended.
3. **Remove the secret from git history** (needs a force-push; tell collaborators to re-clone):
   `git filter-repo --path .env --invert-paths`, then enable GitHub secret scanning and push protection.
4. **Check what else could be affected**: `python scripts/secret_scan.py --history` lists every secret-shaped value and forbidden file ever committed (masked).
5. **Never share archives built by hand.** Build releases with `python scripts/package_release.py --version N`: it packages only files git would publish
   (never `.env`, never `.git`, never databases or backups) and refuses to finish if its own scan finds a secret.
6. Install the commit guard once per clone: `sh scripts/install_hooks.sh` (blocks a commit that adds a secret or a forbidden file).

## What is protected, and how

| Area | Protection |
|---|---|
| Secrets | `.env` is git-ignored and docker-ignored; CI scans the tree and history; releases are built only from tracked files; secret values are never printed in full anywhere |
| Health data at rest | Symptoms, results, notes, chats, interview state and 2FA secrets are encrypted (Fernet: AES-128-CBC + HMAC) before they reach SQLite; old rows are migrated automatically |
| Key rotation | `CHAT_ENCRYPTION_KEY_PREVIOUS` + `python -m quickaid.core.rekey` (see above); the re-key refuses to run if it would make data unreadable |
| Passwords | Slow salted hashes; 8-128 characters; a list of the most common passwords, simple sequences, and passwords containing your username or email name are refused |
| Sign-in | 5 wrong passwords per account per address, 25 per account overall, per 15 minutes (so a stranger cannot easily lock the real owner out); identical response and timing for unknown usernames |
| Sessions | Fixed on sign-in (the old session is discarded); idle timeout (default 60 min) and absolute timeout (default 12 h); changing the password, enabling/disabling 2FA, "sign out other devices" and a password reset all invalidate other devices; `HttpOnly`, `SameSite=Lax`, `Secure` + `__Host-` prefix when `SESSION_COOKIE_SECURE` is on |
| 2FA | TOTP with single-use replay protection, a 5-attempt lockout and a 10-minute window; 10 single-use recovery codes, stored only as hashes, shown once |
| Password reset | Generic response (no account enumeration); at most 3 emails per address per hour; links are single-use and expire in 1 hour; a newer email kills older links; the owner is emailed; every device is signed out |
| Account takeover | Signing in with Google to an *unverified* local account destroys that account's old password and sessions (pre-hijacking defence) |
| CSRF | Every state-changing route requires a token; a test enumerates all routes and fails if one is exempt |
| Uploads | Judged by content, never by name or declared type; size, pixel-count and dimension limits; decompression-bomb defence; fuzz tests; deleted (overwritten, then unlinked) right after analysis, even on a crash |
| Access control | Every query carries the owner's id; someone else's data returns the same 404 as nothing at all; rate limits follow the *person*, not the network address |
| Browser | Strict CSP with per-request nonces, `nosniff`, frame denial, COOP/CORP, HSTS, `no-store` on every page that shows health data |
| Logs | Route *patterns* (never raw paths with tokens), no usernames, emails or text; a filter redacts emails, tokens, API keys and IPs even from tracebacks |
| Error reporting | Off unless `SENTRY_DSN` is set; then no default PII, no bodies, cookies, headers, query strings, variables or breadcrumbs, and all text is redacted |
| Deletion | `secure_delete` zeroes freed database content; the log file is truncated; account/history deletion also vacuums; a deletion log (ids and times only) is replayed after a backup restore so nobody is resurrected |
| Containers | Non-root, read-only filesystem, all capabilities dropped, no privilege escalation, memory/CPU/process limits, no `.env` or `.git` in the image |

## What goes to the AI provider

Only when a person has agreed (the default is that nothing leaves the server): the text they typed, their interview answers, chat messages and
the photo, **re-encoded without location, camera or time metadata**. Never their name, email, account id, notes or other history. Google handles
the data under the terms of the Gemini API plan in use, which can differ by plan; the operator must check which plan that is. A change to what is sent
bumps `AI_CONSENT_VERSION` so everyone who agreed is asked again.

## Known limitations (please read)

* **The translated emergency text has not been reviewed by clinicians.** See [TRANSLATION_REVIEW.md](TRANSLATION_REVIEW.md). Until it is, the app says so on screen.
* **Without an AI key the rules are a floor, not a safety net.** Measured on fresh phrasing the English keyword rules found only about a third of emergencies; the Hindi, Marathi and Kannada
  rules are new and have been tested only against examples written by their authors.
* **Wording checks (softening "you definitely have…", removing doses) work on English text only.** For other languages the structured fields (urgency, emergency flag) are still
  enforced, but free-text phrasing is not rewritten.
* **Signing in with Google skips this app's own 2FA**, by design (Google's own protections apply). Turn that off by not configuring Google sign-in.
* **Rate-limit counters are per process** unless `RATELIMIT_STORAGE_URI` points at Redis. With several workers or instances, use Redis.
* **SQLite is one file on one host.** Backups are verified snapshots; see [OPERATIONS.md](OPERATIONS.md). A deletion also stays in older *backups* until they rotate out.
* **Overwriting a file cannot be guaranteed on SSDs or copy-on-write filesystems.** Photos are kept for seconds; use full-disk encryption on the host.
* **Nothing here is a substitute for a penetration test.** Get one before putting real patients' data in.

## Reporting a vulnerability

Please do not open a public issue. Email the maintainer (see the repository profile) with steps to reproduce; you will get an acknowledgement within a few days.

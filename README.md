# Quick Aid 🏥

AI-powered medical assistant for injury detection and health analysis.

---

## Features

- 🔐 **User Accounts:** Sign up / log in required to use the AI features - each user's history is private to them.
- 🔑 **Account Management:** Password reset via email, email verification, an account settings page (change username/email/password), full GDPR-style account deletion, optional TOTP two-factor authentication, and optional Google OAuth/SSO sign-in.
- 📸 **Image Analysis:** Detect injuries and skin conditions from images, powered by Gemini.
- 🤒 **Symptom Checker:** AI-based health recommendations based on described symptoms.
- 📋 **History:** Past image analyses and symptom checks are saved per account so you can look back at them (stored locally in SQLite).
- 🚨 **Emergency Info:** Quick access to safety tips and urgent care guidance — always public, no login required.
- 💡 **Medical Guidance:** Educational recommendations and health safety guidelines.
- 🛡️ **Basic-mode fallback:** Still works without a Gemini key, using simple rule-based image/symptom heuristics.
- 📈 **Logging:** Structured logs (console + rotating file) covering requests, auth events, and Gemini failures, plus a `/health` endpoint for uptime monitoring.

---

## Configuration

Quick Aid requires a Gemini API key to enable AI-powered analysis (it falls back to a
basic rule-based mode without one).

1. Copy `.env.example` to `.env` (if `.env` doesn't already exist).
2. Add your Gemini API key, generate a `SECRET_KEY`, and set `FLASK_DEBUG`:

   ```env
   GEMINI_API_KEY=your_api_key_here
   SECRET_KEY=<output of: python -c "import secrets; print(secrets.token_hex(32))">
   FLASK_DEBUG=False
   ```

3. Save the file and restart the application. **Never commit `.env`** — it's already excluded via `.gitignore`.

---

## Account Features

Beyond basic username/password auth, Quick Aid supports:

| Feature | Route(s) | Notes |
|---|---|---|
| Password reset | `/forgot-password`, `/reset-password/<token>` | Single-use, hashed token, expires in 1 hour. Same response shown whether or not the email exists (no account enumeration). |
| Email verification | sent on signup & email change, `/verify-email/<token>` | Token expires in 24 hours. Verification is informational, not a login gate — unverified accounts can still use the app; resend it from Account Settings. |
| Account settings | `/account` | Change username, email, or password; see verification/2FA status. |
| Account deletion | `/account/delete` | Permanently deletes the account and all saved history (cascading delete). Requires password + typing `DELETE`. |
| Two-factor auth (TOTP) | `/account/2fa/setup`, `/login/2fa` | Standard authenticator-app codes (Google Authenticator, Authy, 1Password, etc). Login becomes a two-step flow once enabled. |
| Google OAuth/SSO | `/login/google` | Optional — hidden entirely unless `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` are set. |

### Outgoing email

Password-reset and verification links are sent via SMTP. Without SMTP configured, Quick Aid still works for local dev/testing — emails are written to the log instead of sent, so you can copy the link straight out of the console. Set these in `.env` for real delivery:

```env
SMTP_HOST=smtp.yourprovider.com
SMTP_PORT=587
SMTP_USERNAME=you@yourdomain.com
SMTP_PASSWORD=your_smtp_password
SMTP_USE_TLS=True
MAIL_FROM=Quick Aid <no-reply@yourdomain.com>
```

Also set `APP_BASE_URL` (e.g. `https://quickaid.example.com`, no trailing slash) so emailed links point at your real domain rather than whatever host handled the request — important once you're behind a proxy/load balancer.

### Google OAuth/SSO (optional)

To enable "Continue with Google": create an OAuth 2.0 Client ID at the [Google Cloud Console](https://console.cloud.google.com/apis/credentials) with an authorized redirect URI of `<APP_BASE_URL>/login/google/callback`, then set:

```env
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
```

Leave both blank to disable it — the button won't appear on the login/register pages.

---

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Add your Gemini API key in `.env` (see Configuration above).
3. Run the app:

   ```bash
   python app.py
   ```

   For production, run behind a real WSGI server instead of the Flask dev server:

   ```bash
   gunicorn -w 2 -b 0.0.0.0:5000 app:app
   ```

4. Open your browser at:

   ```
   http://localhost:5000
   ```

5. Sign up for an account (`/register`) to use image analysis, symptom checking, and history. The landing page and `/emergency` stay public without an account.

---

## Running with Docker

The included `Dockerfile` runs the app under gunicorn as a non-root user, with
the SQLite database and logs persisted outside the container via volumes.
gunicorn itself is plain HTTP - **nginx sits in front of it and handles TLS
termination**, so the app is only ever reachable over HTTPS from outside the
compose network.

**With docker compose (recommended):**

```bash
cp .env.example .env               # fill in real GEMINI_API_KEY and SECRET_KEY
./nginx/generate-dev-certs.sh      # self-signed cert for local HTTPS
docker compose up --build
```

Open `https://localhost` (your browser will warn about the self-signed
certificate locally - that's expected; see "TLS in production" below for
real certs). Data survives `docker compose down`; only
`docker compose down -v` removes the `quickaid-data` / `quickaid-logs` volumes.

The `web` service (gunicorn) is **not** published to the host - only `nginx`
is, on ports 80/443. This means:
- Port 80 redirects to 443.
- `nginx` forwards to `web:5000` over the private compose network and sets
  `X-Forwarded-For`/`X-Forwarded-Proto`, which the app trusts via
  `BEHIND_PROXY=True` (see `app.py`'s `ProxyFix` setup) to recover the real
  client IP for rate limiting and the real scheme for secure cookies/links.
- To reach gunicorn directly for local debugging (bypassing nginx and TLS),
  uncomment the `ports:` mapping under the `web` service in
  `docker-compose.yml`.

### TLS in production

Swap the self-signed dev cert for a real one before deploying anywhere
public. The simplest path is [certbot](https://certbot.eff.org/) in standalone
or webroot mode - point it at `nginx/certs/` (or run it as its own container
sharing that volume) so it writes `fullchain.pem` and `privkey.pem` there,
and set up a renewal cron/systemd-timer job (certs expire every 90 days).
`nginx/nginx.conf` already serves `/.well-known/acme-challenge/` on port 80
for certbot's HTTP-01 challenge. Alternatively, if you're deploying behind a
cloud load balancer (ALB, Cloud Load Balancing, etc.) that already terminates
TLS for you, you can drop the `nginx` service entirely and point the load
balancer straight at the `web` service - just make sure it forwards
`X-Forwarded-For`/`X-Forwarded-Proto` the same way nginx does here.

**With plain Docker (no TLS - for internal/trusted networks only):**

```bash
docker build -t quickaid .
docker run -p 5000:5000 --env-file .env \
  -v quickaid-data:/app/data \
  -v quickaid-logs:/app/logs \
  quickaid
```

This exposes gunicorn's plain HTTP directly, so only use it somewhere already
behind TLS (e.g. an internal network, or your own reverse proxy in front).

The image exposes a container-level `HEALTHCHECK` that hits `/health`, so
`docker ps` shows the container's actual health status.

---

## Continuous Integration

`.github/workflows/ci.yml` runs on every push/PR to `main` (and weekly on a
schedule): the full `pytest` suite, plus `pip-audit` against
`requirements.txt` to catch newly-disclosed CVEs in pinned dependencies even
when nothing else changes.

---

## Usage

- Create an account or log in.
- Upload images of injuries or skin conditions for AI analysis.
- Enter symptoms in text for personalized health insights.
- View past results any time on the **History** page.
- Access emergency information and safety guidelines any time, logged in or not.

---

## Running Tests

The test suite covers keyword extraction, the Gemini structured-output parsing
(including its fallback paths), database/auth logic, and the full Flask
request/response cycle (auth flow, route protection, input validation).

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests run against an isolated temporary SQLite file and never touch your real
`quickaid.db`, and use a placeholder `GEMINI_API_KEY` so they exercise the
basic-mode fallback logic rather than making real API calls.

---

## Logging & Monitoring

- Logs go to both the console and a rotating file at `logs/app.log` (5MB per file, 5 backups kept).
- Every request is logged with method, path, status code, and duration.
- Gemini failures (bad key, network error, malformed response) are logged with
  full tracebacks instead of failing silently, before falling back to basic-mode analysis.
- Set `LOG_LEVEL` (default `INFO`) and `LOG_DIR` (default `./logs`) via environment variables.
- `GET /health` returns `{"status": "ok", "database": "ok", "gemini_configured": true|false}` — point an uptime monitor or load balancer health check at it.

---

## Data & Privacy

- Uploaded images are analyzed and then deleted immediately — they are not stored.
- Passwords are hashed (never stored in plaintext) using Werkzeug's `generate_password_hash`.
- Analysis *results* (not the images themselves) are saved to a local SQLite database
  (`quickaid.db`) scoped to your account, so you can view your history.
- Clear your history any time from the History page.

---

## Developer Information

- **Developed by:** Santosh Madannavar
- **Email:** [santosh@example.com](mailto:santosh@example.com)
- **GitHub:** [github.com/Santosh02411](https://github.com/Santosh02411)
- **LinkedIn:** [linkedin.com/in/santosh](https://linkedin.com/in/santosh)

---

## Disclaimer

⚠️ Quick Aid is for **educational purposes only**. Always consult healthcare professionals for medical advice. In case of emergency, seek immediate help.

---

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python (Flask), Flask-Login for auth, Flask-Limiter for rate limiting
- **Persistence:** SQLite (built into Python, no separate DB server required)
- **AI:** Google Gemini via the [`google-genai`](https://pypi.org/project/google-genai/) SDK, using structured JSON output (Pydantic schemas) for reliable parsing
- **Fallback analysis:** NumPy/Pillow-based rule heuristics when no Gemini key is configured
- **Testing:** pytest, with a Flask test client and an isolated temp-file SQLite DB per test
- **Logging:** Python's standard `logging` module, console + rotating file handler

---

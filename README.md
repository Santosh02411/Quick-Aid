# Quick Aid 🏥

AI-powered medical assistant for injury detection and health analysis, built with Flask and Google Gemini.

> ⚠️ **Disclaimer:** Quick Aid is for **educational purposes only** and is not a substitute for professional medical advice. Always consult a healthcare professional. In an emergency, call your local emergency number immediately.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Configuration (`.env`)](#configuration-env)
5. [Running the App](#running-the-app)
6. [Docker & Production](#docker--production)
7. [Account Features](#account-features)
8. [How Detection Accuracy Works](#how-detection-accuracy-works)
9. [Testing](#testing)
10. [Logging & Monitoring](#logging--monitoring)
11. [Data & Privacy](#data--privacy)
12. [Tech Stack](#tech-stack)
13. [Developer](#developer)

---

## Features

- 🎨 **Redesigned interface** – calm green identity (coral is used only for emergencies), a **light/dark theme** that follows your system and can be switched any time, and a layout that works on phones (bottom tab bar) and desktops (sidebar).
- 📊 **Dashboard** – activity counters, a photo check and a symptom check side by side, recent checks, and a one-tap emergency call.
- 🚦 **Clear triage** – every result opens with a plain-language level: *Likely manageable at home*, *See a doctor soon* or *Get medical help now*, then what to do, what to avoid and when to see a doctor.
- 🔐 **Better sign-up and sign-in** – show/hide (eye) button on every password field, **confirm password** with a live match check, a password-strength meter, and your typed details kept if something is wrong.
- 📸 **Image analysis** – Gemini with a strict clinical-triage prompt, photo-quality checks and a deterministic safety net (see [How Detection Accuracy Works](#how-detection-accuracy-works)).
- 🤒 **Symptom checker** – understands negations ("no fever, just a cough"), always scans for emergencies, and treats self-harm language with care.
- 🔑 **Account management** – password reset by email, email verification, account settings, account deletion, optional TOTP two-factor auth, optional Google sign-in.
- 📋 **History** – past checks saved per account with what was recommended.
- 🚨 **Emergency guide** – public, region-aware numbers (India by default: 112 / 108) and first-aid basics.
- 🛡️ **Basic-mode fallback** – works without a Gemini key, clearly labelled as a rough check.
- 📈 **Logging** – console + rotating file logs, plus a `/health` endpoint for uptime monitoring.

---

## Project Structure

```
Quick-Aid/
├── quickaid/                    # Application package
│   ├── app.py                   #   Flask app: config, security headers, all routes
│   ├── core/                    #   Shared infrastructure
│   │   ├── database.py          #     SQLite access: users, tokens, history
│   │   ├── logging_config.py    #     Console + rotating-file logging
│   │   └── localization.py      #     Region-aware emergency numbers / units
│   ├── accounts/                #   Authentication & account features
│   │   ├── auth.py              #     Flask-Login user loader
│   │   ├── oauth.py             #     Google OAuth / SSO (optional)
│   │   ├── two_factor.py        #     TOTP 2FA + QR code generation
│   │   └── mailer.py            #     SMTP email (reset / verification)
│   ├── ai/                      #   Gemini-powered analysis
│   │   ├── medical_analyzer.py  #     Image analysis (+ basic-mode fallback)
│   │   ├── symptom_checker.py   #     Symptom analysis (+ basic-mode fallback)
│   │   └── conversation.py      #     Multi-turn follow-up Q&A service
│   ├── static/
│   │   ├── css/app.css          #   Design system (light + dark themes)
│   │   └── js/app.js            #   Theme toggle, password UX, dashboard tools
│   └── templates/               #   Jinja2 HTML templates
│       ├── base.html            #     App shell (sidebar / mobile tabs / public nav)
│       ├── auth_base.html       #     Split-screen layout for sign-in pages
│       ├── _macros.html         #     Icons + password field (eye, strength, match)
│       └── index.html …         #     Landing page (visitors) / dashboard (signed in)
├── tests/                       # pytest suite
├── deploy/
│   └── nginx/                   # nginx.conf, generate-dev-certs.sh, certs/
├── data/                        # SQLite database (runtime, git-ignored)
├── logs/                        # Rotating log files (runtime, git-ignored)
├── uploads/                     # Temporary image uploads (runtime, git-ignored)
├── .github/workflows/ci.yml     # CI: tests + dependency audit
├── run.py                       # Development entry point
├── wsgi.py                      # Production entry point (gunicorn)
├── Dockerfile
├── docker-compose.yml           # web (gunicorn) + nginx (TLS)
├── Procfile                     # Heroku-style platforms
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # + pytest, pip-audit
├── pytest.ini
├── .env.example                 # Documented config template (committed)
└── .env                         # Your real config (NEVER committed)
```

---

## Quick Start

**Prerequisites:** Python 3.10+ (3.12 recommended) and `pip`.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your config and fill it in (see the next section)
cp .env.example .env               # Windows: copy .env.example .env

# 4. Run
python run.py
```

Open **http://localhost:5000**, register an account at `/register`, and start using image analysis, the symptom checker and history. The landing page and `/emergency` stay public.

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and fill it in. Every variable is explained in detail inside `.env.example`; this is the summary.

### Required

| Variable | What to put | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | Your Gemini API key (starts with `AIza…`). Leave the placeholder to run in basic, non-AI mode. | <https://aistudio.google.com/app/apikey> |
| `SECRET_KEY` | A random 64-character string. Keep it private and stable, or everyone is logged out on restart. | Run `python -c "import secrets; print(secrets.token_hex(32))"` |
| `FLASK_DEBUG` | `False` (always in production). `True` only for local development. | – |
| `GEMINI_MODEL` *(optional)* | Leave blank for `gemini-flash-latest`. Set e.g. `gemini-pro-latest` for the best accuracy (slower, costs more). | <https://ai.google.dev/gemini-api/docs/models> |

### Server / deployment (optional)

| Variable | Default | Notes |
|---|---|---|
| `APP_REGION` | `IN` | Emergency numbers shown in the app and used in AI advice: `IN`, `US`, `CA`, `UK`, `EU`, `AU` or `INTL`. |
| `PORT` | `5000` | Dev server port (`run.py`). |
| `APP_BASE_URL` | *(blank)* | Public URL, no trailing slash, e.g. `https://quickaid.example.com`. Used in emailed links and the Google redirect URI. Set it in production. |
| `BEHIND_PROXY` | `False` | `True` only behind a reverse proxy you control (nginx in `docker-compose.yml`). |
| `SESSION_COOKIE_SECURE` | `False` | `True` on any HTTPS deployment; `False` for plain `http://localhost`. |
| `RATELIMIT_STORAGE_URI` | `memory://` | Use `redis://host:6379` in production with more than one worker/instance. |

### Storage & logging (optional)

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_PATH` | `data/quickaid.db` | SQLite file location. |
| `UPLOAD_FOLDER` | `uploads/` | Temporary uploads (deleted after analysis). |
| `LOG_DIR` | `logs/` | Rotating `app.log`. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |

### Outgoing email (optional)

Without SMTP credentials the app still works: reset/verification emails are written to `logs/app.log` so you can copy the link. For real delivery, fill in **all** of `SMTP_HOST`, `SMTP_USERNAME` and `SMTP_PASSWORD`.

| Variable | What to put |
|---|---|
| `SMTP_HOST` | Server hostname, e.g. `smtp.gmail.com` |
| `SMTP_PORT` | `587` (STARTTLS, most common) or `465` (SSL) |
| `SMTP_USERNAME` | SMTP login, usually your full email address |
| `SMTP_PASSWORD` | Password / API key. **Gmail:** a 16-character [App Password](https://myaccount.google.com/apppasswords) (requires 2-Step Verification), not your normal password |
| `SMTP_USE_TLS` | `True` for port 587 |
| `MAIL_FROM` | `Quick Aid <no-reply@yourdomain.com>` – must be an address your provider lets you send from |

### Google sign-in (optional)

Leave `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` blank to hide the "Continue with Google" button. To enable it:

1. Go to <https://console.cloud.google.com/apis/credentials> → **Create credentials → OAuth client ID → Web application**.
2. Add an authorized redirect URI: `http://localhost:5000/login/google/callback` (local) and/or `https://your-domain/login/google/callback` (production).
3. Copy the Client ID and Client secret into `.env`.

> 🔒 **Never commit `.env`.** It is git-ignored and excluded from the Docker image. If a key ever leaks, revoke it and generate a new one.

---

## Running the App

| Mode | Command |
|---|---|
| Development | `python run.py` |
| Production (gunicorn) | `gunicorn -w 2 -b 0.0.0.0:5000 wsgi:app` |
| Docker + nginx + TLS | `docker compose up --build` (see below) |

---

## Docker & Production

`docker-compose.yml` runs two services: `web` (gunicorn, plain HTTP, not published to the host) and `nginx` (TLS termination on ports 80/443, forwarding to `web` with `X-Forwarded-*` headers).

```bash
cp .env.example .env                       # fill in GEMINI_API_KEY and SECRET_KEY
./deploy/nginx/generate-dev-certs.sh       # self-signed cert for local HTTPS
docker compose up --build
```

Open **https://localhost** (your browser will warn about the self-signed certificate locally – expected). The database and logs persist in named volumes; `docker compose down` keeps them, `docker compose down -v` deletes them.

**Real TLS certificates:** place `fullchain.pem` and `privkey.pem` in `deploy/nginx/certs/` (e.g. via [certbot](https://certbot.eff.org/); nginx already serves `/.well-known/acme-challenge/` on port 80) and renew before the 90-day expiry. Behind a cloud load balancer that already terminates TLS you can drop the `nginx` service and point the balancer at `web`, forwarding `X-Forwarded-For` / `X-Forwarded-Proto`.

**Plain Docker (trusted networks only):**

```bash
docker build -t quickaid .
docker run -p 5000:5000 --env-file .env \
  -v quickaid-data:/app/data -v quickaid-logs:/app/logs quickaid
```

**Heroku-style platforms** use the included `Procfile` (`gunicorn … wsgi:app`); set the environment variables in the platform dashboard instead of a `.env` file.

---

## Account Features

| Feature | Route(s) | Notes |
|---|---|---|
| Password reset | `/forgot-password`, `/reset-password/<token>` | Single-use hashed token, expires in 1 hour; same response whether or not the email exists. |
| Email verification | sent on signup/email change, `/verify-email/<token>` | Expires in 24 hours; informational, not a login gate. |
| Account settings | `/account` | Change username, email, password; see verification / 2FA status. |
| Account deletion | `/account/delete` | Permanently deletes the account and all history. Requires password + typing `DELETE`. |
| Two-factor auth (TOTP) | `/account/2fa/setup`, `/login/2fa` | Works with Google Authenticator, Authy, 1Password, etc. |
| Google sign-in | `/login/google` | Optional; hidden unless Google credentials are configured. |

Other routes: `/` (home), `/emergency` (public), `/history`, `/upload` (image analysis), `/analyze_symptoms`, `/api/history`, `/api/history/clear`, `/health`.

---

## How Detection Accuracy Works

No tool can diagnose from a photo or a sentence, so Quick Aid is built to be *careful* rather than confident.

**Photo analysis**
- The photo is auto-rotated (EXIF), flattened and size-capped before analysis, so phone photos are read the right way up.
- Gemini gets a strict system instruction: first decide whether the image is medical at all, then judge photo quality, then **describe what is visible before concluding**, give at most three hedged possibilities, and rate urgency by the *worst plausible* reading (deep wounds, large or facial burns, spreading infection and similar are always high).
- Low temperature and a structured schema keep answers consistent.
- A deterministic safety net runs on the answer: any warning sign forces at least "see a doctor soon", and a blurry or non-medical photo can never be reported with confidence. Poor photos get specific retake advice instead of a guess.
- Without a Gemini key a colour-region check (skin detection, redness, dark discolouration) gives a clearly labelled rough guide, and it will not claim a problem is minor.

**Symptom checking**
- The AI is told to respect negations, ask itself "what is the worst this could be", and avoid definite diagnoses. Your text is passed as quoted data, so it cannot rewrite the instructions.
- A rule-based emergency scanner (chest pain, trouble breathing, stroke signs, severe bleeding, allergic swelling, poisoning, stiff neck with fever, very young babies with fever, self-harm) **always runs on your text, even when the AI says otherwise**, and can only raise urgency, never lower it.
- The offline fallback matches whole words and understands "no fever, just a cough".

For the most accurate results, set a real `GEMINI_API_KEY` (and optionally a Pro model via `GEMINI_MODEL`). Results are still educational guidance, not medical advice.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v
```

Tests run against an isolated temporary SQLite file and a placeholder `GEMINI_API_KEY`, so they use the basic-mode fallback and never call the real API or touch your real database. They cover sign-up validation, the password UI hooks, theme assets, the dashboard, the red-flag scanner and the AI safety nets. CI (`.github/workflows/ci.yml`) runs the suite plus `pip-audit` on every push/PR to `main` and weekly.

---

## Logging & Monitoring

- Logs go to the console and to a rotating file at `logs/app.log` (5 MB × 5 backups).
- Every request is logged with method, path, status and duration; Gemini failures are logged with full tracebacks before falling back to basic mode.
- `GET /health` returns `{"status": "ok", "database": "ok", "gemini_configured": true|false}` – point an uptime monitor or load balancer at it.

---

## Data & Privacy

- Uploaded images are analyzed and then deleted immediately – they are not stored.
- Passwords are hashed with Werkzeug (never stored in plaintext).
- Only analysis *results* are saved to SQLite, scoped to your account; clear them any time from the History page, or delete your account entirely.

---

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript (Jinja2 templates)
- **Backend:** Python, Flask, Flask-Login, Flask-WTF (CSRF), Flask-Limiter
- **Database:** SQLite
- **AI:** Google Gemini via the [`google-genai`](https://pypi.org/project/google-genai/) SDK with structured (Pydantic) output
- **Fallback analysis:** NumPy / Pillow heuristics
- **Auth extras:** PyOTP + qrcode (2FA), Authlib (Google OAuth)
- **Deployment:** gunicorn, Docker, nginx
- **Testing / CI:** pytest, GitHub Actions, pip-audit

---

## Developer

- **Developed by:** Santosh Madannavar
- **Email:** [santoshmadannavar@gmail.com](mailto:santoshmadannavar@gmail.com)
- **GitHub:** [github.com/Santosh02411](https://github.com/Santosh02411)
- **LinkedIn:** [linkedin.com/in/santosh-madannavar-3ba5a7261](https://linkedin.com/in/santosh-madannavar-3ba5a7261)

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
6. [Docker & Production](#docker-production)
7. [Account Features](#account-features)
8. [How Detection Accuracy Works](#how-detection-accuracy-works)
9. [Follow-Up Chat & Interview: How They Stay Safe](#follow-up-chat-interview-how-they-stay-safe)
10. [Triage & Safety Engine](#triage-safety-engine)
11. [Explainable Results & Reliability](#explainable-results-reliability)
12. [Evaluation (and what it honestly shows)](#evaluation-and-what-it-honestly-shows)
13. [AI Evaluation Dashboard](#ai-evaluation-dashboard)
14. [Photo Upload Workflow](#photo-upload-workflow)
15. [Architecture](#architecture)
16. [History, Notes, Export & Retention](#history-notes-export-retention)
17. [API and OpenAPI](#api-and-openapi)
18. [Database, Migrations and Backups](#database-migrations-and-backups)
19. [Security and Privacy](#security-and-privacy)
20. [Languages](#languages)
21. [Monitoring and Deployment](#monitoring-and-deployment)
22. [Testing](#testing)
23. [Logging & Monitoring](#logging-monitoring)
24. [Data & Privacy](#data-privacy)
25. [Tech Stack](#tech-stack)
26. [Developer](#developer)

---

## Features

- 🎨 **Redesigned interface** – calm green identity (coral is used only for emergencies), a **light/dark theme** that follows your system and can be switched any time, and a layout that works on phones (bottom tab bar) and desktops (sidebar).
- 📊 **Dashboard** – activity counters, a photo check and a symptom check side by side, recent checks, and a one-tap emergency call.
- 🚦 **Clear triage** – every result opens with a plain-language level: *Likely manageable at home*, *See a doctor soon* or *Get medical help now*, then what to do, what to avoid and when to see a doctor.
- 🔐 **Better sign-up and sign-in** – show/hide (eye) button on every password field, **confirm password** with a live match check, a password-strength meter, and your typed details kept if something is wrong.
- 📸 **Image analysis** – Gemini with a strict clinical-triage prompt, photo-quality checks and a deterministic safety net (see [How Detection Accuracy Works](#how-detection-accuracy-works)).
- 🤒 **Adaptive symptom interview** – instead of judging one sentence, it asks the short follow-up questions a nurse would (who it's for, how long, how bad, getting better or worse, what else is happening, and medical history *when it matters*), skips anything you already said, then gives a structured assessment: possible explanations with likelihood, recommended next steps, warning signs, urgency and timeframe. Emergency signs **stop the interview instantly** with guidance, and you can always press "Get my assessment now".
- 💬 **Follow-up chat** – ask follow-up questions about any past symptom check or photo analysis (or a general health question). Dedicated chat page with conversation history, new chat, delete, suggested questions, and medical safety limits. Chats are stored **encrypted**.
- 🔑 **Account management** – password reset by email, email verification, account settings, account deletion, optional TOTP two-factor auth, optional Google sign-in.
- 📋 **History** – past checks saved per account with what was recommended.
- 🚨 **Emergency guide** – public, region-aware numbers (India by default: 112 / 108) and first-aid basics.
- 🧭 **Central triage & safety engine** – every AI result is validated before it is shown: urgency can only be raised by safety rules, danger signs in the AI's own words force a minimum urgency, overconfident wording is softened, medicine doses are removed, and every change is logged on the result.
- 🔍 **Explainable results** – what you reported, what a photo shows, what it could be (with the evidence for each), what to do and what could not be assessed are kept apart, with a reliability panel and an "uncertainty" statement. Nothing is presented as a diagnosis, and the AI's confidence is never shown as a probability.
- 📷 **Photo-quality gate** – blurry, dark, tiny or blank photos are caught automatically and get a "please retake" answer instead of a guess.
- 🧪 **Measured, not assumed** – a curated evaluation with held-out sets and a CI gate; the results (including the weak ones) are documented below.
- 📊 **Internal AI evaluation dashboard** – emergency recall, false-negative and false-positive rates, structured-output validity, API failure rate, latency and cost per analysis, from versioned datasets and live telemetry, with run-to-run comparison across models and prompt changes. Admins only.
- 📷 **Photo upload workflow** – drag and drop, paste, camera capture, instant preview with quick quality hints, real upload progress, cancel and retry, content-based validation and plain-language error messages. Visible findings and AI interpretations are shown separately, and basic image processing is never presented as diagnosis.
- 🗂️ **History you control** – search, filter, sort and paginate every check; open a full printable report; add private notes; export everything (JSON/CSV); delete one entry, all history, or your account. Health text and notes are encrypted at rest, and strict per-user access means nobody else can read, change or even detect your entries.
- ⏳ **Clear retention policy** – photos deleted immediately; saved checks kept for a period you choose (default 1 year); an automatic daily clean-up enforces it; the `/privacy` page states the live policy.
- 🧱 **Modular backend** – Blueprints, service layer, one standard API response and error shape, request validation, request ids, and an OpenAPI description generated from the code.
- 🗄️ **Safer database** – versioned transactional migrations with an automatic pre-upgrade backup, indexes for the history queries, WAL + explicit transactions, verified online backups and a guarded restore.
- 🛡️ **Basic-mode fallback** – works without a Gemini key, clearly labelled as a rough check.
- 📈 **Logging** – console + rotating file logs, plus a `/health` endpoint for uptime monitoring.

---

## Project Structure

```
Quick-Aid/
├── quickaid/
│   ├── app.py                   # Application factory: wires everything together (no routes, no logic)
│   ├── config.py, extensions.py # Constants and policy / Flask extensions (CSRF, rate limiter, login)
│   ├── blueprints/              # HTTP views, one Blueprint per area (thin: parse -> call a service -> respond)
│   │   ├── pages.py             #   home, emergency, privacy
│   │   ├── auth.py              #   register, sign-in, 2FA, Google, password reset, account settings
│   │   ├── symptoms.py          #   one-shot check + adaptive interview
│   │   ├── images.py            #   photo upload and analysis
│   │   ├── conversations.py     #   follow-up chats (search, rename, export, delete)
│   │   ├── history.py           #   search / filter / paginate, reports, notes, export, delete
│   │   ├── privacy.py           #   retention policy and each person's choice
│   │   ├── admin.py             #   internal AI evaluation dashboard
│   │   └── system.py            #   health, regions, API docs
│   ├── services/                # Business logic, no HTTP: analysis, chats, history, retention, safety facade
│   ├── api/                     # Standard responses, structured errors, request validation (Pydantic), OpenAPI
│   ├── web/hooks.py             # Request ids, security headers, no-store caching, regional context, maintenance trigger
│   ├── ai/                      # Models, the triage/safety engine, prompts, telemetry, photo validation, reliability
│   ├── core/                    # database.py, migrations.py, backup.py, history_repo.py, crypto.py, localization
│   ├── i18n/                    # Hindi / Marathi / Kannada: UI strings, emergency guidance + detection, review status
│   ├── observability/           # Prometheus metrics, readiness checks, privacy-safe error reporting
│   ├── evaluation/              # Datasets, runs and the evaluation dashboard back end
│   ├── accounts/                # Auth helpers, mail, 2FA, Google OAuth
│   ├── static/ templates/       # Front end (css/app.css; js: app, chat, history, admin, docs)
├── evaluation/                  # Versioned datasets, saved runs, reference-image docs, CLI tools
├── docs/                        # SECURITY.md, OPERATIONS.md, DATABASE.md, TRANSLATION_REVIEW.md
├── scripts/                     # secret_scan.py, package_release.py (safe archives), git hook installer
├── tests/                       # pytest suite (mocked models only: no network); tests/e2e = real-browser journeys + accessibility audit
├── deploy/ docker-compose.yml Dockerfile  # nginx + gunicorn, optional scheduled backups
└── .env.example                 # every setting, documented
```

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
| `CHAT_ENCRYPTION_KEY` *(optional)* | Long random string used to encrypt chats and interview answers. Blank = derived from `SECRET_KEY`. **Keep it stable** or old chats become unreadable. | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `ADMIN_USERNAMES` *(optional)* | Comma-separated usernames allowed to open the evaluation dashboard (`/admin/evaluation`) **and** the user-management dashboard (`/admin/users` - see "Admin: managing users" below, e.g. for recovering a 2FA-locked account). Blank = nobody. Everyone else sees "not found". | – |
| `DEFAULT_RETENTION_DAYS` / `HISTORY_MAX_RETENTION_DAYS` *(optional)* | Days saved checks and chats are kept until a person chooses (default 365; 0 = until deleted) / an operator ceiling nobody can exceed (0 = none) | – |
| `UPLOAD_MAX_AGE_MINUTES`, `MAINTENANCE_INTERVAL_HOURS` *(optional)* | Leftover-photo cleanup age (30) and how often the automatic clean-up runs (24) | – |
| `BACKUP_DIR`, `BACKUP_KEEP` *(optional)* | Where backups go and how many scheduled ones to keep (14) | – |
| `API_DOCS_PUBLIC` *(optional)* | `true` shows `/api/docs` to every signed-in user instead of admins only | – |
| `GEMINI_PRICE_INPUT_PER_M` / `GEMINI_PRICE_OUTPUT_PER_M` *(optional)* | USD per **million** tokens, to show cost per analysis. Tokens are always counted; cost is shown only if both are set. We don't guess prices. | your provider's pricing page |
| `PROMPT_VERSION_LABEL`, `TELEMETRY_RETENTION_DAYS`, `EVAL_DIR` *(optional)* | A human tag stored with each run; days to keep telemetry (default 90); where evaluation files live | – |
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

Chat and interview API (all require login, all scoped to your account; other people's ids return 404):

| Endpoint | Purpose |
|---|---|
| `POST /api/interview/start` | Begin the interview (or return emergency guidance immediately) |
| `POST /api/interview/answer` | Answer or skip the current question |
| `POST /api/interview/finish` | Get the assessment now with what has been answered |
| `GET/POST /api/chats` | List chats / start one (about an analysis, or general) |
| `GET/DELETE /api/chats/<id>` | Open or delete a chat |
| `POST /api/chats/<id>/messages` | Send a message and get the reply |
| `POST /api/follow_up` | One-call follow-up on an analysis (finds or creates its chat) |
| `GET /api/regions`, `POST /api/region` | List regions / set yours (emergency numbers follow it) |

Other routes: `/` (home), `/chat`, `/emergency` (public), `/history`, `/upload` (image analysis), `/analyze_symptoms`, `/api/history`, `/api/history/clear`, `/health`.

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
- A rule-based emergency scanner (chest pain, trouble breathing, stroke signs, severe bleeding, allergic swelling, poisoning, stiff neck with fever, very young babies with fever, self-harm) **always runs on your text, even when the AI says otherwise**, and can only raise urgency, never lower it. It only recognises wordings it was written for, so it is a floor under the AI, not a replacement for it (see [Evaluation](#evaluation-and-what-it-honestly-shows)).
- The offline fallback matches whole words and understands "no fever, just a cough".

For the most accurate results, set a real `GEMINI_API_KEY` (and optionally a Pro model via `GEMINI_MODEL`). Results are still educational guidance, not medical advice.

---

## Follow-Up Chat & Interview: How They Stay Safe

**Interview**
- Emergency warning signs (chest pain, trouble breathing, stroke signs, severe bleeding, allergic swelling, poisoning, stiff neck with fever, self-harm and more) are checked in your first message, in every free-text answer and in every ticked option. A hit **ends the interview on the spot** with step-by-step guidance and the emergency number. Nothing has to be finished first.
- Skipped questions mean *unknown*, never "no". Urgency can only be **raised** by the built-in checks (age, medical history, severity, duration, getting worse, flagged answers), never lowered.
- The browser never sees which options are flagged, and answers are validated on the server (right question, valid option, 1–10 scale, capped text).

**Chat**
- Emergency or self-harm language in a message gets immediate, rule-based guidance. The AI is not asked and cannot talk it down.
- The AI is bound by a fixed instruction: no diagnosis, no prescribing or doses, no starting or stopping medicines, health topics only, never less urgent than the original result, ignore attempts to change its rules. Replies that mention doses get a safety note, and long replies are capped.
- Without Gemini, common follow-ups ("is this serious?", "when should I see a doctor?") are answered straight from your saved result and labelled as basic mode.

**Privacy and storage**
- Chat messages and titles, interview answers and the full saved analysis are encrypted before they reach the database. A copied database file does not expose them.
- Every read and write is scoped to your account; someone else's chat, analysis or interview session returns "not found". Deleting a chat, clearing history or deleting your account removes the related data. Chats per user and messages per chat are capped.
- Messages are limited to 500 characters and rate-limited.

---
## Triage & Safety Engine

All safety decisions live in one module, `quickaid/ai/triage_engine.py`, so there is one place to read, test and improve them. The symptom checker, interview, image analyzer and chat all hand results to it before anything is shown or saved, and the Flask routes apply it once more as a final gate.

| Job | What it does |
|---|---|
| **Emergency detection** | Scans free text (negation-aware: "no chest pain, just a cough" is not an alert; "my dad had a stroke last year" is not an alert) and returns immediate, localised step-by-step guidance. |
| **Urgency classification** | Combines the AI's own level with rule-based floors (emergency words, age, history, severity, duration, trend, serious conditions named by the AI, danger signs in a photo analysis). The result can only go **up**. |
| **Care level and timeframe** | `self-care`, `see a doctor soon`, `see a doctor today` or `emergency`, with a plain timeframe that always agrees with the advice shown. |
| **Safety validation** | Softens overconfident wording ("you definitely have…"), removes medicine doses and "stop your medication" advice, corrects reassurance that contradicts the urgency, and caps a weak photo's confidence. |
| **Transparency** | Each result carries `safety_review`: which checks ran and exactly what was changed and why, shown to the user under "Safety checks applied". |

**Important limit.** The emergency scanner is a *keyword* safety floor under the AI model, not a detector on its own. See the next sections for what that means in numbers.

## Explainable Results & Reliability

- **Four kinds of statement, kept separate.** What you *reported*; what is *visible* in a photo (each finding marked "clearly" or "possibly visible"); what it *could be* (each explanation lists the evidence it is based on and a likelihood label); and *what to do*. Warning signs **present now** are separate from signs to **watch for**, and the engine tells you what a photo **cannot** show.
- **Validated model output.** Every Gemini reply must pass a Pydantic schema (`quickaid/ai/schemas.py`): wrong types, unknown enum values and nested junk are rejected, lists are cleaned and capped, and a missing urgency defaults to *medium*, never *low*. An answer that cannot be validated is discarded and replaced by a safe "could not assess" result, never shown.
- **Photo-quality checks** (resolution, sharpness, exposure, glare, contrast) run before the model. An unusable photo never reaches the AI. A weak one is passed to it with the findings, and caps confidence and reliability.
- **Reliability, not probability.** The panel says how much information the result was based on ("Limited", "Moderate" or "Reasonably well-informed") and why. The AI's own confidence is shown only as its rough self-rating, explained as unverified. It is never converted into a percentage.
- **Safe fallbacks.** Too little information, a poor photo, a non-medical image or a malformed AI reply all give "Not enough to assess" with retake or next-step advice and the emergency number, never reassurance.

## Evaluation (and what it honestly shows)

```bash
python evaluation/run_evaluation.py               # offline, no API key
python evaluation/run_evaluation.py --weak-model  # also simulate an unsafe AI that always says "low"
python evaluation/run_evaluation.py --live        # also run the real Gemini model (needs GEMINI_API_KEY)
```

For running against the real model specifically — exact commands, what each metric means, and how to compare prompt/model versions without accidentally comparing apples to oranges — see **`evaluation/LIVE_EVALUATION.md`**.

It exits non-zero if a safety threshold is missed, and runs in CI. Datasets are in `evaluation/datasets/`; the full report is `evaluation/reports/latest.md`.

| What was measured | Result |
|---|---|
| Main set (97 cases, 29 emergencies): emergency recall / false alarms / severe under-triage | 100% / 0% / 0% |
| Unsafe or malformed **model outputs** pushed through the real code (30 cases, incl. poisoning / severe allergic reaction / unconsciousness / breathing difficulty) | 30 / 30 handled safely |
| **Photo-quality** (is this a usable photo?) checks on synthetic photos across a realistic skin-tone range (12 scored cases + a dedicated fairness suite) | 12 / 12 right; see the skin-tone note below - **this is not a measurement of medical interpretation accuracy** |
| **Held-out set 1** (written after tuning, run once, then fixed, now no longer fresh) | recall **11.8% → 100%** after the fixes (`HOLDOUT_FIRST_RUN.md`) |
| **Held-out set 2** (fresh, incl. Hindi/Kannada, run once → fixed, now no longer fresh) | recall **30.8% → 100%** after the fixes (`HOLDOUT2_RESULT.md`) |
| **Held-out set 3** (fresh, incl. Devanagari/Kannada script, typos, colloquial phrasing, run once → fixed, now no longer fresh) | recall **37.9% → 100%** after the fixes (`HOLDOUT3_RESULT.md`) |
| **Held-out set 4** (fresh: children/pregnancy/older adults, several cases grounded in WHO/NHS published danger signs, plus contradictory phrasing; run once, **left unfixed by deliberate choice** - see below) | recall **22.2%**, not improved (`HOLDOUT4_RESULT.md`) |
| Live Gemini model | **not measured** (needs your key and real network access: see `evaluation/LIVE_EVALUATION.md`) |

**How to read this.**

- The 100% on the main set is **in-sample**: those cases were written while the rules were being tuned, so it mostly shows the rules fit their own tests.
- **Each held-out set tells the same story once, honestly, before any fix:** real recall on wording the rules have never seen is much lower than the main set suggests (11.8%, then 30.8%, then 37.9% — each a *different* fresh set, each written after the previous one's fixes). Every one of those first-contact results is preserved exactly as measured, specifically so they cannot be quietly improved away: `HOLDOUT_FIRST_RUN.md`, `HOLDOUT2_RESULT.md`, `HOLDOUT3_RESULT.md`. A round of fixes then brought each set to 100% recall / 0% severe under-triage / 0% false alarms — verified, not assumed, by rerunning after every change.
- **What that pattern means:** three cycles of "fresh set finds gaps → fixes close them → recall returns to 100%" is evidence the fixes are real, not evidence the rules have become complete. A **fourth** fresh set would very plausibly still find new gaps in wording nobody anticipated — hand-written patterns cannot be made complete by writing more patterns, only narrower. Detection now goes beyond single-phrase keyword matching in three ways: a **combination-signal layer** (e.g. "went pale" + "clutched his arm" + "fell down" together imply a cardiac emergency with no single strong phrase), a **typo-tolerance layer** (a curated misspelling table, plus careful fuzzy matching restricted to long words individually checked against real-word collisions), and **one-shot context parsing** (duration/age/severity now apply to a single message, not only the question-and-answer interview).
- Therefore: **the AI model is the primary reader and the rules are only a floor.** Without an AI model (basic mode) the app can still miss emergencies phrased in genuinely novel words or other languages. The interface says this on the dashboard and on every basic-mode result, keeps the emergency number visible during every check (including when the pipeline fails outright — a last-resort text scan runs before any generic error is shown, at every stage: one-shot check, interview start/answer/finish, photo check, and chat), and never answers an unrecognised description with reassurance.
- A new `false_reassurance_rate` metric measures how often the **model's own raw text** claims reassurance ("nothing to worry about", "this will clear up on its own"...) on a case that genuinely needed caution — captured before the safety net rewrites it, so it is a read on model quality, not on what the person actually sees (the net corrects it every time, which is what the guardrail set checks separately).
- **Held-out set 4 was deliberately left unfixed.** It specifically targets children, pregnancy and older adults (several cases grounded in WHO/UNICEF IMCI, WHO/Government-of-India obstetric, and NHS geriatric published danger-sign criteria, not just developer judgement), plus contradictory and hedging descriptions. Its 22.2% recall is **not** a bug that slipped through; it is the honest result of running a fresh set **and not tuning the rules against it**, by explicit choice, so the project keeps at least one genuinely untouched measurement rather than converging every set toward 100% through repeated exposure. See `HOLDOUT4_RESULT.md` for exactly what was missed and the sources used.
- **Image quality and medical interpretation are different claims, and only one of them is measured here.** The automatic photo-quality gate (blur, darkness, contrast, resolution, glare) runs on-server before any AI call and is what the table above tests - it answers "is this photo usable at all", never "what does it show". Testing it found a real fairness problem: under identical lighting, the darkness/contrast checks penalised photos of darker skin tones far more readily, because absolute pixel brightness is a biased proxy for "not enough light" once different skin tones (which naturally reflect different amounts of light) are considered. That has been **measurably reduced** (see `quickaid/ai/image_quality.py` and `tests/test_image_quality_fairness.py`) but **not eliminated** - it is a structural limit of judging brightness from one photo with no reference, not something threshold-tuning alone can fully close. Separately, EXIF-based photo rotation correction existed but had never been tested; it now is (`tests/test_image_orientation_and_content.py`), alongside photos with no EXIF data, and photos that are technically "good quality" but unrelated to any medical concern (the model's own prompt handles that classification, not the pixel-level quality gate - see the same test file for why that split exists). **None of this tests medical interpretation accuracy - whether the AI correctly identifies what a wound or rash actually is - which has no real, independently-labelled photo set to measure against yet** (`evaluation/reference/README.md` explains what one would need and why none ships with this app). Do not describe image analysis as validated injury or disease detection; describe it as what has actually been measured, which is photo-usability screening.
- The datasets were written by the developers, are **not clinician-validated**, and are small. These numbers are an engineering regression check, **not evidence that the app is clinically accurate or safe for real patients**.
- Next steps: run `--live` and publish the numbers (see `evaluation/LIVE_EVALUATION.md` for exact commands and what each metric means); have a clinician review and extend the datasets; build the children/pregnancy/older-adult vocabulary gaps `HOLDOUT4_RESULT.md` lists, then measure against a fresh, not-yet-tuned held-out set 5; get a real, independently-labelled photo set for medical-interpretation accuracy (not just photo-quality) testing.

---
## AI Evaluation Dashboard

**Enable it:** put your username in `ADMIN_USERNAMES` in `.env`, restart, and open **AI evaluation** in the sidebar (`/admin/evaluation`). Anyone else, signed in or not, gets a "not found" page, and the API routes behave the same way.

Two kinds of numbers, deliberately kept apart:

| | **Evaluation runs** | **Live telemetry** |
|---|---|---|
| Source | the system scored on a curated dataset with known answers | what really happens in the running app |
| Can show | emergency recall, false-negative rate, false-positive rate, precision, under-triage | structured-output validity, API failure rate, latency (p50/p95), tokens, cost per analysis, fallback rate, how often the safety engine raised urgency |
| Cannot show | how it does on real traffic | whether answers were *right*: nobody labels real conversations |

**Definitions.**
*Emergency recall* = emergencies that raised the alert ÷ all emergencies. *False-negative rate* = 1 − recall (missed emergencies). *False-positive rate* = false alarms ÷ cases where there is no emergency. *Structured-output validity* = model answers that passed the Pydantic schema ÷ answers received (an invalid answer is discarded and the app falls back). *API failure rate* = failed model calls ÷ calls (each retry counts). *Cost per analysis* = total token cost ÷ finished analyses (symptom, photo, interview assessment, chat reply), including planning calls.

**What is stored.** Per model call and per analysis: kind, model, prompt version, timings, token counts, validity and outcome labels. **Never** user text, images or identities. Old rows are deleted after `TELEMETRY_RETENTION_DAYS`.

**Versioned everything.** Each saved run (`evaluation/runs/*.json`) records the model, a hash **fingerprint of every prompt** (so any prompt edit is tracked automatically), the safety-engine version, and the **version and SHA-256 of each dataset used**. Pick two runs in the dashboard (or `--compare A B`) to see metric changes with better/worse verdicts, which cases regressed or improved, exactly which prompt or dataset changed, and warnings when the comparison is unfair (different dataset versions, mock model, model *and* prompt changed together).

**Datasets are versioned too** (`evaluation/datasets/MANIFEST.json`: version, hash, role, who labelled it, clinician-reviewed?, changelog). Editing a file without `python evaluation/manage_datasets.py bump NAME --note "..."` fails CI. Held-out test sets are **frozen** and refuse to be bumped, because tuning against a test set turns it into a training set.

**Mocks for tests, real outputs for judgement.**

```bash
python evaluation/run_evaluation.py --model offline                 # rules only
python evaluation/run_evaluation.py --model mock-flaky              # synthetic API errors + invalid JSON (no network)
python evaluation/run_evaluation.py --model live --record evaluation/recordings/r1.jsonl --label "prompt v2"   # REAL model, costs money
python evaluation/run_evaluation.py --model replay --replay evaluation/recordings/r1.jsonl                       # re-score recorded real output for free
python evaluation/run_evaluation.py --list-runs ;  python evaluation/run_evaluation.py --compare RUN_A RUN_B
```

Automated tests use only mocked clients and never touch the network. Real model output is evaluated against **reference data the model and prompt author did not write**: the curated symptom datasets (written by the developers and *not* clinician-reviewed, which the dashboard says) and, for photos, a reference set **you** supply with provenance and independent labels (`evaluation/reference/README.md`). Recording real responses means a safety-rule change can be re-scored on the *same* outputs without spending anything. Live runs stay on the command line: the dashboard can start only offline and mock runs.

**Evaluation traffic is isolated** from live telemetry, so running an evaluation never pollutes production numbers.

## Admin: managing users

There is no "become an admin" option anywhere a signed-up user could reach, on purpose - an admin account is created the same way as the evaluation dashboard's: put a username in `ADMIN_USERNAMES` in `.env` (comma-separated for more than one) and restart. That account then sees **Users** in the sidebar alongside **AI evaluation**; everyone else, signed in or not, gets a "not found" page on both the page and its API routes.

**What it is for:** account-level administration, never health data. The dashboard (`/admin/users`) shows username, email, verification and 2FA status, account age - never anyone's symptoms, photos, chats or notes, and that boundary is tested (`tests/test_admin_users.py`). From it you can:

| Action | What it does |
|---|---|
| **Reset 2FA** | Disables two-factor authentication and clears recovery codes, signs the account out everywhere. **This is how to recover a locked-out account** - lost authenticator app, no saved recovery codes, codes rejected. The person can set 2FA up again once they are back in. |
| **Sign out everywhere** | Invalidates every device currently signed in as that account. |
| **Email password reset** | Sends the normal password-reset email on the person's behalf. |
| **Delete** | Permanently deletes the account and everything it owns, via the same secure-deletion path (overwrite, deletion tombstone) a person deleting their own account uses. Requires typing the username again as confirmation; cannot be undone. |

Every action is logged to **that person's own** "recent security activity" (visible to them on their Account page) and, where there is somewhere to send it, emails them - an admin action is never invisible to the person it affects.

**The same actions from a terminal**, for when you only have shell access (or the account that would do the fixing is itself the one that is locked out):

```bash
python -m quickaid.core.accounts_cli list                                  # every account: username, email, verified, 2FA, admin, created
python -m quickaid.core.accounts_cli show USERNAME_OR_EMAIL                # one account's detail + recent security activity
python -m quickaid.core.accounts_cli reset-2fa USERNAME_OR_EMAIL           # recover a locked-out account
python -m quickaid.core.accounts_cli force-logout USERNAME_OR_EMAIL
python -m quickaid.core.accounts_cli send-password-reset USERNAME_OR_EMAIL [--site https://your-domain.example]
python -m quickaid.core.accounts_cli delete USERNAME_OR_EMAIL --confirm USERNAME_OR_EMAIL
```

The web dashboard and the CLI call the exact same code (`quickaid/services/admin_accounts.py`), so the two can never behave differently.

**Why there is no "forgot my 2FA, email me a one-time code instead" option:** that would be a real security downgrade, not a convenience - it would mean a compromised email account (which is *also* the password-reset channel) bypasses two-factor authentication entirely. Recovery codes are the correct self-service path for a lost authenticator; admin reset (CLI or dashboard) is the correct last resort when those are not available either, the same way most real services handle it (contact whoever can verify it is really you, out of band).

## Photo Upload Workflow

- **Add a photo any way you like:** drag and drop, paste from the clipboard, *Choose a photo*, or *Take a photo*. On phones this opens the native camera; on computers, a live camera preview with capture (and *Switch camera* when there is more than one). If camera access is blocked the app says exactly why and how to fix it.
- **Instant, honest feedback:** a preview with size and a quick on-device check of lighting, focus and resolution, so you can retake a weak photo *before* uploading. This check is only a hint and is labelled as not medical.
- **Clear errors, in plain words, with what to do next:** wrong type, empty file, over 16 MB, iPhone HEIC, corrupt or truncated image, too small, absurdly large dimensions. The server repeats every check and judges by the file's **real content**, never its name or the type the browser claims.
- **Real upload progress** with cancel, then "checking photo quality and analysing…", and a retry button on network trouble.
- **Unsuitable photos are not guessed at.** A blurry, dark, glared, tiny or blank photo gets "Not enough to assess" with specific retake tips and *Take another photo* / *Choose another photo* buttons. Nothing is saved to history for a photo that was rejected.
- **Observation and interpretation are separate cards.** *What is visible in the photo* (numbered findings, each "clearly" or "possibly visible", labelled **Observed**) is kept apart from *What it could mean* (labelled **AI interpretation · unverified**, each guess pointing at the findings it rests on). *What a photo cannot tell us* is always shown.
- **Basic image processing is never presented as diagnosis.** Without AI, the colour check reports measurements only ("about 13% of the skin area is redder than the rest"), offers **no** explanation ("Not available"), and gives general first aid that does not depend on guessing what is wrong. AI interpretations are labelled as not validated against clinical outcomes.
- **Privacy:** the uploaded file is validated in memory, analysed, and **deleted immediately, even if the analysis crashes**. The camera permission is granted to this site only; microphone and location stay blocked.

---
## Architecture

Request flow: **Blueprint (HTTP) → service (business logic) → repository (SQL) / AI layer**, with medical safety applied inside the services through a single facade.

* **Blueprints** only parse the request, call one service method, and return the standard response. They contain no SQL and do not import the database or the AI classes, and a test enforces this.
* **Services** hold the logic and know nothing about HTTP (no `request`, no `jsonify`): `analysis` (symptoms, photos, interview), `chats`, `history`, `retention`, and `safety`, the only door to the triage engine.
* **One API shape.** Success: `{"success": true, ...payload, "request_id": "..."}`. Failure: `{"success": false, "error": "...", "code": "entry_not_found", "request_id": "...", "hints": [...], "details": [{"field","message"}]}`. Services raise `ApiError`; one handler turns it into that shape, so no endpoint invents its own. HTTP errors are JSON for API calls and HTML for pages, and an unexpected error never leaks internals (the user sees a reference id; the log has the trace).
* **Request validation** with Pydantic models (`api/schemas.py`) returns per-field `details`; the app's long-standing message wording is kept in `api/validation.py`.
* **Per-app services.** `create_app()` builds fresh services and extensions for each app instance, so tests and workers are isolated. `from quickaid.app import app` still works for `wsgi.py` and `run.py`.

## History, Notes, Export & Retention

* **Search and filters** (`/history`, `GET /api/history`): words that must all appear in symptoms, conditions, advice, filenames or your notes; type, urgency, emergency-only, date range, sort, page size. Plain GET form, so it works without JavaScript and links are shareable. Invalid input gets a clear message, never a crash.
* **Report** (`/history/<type>/<id>`): the full analysis (visible findings kept apart from interpretations for photos), reliability and safety checks, your interview answers, linked conversations, print/save as PDF, JSON download.
* **Notes:** up to 2000 characters per entry, encrypted, searchable by you, included in your export.
* **Export:** `GET /api/history/export?format=json|csv` (everything, including notes and chats) and per-entry/per-chat downloads. CSV cells that start with `=`, `+`, `-` or `@` are neutralised so a spreadsheet cannot run them as formulas.
* **Delete:** one entry (with the chats about it, atomically), all history, or the whole account, all immediate.
* **Access control:** every query carries the owner's id in its `WHERE` clause. Someone else's entry gives the **same 404 as one that never existed**, so ids cannot be probed. A two-user test matrix covers every history, chat and privacy endpoint. Pages and APIs that carry medical data are sent `Cache-Control: no-store`.
* **Retention:** see the table in [docs/OPERATIONS.md](docs/OPERATIONS.md#data-retention). Photos are deleted immediately; leftovers from a crash are swept after 30 minutes. Each person picks 30 / 90 / 365 days or "until I delete it" (Account → Your data). A daily sweep, coordinated through the database so it runs once however many workers exist, enforces it. **Honest caveat:** a deleted entry can persist in older database backups until they rotate out (14 by default), and the privacy page says so.

## API and OpenAPI

Endpoints are documented right next to their code with `@doc(...)`, and the OpenAPI 3.0 document is generated from the live routing table and the same Pydantic models that validate requests, so it cannot drift. A test fails if any API route lacks documentation and validates the result with `openapi-spec-validator`.

* `/api/openapi.json` and a built-in viewer at `/api/docs` (admins by default; `API_DOCS_PUBLIC=true` for any signed-in user). No external CDN is used, so the strict security policy stays intact.
* Authentication is the session cookie; state-changing requests also need the CSRF token in `X-CSRFToken`.

## Database, Migrations and Backups

SQLite remains the database (WAL, busy timeout, foreign keys, explicit transactions). Full procedures are in [docs/OPERATIONS.md](docs/OPERATIONS.md); the PostgreSQL path (a plan, **not implemented**) is in [docs/DATABASE.md](docs/DATABASE.md).

```bash
python -m quickaid.core.migrations status|upgrade|check
python -m quickaid.core.backup create|list|verify FILE|prune --keep N|restore FILE
python -m quickaid.services.retention          # run the retention sweep now
```

* **Migrations are versioned and transactional.** Each runs in its own transaction and rolls back completely on failure. Before upgrading a database that holds data, a verified backup is written automatically, **including databases from older releases**. This release's migrations add notes, an emergency flag on photo analyses, history and maintenance indexes, per-user retention, and **encrypt the health text that earlier releases stored readable** (idempotent).
* **Backups are consistent snapshots** taken while the app runs (SQLite online backup API), integrity-checked, compressed, private (`0600`), checksummed, and rotated. Restore is guarded: it verifies the backup, refuses while the database is in use, and saves what it replaced. **The encryption key is not inside a backup. Store it separately,** or restored data cannot be read.

---
## Security and Privacy

The full policy, the credential-leak runbook and an honest list of limitations are in **[docs/SECURITY.md](docs/SECURITY.md)**. In short:

* **Secrets never leave your machine by accident.** `.env` is git- and docker-ignored. `python scripts/secret_scan.py` checks the tree, staged changes, **full history** (`--history`) and finished archives, printing findings masked. `python scripts/package_release.py --version N` builds archives only from files git would publish (never `.env`, never `.git`) and refuses to finish if it finds a secret. Install the commit guard with `sh scripts/install_hooks.sh`.
* **If a secret *was* committed**, rotate it at its source first, then follow the runbook (key rotation without data loss: `CHAT_ENCRYPTION_KEY_PREVIOUS` + `python -m quickaid.core.rekey`).
* **Nothing is sent to an external AI until the person agrees.** Until then everything is handled on the server in a basic mode. When they agree, only what they typed and the photo (with location, camera and time details removed) is sent to Google Gemini, never their name, email or account details. A "no" is remembered and can be changed in Account.
* **Sign-in hardening:** lockout that a stranger can't use to lock you out, session versioning and idle/absolute timeouts, 2FA with single-use recovery codes and replay protection, safer password reset, a password blocklist, and a "recent security activity" list on the Account page.
* **Deletion is physical:** freed database content is zeroed, uploads are overwritten before removal, and a deletion log makes sure restoring an old backup cannot bring deleted data back.
* **Logs are privacy-safe** (route patterns, no usernames, emails or health text, plus a redaction filter), and error reporting (optional Sentry) sends no personal data.

## First aid, built in, and speed

* **Everyday first aid works without AI.** Fifteen guides cover the problems people actually open the app for: cuts and scrapes, burns and scalds, nosebleeds, sprains, bruises, insect stings, animal bites, blisters, splinters, sunburn, something in the eye, a bump on the head, a suspected broken bone, heat exhaustion, and **choking**. Each gives what it probably is, ordered first-aid steps, what *not* to do, and exactly when to get it seen, and moves up to "see a doctor today" or "call the emergency number" when the description contains a danger sign (a deep cut that won't stop bleeding, a dog bite, swollen lips after a sting, a bone showing). They are standard published first-aid steps, written conservatively and **in English only**; the safety engine reviews them and can only make advice more cautious.
* **Injuries skip the questionnaire** (it asks about fever and cough), giving the steps immediately; illnesses still get follow-up questions, and the first question appears instantly.
* **Time limits keep results fast:** a hard per-request timeout, the model's "thinking" switched off for these structured tasks, no pointless retry after a timeout, and a total budget, then the built-in result is shown rather than an error. Settings and troubleshooting are in [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Languages

The interface and, above all, the **emergency guidance** work in English, **हिन्दी (Hindi), मराठी (Marathi) and ಕನ್ನಡ (Kannada)**. The language follows the person's saved choice, then a cookie, then their browser; a menu on every page switches it (no JavaScript needed).

* **Detecting emergencies** works in each script *and* in Roman letters (how many people type on phones): about 50 patterns across 12 warning signs plus self-harm wording, with negation ("सीने में दर्द *नहीं* है" is not an alarm), spelling variants, and clause boundaries. About 70 example phrases are tested, plus 17 everyday complaints that must **not** raise an alarm.
* **The instructions** for every warning sign are translated with the same number of steps as English, the emergency number always inserted from a placeholder, and the key "do not" steps checked by tests. The AI model is told to write in the person's language while keeping enum values, numbers and drug names untouched.
* **Honest limits:** **no native-speaking clinician has reviewed these translations yet**, so every translated page says so. Detailed results from the basic (no-AI) mode, and the first-aid guides on the emergency page, remain in English and are labelled as such. Text-wording safety rewrites apply to English only. See [docs/TRANSLATION_REVIEW.md](docs/TRANSLATION_REVIEW.md) for the review checklist.

## Monitoring and Deployment

* `/health` (liveness), `/ready` (database writable, schema current, uploads writable, disk space) and `/metrics` (Prometheus; token or admin only; route patterns and outcomes, never user data). Every response carries `X-Request-ID`, also in the logs.
* Production container: non-root, **read-only filesystem, all capabilities dropped, no privilege escalation, resource limits**, gunicorn with recycled workers and request limits (`deploy/gunicorn.conf.py`), aggregated multi-worker metrics, and a health check. Details and an alert list in [docs/OPERATIONS.md](docs/OPERATIONS.md).
* **CI** (`.github/workflows/ci.yml`): lint + Bandit, secret scan with full history, tests on Python 3.11 and 3.12 with a **90% coverage floor**, migrations and dataset checks, the safety evaluation gate, a real-browser job with an automated **accessibility audit**, dependency audit (`pip-audit`), and a container build that checks the image has no `.env` or `.git`, runs non-root and starts read-only. *The workflow and Dockerfile are written and validated for syntax, but could not be run in the sandbox where this was built, so expect to fix small things on first run.*

---

## Testing

Tests need more than the app itself (pytest, playwright, lint/security tools), so install the **dev** requirements first — in a **fresh
virtual environment**, not on top of one that already has `requirements.txt` installed from Quick Start, to avoid a stale or partial
environment masking a real problem:

```bash
python -m venv .venv-test && source .venv-test/bin/activate   # a clean environment, separate from Quick Start's
pip install -r requirements-dev.txt       # installs requirements.txt too, plus pytest, playwright, ruff, bandit...
```

```bash
pytest                                    # ~1,300 tests, mocked AI only (no network), about 94% coverage
pytest --cov=quickaid --cov-report=term-missing
python -m playwright install chromium && npm install axe-core   # once, for the browser suite
AXE_PATH=node_modules/axe-core/axe.min.js pytest -m browser tests/e2e   # real browser journeys + accessibility audit
python scripts/secret_scan.py             # secrets and forbidden files
python evaluation/run_evaluation.py --weak-model --no-write   # safety evaluation gate
```

What is covered end to end: registration, sign-in, 2FA and recovery, password reset, symptom and photo analysis (including AI outage, garbage model output and a model that lies about an emergency), follow-up chats, history search/notes/export/deletion, emergency handling through every entry point in four languages, access control between users, retention and physical deletion, AI consent, and, in a real browser, the dashboard, upload, history, report, language switching and an axe-core audit of every key page at three screen sizes in light and dark themes.

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

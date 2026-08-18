# Quick Aid 🏥

AI-powered medical assistant for injury detection and health analysis.

---

## Features

- 🔐 **User Accounts:** Sign up / log in required to use the AI features - each user's history is private to them.
- 📸 **Image Analysis:** Detect injuries and skin conditions from images, powered by Gemini.
- 🤒 **Symptom Checker:** AI-based health recommendations based on described symptoms.
- 💬 **Follow-up Questions:** Ask follow-up questions about any analysis ("is this serious?") with full conversational context - requires Gemini to be configured.
- 🌍 **Localization:** Pick your region at signup (or any time from the navbar) for the correct local emergency number (US 911, UK 999, EU/India 112, Australia 000, or a generic international fallback) and temperature-unit ordering, in both AI-generated and fallback content.
- 📋 **History:** Past image analyses, symptom checks, and their follow-up conversations are saved per account so you can look back at them (stored locally in SQLite).
- 🚨 **Emergency Info:** Quick access to safety tips and urgent care guidance — always public, no login required, and shows every region's number for reference.
- 💡 **Medical Guidance:** Educational recommendations and health safety guidelines.
- 🛡️ **Basic-mode fallback:** Still works without a Gemini key, using simple rule-based image/symptom heuristics.
- 📈 **Logging:** Structured logs (console + rotating file) covering requests, auth events, and Gemini failures, plus a `/health` endpoint for uptime monitoring.
- ♿ **Accessibility:** Audited with axe-core (0 violations across all pages), WCAG AA color contrast, keyboard-accessible mobile navigation, skip-to-content link, and screen-reader landmarks.
- 📱 **Mobile-responsive:** Card layouts, forms, and navigation (via an accessible hamburger menu) all adapt below 768px.
- 🐳 **Docker:** Production-ready `Dockerfile` + `docker-compose.yml`, running under gunicorn as a non-root user with persistent volumes.

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

**With docker compose (recommended):**

```bash
cp .env.example .env   # fill in real GEMINI_API_KEY and SECRET_KEY
docker compose up --build
```

Open `http://localhost:5000`. Data survives `docker compose down`; only
`docker compose down -v` removes the `quickaid-data` / `quickaid-logs` volumes.

**With plain Docker:**

```bash
docker build -t quickaid .
docker run -p 5000:5000 --env-file .env \
  -v quickaid-data:/app/data \
  -v quickaid-logs:/app/logs \
  quickaid
```

The image exposes a container-level `HEALTHCHECK` that hits `/health`, so
`docker ps` shows the container's actual health status.

---

## Usage

- Create an account or log in, picking your region (used for the correct emergency number and units).
- Upload images of injuries or skin conditions for AI analysis.
- Enter symptoms in text for personalized health insights.
- Ask follow-up questions about any result right below it (needs Gemini configured).
- Change your region any time from the dropdown in the navbar.
- View past results and their follow-up conversations any time on the **History** page.
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

## Accessibility

Audited with [axe-core](https://github.com/dequelabs/axe-core) against the real rendered
HTML of every page (landing, emergency, history, login, register) - **0 violations**.
Also includes:

- A skip-to-content link and proper `<main>` landmarks on every page.
- WCAG AA color contrast (4.5:1+) - several of the original theme colors were
  darkened after failing contrast checks (most notably on the emergency page).
- A keyboard-accessible mobile navigation menu (proper `aria-expanded`/`aria-controls`,
  closes on Escape or an outside click) - the original mobile menu had no way to open it at all.
- 44px-minimum touch targets for interactive nav elements.

To re-run the audit yourself: render a page's HTML (e.g. via the Flask test client),
then feed it to `axe.run()` in a jsdom environment (color-contrast needs a real browser,
so that rule needs manual/Lighthouse verification instead).

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
- **Persistence:** SQLite (built into Python, no separate DB server required), with an automatic schema migration path for upgrading existing databases
- **AI:** Google Gemini via the [`google-genai`](https://pypi.org/project/google-genai/) SDK, using structured JSON output (Pydantic schemas) for reliable parsing, plus multi-turn chat for follow-up questions (`conversation.py`)
- **Localization:** Region-aware emergency numbers and units (`localization.py`), applied to both AI-generated and fallback content
- **Fallback analysis:** NumPy/Pillow-based rule heuristics when no Gemini key is configured
- **Testing:** pytest, with a Flask test client and an isolated temp-file SQLite DB per test
- **Logging:** Python's standard `logging` module, console + rotating file handler
- **Deployment:** Docker + docker-compose, gunicorn, non-root container user

---

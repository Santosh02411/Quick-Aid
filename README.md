# Quick Aid 🏥

AI-powered medical assistant for injury detection and health analysis.

---

## Features

- 📸 **Image Analysis:** Detect injuries and skin conditions from images, powered by Gemini.
- 🤒 **Symptom Checker:** AI-based health recommendations based on described symptoms.
- 📋 **History:** Past image analyses and symptom checks are saved per browser session so you can look back at them (stored locally in SQLite).
- 🚨 **Emergency Info:** Quick access to safety tips and urgent care guidance.
- 💡 **Medical Guidance:** Educational recommendations and health safety guidelines.
- 🛡️ **Basic-mode fallback:** Still works without a Gemini key, using simple rule-based image/symptom heuristics.

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

---

## Usage

- Upload images of injuries or skin conditions for AI analysis.
- Enter symptoms in text for personalized health insights.
- View past results any time on the **History** page.
- Access emergency information and safety guidelines as needed.

---

## Data & Privacy

- Uploaded images are analyzed and then deleted immediately — they are not stored.
- Analysis _results_ (not the images themselves) are saved to a local SQLite database
  (`quickaid.db`) so you can view your history. Entries are scoped to an anonymous
  per-browser session cookie, since there's no login system.
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
- **Backend:** Python , Flask, Flask-Limiter for rate limiting
- **Persistence:** SQLite (built into Python, no separate DB server required)
- **AI:** Google Gemini via the [`google-genai`](https://pypi.org/project/google-genai/) SDK, using structured JSON output (Pydantic schemas) for reliable parsing
- **Fallback analysis:** NumPy/Pillow-based rule heuristics when no Gemini key is configured

---

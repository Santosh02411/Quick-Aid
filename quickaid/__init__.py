"""Quick Aid - AI-powered medical assistant (Flask)."""

from pathlib import Path

from dotenv import load_dotenv

# Repository root (the folder that contains .env, data/, logs/, uploads/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env before any submodule reads os.getenv() at import time
# (database path, log dir, SMTP, OAuth...). Real environment variables
# (Docker, Heroku, CI) always win over values in the file.
load_dotenv(PROJECT_ROOT / '.env')

__all__ = ['PROJECT_ROOT']

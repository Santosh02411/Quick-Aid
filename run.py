"""Development entry point:  python run.py

For production use gunicorn instead (see wsgi.py / Procfile / Dockerfile).
"""

import os

from quickaid.app import app, logger


def main():
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('1', 'true', 'yes')
    port = int(os.getenv('PORT', '5000'))

    logger.info("Starting Quick Aid Medical Assistant...")
    print("Starting Quick Aid Medical Assistant...")
    print("⚠️  DISCLAIMER: This tool is for educational purposes only.")
    print("⚠️  Always consult healthcare professionals for medical advice.")
    if debug_mode:
        logger.warning("Running in DEBUG mode - never do this in production, it exposes an interactive debugger/RCE risk.")
        print("⚠️  Running in DEBUG mode — never do this in production, "
              "it exposes an interactive debugger/RCE risk.")
    else:
        print("Running with debug mode OFF. For production, use a WSGI server "
              "instead of this dev server, e.g.:")
        print(f"    gunicorn -w 2 -b 0.0.0.0:{port} wsgi:app")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()

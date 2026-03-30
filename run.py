"""
run.py
Start the NFL Predictor.

Development (Flask dev server):
    python run.py

Production (Gunicorn):
    python run.py --prod
    -- or --
    gunicorn -c gunicorn.conf.py "app.app:create_app()"
    -- or with Docker --
    make docker-up
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import PORT, HOST

if "--prod" in sys.argv:
    # Launch Gunicorn programmatically
    try:
        from gunicorn.app.wsgiapp import WSGIApplication
        sys.argv = [
            "gunicorn",
            "-c", "gunicorn.conf.py",
            "app.app:create_app()",
        ]
        WSGIApplication("%(prog)s [OPTIONS] [APP_MODULE]").run()
    except ImportError:
        print("gunicorn not installed. Run: pip install gunicorn")
        sys.exit(1)
else:
    # Flask development server
    from app.app import create_app
    app = create_app()
    print(f"\n🏈  NFL Predictor running at http://localhost:{PORT}")
    print("    For production use: python run.py --prod\n")
    app.run(debug=False, host=HOST, port=PORT)

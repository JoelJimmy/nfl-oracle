"""
app.py — Flask application factory.

Production: gunicorn -w 2 -b 0.0.0.0:$PORT "app.app:create_app()"
Development: python run.py
"""

import os
from flask import Flask, send_from_directory, jsonify, render_template_string
from flask_cors import CORS
from app.logger import setup_logging, get_logger
from app.cache import init_cache
from app.routes.api import api_bp

setup_logging()
logger = get_logger(__name__)


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    CORS(app)

    # ── Cache ─────────────────────────────────────────────────────────────────
    init_cache(app)

    # ── Routes ────────────────────────────────────────────────────────────────
    app.register_blueprint(api_bp)

    @app.route("/")
    def index():
        return send_from_directory(app.template_folder, "index.html")

    @app.route("/rosters")
    def rosters_page():
        return send_from_directory(app.template_folder, "rosters.html")

    @app.route("/power-rankings")
    def power_rankings_page():
        return send_from_directory(app.template_folder, "power_rankings.html")

    @app.route("/team/<team_abbr>")
    def team_page(team_abbr):
        return send_from_directory(app.template_folder, "team.html")

    @app.route("/sw.js")
    def service_worker():
        """Serve service worker from root scope so it can control all pages."""
        from flask import Response
        sw_path = os.path.join(app.static_folder, "sw.js")
        with open(sw_path, "r") as f:
            return Response(f.read(), mimetype="application/javascript",
                            headers={"Service-Worker-Allowed": "/"})

    @app.route("/privacy")
    def privacy():
        return send_from_directory(app.template_folder, "privacy.html")

    @app.route("/terms")
    def terms():
        return send_from_directory(app.template_folder, "terms.html")

    @app.route("/about")
    def about():
        return send_from_directory(app.template_folder, "about.html")

    @app.route("/unsubscribe")
    def unsubscribe():
        from flask import request
        email = request.args.get("email", "")
        if email:
            from app.database import remove_subscriber
            remove_subscriber(email)
        return render_template_string("""
        <html><body style="font-family:sans-serif;background:#080b10;color:#e8eef8;display:flex;
        align-items:center;justify-content:center;min-height:100vh;margin:0">
        <div style="text-align:center">
          <div style="font-size:2rem;margin-bottom:1rem">✓</div>
          <h2>Unsubscribed</h2>
          <p style="color:#7a8aa0">{{ email }} has been removed from NFL Oracle alerts.</p>
          <a href="/" style="color:#00e5ff">← Back to predictions</a>
        </div></body></html>
        """, email=email or "Your email")

    # ── Error handlers ────────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        if _wants_json():
            return jsonify({"error": "Not found", "status": 404}), 404
        return send_from_directory(app.template_folder, "error.html"), 404

    @app.errorhandler(500)
    def server_error(e):
        logger.error(f"500 error: {e}")
        if _wants_json():
            return jsonify({"error": "Server error — the model may be retraining", "status": 500}), 500
        return send_from_directory(app.template_folder, "error.html"), 500

    @app.errorhandler(503)
    def service_unavailable(e):
        if _wants_json():
            return jsonify({"error": "Service unavailable — model loading", "status": 503}), 503
        return send_from_directory(app.template_folder, "error.html"), 503

    # ── Database ──────────────────────────────────────────────────────────────
    try:
        from app.database import init_db
        init_db()
    except Exception as e:
        logger.warning(f"Database init failed: {e}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_path = "app/models/model.pkl"
    if os.path.exists(model_path):
        try:
            from app.services.predictor import load_model
            load_model()
        except Exception as e:
            logger.error(f"Could not load model: {e}")
            logger.info("Run `python -m scripts.train` to train first.")
    else:
        logger.warning("No model found. Run `python -m scripts.train` first.")

    # ── Rosters ───────────────────────────────────────────────────────────────
    try:
        from app.services.rosters import load_rosters, load_full_rosters
        load_rosters()
        load_full_rosters()
    except Exception as e:
        logger.warning(f"Roster load failed: {e}")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # Only start scheduler in the main process (not in gunicorn workers)
    if not _is_gunicorn_worker():
        try:
            from app.services.scheduler import start_scheduler
            start_scheduler()
        except Exception as e:
            logger.warning(f"Scheduler failed to start: {e}")

    return app


def _is_gunicorn_worker() -> bool:
    """Detect if running inside a gunicorn worker process."""
    return "gunicorn" in os.environ.get("SERVER_SOFTWARE", "") or \
           os.environ.get("GUNICORN_WORKER", "") == "1"


def _wants_json() -> bool:
    """Return True if the request prefers a JSON response."""
    from flask import request
    return (request.path.startswith("/api/") or
            "application/json" in request.headers.get("Accept", ""))
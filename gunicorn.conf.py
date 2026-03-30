"""
gunicorn.conf.py
Production server configuration for Gunicorn.

Start with:
    gunicorn -c gunicorn.conf.py "app.app:create_app()"

Or use the Makefile:
    make prod
"""

import os
import multiprocessing

# ── Binding ───────────────────────────────────────────────────────────────────
bind    = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
backlog = 64

# ── Workers ───────────────────────────────────────────────────────────────────
# 2 workers is enough for this app — the model is loaded per-worker so don't
# go too high or you'll use a lot of RAM.
workers         = int(os.environ.get("WEB_CONCURRENCY", 2))
worker_class    = "sync"
worker_connections = 100
timeout         = 120   # long timeout for the first predict call after cache miss
keepalive       = 5

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog   = "-"     # stdout
errorlog    = "-"     # stderr
loglevel    = os.environ.get("LOG_LEVEL", "info").lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(D)sus'

# ── Process ───────────────────────────────────────────────────────────────────
daemon      = False
pidfile     = None
umask       = 0
user        = None
group       = None
tmp_upload_dir = None

# ── Worker lifecycle ──────────────────────────────────────────────────────────
# Tell each worker to set GUNICORN_WORKER=1 so app.py skips the scheduler
# (scheduler only runs in the first/main process)
def post_fork(server, worker):
    import os
    os.environ["GUNICORN_WORKER"] = "1"

def on_starting(server):
    server.log.info("NFL Predictor starting up...")

def on_exit(server):
    server.log.info("NFL Predictor shutting down.")

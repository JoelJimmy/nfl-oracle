# NFL Predictor — Makefile
# Usage: make <target>

.PHONY: install train run prod docker-build docker-up docker-down docker-train clean

# ── Local development ─────────────────────────────────────────────────────────

install:
	python3.11 -m venv venv
	./venv/bin/pip install -r requirements.txt
	@echo "\nDone. Activate with: source venv/bin/activate"

train:
	python -m scripts.train

run:
	python run.py

# ── Production (Gunicorn) ─────────────────────────────────────────────────────

prod:
	gunicorn -c gunicorn.conf.py "app.app:create_app()"

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	docker compose up -d
	@echo "\nNFL Predictor running at http://localhost:$${PORT:-5000}"

docker-down:
	docker compose down

docker-train:
	docker compose run --rm train

docker-logs:
	docker compose logs -f web

# ── Utility ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."

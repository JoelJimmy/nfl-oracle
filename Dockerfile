FROM python:3.11-slim

# System deps for XGBoost / LightGBM on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories
RUN mkdir -p app/models logs

# Expose port
EXPOSE 5000

# Default: run with Gunicorn
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.app:create_app()"]

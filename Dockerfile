FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT at runtime (default 8080).
# One worker + 8 threads keeps memory low while handling concurrent Gemini calls.
# 120-second timeout covers a full audit + Gemini narration round-trip.
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 120 api.routes:app

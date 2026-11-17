FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000

# pydub decodes and encodes through ffmpeg/ffprobe.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY audio_processing_engine ./audio_processing_engine
COPY wsgi.py .

EXPOSE 10000
# Threads, not processes: pydub holds whole recordings in memory, so one process keeps RAM bounded.
# Long timeout: filtering or encoding a long recording can take minutes.
CMD gunicorn wsgi:app --bind 0.0.0.0:${PORT} --workers ${WEB_CONCURRENCY:-1} --threads ${GUNICORN_THREADS:-4} \
    --timeout ${GUNICORN_TIMEOUT:-600}

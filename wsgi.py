"""WSGI entry point: `gunicorn wsgi:app`."""

from audio_processing_engine.blueprint import create_app

app = create_app()

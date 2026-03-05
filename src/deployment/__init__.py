"""Deployment module for music generation."""

from .flask_app import FlaskApp, create_flask_app
from .fastapi_app import FastAPIApp, create_fastapi_app

__all__ = ["FlaskApp", "create_flask_app", "FastAPIApp", "create_fastapi_app"]

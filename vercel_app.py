"""
Vercel serverless entry point.

Imports the Flask app from the project root and exposes it
as the WSGI handler that Vercel's Python runtime expects.
"""

from api import app

# Vercel looks for a variable called `app` (WSGI/ASGI) or `handler`.
# Flask is WSGI, so we just expose `app` directly.

"""
Shared configuration for the Learning Engine.
"""
import os

# Where all persistent learning data lives.
# On Vercel (serverless), the deployed filesystem is read-only except /tmp,
# so we redirect writable data there.  For local / Docker, we keep the
# original project-relative path.
if os.environ.get("VERCEL"):
    DATA_DIR = os.path.join("/tmp", "learning_data")
else:
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "learning_data")

os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_URI = f"sqlite:///{os.path.join(DATA_DIR, 'deepscan.db')}"

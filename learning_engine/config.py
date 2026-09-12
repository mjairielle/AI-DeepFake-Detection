"""
Shared configuration for the Learning Engine.
"""
import os

# Where all persistent learning data lives
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "learning_data")
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_URI = f"sqlite:///{os.path.join(DATA_DIR, 'deepscan.db')}"

# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""Test simulation suite for dead reckoning backend."""


import os

# app.config requires SECRET_KEY; give test runs a throwaway one when none is set
os.environ.setdefault("SECRET_KEY", "test-only-secret-key")

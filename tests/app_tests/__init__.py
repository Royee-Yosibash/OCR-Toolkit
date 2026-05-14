"""Helpers shared by the OCR tagging app tests.

Exposes feature flags used to conditionally skip the test classes when
the optional app-related libraries are not installed in the active
environment.
"""

from importlib.util import find_spec

HAS_APP_DEPS = find_spec("flask") is not None and find_spec("werkzeug") is not None
HAS_PLAYWRIGHT = find_spec("playwright") is not None

APP_DEPS_REASON = "Flask app dependencies (flask, werkzeug) not installed"
PLAYWRIGHT_REASON = "playwright not installed"

"""Helpers shared by the OCR tagging app tests.

Exposes feature flags used to conditionally skip the test classes when
the optional app-related libraries are not installed in the active
environment.
"""

try:
    import flask  # noqa: F401
    import werkzeug  # noqa: F401

    HAS_APP_DEPS = True
except ImportError:
    HAS_APP_DEPS = False

try:
    import playwright  # noqa: F401

    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

APP_DEPS_REASON = "Flask app dependencies (flask, werkzeug) not installed"
PLAYWRIGHT_REASON = "playwright not installed"

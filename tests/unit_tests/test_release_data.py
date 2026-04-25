import json
import re
import unittest
from datetime import datetime

from consts import APP_ROOT

RELEASE_DATA_PATH = APP_ROOT / "release_data.json"
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:(?:rc|alpha|beta)\d+)?$")
ISO_DATE_FORMAT = "%Y-%m-%d"


class TestReleaseData(unittest.TestCase):
    """Tests for release_data.json structure and version validity."""

    def setUp(self):
        with open(RELEASE_DATA_PATH) as f:
            self.data = json.load(f)

    def test_required_keys_exist(self):
        """Verifies that all required keys are present in release_data.json."""
        for key in ("creator", "version", "release_date"):
            self.assertIn(key, self.data, f"Missing required key: {key}")

    def test_version_is_valid_semver(self):
        """Verifies that the version string matches the MAJOR.MINOR.PATCH format."""
        version = self.data["version"]
        self.assertRegex(version, SEMVER_PATTERN, f"Invalid semver format: {version}")

    def test_version_segments_are_non_negative(self):
        """Verifies that each semver segment is a non-negative integer."""
        parts = self.data["version"].split(".")
        for part in parts:
            value = int(part)
            self.assertGreaterEqual(value, 0, f"Negative version segment: {value}")

    def test_release_date_is_valid_iso8601(self):
        """Verifies that the release_date is a valid ISO 8601 date (YYYY-MM-DD)."""
        date_str = self.data["release_date"]
        try:
            datetime.strptime(date_str, ISO_DATE_FORMAT)
        except ValueError:
            self.fail(f"Invalid ISO 8601 date format: {date_str}. Expected YYYY-MM-DD.")

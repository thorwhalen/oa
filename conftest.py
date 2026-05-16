"""Set dummy env vars before pytest collection so module-level config2py
lookups (e.g. OPENAI_API_KEY) succeed in CI without requiring real secrets."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key-for-ci")

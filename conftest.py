"""Set a dummy OPENAI_API_KEY before pytest collection so tests that construct
clients succeed in CI without real secrets. (``import oa`` itself no longer
needs a key; see oa/tests/test_lazy_api_key.py.)"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key-for-ci")

"""``import oa`` must not need an API key; the key is resolved on first use.

Regression tests for thorwhalen/oa#16: dependents (imbed, newsmood) failed at
import when ``OPENAI_API_KEY`` was unset.
"""

import os
import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import oa, oa.util
    assert oa.util.openai.api_key is None, "key resolved at import"
    assert oa.util.dflt_client is None  # no key: None, and no prompt
    print("IMPORT_OK")
    """
)


def _run_without_key(tmp_path, script=_SCRIPT):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OPENAI_API_KEY", "OA_APP_DATA_DIR")
    }
    # Fresh HOME so no key saved in a real oa config store can be found
    env.update(HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "cfg"))
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_import_oa_with_no_api_key(tmp_path):
    result = _run_without_key(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout


def test_key_resolved_on_first_use(monkeypatch):
    import oa.util as util

    monkeypatch.setattr(util.openai, "api_key", None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-lazy")
    util.get_api_key_from_config.cache_clear()
    try:
        assert util.ensure_openai_api_key() == "sk-test-lazy"
        assert util.openai.api_key == "sk-test-lazy"
    finally:
        util.get_api_key_from_config.cache_clear()


def test_explicitly_set_openai_key_is_respected(monkeypatch):
    import oa.util as util

    monkeypatch.setattr(util.openai, "api_key", "sk-user-set")
    assert util.ensure_openai_api_key() == "sk-user-set"

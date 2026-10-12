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
    env.update(
        HOME=str(tmp_path),
        XDG_CONFIG_HOME=str(tmp_path / "cfg"),
        APPDATA=str(tmp_path / "appdata"),
        CONFIG2PY_CONFIG_DIR=str(tmp_path / "config2py"),
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # no controlling terminal, even if pytest has one
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_import_oa_with_no_api_key(tmp_path):
    result = _run_without_key(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout


def _fresh(monkeypatch, *, store=None, api_key=None):
    """Hermetic state: given oa config store, no env key, ``ensure`` not yet run."""
    import oa.util as util

    monkeypatch.setattr(util, "configs_local_store", dict(store or {}))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(util.openai, "api_key", api_key)
    monkeypatch.setattr(util, "_openai_api_key_ensured", False)
    util.get_api_key_from_config.cache_clear()
    return util


def test_key_resolved_on_first_use(monkeypatch):
    util = _fresh(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-lazy")
    assert util.ensure_openai_api_key() == "sk-test-lazy"
    assert util.openai.api_key == "sk-test-lazy"
    util.get_api_key_from_config.cache_clear()


def test_store_key_wins_over_env_even_if_openai_mirrored_env(monkeypatch):
    # openai's module client copies the env key into openai.api_key when first touched
    util = _fresh(monkeypatch, store={"OPENAI_API_KEY": "sk-test-store"})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-env")
    monkeypatch.setattr(util.openai, "api_key", "sk-test-env")
    assert util.ensure_openai_api_key() == "sk-test-store"


def test_key_set_on_openai_is_kept_when_none_stored(monkeypatch):
    util = _fresh(monkeypatch, api_key="sk-test-user-set")
    assert util.ensure_openai_api_key() == "sk-test-user-set"  # and no prompt


def test_key_set_on_openai_after_first_use_is_respected(monkeypatch):
    util = _fresh(monkeypatch, store={"OPENAI_API_KEY": "sk-test-store"})
    util.ensure_openai_api_key()
    util.openai.api_key = "sk-test-later"
    assert util.ensure_openai_api_key() == "sk-test-later"


# --- first-use prompt: no blocking prompt without a terminal; one prompt across threads


def test_no_key_and_no_terminal_raises_informative_error_instead_of_prompting(tmp_path):
    script = textwrap.dedent(
        """
        import oa
        from oa.util import OpenAIKeyNotFoundError
        for call in (oa.util.get_api_key_from_config, lambda: oa.chat("hi")):
            try:
                call()
            except OpenAIKeyNotFoundError as e:
                assert "OPENAI_API_KEY" in str(e), e
            else:
                raise SystemExit("no error raised")
        print("ERRORS_OK")
        """
    )
    result = _run_without_key(tmp_path, script)  # stdin is /dev/null: must not hang
    assert result.returncode == 0, result.stderr
    assert "ERRORS_OK" in result.stdout


def test_concurrent_first_use_asks_once(monkeypatch):
    import threading
    import time

    util = _fresh(monkeypatch)
    monkeypatch.setenv("OA_PROMPT_FOR_API_KEY", "1")
    asked = []

    def slow_ask(prompt, **kwargs):
        asked.append(prompt)
        time.sleep(0.2)
        return "  sk-test-asked \n"

    monkeypatch.setattr(util, "ask_user_for_input", slow_ask)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(util.get_api_key_from_config()))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert results == ["sk-test-asked"] * 8
    assert len(asked) == 1
    assert util.configs_local_store["OPENAI_API_KEY"] == "sk-test-asked"
    util.get_api_key_from_config.cache_clear()


def test_prompting_is_main_thread_only_unless_forced(monkeypatch):
    import threading

    util = _fresh(monkeypatch)
    monkeypatch.delenv("OA_PROMPT_FOR_API_KEY", raising=False)
    monkeypatch.setattr(util, "_has_controlling_terminal", lambda: True)
    monkeypatch.setattr(util, "_in_jupyter_kernel", lambda: False)
    assert util._can_prompt_user() is True  # main thread with a terminal
    seen = []
    t = threading.Thread(target=lambda: seen.append(util._can_prompt_user()))
    t.start(), t.join()
    assert seen == [False]  # e.g. a server's request handler
    monkeypatch.setenv("OA_PROMPT_FOR_API_KEY", "1")
    t = threading.Thread(target=lambda: seen.append(util._can_prompt_user()))
    t.start(), t.join()
    assert seen == [False, True]


def test_closed_stdin_means_cannot_prompt_not_a_crash(monkeypatch):
    import io

    util = _fresh(monkeypatch)
    stdin = io.StringIO()
    stdin.close()
    monkeypatch.setattr(util.sys, "stdin", stdin)
    monkeypatch.setattr(util.os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError))
    assert util._has_controlling_terminal() is False


def test_ensure_does_not_wait_behind_a_prompt_once_resolved(monkeypatch):
    import threading

    util = _fresh(monkeypatch, store={"OPENAI_API_KEY": "sk-test-store"})
    util.ensure_openai_api_key()  # resolved
    with util._api_key_lock:  # another thread is stuck in a prompt
        got = []
        t = threading.Thread(target=lambda: got.append(util.ensure_openai_api_key()))
        t.start()
        t.join(timeout=5)
        assert got == ["sk-test-store"]

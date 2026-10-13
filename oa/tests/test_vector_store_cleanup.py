"""Cleanup behavior of ``docs_to_vector_store`` when it fails partway (mocked client)."""

import glob
import tempfile
from unittest.mock import MagicMock

import openai
import pytest

from oa.vector_stores import docs_to_vector_store

DOCS = {"a": "alpha text", "b": "beta text", "c": "gamma text"}


def _mk_client(*, fail_upload_at=None, fail_attach_at=None):
    """A mock openai client; the n-th (0-based) upload/attach raises if requested."""
    client = MagicMock(spec=openai.OpenAI)
    client.vector_stores = MagicMock()
    client.vector_stores.create.return_value = MagicMock(id="vs_1")
    client.files = MagicMock()
    uploads = []

    def create_file(*, file, purpose, **kw):
        if fail_upload_at == len(uploads):
            raise RuntimeError("upload boom")
        uploads.append(f"file_{len(uploads)}")
        return MagicMock(id=uploads[-1])

    client.files.create.side_effect = create_file
    attached = []

    def attach(*, vector_store_id, file_id, **kw):
        if fail_attach_at == len(attached):
            raise RuntimeError("attach boom")
        attached.append(file_id)

    client.vector_stores.files.create.side_effect = attach
    return client


def _tmp_docs_files():
    return set(glob.glob(f"{tempfile.gettempdir()}/[abc]_*.txt"))


def test_success_keeps_store_and_files_and_leaves_no_temp_files():
    before = _tmp_docs_files()
    client = _mk_client()
    vs_id, mapping = docs_to_vector_store(DOCS, "x", client=client)
    assert vs_id == "vs_1"
    assert sorted(mapping.values()) == ["a", "b", "c"]
    client.vector_stores.delete.assert_not_called()
    client.files.delete.assert_not_called()
    assert _tmp_docs_files() == before


def test_client_is_reused_not_recreated_per_document(monkeypatch):
    import oa.stores

    client = _mk_client()
    monkeypatch.setattr(
        oa.stores, "mk_client", lambda *a, **k: pytest.fail("new client made")
    )
    docs_to_vector_store(DOCS, "x", client=client)
    assert client.files.create.call_count == len(DOCS)


@pytest.mark.parametrize(
    "kwargs, expected_deleted_files",
    [
        (dict(fail_upload_at=2), ["file_0", "file_1"]),
        # the third file is uploaded, then fails to attach: it must be deleted too
        (dict(fail_attach_at=2), ["file_0", "file_1", "file_2"]),
        (dict(fail_upload_at=0), []),
    ],
)
def test_failure_deletes_store_and_uploaded_files(kwargs, expected_deleted_files):
    before = _tmp_docs_files()
    client = _mk_client(**kwargs)
    with pytest.raises(RuntimeError, match="boom"):
        docs_to_vector_store(DOCS, "x", client=client)
    client.vector_stores.delete.assert_called_once_with("vs_1")
    deleted = [c.args[0] for c in client.files.delete.call_args_list]
    assert deleted == expected_deleted_files
    assert _tmp_docs_files() == before


def test_cleanup_errors_do_not_mask_original_error():
    client = _mk_client(fail_upload_at=1)
    client.vector_stores.delete.side_effect = ValueError("cleanup failed")
    client.files.delete.side_effect = ValueError("cleanup failed")
    with pytest.raises(RuntimeError, match="upload boom"):
        docs_to_vector_store(DOCS, "x", client=client)
    client.files.delete.assert_called_once_with("file_0")

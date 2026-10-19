from pathlib import Path

import pytest
from google.api_core.exceptions import PreconditionFailed

from sonnet_geometry.storage import IntegrityError, RunStore, upload_run_gcs, verify_run


def make_run(tmp_path):
    store = RunStore(tmp_path, "run-one")
    store.write_json("tokens.json", {"tokens": [1, 2, 3]})
    store.write_bytes("data.bin", b"raw data")
    store.complete({"schema_version": 1})
    return store.path


def test_immutable_run_and_checksum_checkpoint(tmp_path):
    run = make_run(tmp_path)
    assert verify_run(run)["run_id"] == "run-one"
    with pytest.raises(FileExistsError):
        RunStore(tmp_path, "run-one")
    (run / "data.bin").write_bytes(b"bad data")
    with pytest.raises(IntegrityError, match="checksum"):
        verify_run(run)


def test_incomplete_run_cannot_resume(tmp_path):
    store = RunStore(tmp_path, "incomplete")
    store.write_bytes("data.bin", b"incomplete")
    with pytest.raises(IntegrityError):
        verify_run(store.path)
    with pytest.raises(ValueError):
        store.write_bytes("../escape", b"bad")
    with pytest.raises(ValueError):
        RunStore(tmp_path, "../escape")


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        self.metadata = None
        self.generation = None
        self.size = None

    def upload_from_filename(self, filename, *, if_generation_match, checksum):
        assert if_generation_match == 0
        assert checksum == "auto"
        if self.name in self.bucket.data:
            raise PreconditionFailed("already exists")
        payload = Path(filename).read_bytes()
        self.generation = 42
        self.size = len(payload)
        self.bucket.data[self.name] = payload
        self.bucket.order.append(self.name)

    def reload(self):
        self.size = len(self.bucket.data[self.name])
        self.generation = 42

    def download_to_file(self, file, *, if_generation_match, checksum):
        assert if_generation_match == 42
        assert checksum == "auto"
        self.bucket.downloads += 1
        file.write(self.bucket.data[self.name])


class FakeClient:
    def __init__(self):
        self.data, self.order, self.downloads = {}, [], 0

    def bucket(self, name):
        assert name == "private-research"
        return self

    def blob(self, name):
        return FakeBlob(self, name)


def test_gcs_create_only_and_resume_validates_remote_bytes(tmp_path):
    run = make_run(tmp_path)
    client = FakeClient()
    receipt = upload_run_gcs(run, "private-research", "qwen3/v1/001/run-one", client=client)
    assert client.order[-1].endswith("checkpoint.json")
    assert client.order[-2].endswith("manifest.json")
    assert len(receipt["objects"]) == 4
    again = upload_run_gcs(run, "private-research", "qwen3/v1/001/run-one", client=client)
    assert again == receipt
    assert client.downloads == 4
    key = "qwen3/v1/001/run-one/data.bin"
    client.data[key] = b"bad data"
    with pytest.raises(IntegrityError, match="differs"):
        upload_run_gcs(run, "private-research", "qwen3/v1/001/run-one", client=client)

import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Keep every test out of the real session library.

    ``create_app`` falls back to a store at ``settings.data_dir``, so a test
    that does not inject one writes into whatever the user's machine uses --
    which is how a test run ended up creating rows in the real database. This
    is autouse rather than opt-in because the failure is silent: the tests pass
    either way, and the only symptom is somebody else's data.
    """
    monkeypatch.setenv("STT_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"

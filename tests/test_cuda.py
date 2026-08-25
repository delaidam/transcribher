"""Tests for putting the pip-installed CUDA libraries where CTranslate2 looks."""

import os
import sys

from delaida_transcriber import cuda


def test_nothing_to_do_off_windows(monkeypatch) -> None:
    """Linux wheels carry an RPATH the loader follows, so touching PATH there
    would be meddling for no reason."""
    monkeypatch.setattr(sys, "platform", "linux")

    assert cuda.add_library_path() == []


def test_missing_packages_are_not_an_error(monkeypatch) -> None:
    """A CPU-only install has no nvidia packages at all, and that is a normal
    way to run this: it must not raise on import of the package."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cuda.importlib.util, "find_spec", lambda name: None)

    assert cuda.add_library_path() == []


def test_directories_land_on_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    for name in ("cublas", "cudnn"):
        (tmp_path / name / "bin").mkdir(parents=True)
    (tmp_path / "cuda_runtime" / "lib").mkdir(parents=True)  # no bin: not ours

    class Spec:
        submodule_search_locations = [str(tmp_path)]

    monkeypatch.setattr(cuda.importlib.util, "find_spec", lambda name: Spec())
    monkeypatch.setenv("PATH", "C:\\Windows")

    added = cuda.add_library_path()

    assert added == [str(tmp_path / "cublas" / "bin"), str(tmp_path / "cudnn" / "bin")]
    assert os.environ["PATH"].startswith(os.pathsep.join(added))
    assert os.environ["PATH"].endswith("C:\\Windows")


def test_adding_twice_does_not_grow_path(monkeypatch, tmp_path) -> None:
    """The package calls this on import, and a test run imports it many times."""
    monkeypatch.setattr(sys, "platform", "win32")
    (tmp_path / "cublas" / "bin").mkdir(parents=True)

    class Spec:
        submodule_search_locations = [str(tmp_path)]

    monkeypatch.setattr(cuda.importlib.util, "find_spec", lambda name: Spec())
    monkeypatch.setenv("PATH", "C:\\Windows")

    assert cuda.add_library_path() != []
    after_first = os.environ["PATH"]

    assert cuda.add_library_path() == []
    assert os.environ["PATH"] == after_first

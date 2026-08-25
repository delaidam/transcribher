"""Make the CUDA libraries installed by pip visible to CTranslate2 on Windows.

`nvidia-cublas-cu12` and `nvidia-cudnn-cu12` put their DLLs inside site-packages,
where nothing looks for them. CTranslate2 registers its own directory with
`os.add_dll_directory` when it is imported but not theirs, so a GPU that is
present, detected and configured still fails at the first operation with
`Library cublas64_12.dll is not found`.

PATH rather than `os.add_dll_directory`, which would be the tidier tool:
CTranslate2 resolves cuBLAS lazily through a plain `LoadLibrary`, and that
searches PATH while ignoring the directory list `add_dll_directory` maintains.

Mutating the environment on import is a liberty for a library to take, so it is
kept to the narrowest version that works: Windows only, additive, idempotent,
and silent when there is nothing to add. Linux needs none of it, because the
wheels there ship an RPATH that the loader follows on its own.
"""

import importlib.util
import os
import sys
from pathlib import Path


def add_library_path() -> list[str]:
    """Prepend the NVIDIA wheels' DLL directories to PATH; returns what it added.

    Returns an empty list when there is nothing to do -- not Windows, the
    packages are not installed, or their directories are already on PATH.
    """
    if sys.platform != "win32":
        return []

    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []

    current = os.environ.get("PATH", "")
    known = {entry.lower() for entry in current.split(os.pathsep)}
    directories = [
        str(path)
        for root in spec.submodule_search_locations
        for path in sorted(Path(root).glob("*/bin"))
        if path.is_dir() and str(path).lower() not in known
    ]
    if not directories:
        return []

    os.environ["PATH"] = os.pathsep.join(directories) + os.pathsep + current
    return directories

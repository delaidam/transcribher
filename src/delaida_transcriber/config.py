"""Environment-backed configuration."""

import os
import shutil


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def has_cuda() -> bool:
    return shutil.which("nvidia-smi") is not None


class Settings:
    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        host: str | None = None,
        port: int | None = None,
        max_upload_mb: int | None = None,
    ) -> None:
        gpu = has_cuda()
        self.model = model or _env("STT_MODEL", "large-v3" if gpu else "base")
        self.device = device or _env("STT_DEVICE", "cuda" if gpu else "cpu")
        self.compute_type = compute_type or _env("STT_COMPUTE_TYPE", "float16" if gpu else "int8")
        self.host = host or _env("HOST", "127.0.0.1")
        self.port = port or int(_env("PORT", "8765"))
        self.max_upload_mb = max_upload_mb or int(_env("MAX_UPLOAD_MB", "100"))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

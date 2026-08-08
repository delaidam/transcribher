from delaida_transcriber import config


def test_cpu_defaults_use_supported_compute_type(monkeypatch) -> None:
    monkeypatch.setattr(config, "has_cuda", lambda: False)
    monkeypatch.delenv("STT_DEVICE", raising=False)
    monkeypatch.delenv("STT_COMPUTE_TYPE", raising=False)

    settings = config.Settings()

    assert settings.device == "cpu"
    assert settings.compute_type == "int8"


def test_cpu_normalizes_float16_from_environment(monkeypatch) -> None:
    monkeypatch.setattr(config, "has_cuda", lambda: False)
    monkeypatch.setenv("STT_DEVICE", "cpu")
    monkeypatch.setenv("STT_COMPUTE_TYPE", "float16")

    settings = config.Settings()

    assert settings.compute_type == "int8"

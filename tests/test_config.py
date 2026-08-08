from delaida_transcriber import config


def test_cpu_defaults_use_supported_compute_type(monkeypatch) -> None:
    monkeypatch.setattr(config, "has_cuda", lambda: False)
    monkeypatch.delenv("STT_DEVICE", raising=False)
    monkeypatch.delenv("STT_COMPUTE_TYPE", raising=False)

    settings = config.Settings()

    assert settings.device == "cpu"
    assert settings.compute_type == "int8"


def test_default_model_is_turbo_on_cpu_and_gpu(monkeypatch) -> None:
    """large-v3-turbo measured better *and* faster than large-v3, so it is the
    default on both devices. Guards against a regression to ``base``."""
    monkeypatch.delenv("STT_MODEL", raising=False)
    for available in (False, True):
        monkeypatch.setattr(config, "has_cuda", lambda available=available: available)
        assert config.Settings().model == "large-v3-turbo"


def test_priming_defaults_are_populated_and_overridable(monkeypatch) -> None:
    monkeypatch.setattr(config, "has_cuda", lambda: False)
    monkeypatch.delenv("STT_INITIAL_PROMPT", raising=False)
    monkeypatch.delenv("STT_HOTWORDS", raising=False)

    settings = config.Settings()
    assert "bosanskom" in settings.initial_prompt
    assert "GPT" in settings.hotwords

    # An explicit empty string disables priming rather than falling back.
    assert config.Settings(initial_prompt="", hotwords="").initial_prompt == ""
    assert config.Settings(initial_prompt="", hotwords="").hotwords == ""


def test_cpu_normalizes_float16_from_environment(monkeypatch) -> None:
    monkeypatch.setattr(config, "has_cuda", lambda: False)
    monkeypatch.setenv("STT_DEVICE", "cpu")
    monkeypatch.setenv("STT_COMPUTE_TYPE", "float16")

    settings = config.Settings()

    assert settings.compute_type == "int8"


def test_cpu_normalizes_int8_float16_from_environment(monkeypatch) -> None:
    monkeypatch.setattr(config, "has_cuda", lambda: False)
    monkeypatch.setenv("STT_DEVICE", "cpu")
    monkeypatch.setenv("STT_COMPUTE_TYPE", "int8_float16")

    settings = config.Settings()

    assert settings.compute_type == "int8"

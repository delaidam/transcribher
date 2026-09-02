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


def test_language_defaults_to_auto_and_is_overridable(monkeypatch) -> None:
    """STT_LANGUAGE is what lets a machine keep forcing hr without passing
    --language to every run."""
    monkeypatch.delenv("STT_LANGUAGE", raising=False)
    assert config.Settings().language == "auto"

    monkeypatch.setenv("STT_LANGUAGE", "hr")
    assert config.Settings().language == "hr"
    assert config.Settings(language="en").language == "en"


def test_ollama_context_and_keep_alive_default_and_override(monkeypatch) -> None:
    """A missing num_ctx is not a neutral default. Ollama loads the model at its
    own context -- 4096 for qwen3:8b, against the 40960 its card advertises --
    and truncates a longer prompt to fit rather than refusing it, so the setting
    has to have a value here rather than being left to the server."""
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)

    settings = config.Settings()
    assert settings.ollama_num_ctx == 8192
    assert settings.ollama_keep_alive == "30m"

    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
    assert config.Settings().ollama_num_ctx == 16384
    assert config.Settings(ollama_num_ctx=4096).ollama_num_ctx == 4096


def test_norwegian_gets_its_own_priming(monkeypatch) -> None:
    """Priming steers what Whisper reaches for -- worth ~2.6 points by the
    README's measurements -- so pointing a Norwegian recording at a Bosnian
    prompt is not neutral, it is a penalty."""
    monkeypatch.delenv("STT_INITIAL_PROMPT", raising=False)
    monkeypatch.delenv("STT_HOTWORDS", raising=False)
    settings = config.Settings()

    prompt, hotwords = settings.priming_for("no")

    assert "norsk" in prompt
    assert "nettleser" in hotwords


def test_every_other_language_keeps_the_measured_default(monkeypatch) -> None:
    """The README's accuracy table was measured under this exact configuration.
    Changing what bs/hr/en/auto are primed with would make every number in it
    unverifiable, so only Norwegian was added."""
    monkeypatch.delenv("STT_INITIAL_PROMPT", raising=False)
    monkeypatch.delenv("STT_HOTWORDS", raising=False)
    settings = config.Settings()

    for language in ("bs", "hr", "en", None):
        assert settings.priming_for(language) == (settings.initial_prompt, settings.hotwords)
    assert settings.initial_prompt == config.DEFAULT_INITIAL_PROMPT
    assert settings.hotwords == config.DEFAULT_HOTWORDS


def test_an_explicit_prompt_wins_over_every_profile(monkeypatch) -> None:
    """Set deliberately, so it is not second-guessed per language."""
    monkeypatch.setenv("STT_INITIAL_PROMPT", "moj vlastiti prompt")
    settings = config.Settings()

    assert settings.priming_for("no")[0] == "moj vlastiti prompt"
    assert config.Settings(initial_prompt="drugi").priming_for("no")[0] == "drugi"


def test_the_llm_backend_defaults_to_local(monkeypatch) -> None:
    """Sending a transcript off the machine has to be chosen, not inherited."""
    monkeypatch.delenv("LLM_BACKEND", raising=False)

    assert config.Settings().llm_backend == "ollama"
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    assert config.Settings().llm_backend == "anthropic"

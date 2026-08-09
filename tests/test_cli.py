import argparse

from delaida_transcriber.config import BEST_MODEL, Settings


def _settings_for(best: bool, model: str | None = None) -> Settings:
    """Mirror how cli._run resolves the model, without loading anything."""
    args = argparse.Namespace(best=best, model=model, cpu=True)
    return Settings(
        model=args.model or (BEST_MODEL if args.best else None),
        device="cpu",
        compute_type="int8",
    )


def test_best_flag_selects_the_most_accurate_model(monkeypatch) -> None:
    monkeypatch.delenv("STT_MODEL", raising=False)
    assert _settings_for(best=True).model == "large-v3"


def test_default_stays_on_the_fast_model(monkeypatch) -> None:
    monkeypatch.delenv("STT_MODEL", raising=False)
    assert _settings_for(best=False).model == "large-v3-turbo"


def test_explicit_model_wins_over_best(monkeypatch) -> None:
    monkeypatch.delenv("STT_MODEL", raising=False)
    assert _settings_for(best=True, model="small").model == "small"

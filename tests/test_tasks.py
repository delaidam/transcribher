import pytest

from delaida_transcriber import tasks


def test_every_task_is_addressable_and_labelled() -> None:
    ids = [task.id for task in tasks.TASKS]

    assert len(ids) == len(set(ids)), "task ids reach the client as values; they must be unique"
    assert all(task.label.strip() for task in tasks.TASKS)
    assert tasks.DEFAULT_TASK_ID in tasks.BY_ID


def test_only_ask_expects_the_user_to_supply_the_instruction() -> None:
    """Every preset carries its own instruction. ``ask`` is the one that does
    not, which is what the page keys its textarea off."""
    without = [task.id for task in tasks.TASKS if not task.instruction]

    assert without == ["ask"]


def test_structured_tasks_declare_the_fields_they_render() -> None:
    for task in tasks.TASKS:
        if not task.structured:
            continue
        assert task.keys, f"{task.id} claims to be structured but names no fields"
        assert all(label.strip() for _, label in task.fields)


def test_refine_keeps_the_shape_the_page_already_knew() -> None:
    """The original single-purpose endpoint returned these four. Keeping them
    means the preset is a superset of what shipped, not a replacement."""
    assert tasks.get("refine").keys == (
        "cleaned_text",
        "summary",
        "key_points",
        "unclear_parts",
    )


def test_the_mixed_language_preset_exists_and_is_free_text() -> None:
    """One coherent note out of a recording that switches languages is prose,
    not fields."""
    unify = tasks.get("unify")

    assert not unify.structured
    assert "jezik" in unify.instruction


def test_an_unknown_task_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="Dostupne su:"):
        tasks.get("nepostojeca")

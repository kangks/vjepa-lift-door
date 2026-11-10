import pytest

from lda_cost_joiner.apportionment import apportion, even_split


def test_absorbed_shares_sum_to_day_cost_and_follow_busy_ratio():
    a = apportion("lda-embed", 1.0, 1000.0, {"r1": 30.0, "r2": 10.0})
    assert a.absorbed_by_run["r1"] == pytest.approx(0.75)
    assert a.absorbed_by_run["r2"] == pytest.approx(0.25)
    assert sum(a.absorbed_by_run.values()) == pytest.approx(1.0)


def test_marginal_uses_billed_task_time_and_idle_is_the_residual():
    a = apportion("lda-embed", 1.0, 1000.0, {"r1": 30.0, "r2": 10.0})
    assert a.marginal_by_run["r1"] == pytest.approx(0.03)
    assert a.marginal_by_run["r2"] == pytest.approx(0.01)
    assert a.idle_residual_usd == pytest.approx(0.96)
    assert a.utilisation == pytest.approx(0.04)
    # marginal + idle reconstructs the day cost
    assert sum(a.marginal_by_run.values()) + a.idle_residual_usd == pytest.approx(1.0)


def test_full_day_capacity_reduces_to_the_original_86400s_formula():
    a = apportion("svc", 0.699, 86_400.0, {"r": 2.1})
    assert a.marginal_by_run["r"] == pytest.approx(0.699 * 2.1 / 86_400)


def test_day_without_runs_is_an_explicit_unallocated_line():
    a = apportion("lda-embed", 0.5, 600.0, {})
    assert a.unallocated
    assert a.idle_residual_usd == 0.5
    assert a.absorbed_by_run == {}


def test_busy_exceeding_billed_task_time_raises_instead_of_negative_idle():
    with pytest.raises(ValueError, match="exceeds billed task time"):
        apportion("lda-embed", 1.0, 50.0, {"r1": 40.0, "r2": 40.0})


def test_busy_without_task_time_raises():
    with pytest.raises(ValueError):
        apportion("lda-embed", 0.0, 0.0, {"r1": 5.0})


def test_even_split():
    assert even_split(0.0018, ["a", "b", "c"]) == pytest.approx({"a": 0.0006, "b": 0.0006, "c": 0.0006})
    assert even_split(1.0, []) == {}

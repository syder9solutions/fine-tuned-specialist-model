"""Metric correctness: these tests exist because a wrong significance test or a
circular metric silently destroys the claim the whole project makes."""
from src.metrics import (
    Prediction,
    accuracy,
    brier_score,
    confusion_matrix,
    escalation_metrics,
    expected_calibration_error,
    macro_f1,
    mcnemar_test,
    paired_bootstrap_delta,
    per_class_metrics,
)


def p(text, gold, pred, conf=0.9, kind="intent"):
    return Prediction(text=text, gold=gold, pred=pred, confidence=conf, kind=kind)


def test_accuracy_is_live():
    preds = [p("a", "code", "code"), p("b", "math", "math"), p("c", "qa", "code")]
    assert accuracy(preds) == pytest_approx(2 / 3)


def pytest_approx(x):
    from pytest import approx

    return approx(x, abs=1e-3)  # metrics round to 4 decimals


def test_per_class_counts():
    preds = [
        p("1", "code", "code"),
        p("2", "code", "code"),
        p("3", "code", "math"),
        p("4", "math", "math"),
    ]
    per = per_class_metrics(preds, ["code", "math"])
    assert per["code"]["precision"] == 1.0
    assert per["code"]["recall"] == pytest_approx(2 / 3)
    assert per["code"]["f1"] == pytest_approx(0.8)
    assert per["code"]["support"] == 3
    assert per["math"]["recall"] == 1.0
    assert per["math"]["precision"] == 0.5


def test_macro_f1_averages_all_labels():
    preds = [p("1", "code", "code"), p("2", "math", "math")]
    per = per_class_metrics(preds, ["code", "math", "qa"])
    assert per["qa"]["f1"] == 0.0  # absent label must not silently vanish
    assert macro_f1(preds, ["code", "math", "qa"]) == pytest_approx(2 / 3)


def test_confusion_matrix_is_complete():
    preds = [p("1", "code", "math")]
    cm = confusion_matrix(preds, ["code", "math"])
    assert cm["code"]["math"] == 1
    assert cm["code"]["code"] == 0
    assert sum(v for row in cm.values() for v in row.values()) == len(preds)


def test_escalation_metrics_directions():
    preds = [
        p("1", "escalate", "escalate", kind="ambiguous"),  # caught
        p("2", "escalate", "code", kind="ambiguous"),      # missed (unsafe)
        p("3", "code", "code"),                            # clean route
        p("4", "math", "escalate"),                        # false escalation
    ]
    em = escalation_metrics(preds)
    assert em["escalation_recall"] == 0.5
    assert em["false_escalation_rate"] == 0.5


def test_ece_is_zero_when_perfectly_calibrated():
    preds = [
        p("1", "code", "code", conf=1.0),
        p("2", "math", "math", conf=1.0),
    ]
    assert expected_calibration_error(preds)["ece"] == 0.0


def test_ece_penalises_overconfidence():
    # 50% accurate predictions that all claim 1.0 confidence -> ECE = 0.5
    preds = [p(str(i), "code", "code", conf=1.0) for i in range(10)] + [
        p(str(i + 10), "code", "math", conf=1.0) for i in range(10)
    ]
    assert expected_calibration_error(preds)["ece"] == pytest_approx(0.5)


def test_ece_decomposes_gap_signs():
    preds = [p("1", "code", "code", conf=0.6)]  # underconfident: gap +0.4
    bins = expected_calibration_error(preds)["bins"]
    assert bins[0]["gap"] == pytest_approx(0.4)


def test_brier_penalises_confident_wrong():
    right = brier_score([p("1", "code", "code", conf=1.0)])
    wrong = brier_score([p("1", "code", "math", conf=1.0)])
    unsure = brier_score([p("1", "code", "code", conf=0.5)])
    assert wrong > unsure > 0.0
    assert right == 0.0


def test_paired_bootstrap_detects_perfect_separation():
    a = [p(str(i), "code", "code") for i in range(20)]  # always right
    b = [p(str(i), "code", "math") for i in range(20)]  # always wrong
    res = paired_bootstrap_delta(a, b, n_boot=500, seed=1)
    assert res["observed_delta"] == 1.0
    assert res["ci_low"] == 1.0
    assert res["p_delta_le_zero"] == 0.0


def test_paired_bootstrap_handles_null_case():
    # identical arms -> CI must straddle 0
    a = [p(str(i), "code", "code" if i % 2 else "math") for i in range(30)]
    res = paired_bootstrap_delta(a, list(a), n_boot=300, seed=1)
    assert res["observed_delta"] == 0.0
    assert res["ci_low"] <= 0.0 <= res["ci_high"]


def test_paired_bootstrap_requires_same_length():
    import pytest

    a = [p("1", "code", "code")]
    with pytest.raises(ValueError):
        paired_bootstrap_delta(a, a + a)


def test_paired_bootstrap_is_paired_not_independent():
    # b differs from a on exactly one of 20 paired examples (the last one:
    # a predicts code correctly, b predicts math). The paired design sees the
    # delta, but a single discordant example is excluded from ~36% of
    # resamples, so the 2.5th percentile honestly lands at 0.
    a = [p(str(i), "code", "code") for i in range(20)]
    b = [p(str(i), "code", "code") for i in range(19)] + [p("19", "code", "math")]
    res = paired_bootstrap_delta(a, b, n_boot=800, seed=3)
    assert res["observed_delta"] == pytest_approx(1 / 20)
    # honest CI: a single discordant example cannot clear zero
    assert res["ci_low"] == 0.0
    assert res["ci_high"] <= 0.2
    assert res["p_delta_le_zero"] > 0.05  # not significant at n=1 discordance


def test_paired_bootstrap_significant_with_enough_discordance():
    # 10 discordant examples all favouring arm a -> CI clears 0
    a = [p(str(i), "code", "code") for i in range(10)] * 3  # 30 always right
    b = (
        [p(str(i), "code", "code") for i in range(10)]
        + [p(f"x{i}", "code", "math") for i in range(10)]
        + [p(str(i), "code", "code") for i in range(10)]
    )
    res = paired_bootstrap_delta(a, b, n_boot=1000, seed=3)
    assert res["observed_delta"] == pytest_approx(10 / 30)
    assert res["ci_low"] > 0.0
    assert res["p_delta_le_zero"] == 0.0


def test_mcnemar_exact():
    # a is correct on 2 examples where b is wrong; b is never right where a is
    # wrong -> a_only=2, b_only=0, exact two-sided binomial p = 2*(1/2)^2
    a = [p(str(i), "code", "code") for i in range(8)] + [
        p(str(i + 8), "math", "math") for i in range(2)
    ]
    b = [p(str(i), "code", "code") for i in range(8)] + [
        p(str(i + 8), "math", "code") for i in range(2)
    ]
    res = mcnemar_test(a, b)
    assert res["a_only_correct"] == 2
    assert res["b_only_correct"] == 0
    expected = min(1.0, 2 * (1 / 2) ** 2)
    assert res["p_value"] == pytest_approx(expected)


def test_mcnemar_asymmetric_discordance_is_significant():
    # a-only = 10, b-only = 0 -> p = 2 * 0.5^10 = 0.001953, significant
    a = [p(str(i), "code", "code") for i in range(10)] + [
        p(f"m{i}", "math", "math") for i in range(10)
    ]
    b = [p(str(i), "code", "code") for i in range(10)] + [
        p(f"m{i}", "math", "smalltalk") for i in range(10)
    ]
    res = mcnemar_test(a, b)
    assert res["a_only_correct"] == 10
    assert res["b_only_correct"] == 0
    assert res["p_value"] == pytest_approx(min(1.0, 2 * (0.5) ** 10))
    assert res["significant"] is True


def test_mcnemar_no_discordant_pairs():
    a = [p("1", "code", "code")]
    res = mcnemar_test(a, list(a))
    assert res["p_value"] == 1.0
    assert "no discordant" in res["note"]

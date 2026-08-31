import numpy as np
import pytest

from dreems.null_shrinkage import (
    NullShrinkageRule,
    mask_stability,
    pattern_correlation,
    posterior_signal_ratio,
)


def test_pattern_correlation_ignores_offset_and_detects_sign() -> None:
    values = np.array([-2.0, -0.5, 0.5, 2.0])
    assert pattern_correlation(values, 3.0 * values + 8.0) == pytest.approx(1.0)
    assert pattern_correlation(values, -values) == pytest.approx(-1.0)


def test_mask_stability_is_median_anchor_agreement() -> None:
    anchor = np.array([-1.0, 0.0, 1.0])
    masked = [anchor, 2.0 * anchor, -anchor]
    assert mask_stability(anchor, masked) == pytest.approx(1.0)


def test_posterior_signal_ratio_uses_centered_rms() -> None:
    mean = np.array([-1.0, 1.0]) + 20.0
    sd = np.full(2, 0.5)
    assert posterior_signal_ratio(mean, sd) == pytest.approx(2.0)


def test_rule_shrinks_low_evidence_more_than_high_evidence() -> None:
    rule = NullShrinkageRule(
        stability_weight=0.5,
        threshold=0.0,
        slope=8.0,
    )
    sd = np.ones(4)
    low = np.array([-0.1, -0.05, 0.05, 0.1])
    high = 20.0 * low
    assert rule.alpha(low, sd, stability=0.0) < rule.alpha(high, sd, stability=0.8)


def test_rule_preserves_pattern_and_centers_output() -> None:
    rule = NullShrinkageRule(
        stability_weight=1.0,
        threshold=0.5,
        slope=10.0,
        minimum_alpha=0.1,
    )
    mean = np.array([1.0, 2.0, 4.0])
    shrunk = rule.apply(mean, np.ones(3), stability=0.0)
    assert np.mean(shrunk) == pytest.approx(0.0)
    assert pattern_correlation(mean, shrunk) == pytest.approx(1.0)
    assert np.linalg.norm(shrunk) < np.linalg.norm(mean - np.mean(mean))


def test_stability_rule_requires_stability() -> None:
    rule = NullShrinkageRule(stability_weight=0.5, threshold=0.0, slope=1.0)
    with pytest.raises(ValueError, match="requires finite mask stability"):
        rule.alpha(np.array([-1.0, 1.0]), np.ones(2))

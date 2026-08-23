from __future__ import annotations

import numpy as np
import pytest

from dreems.metrics import edge_log_metrics


def test_edge_metrics_reports_signed_correlation() -> None:
    truth = np.asarray([2.0, 1.0, -1.0, -2.0], dtype=np.float64)

    assert edge_log_metrics(truth, truth).corr == pytest.approx(1.0)
    assert edge_log_metrics(truth, -truth).corr == pytest.approx(-1.0)

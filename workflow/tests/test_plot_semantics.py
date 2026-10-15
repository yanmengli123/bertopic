"""Check misleading edge cases without fitting another model."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from bertopic_workflow.plots import Renderer


def renderer():
    result = object.__new__(Renderer)
    result.config = SimpleNamespace(report=SimpleNamespace(year_start=None, year_end=2026, incomplete_year=2026))
    result.docs = [{'source': 'VIP', 'year': y, 'topic': 0, 'nearest_cosine': .9} for y in [2020, 2022]]
    result.ids = result.top = [0]
    result.metrics = {'outliers': 0}
    return result


def test_empty_year_is_a_gap_not_a_zero_or_interpolated_observation():
    subject = renderer()
    captured = []
    subject.emit = lambda *args: captured.append(args)
    try:
        subject.trends()
        _, figure, interactive, evidence = captured[0]
        trace = interactive.data[0]
        assert list(trace.x) == [2020, 2021, 2022]
        assert np.isnan(trace.y[1]) and trace.connectgaps is False
        assert evidence['annual_denominators'] == [1, 1] and evidence['missing_years'] == [2021]
        # Missing years must not acquire a stack that implies an observed corpus.
        assert [p.get_x() + p.get_width() / 2 for p in figure.axes[1].containers[0]] == [2020, 2022]
    finally:
        plt.close('all')


def test_zero_outlier_rate_has_no_negative_or_fabricated_values():
    subject = renderer()
    captured = []
    subject.emit = lambda *args: captured.append(args)
    try:
        subject.outliers()
        _, figure, interactive, evidence = captured[0]
        assert figure.axes[1].get_ylim() == (0, 100)
        assert figure.axes[2].get_ylim() == (0, 100)
        assert list(interactive.layout.yaxis.range) == [0, 1]
        assert evidence['rates'] == [0] and evidence['outliers'] == 0
    finally:
        plt.close('all')

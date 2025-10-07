from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from qlib.data.base import Expression
from qlib.data.ops import CSRank


class DummyFeature(Expression):
    """Minimal expression stub that returns predefined series per instrument."""

    def __init__(self, data: Dict[str, pd.Series]):
        self._data = {symbol: series.copy() for symbol, series in data.items()}
        self.load_counts: Counter = Counter()

    def get_longest_back_rolling(self) -> int:
        return 0

    def get_extended_window_size(self) -> Tuple[int, int]:
        return 0, 0

    def _load_internal(self, instrument, start_index, end_index, *args):
        self.load_counts[instrument] += 1
        series = self._data[instrument]
        return series.loc[start_index:end_index]


@pytest.fixture(autouse=True)
def reset_csrank_caches():
    CSRank._result_cache.clear()
    CSRank._instrument_cache.clear()
    yield
    CSRank._result_cache.clear()
    CSRank._instrument_cache.clear()


@pytest.fixture
def patch_universe(monkeypatch):
    def _patch(symbols):
        def _universe(cls, freq):  # pragma: no cover - trivial wrapper
            return list(symbols)

        monkeypatch.setattr(CSRank, "_get_instrument_universe", classmethod(_universe))

    return _patch


def _make_feature():
    index = pd.Index([0, 1, 2], name="index")
    data = {
        "AAA": pd.Series([1.0, 5.0, 3.0], index=index, dtype=float),
        "BBB": pd.Series([2.0, 3.0, 4.0], index=index, dtype=float),
        "CCC": pd.Series([3.0, 1.0, 5.0], index=index, dtype=float),
    }
    return DummyFeature(data)


def test_csrank_basic_cross_sectional_order(patch_universe):
    patch_universe(["AAA", "BBB", "CCC"])
    feature = _make_feature()
    op = CSRank(feature)

    result_aaa = op.load("AAA", 0, 2, "day")
    expected_df = pd.DataFrame({symbol: series for symbol, series in feature._data.items()})
    expected = expected_df.rank(axis=1, pct=True, method="average")["AAA"].rename(str(op))

    pd.testing.assert_series_equal(result_aaa, expected)
    assert (0, 2, "day") in CSRank._result_cache


def test_csrank_reuses_cached_window(patch_universe):
    patch_universe(["AAA", "BBB", "CCC"])
    feature = _make_feature()
    op = CSRank(feature)

    op.load("AAA", 0, 2, "day")
    initial_counts = feature.load_counts.copy()

    result_bbb = op.load("BBB", 0, 2, "day")
    expected_df = pd.DataFrame({symbol: series for symbol, series in feature._data.items()})
    expected_bbb = expected_df.rank(axis=1, pct=True, method="average")["BBB"].rename(str(op))

    pd.testing.assert_series_equal(result_bbb, expected_bbb)
    assert feature.load_counts == initial_counts


def test_csrank_handles_nan_inputs(patch_universe):
    patch_universe(["AAA", "BBB", "CCC"])
    index = pd.Index([0, 1, 2], name="index")
    data = {
        "AAA": pd.Series([1.0, np.nan, 3.0], index=index, dtype=float),
        "BBB": pd.Series([2.0, 3.0, 4.0], index=index, dtype=float),
        "CCC": pd.Series([3.0, 4.0, np.nan], index=index, dtype=float),
    }
    feature = DummyFeature(data)
    op = CSRank(feature)

    result_bbb = op.load("BBB", 0, 2, "day")
    expected_df = pd.DataFrame(data)
    expected_bbb = expected_df.rank(axis=1, pct=True, method="average")["BBB"].rename(str(op))

    pd.testing.assert_series_equal(result_bbb, expected_bbb)

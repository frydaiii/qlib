from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from qlib.data.base import Expression
from qlib.data.ops import CSRank, Product, Scale, SignedPower


class DummyFeature(Expression):
    """Expression stub returning predefined series per instrument."""

    def __init__(self, data: Dict[str, pd.Series]):
        self._data = {symbol: series.copy() for symbol, series in data.items()}
        self.load_counts: Counter = Counter()
        self._repr = f"DummyFeature<{id(self)}>"

    def get_longest_back_rolling(self) -> int:
        return 0

    def get_extended_window_size(self) -> Tuple[int, int]:
        return 0, 0

    def __str__(self) -> str:
        return self._repr

    def _load_internal(self, instrument, start_index, end_index, *args):
        self.load_counts[instrument] += 1
        series = self._data[instrument]
        return series.loc[start_index:end_index]


@pytest.fixture(autouse=True)
def reset_scale_cache():
    Scale._result_cache.clear()
    yield
    Scale._result_cache.clear()


@pytest.fixture
def patch_universe(monkeypatch):
    def _patch(symbols):
        def _universe(cls, freq):  # pragma: no cover - trivial wrapper
            return list(symbols)

        monkeypatch.setattr(CSRank, "_get_instrument_universe", classmethod(_universe))

    return _patch


def test_signed_power_series_exponent():
    index = pd.Index([0, 1, 2, 3, 4], name="index")
    base_feature = DummyFeature({"AAA": pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0], index=index)})
    exp_feature = DummyFeature({"AAA": pd.Series([2.0, 0.5, 3.0, 1.0, 2.0], index=index)})

    op = SignedPower(base_feature, exp_feature)
    result = op.load("AAA", 0, 4, "day")

    expected = np.sign(base_feature._data["AAA"]) * np.power(np.abs(base_feature._data["AAA"]), exp_feature._data["AAA"])
    expected = expected.rename(str(op)).astype(float)

    pd.testing.assert_series_equal(result, expected)


def test_signed_power_scalar_exponent():
    index = pd.Index([0, 1, 2], name="index")
    base_feature = DummyFeature({"AAA": pd.Series([-3.0, 0.0, 2.0], index=index)})

    op = SignedPower(base_feature, 2)
    result = op.load("AAA", 0, 2, "day")

    data = base_feature._data["AAA"]
    expected = np.sign(data) * np.power(np.abs(data), 2)
    expected = pd.Series(expected, index=index, name=str(op), dtype=float)
    pd.testing.assert_series_equal(result, expected)


def test_product_matches_rolling_prod():
    index = pd.Index([0, 1, 2, 3], name="index")
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    feature = DummyFeature({"AAA": series})

    op = Product(feature, 3)
    result = op.load("AAA", 0, 3, "day")

    rolling_obj = series.rolling(3, min_periods=1)
    prod_func = getattr(rolling_obj, "prod", None)
    if callable(prod_func):
        expected = prod_func()
    else:
        expected = rolling_obj.apply(np.prod, raw=True)
    expected.name = str(op)

    pd.testing.assert_series_equal(result, expected)


def test_scale_normalizes_cross_section(patch_universe):
    patch_universe(["AAA", "BBB", "CCC"])
    index = pd.Index([0, 1, 2], name="index")
    data = {
        "AAA": pd.Series([1.0, -2.0, 0.0], index=index),
        "BBB": pd.Series([2.0, 1.0, 0.0], index=index),
        "CCC": pd.Series([-1.0, 1.0, 0.0], index=index),
    }
    feature = DummyFeature(data)

    op = Scale(feature)
    result = op.load("AAA", 0, 2, "day")

    df = pd.DataFrame(data)
    denominator = df.abs().sum(axis=1)
    expected_df = df.mul(1.0 / denominator.replace(0.0, np.nan), axis=0).fillna(0.0)
    expected = expected_df["AAA"].rename(str(op))

    pd.testing.assert_series_equal(result, expected)
    scaled_sum = expected_df.abs().sum(axis=1)
    positive_mask = denominator > 0
    np.testing.assert_allclose(scaled_sum[positive_mask], np.ones(positive_mask.sum()))
    assert (scaled_sum[~positive_mask] == 0).all()


def test_scale_respects_target_and_cache(patch_universe):
    patch_universe(["AAA", "BBB"])
    index = pd.Index([0, 1], name="index")
    data = {
        "AAA": pd.Series([1.0, 1.0], index=index),
        "BBB": pd.Series([-1.0, 1.0], index=index),
    }
    feature = DummyFeature(data)

    op = Scale(feature, target=2)
    result_aaa = op.load("AAA", 0, 1, "day")
    counts_after_first = feature.load_counts.copy()
    result_bbb = op.load("BBB", 0, 1, "day")

    df = pd.DataFrame(data)
    expected_df = df.mul(2.0 / df.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    pd.testing.assert_series_equal(result_aaa, expected_df["AAA"].rename(str(op)))
    pd.testing.assert_series_equal(result_bbb, expected_df["BBB"].rename(str(op)))

    assert feature.load_counts == counts_after_first

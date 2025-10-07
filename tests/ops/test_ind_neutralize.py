import numpy as np
import pandas as pd
import pytest

from qlib.data.base import Expression
from qlib.data.ops import IndNeutralize


class DummyFeature(Expression):
    def __init__(self, data):
        self._data = data
        first_series = next(iter(data.values()))
        self.index = first_series.index

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self._data[instrument].copy()

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


@pytest.fixture(autouse=True)
def mock_listing(monkeypatch):
    import vnstock

    class FakeListing:
        industries_calls = 0
        group_calls = 0

        def symbols_by_industries(self):
            FakeListing.industries_calls += 1
            return pd.DataFrame(
                {
                    "symbol": ["AAA", "BBB", "CCC"],
                    "icb_code2": ["10", "10", "20"],
                    "icb_code3": ["1010", "1010", "2010"],
                    "icb_code4": ["101020", "101030", "201010"],
                }
            )

        def symbols_by_group(self, group):
            assert group == "HOSE"
            FakeListing.group_calls += 1
            return pd.Series(["AAA", "BBB", "CCC"], name="symbol")

    monkeypatch.setattr(vnstock, "Listing", FakeListing)
    IndNeutralize._industries_df = None
    IndNeutralize._hose_symbols = None
    IndNeutralize._mapping_cache.clear()
    IndNeutralize._group_cache.clear()
    yield FakeListing


@pytest.fixture
def dummy_feature():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    data = {
        "AAA": pd.Series([1.0, 2.0, 3.0], index=index),
        "BBB": pd.Series([2.0, 4.0, 6.0], index=index),
        "CCC": pd.Series([10.0, 20.0, 30.0], index=index),
        "DDD": pd.Series([5.0, 5.0, 5.0], index=index),
    }
    return DummyFeature(data)


def test_ind_neutralize_removes_group_mean(dummy_feature):
    op = IndNeutralize(dummy_feature, 2)
    result = op.load("AAA", None, None, "day")
    expected = dummy_feature._data["AAA"] - (
        dummy_feature._data["AAA"] + dummy_feature._data["BBB"]
    ) / 2
    expected.name = result.name
    pd.testing.assert_series_equal(result, expected)

    unique_group = op.load("CCC", None, None, "day")
    assert np.allclose(unique_group.fillna(0).values, 0.0)


def test_ind_neutralize_returns_nan_for_unknown_symbol(dummy_feature):
    op = IndNeutralize(dummy_feature, 2)
    result = op.load("DDD", None, None, "day")
    assert result.isna().all()


def test_ind_neutralize_validates_level(dummy_feature):
    with pytest.raises(ValueError):
        IndNeutralize(dummy_feature, 1)


def test_group_symbols_cached(mock_listing, dummy_feature):
    IndNeutralize(dummy_feature, 2)
    IndNeutralize(dummy_feature, 3)
    assert mock_listing.group_calls == 1

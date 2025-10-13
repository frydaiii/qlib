import numpy as np
import pandas as pd
import pandas.testing as pdt
import unittest
import pytest

from qlib.data import DatasetProvider
from qlib.data.data import ExpressionD
from qlib.data.base import Expression
from qlib.data.ops import If
from qlib.tests import TestOperatorData, TestMockData, MOCK_DF
from qlib.config import C


class TestElementOperator(TestMockData):
    def setUp(self) -> None:
        self.instrument = "0050"
        self.start_time = "2022-01-01"
        self.end_time = "2022-02-01"
        self.freq = "day"
        self.mock_df = MOCK_DF[MOCK_DF["symbol"] == self.instrument]

    def test_Abs(self):
        field = "Abs($close-Ref($close, 1))"
        result = ExpressionD.expression(self.instrument, field, self.start_time, self.end_time, self.freq)
        self.assertGreaterEqual(result.min(), 0)
        result = result.to_numpy()
        prev_close = self.mock_df["close"].shift(1)
        close = self.mock_df["close"]
        change = prev_close - close
        golden = change.abs().to_numpy()
        self.assertIsNone(np.testing.assert_allclose(result, golden))

    def test_Sign(self):
        field = "Sign($close-Ref($close, 1))"
        result = ExpressionD.expression(self.instrument, field, self.start_time, self.end_time, self.freq)
        result = result.to_numpy()
        prev_close = self.mock_df["close"].shift(1)
        close = self.mock_df["close"]
        change = close - prev_close
        change[change > 0] = 1.0
        change[change < 0] = -1.0
        golden = change.to_numpy()
        self.assertIsNone(np.testing.assert_allclose(result, golden))


class TestOperatorDataSetting(TestOperatorData):
    def test_setting(self):
        self.assertEqual(len(self.instruments_d), 1)
        self.assertGreater(len(self.cal), 0)


class TestInstElementOperator(TestOperatorData):
    def setUp(self) -> None:
        freq = "day"
        expressions = [
            "$change",
            "Abs($change)",
        ]
        columns = ["change", "abs"]
        self.data = DatasetProvider.inst_calculator(
            self.inst, self.start_time, self.end_time, freq, expressions, self.spans, C, []
        )
        self.data.columns = columns

    @pytest.mark.slow
    def test_abs(self):
        abs_values = self.data["abs"]
        self.assertGreater(abs_values[2], 0)


class _StaticExpression(Expression):
    """Simple expression that always returns the provided series."""

    def __init__(self, series: pd.Series):
        self._series = series
        self._repr = f"StaticExpression<{id(self)}>"

    def __str__(self):
        return self._repr

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self._series

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


class TestIfOperatorAlignment(unittest.TestCase):
    def test_if_uses_right_branch_when_condition_empty(self):
        index = pd.date_range("2022-01-01", periods=3, freq="D")
        cond = pd.Series(dtype=bool)
        left = pd.Series(dtype=float)
        right = pd.Series([1.0, 2.0, 3.0], index=index)

        expr = If(_StaticExpression(cond), _StaticExpression(left), _StaticExpression(right))
        result = expr.load("any", None, None, "day")

        right = right.rename(result.name)
        pdt.assert_series_equal(result, right)

    def test_if_aligns_mismatched_indices(self):
        right_index = pd.date_range("2022-01-01", periods=4, freq="D")
        left_index = right_index[[1, 3]]

        cond = pd.Series([True, True], index=left_index)
        left = pd.Series([10.0, 20.0], index=left_index)
        right = pd.Series([1.0, 2.0, 3.0, 4.0], index=right_index)

        expr = If(_StaticExpression(cond), _StaticExpression(left), _StaticExpression(right))
        result = expr.load("any", None, None, "day")

        expected = right.copy()
        expected.loc[left_index] = left
        expected = expected.rename(result.name)
        pdt.assert_series_equal(result, expected)


if __name__ == "__main__":
    unittest.main()

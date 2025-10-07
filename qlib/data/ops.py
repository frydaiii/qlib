# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Set, Tuple, Type, Union
from scipy.stats import percentileofscore
from .base import Expression, ExpressionOps, Feature, PFeature
from ..log import get_module_logger
from ..utils import get_callable_kwargs

try:
    from ._libs.rolling import rolling_slope, rolling_rsquare, rolling_resi
    from ._libs.expanding import expanding_slope, expanding_rsquare, expanding_resi
except ImportError:
    print(
        "#### Do not import qlib package in the repository directory in case of importing qlib from . without compiling #####"
    )
    raise
except ValueError:
    print("!!!!!!!! A error occurs when importing operators implemented based on Cython.!!!!!!!!")
    print("!!!!!!!! They will be disabled. Please Upgrade your numpy to enable them     !!!!!!!!")
    # We catch this error because some platform can't upgrade there package (e.g. Kaggle)
    # https://www.kaggle.com/general/293387
    # https://www.kaggle.com/product-feedback/98562


np.seterr(invalid="ignore")


#################### Element-Wise Operator ####################
class ElemOperator(ExpressionOps):
    """Element-wise Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        feature operation output
    """

    def __init__(self, feature):
        self.feature = feature

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.feature)

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


class ChangeInstrument(ElemOperator):
    """Change Instrument Operator
    In some case, one may want to change to another instrument when calculating, for example, to
    calculate beta of a stock with respect to a market index.
    This would require changing the calculation of features from the stock (original instrument) to
    the index (reference instrument)
    Parameters
    ----------
    instrument: new instrument for which the downstream operations should be performed upon.
                i.e., SH000300 (CSI300 index), or ^GPSC (SP500 index).

    feature: the feature to be calculated for the new instrument.
    Returns
    ----------
    Expression
        feature operation output
    """

    def __init__(self, instrument, feature):
        self.instrument = instrument
        self.feature = feature

    def __str__(self):
        return "{}('{}',{})".format(type(self).__name__, self.instrument, self.feature)

    def load(self, instrument, start_index, end_index, *args):
        # the first `instrument` is ignored
        return super().load(self.instrument, start_index, end_index, *args)

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self.feature.load(instrument, start_index, end_index, *args)


class NpElemOperator(ElemOperator):
    """Numpy Element-wise Operator

    Parameters
    ----------
    feature : Expression
        feature instance
    func : str
        numpy feature operation method

    Returns
    ----------
    Expression
        feature operation output
    """

    def __init__(self, feature, func):
        self.func = func
        super(NpElemOperator, self).__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        return getattr(np, self.func)(series)


class Abs(NpElemOperator):
    """Feature Absolute Value

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with absolute output
    """

    def __init__(self, feature):
        super(Abs, self).__init__(feature, "abs")


class Sign(NpElemOperator):
    """Feature Sign

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(self, feature):
        super(Sign, self).__init__(feature, "sign")

    def _load_internal(self, instrument, start_index, end_index, *args):
        """
        To avoid error raised by bool type input, we transform the data into float32.
        """
        series = self.feature.load(instrument, start_index, end_index, *args)
        # TODO:  More precision types should be configurable
        series = series.astype(np.float32)
        return getattr(np, self.func)(series)


class Log(NpElemOperator):
    """Feature Log

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with log
    """

    def __init__(self, feature):
        super(Log, self).__init__(feature, "log")


class Mask(NpElemOperator):
    """Feature Mask

    Parameters
    ----------
    feature : Expression
        feature instance
    instrument : str
        instrument mask

    Returns
    ----------
    Expression
        a feature instance with masked instrument
    """

    def __init__(self, feature, instrument):
        super(Mask, self).__init__(feature, "mask")
        self.instrument = instrument

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.instrument.lower())

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self.feature.load(self.instrument, start_index, end_index, *args)


class Not(NpElemOperator):
    """Not Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Feature:
        feature elementwise not output
    """

    def __init__(self, feature):
        super(Not, self).__init__(feature, "bitwise_not")


#################### Pair-Wise Operator ####################


class IndNeutralize(ElemOperator):
    """Cross-sectional industry neutralization for Vietnamese HOSE symbols.

    Parameters
    ----------
    feature : Expression
        Feature expression to neutralize.
    level : int
        Industry classification level to use. Must be 2, 3, or 4 corresponding to
        ``icb_code2``, ``icb_code3``, ``icb_code4`` from ``vnstock``.
    """

    _LEVEL_TO_COLUMN = {2: "icb_code2", 3: "icb_code3", 4: "icb_code4"}
    _industries_df: Optional[pd.DataFrame] = None
    _hose_symbols: Optional[Set[str]] = None
    _mapping_cache: Dict[str, Tuple[Dict[str, str], Dict[str, List[str]]]] = {}
    _group_cache: Dict[str, Set[str]] = {}

    def __init__(self, feature, level: int):
        super().__init__(feature)
        if level not in self._LEVEL_TO_COLUMN:
            raise ValueError("IndNeutralize level must be one of {2, 3, 4}")
        self.level = level
        self._code_column = self._LEVEL_TO_COLUMN[level]
        self._symbol_to_code, self._code_to_symbols = self._build_mappings(self._code_column)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if not isinstance(symbol, str):
            return symbol
        return symbol.split(".")[0].upper()

    @classmethod
    def _ensure_base_data(cls):
        if cls._industries_df is not None and cls._hose_symbols is not None:
            return
        try:
            from vnstock import Listing  # type: ignore import
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise ImportError("IndNeutralize requires the 'vnstock' package") from exc

        listing = None
        if cls._industries_df is None:
            listing = Listing()
            industries = listing.symbols_by_industries()
            if not isinstance(industries, pd.DataFrame):
                raise TypeError("symbols_by_industries must return a pandas DataFrame")
            if "symbol" not in industries.columns:
                raise KeyError("symbols_by_industries must provide a 'symbol' column")
            industries_df = industries.copy()
            industries_df["symbol"] = industries_df["symbol"].astype(str).str.upper()
            cls._industries_df = industries_df

        if cls._hose_symbols is None:
            cls._hose_symbols = cls._get_group_symbols("HOSE", listing=listing)

    @classmethod
    def _get_group_symbols(cls, group: str, listing=None) -> Set[str]:
        if group in cls._group_cache:
            return cls._group_cache[group]
        if listing is None:
            try:
                from vnstock import Listing  # type: ignore import
            except ImportError as exc:  # pragma: no cover - dependency missing
                raise ImportError("IndNeutralize requires the 'vnstock' package") from exc
            listing = Listing()
        raw = listing.symbols_by_group(group)
        if isinstance(raw, pd.Series):
            iterable = raw.dropna().tolist()
        elif isinstance(raw, pd.DataFrame):
            if "symbol" not in raw.columns:
                raise KeyError("symbols_by_group DataFrame must include a 'symbol' column")
            iterable = raw["symbol"].dropna().tolist()
        elif isinstance(raw, (list, tuple, set)):
            iterable = [item for item in raw if item is not None]
        else:
            raise TypeError("symbols_by_group must return a pandas Series/DataFrame or a sequence")

        symbols: Set[str] = set()
        for sym in iterable:
            sym_str = str(sym).strip()
            if sym_str:
                symbols.add(sym_str.upper())
        cls._group_cache[group] = symbols
        return symbols

    @classmethod
    def _build_mappings(cls, code_column: str):
        cls._ensure_base_data()
        if cls._industries_df is None or cls._hose_symbols is None:
            raise RuntimeError("vnstock listing information is not available")

        if code_column in cls._mapping_cache:
            return cls._mapping_cache[code_column]

        industries_df = cls._industries_df
        hose_symbols = cls._hose_symbols

        filtered = industries_df[industries_df["symbol"].isin(hose_symbols)].copy()
        if code_column not in filtered.columns:
            raise KeyError(f"symbols_by_industries missing expected column '{code_column}'")

        filtered = filtered.dropna(subset=[code_column])
        filtered[code_column] = filtered[code_column].astype(str).str.strip()
        filtered = filtered[filtered[code_column] != ""]

        symbol_to_code = filtered.set_index("symbol")[code_column].to_dict()
        code_to_symbols = filtered.groupby(code_column)["symbol"].apply(list).to_dict()

        cls._mapping_cache[code_column] = (symbol_to_code, code_to_symbols)
        return cls._mapping_cache[code_column]

    def _load_symbol_series(self, symbol: str, instrument: str, start_index, end_index, args):
        try:
            return self.feature.load(symbol, start_index, end_index, *args)
        except Exception:
            if symbol == self._normalize_symbol(instrument):
                return self.feature.load(instrument, start_index, end_index, *args)
            raise

    def _load_internal(self, instrument, start_index, end_index, *args):
        target_series = self.feature.load(instrument, start_index, end_index, *args)
        norm_symbol = self._normalize_symbol(instrument)
        code = self._symbol_to_code.get(norm_symbol)

        if code is None:
            return pd.Series(np.nan, index=target_series.index, dtype=float)

        group_symbols = self._code_to_symbols.get(code)
        if not group_symbols:
            return pd.Series(np.nan, index=target_series.index, dtype=float)

        data = {}
        for symbol in group_symbols:
            try:
                data[symbol] = self._load_symbol_series(symbol, instrument, start_index, end_index, args)
            except Exception as exc:  # pragma: no cover - logging path only
                get_module_logger(self.__class__.__name__).debug(
                    "Skipping symbol %s in IndNeutralize due to error: %s", symbol, exc
                )

        if not data:
            return pd.Series(np.nan, index=target_series.index, dtype=float)

        if norm_symbol not in data:
            data[norm_symbol] = target_series

        df = pd.DataFrame(data)
        if df.empty or norm_symbol not in df:
            return pd.Series(np.nan, index=target_series.index, dtype=float)

        group_mean = df.mean(axis=1)
        neutralized = df[norm_symbol] - group_mean
        neutralized.name = str(self)
        return neutralized.reindex(target_series.index)


class CSRank(ElemOperator):
    """Cross-Sectional Rank

    Computes the percentile rank of each value across all instruments at each
    timestamp, matching the Alpha101 ``rank(x)`` operator semantics.

    Parameters
    ----------
    feature : Expression
        Feature expression to rank.

    Returns
    -------
    Expression
        Cross-sectional percentile rank (0 to 1) for each value at each
        timestamp.
    """

    _result_cache: "OrderedDict[Tuple[int, int, Optional[str]], Dict[str, pd.Series]]" = OrderedDict()
    _instrument_cache: Dict[str, List[str]] = {}
    _MAX_CACHE_KEYS = 16

    def __init__(self, feature):
        super().__init__(feature)

    @classmethod
    def _get_instrument_universe(cls, freq: Optional[str]) -> List[str]:
        freq_key = str(freq) if freq is not None else "day"
        if freq_key not in cls._instrument_cache:
            instruments: List[str] = []
            try:
                from qlib.data import Inst  # type: ignore
            except Exception as exc:  # pragma: no cover - logging path only
                get_module_logger(cls.__name__).debug(
                    "Unable to import instrument provider for CSRank: %s", exc
                )
            else:
                try:
                    inst_conf = Inst.instruments("all")
                    instruments = list(Inst.list_instruments(inst_conf, freq=freq_key, as_list=True) or [])
                except Exception as exc:  # pragma: no cover - logging path only
                    get_module_logger(cls.__name__).debug(
                        "Unable to list instruments for CSRank(freq=%s): %s", freq_key, exc
                    )
            cls._instrument_cache[freq_key] = [str(sym) for sym in instruments]
        return cls._instrument_cache[freq_key]

    @staticmethod
    def _empty_like(series: pd.Series, name: str) -> pd.Series:
        empty = pd.Series(np.nan, index=series.index, dtype=float)
        empty.name = name
        return empty

    @classmethod
    def _store_cache(cls, cache_key: Tuple[int, int, Optional[str]], data: Dict[str, pd.Series]):
        cls._result_cache[cache_key] = {sym: s.copy() for sym, s in data.items()}
        cls._result_cache.move_to_end(cache_key)
        while len(cls._result_cache) > cls._MAX_CACHE_KEYS:
            cls._result_cache.popitem(last=False)

    def _build_cross_sectional_cache(
        self,
        cache_key: Tuple[int, int, Optional[str]],
        instrument: str,
        start_index,
        end_index,
        freq: Optional[str],
        args: Tuple,
    ) -> Dict[str, pd.Series]:
        logger = get_module_logger(self.__class__.__name__)
        target_series = self.feature.load(instrument, start_index, end_index, *args)
        if target_series.empty:
            result = {instrument: self._empty_like(target_series, str(self))}
            self._store_cache(cache_key, result)
            return self._result_cache[cache_key]

        universe = self._get_instrument_universe(freq)
        data: Dict[str, pd.Series] = {}
        for symbol in universe:
            if symbol == instrument:
                data[symbol] = target_series
                continue
            try:
                series = self.feature.load(symbol, start_index, end_index, *args)
            except Exception as exc:  # pragma: no cover - logging path only
                logger.debug("Skipping symbol %s in CSRank due to error: %s", symbol, exc)
                continue
            if series is None or series.empty:
                continue
            data[symbol] = series

        if instrument not in data:
            data[instrument] = target_series

        if not data:
            result = {instrument: self._empty_like(target_series, str(self))}
            self._store_cache(cache_key, result)
            return self._result_cache[cache_key]

        df = pd.DataFrame(data).dropna(axis=1, how="all")
        if df.empty or instrument not in df:
            result = {instrument: self._empty_like(target_series, str(self))}
            self._store_cache(cache_key, result)
            return self._result_cache[cache_key]

        ranked = df.rank(axis=1, pct=True, method="average").reindex(target_series.index)
        result_map = {sym: ranked[sym].rename(str(self)) for sym in ranked.columns}
        self._store_cache(cache_key, result_map)
        return self._result_cache[cache_key]

    def _load_internal(self, instrument, start_index, end_index, *args):
        freq = args[0] if args else None
        cache_key = (start_index, end_index, freq)
        cached = self._result_cache.get(cache_key)
        if cached is None:
            cached = self._build_cross_sectional_cache(cache_key, instrument, start_index, end_index, freq, args)
        else:
            self._result_cache.move_to_end(cache_key)

        result = cached.get(instrument)
        if result is not None:
            return result

        series = self.feature.load(instrument, start_index, end_index, *args)
        return self._empty_like(series, str(self))


class PairOperator(ExpressionOps):
    """Pair-wise operator

    Parameters
    ----------
    feature_left : Expression
        feature instance or numeric value
    feature_right : Expression
        feature instance or numeric value

    Returns
    ----------
    Feature:
        two features' operation output
    """

    def __init__(self, feature_left, feature_right):
        self.feature_left = feature_left
        self.feature_right = feature_right

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature_left, self.feature_right)

    def get_longest_back_rolling(self):
        if isinstance(self.feature_left, (Expression,)):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, (Expression,)):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0
        return max(left_br, right_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, (Expression,)):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0

        if isinstance(self.feature_right, (Expression,)):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0
        return max(ll, rl), max(lr, rr)


class NpPairOperator(PairOperator):
    """Numpy Pair-wise operator

    Parameters
    ----------
    feature_left : Expression
        feature instance or numeric value
    feature_right : Expression
        feature instance or numeric value
    func : str
        operator function

    Returns
    ----------
    Feature:
        two features' operation output
    """

    def __init__(self, feature_left, feature_right, func):
        self.func = func
        super(NpPairOperator, self).__init__(feature_left, feature_right)

    def _load_internal(self, instrument, start_index, end_index, *args):
        assert any(
            [isinstance(self.feature_left, (Expression,)), self.feature_right, Expression]
        ), "at least one of two inputs is Expression instance"
        if isinstance(self.feature_left, (Expression,)):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left  # numeric value
        if isinstance(self.feature_right, (Expression,)):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right
        check_length = isinstance(series_left, (np.ndarray, pd.Series)) and isinstance(
            series_right, (np.ndarray, pd.Series)
        )
        if check_length:
            warning_info = (
                f"Loading {instrument}: {str(self)}; np.{self.func}(series_left, series_right), "
                f"The length of series_left and series_right is different: ({len(series_left)}, {len(series_right)}), "
                f"series_left is {str(self.feature_left)}, series_right is {str(self.feature_right)}. Please check the data"
            )
        else:
            warning_info = (
                f"Loading {instrument}: {str(self)}; np.{self.func}(series_left, series_right), "
                f"series_left is {str(self.feature_left)}, series_right is {str(self.feature_right)}. Please check the data"
            )
        try:
            res = getattr(np, self.func)(series_left, series_right)
        except ValueError as e:
            get_module_logger("ops").debug(warning_info)
            raise ValueError(f"{str(e)}. \n\t{warning_info}") from e
        else:
            if check_length and len(series_left) != len(series_right):
                get_module_logger("ops").debug(warning_info)
        return res


class Power(NpPairOperator):
    """Power Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        The bases in feature_left raised to the exponents in feature_right
    """

    def __init__(self, feature_left, feature_right):
        super(Power, self).__init__(feature_left, feature_right, "power")


class Add(NpPairOperator):
    """Add Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' sum
    """

    def __init__(self, feature_left, feature_right):
        super(Add, self).__init__(feature_left, feature_right, "add")


class Sub(NpPairOperator):
    """Subtract Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' subtraction
    """

    def __init__(self, feature_left, feature_right):
        super(Sub, self).__init__(feature_left, feature_right, "subtract")


class Mul(NpPairOperator):
    """Multiply Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' product
    """

    def __init__(self, feature_left, feature_right):
        super(Mul, self).__init__(feature_left, feature_right, "multiply")


class Div(NpPairOperator):
    """Division Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' division
    """

    def __init__(self, feature_left, feature_right):
        super(Div, self).__init__(feature_left, feature_right, "divide")


class Greater(NpPairOperator):
    """Greater Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        greater elements taken from the input two features
    """

    def __init__(self, feature_left, feature_right):
        super(Greater, self).__init__(feature_left, feature_right, "maximum")


class Less(NpPairOperator):
    """Less Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        smaller elements taken from the input two features
    """

    def __init__(self, feature_left, feature_right):
        super(Less, self).__init__(feature_left, feature_right, "minimum")


class Gt(NpPairOperator):
    """Greater Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left > right`
    """

    def __init__(self, feature_left, feature_right):
        super(Gt, self).__init__(feature_left, feature_right, "greater")


class Ge(NpPairOperator):
    """Greater Equal Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left >= right`
    """

    def __init__(self, feature_left, feature_right):
        super(Ge, self).__init__(feature_left, feature_right, "greater_equal")


class Lt(NpPairOperator):
    """Less Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left < right`
    """

    def __init__(self, feature_left, feature_right):
        super(Lt, self).__init__(feature_left, feature_right, "less")


class Le(NpPairOperator):
    """Less Equal Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left <= right`
    """

    def __init__(self, feature_left, feature_right):
        super(Le, self).__init__(feature_left, feature_right, "less_equal")


class Eq(NpPairOperator):
    """Equal Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left == right`
    """

    def __init__(self, feature_left, feature_right):
        super(Eq, self).__init__(feature_left, feature_right, "equal")


class Ne(NpPairOperator):
    """Not Equal Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left != right`
    """

    def __init__(self, feature_left, feature_right):
        super(Ne, self).__init__(feature_left, feature_right, "not_equal")


class And(NpPairOperator):
    """And Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' row by row & output
    """

    def __init__(self, feature_left, feature_right):
        super(And, self).__init__(feature_left, feature_right, "bitwise_and")


class Or(NpPairOperator):
    """Or Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' row by row | outputs
    """

    def __init__(self, feature_left, feature_right):
        super(Or, self).__init__(feature_left, feature_right, "bitwise_or")


#################### Triple-wise Operator ####################
class If(ExpressionOps):
    """If Operator

    Parameters
    ----------
    condition : Expression
        feature instance with bool values as condition
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    """

    def __init__(self, condition, feature_left, feature_right):
        self.condition = condition
        self.feature_left = feature_left
        self.feature_right = feature_right

    def __str__(self):
        return "If({},{},{})".format(self.condition, self.feature_left, self.feature_right)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series_cond = self.condition.load(instrument, start_index, end_index, *args)
        if isinstance(self.feature_left, (Expression,)):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left
        if isinstance(self.feature_right, (Expression,)):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right
        series = pd.Series(np.where(series_cond, series_left, series_right), index=series_cond.index)
        return series

    def get_longest_back_rolling(self):
        if isinstance(self.feature_left, (Expression,)):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, (Expression,)):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0

        if isinstance(self.condition, (Expression,)):
            c_br = self.condition.get_longest_back_rolling()
        else:
            c_br = 0
        return max(left_br, right_br, c_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, (Expression,)):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0

        if isinstance(self.feature_right, (Expression,)):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0

        if isinstance(self.condition, (Expression,)):
            cl, cr = self.condition.get_extended_window_size()
        else:
            cl, cr = 0, 0
        return max(ll, rl, cl), max(lr, rr, cr)


#################### Rolling ####################
# NOTE: methods like `rolling.mean` are optimized with cython,
# and are super faster than `rolling.apply(np.mean)`


class Rolling(ExpressionOps):
    """Rolling Operator
    The meaning of rolling and expanding is the same in pandas.
    When the window is set to 0, the behaviour of the operator should follow `expanding`
    Otherwise, it follows `rolling`

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size
    func : str
        rolling method

    Returns
    ----------
    Expression
        rolling outputs
    """

    def __init__(self, feature, N, func):
        self.feature = feature
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # NOTE: remove all null check,
        # now it's user's responsibility to decide whether use features in null days
        # isnull = series.isnull() # NOTE: isnull = NaN, inf is not null
        if isinstance(self.N, int) and self.N == 0:
            series = getattr(series.expanding(min_periods=1), self.func)()
        elif isinstance(self.N, float) and 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = getattr(series.rolling(self.N, min_periods=1), self.func)()
            # series.iloc[:self.N-1] = np.nan
        # series[isnull] = np.nan
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        if 0 < self.N < 1:
            return int(np.log(1e-6) / np.log(1 - self.N))  # (1 - N)**window == 1e-6
        return self.feature.get_longest_back_rolling() + self.N - 1

    def get_extended_window_size(self):
        if self.N == 0:
            # FIXME: How to make this accurate and efficiently? Or  should we
            # remove such support for N == 0?
            get_module_logger(self.__class__.__name__).warning("The Rolling(ATTR, 0) will not be accurately calculated")
            return self.feature.get_extended_window_size()
        elif 0 < self.N < 1:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            size = int(np.log(1e-6) / np.log(1 - self.N))
            lft_etd = max(lft_etd + size - 1, lft_etd)
            return lft_etd, rght_etd
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N - 1, lft_etd)
            return lft_etd, rght_etd


class Ref(Rolling):
    """Feature Reference

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        N = 0, retrieve the first data; N > 0, retrieve data of N periods ago; N < 0, future data

    Returns
    ----------
    Expression
        a feature instance with target reference
    """

    def __init__(self, feature, N):
        super(Ref, self).__init__(feature, N, "ref")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # N = 0, return first day
        if series.empty:
            return series  # Pandas bug, see: https://github.com/pandas-dev/pandas/issues/21049
        elif self.N == 0:
            series = pd.Series(series.iloc[0], index=series.index)
        else:
            series = series.shift(self.N)  # copy
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        return self.feature.get_longest_back_rolling() + self.N

    def get_extended_window_size(self):
        if self.N == 0:
            get_module_logger(self.__class__.__name__).warning("The Ref(ATTR, 0) will not be accurately calculated")
            return self.feature.get_extended_window_size()
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N, lft_etd)
            rght_etd = max(rght_etd - self.N, rght_etd)
            return lft_etd, rght_etd


class Mean(Rolling):
    """Rolling Mean (MA)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling average
    """

    def __init__(self, feature, N):
        super(Mean, self).__init__(feature, N, "mean")


class Sum(Rolling):
    """Rolling Sum

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling sum
    """

    def __init__(self, feature, N):
        super(Sum, self).__init__(feature, N, "sum")


class Std(Rolling):
    """Rolling Std

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling std
    """

    def __init__(self, feature, N):
        super(Std, self).__init__(feature, N, "std")


class Var(Rolling):
    """Rolling Variance

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling variance
    """

    def __init__(self, feature, N):
        super(Var, self).__init__(feature, N, "var")


class Skew(Rolling):
    """Rolling Skewness

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling skewness
    """

    def __init__(self, feature, N):
        if N != 0 and N < 3:
            raise ValueError("The rolling window size of Skewness operation should >= 3")
        super(Skew, self).__init__(feature, N, "skew")


class Kurt(Rolling):
    """Rolling Kurtosis

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling kurtosis
    """

    def __init__(self, feature, N):
        if N != 0 and N < 4:
            raise ValueError("The rolling window size of Kurtosis operation should >= 5")
        super(Kurt, self).__init__(feature, N, "kurt")


class Max(Rolling):
    """Rolling Max

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max
    """

    def __init__(self, feature, N):
        super(Max, self).__init__(feature, N, "max")


class IdxMax(Rolling):
    """Rolling Max Index

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max index
    """

    def __init__(self, feature, N):
        super(IdxMax, self).__init__(feature, N, "idxmax")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        return series


class Min(Rolling):
    """Rolling Min

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling min
    """

    def __init__(self, feature, N):
        super(Min, self).__init__(feature, N, "min")


class IdxMin(Rolling):
    """Rolling Min Index

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling min index
    """

    def __init__(self, feature, N):
        super(IdxMin, self).__init__(feature, N, "idxmin")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        return series


class Quantile(Rolling):
    """Rolling Quantile

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling quantile
    """

    def __init__(self, feature, N, qscore):
        super(Quantile, self).__init__(feature, N, "quantile")
        self.qscore = qscore

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature, self.N, self.qscore)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).quantile(self.qscore)
        else:
            series = series.rolling(self.N, min_periods=1).quantile(self.qscore)
        return series


class Med(Rolling):
    """Rolling Median

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling median
    """

    def __init__(self, feature, N):
        super(Med, self).__init__(feature, N, "median")


class Mad(Rolling):
    """Rolling Mean Absolute Deviation

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling mean absolute deviation
    """

    def __init__(self, feature, N):
        super(Mad, self).__init__(feature, N, "mad")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # TODO: implement in Cython

        def mad(x):
            x1 = x[~np.isnan(x)]
            return np.mean(np.abs(x1 - x1.mean()))

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(mad, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(mad, raw=True)
        return series


class Rank(Rolling):
    """Rolling Rank (Percentile)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling rank
    """

    def __init__(self, feature, N):
        super(Rank, self).__init__(feature, N, "rank")

    # for compatiblity of python 3.7, which doesn't support pandas 1.4.0+ which implements Rolling.rank
    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        rolling_or_expending = series.expanding(min_periods=1) if self.N == 0 else series.rolling(self.N, min_periods=1)
        if hasattr(rolling_or_expending, "rank"):
            return rolling_or_expending.rank(pct=True)

        def rank(x):
            if np.isnan(x[-1]):
                return np.nan
            x1 = x[~np.isnan(x)]
            if x1.shape[0] == 0:
                return np.nan
            return percentileofscore(x1, x1[-1]) / 100

        return rolling_or_expending.apply(rank, raw=True)


class Count(Rolling):
    """Rolling Count

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling count of number of non-NaN elements
    """

    def __init__(self, feature, N):
        super(Count, self).__init__(feature, N, "count")


class Delta(Rolling):
    """Rolling Delta

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with end minus start in rolling window
    """

    def __init__(self, feature, N):
        super(Delta, self).__init__(feature, N, "delta")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series - series.iloc[0]
        else:
            series = series - series.shift(self.N)
        return series


# TODO:
# support pair-wise rolling like `Slope(A, B, N)`
class Slope(Rolling):
    """Rolling Slope
    This operator calculate the slope between `idx` and `feature`.
    (e.g. [<feature_t1>, <feature_t2>, <feature_t3>] and [1, 2, 3])

    Usage Example:
    - "Slope($close, %d)/$close"

    # TODO:
    # Some users may want pair-wise rolling like `Slope(A, B, N)`

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with linear regression slope of given window
    """

    def __init__(self, feature, N):
        super(Slope, self).__init__(feature, N, "slope")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_slope(series.values), index=series.index)
        else:
            series = pd.Series(rolling_slope(series.values, self.N), index=series.index)
        return series


class Rsquare(Rolling):
    """Rolling R-value Square

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with linear regression r-value square of given window
    """

    def __init__(self, feature, N):
        super(Rsquare, self).__init__(feature, N, "rsquare")

    def _load_internal(self, instrument, start_index, end_index, *args):
        _series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_rsquare(_series.values), index=_series.index)
        else:
            series = pd.Series(rolling_rsquare(_series.values, self.N), index=_series.index)
            series.loc[np.isclose(_series.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)] = np.nan
        return series


class Resi(Rolling):
    """Rolling Regression Residuals

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with regression residuals of given window
    """

    def __init__(self, feature, N):
        super(Resi, self).__init__(feature, N, "resi")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_resi(series.values), index=series.index)
        else:
            series = pd.Series(rolling_resi(series.values, self.N), index=series.index)
        return series


class WMA(Rolling):
    """Rolling WMA

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with weighted moving average output
    """

    def __init__(self, feature, N):
        super(WMA, self).__init__(feature, N, "wma")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # TODO: implement in Cython

        def weighted_mean(x):
            w = np.arange(len(x)) + 1
            w = w / w.sum()
            return np.nanmean(w * x)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(weighted_mean, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(weighted_mean, raw=True)
        return series


class EMA(Rolling):
    """Rolling Exponential Mean (EMA)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int, float
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with regression r-value square of given window
    """

    def __init__(self, feature, N):
        super(EMA, self).__init__(feature, N, "ema")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        def exp_weighted_mean(x):
            a = 1 - 2 / (1 + len(x))
            w = a ** np.arange(len(x))[::-1]
            w /= w.sum()
            return np.nansum(w * x)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(exp_weighted_mean, raw=True)
        elif 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = series.ewm(span=self.N, min_periods=1).mean()
        return series


class Adv(ExpressionOps):
    """Average Daily Dollar Volume

    Parameters
    ----------
    N : int
        rolling window size. When set to 0, fall back to expanding window.
    price_feature : Expression, optional
        price feature used to translate volume into dollar notional. Defaults to ``$close``.
    volume_feature : Expression, optional
        volume feature that will be combined with ``price_feature``. Defaults to ``$volume``.
    value_feature : Expression, optional
        Precomputed dollar value feature. If not provided, it will be derived from
        ``price_feature * volume_feature``.

    Returns
    ----------
    Expression
        rolling average of dollar volume
    """

    def __init__(self, N, price_feature=None, volume_feature=None, value_feature=None):
        if price_feature is None:
            price_feature = Feature("close")
        if volume_feature is None:
            volume_feature = Feature("volume")

        self.N = N
        self.price_feature = price_feature
        self.volume_feature = volume_feature

        if value_feature is None:
            value_feature = Mul(self.price_feature, self.volume_feature)

        self.value_feature = value_feature
        self._rolling_expr = Mean(self.value_feature, self.N)

    def __str__(self):
        return f"adv{self.N}"

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self._rolling_expr.load(instrument, start_index, end_index, *args)

    def get_longest_back_rolling(self):
        return self._rolling_expr.get_longest_back_rolling()

    def get_extended_window_size(self):
        return self._rolling_expr.get_extended_window_size()


#################### Pair-Wise Rolling ####################
class PairRolling(ExpressionOps):
    """Pair Rolling Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling output of two input features
    """

    def __init__(self, feature_left, feature_right, N, func):
        # TODO: in what case will a const be passed into `__init__` as `feature_left` or `feature_right`
        self.feature_left = feature_left
        self.feature_right = feature_right
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature_left, self.feature_right, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        assert any(
            [isinstance(self.feature_left, Expression), self.feature_right, Expression]
        ), "at least one of two inputs is Expression instance"

        if isinstance(self.feature_left, Expression):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left  # numeric value
        if isinstance(self.feature_right, Expression):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right

        if self.N == 0:
            series = getattr(series_left.expanding(min_periods=1), self.func)(series_right)
        else:
            series = getattr(series_left.rolling(self.N, min_periods=1), self.func)(series_right)
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        if isinstance(self.feature_left, Expression):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, Expression):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0
        return max(left_br, right_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, Expression):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0
        if isinstance(self.feature_right, Expression):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0
        if self.N == 0:
            get_module_logger(self.__class__.__name__).warning(
                "The PairRolling(ATTR, 0) will not be accurately calculated"
            )
            return -np.inf, max(lr, rr)
        else:
            return max(ll, rl) + self.N - 1, max(lr, rr)


class Corr(PairRolling):
    """Rolling Correlation

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling correlation of two input features
    """

    def __init__(self, feature_left, feature_right, N):
        super(Corr, self).__init__(feature_left, feature_right, N, "corr")

    def _load_internal(self, instrument, start_index, end_index, *args):
        res: pd.Series = super(Corr, self)._load_internal(instrument, start_index, end_index, *args)

        # NOTE: Load uses MemCache, so calling load again will not cause performance degradation
        series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        res.loc[
            np.isclose(series_left.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
            | np.isclose(series_right.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
        ] = np.nan
        return res


class Cov(PairRolling):
    """Rolling Covariance

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max of two input features
    """

    def __init__(self, feature_left, feature_right, N):
        super(Cov, self).__init__(feature_left, feature_right, N, "cov")


#################### Operator which only support data with time index ####################
# Convention
# - The name of the operators in this section will start with "T"


class TResample(ElemOperator):
    def __init__(self, feature, freq, func):
        """
        Resampling the data to target frequency.
        The resample function of pandas is used.

        - the timestamp will be at the start of the time span after resample.

        Parameters
        ----------
        feature : Expression
            An expression for calculating the feature
        freq : str
            It will be passed into the resample method for resampling basedn on given frequency
        func : method
            The method to get the resampled values
            Some expression are high frequently used
        """
        self.feature = feature
        self.freq = freq
        self.func = func

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.freq)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        if series.empty:
            return series
        else:
            if self.func == "sum":
                return getattr(series.resample(self.freq), self.func)(min_count=1)
            else:
                return getattr(series.resample(self.freq), self.func)()


TOpsList = [TResample]
OpsList = [
    ChangeInstrument,
    Rolling,
    Ref,
    Max,
    Min,
    Sum,
    Mean,
    Std,
    Var,
    Skew,
    Kurt,
    Med,
    Mad,
    Slope,
    Rsquare,
    Resi,
    Rank,
    Quantile,
    Count,
    EMA,
    WMA,
    Adv,
    Corr,
    Cov,
    Delta,
    Abs,
    Sign,
    Log,
    Power,
    Add,
    Sub,
    Mul,
    Div,
    Greater,
    Less,
    And,
    Or,
    Not,
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
    Mask,
    IndNeutralize,
    CSRank,
    IdxMax,
    IdxMin,
    If,
    Feature,
    PFeature,
] + [TResample]


class OpsWrapper:
    """Ops Wrapper"""

    def __init__(self):
        self._ops = {}

    def reset(self):
        self._ops = {}

    def register(self, ops_list: List[Union[Type[ExpressionOps], dict]]):
        """register operator

        Parameters
        ----------
        ops_list : List[Union[Type[ExpressionOps], dict]]
            - if type(ops_list) is List[Type[ExpressionOps]], each element of ops_list represents the operator class, which should be the subclass of `ExpressionOps`.
            - if type(ops_list) is List[dict], each element of ops_list represents the config of operator, which has the following format:

                .. code-block:: text

                    {
                        "class": class_name,
                        "module_path": path,
                    }

                Note: `class` should be the class name of operator, `module_path` should be a python module or path of file.
        """
        for _operator in ops_list:
            if isinstance(_operator, dict):
                _ops_class, _ = get_callable_kwargs(_operator)
            else:
                _ops_class = _operator

            if not issubclass(_ops_class, (Expression,)):
                raise TypeError("operator must be subclass of ExpressionOps, not {}".format(_ops_class))

            if _ops_class.__name__ in self._ops:
                get_module_logger(self.__class__.__name__).warning(
                    "The custom operator [{}] will override the qlib default definition".format(_ops_class.__name__)
                )
            self._ops[_ops_class.__name__] = _ops_class

    def __getattr__(self, key):
        if key not in self._ops:
            raise AttributeError("The operator [{0}] is not registered".format(key))
        return self._ops[key]


Operators = OpsWrapper()


def register_all_ops(C):
    """register all operator"""
    logger = get_module_logger("ops")

    from qlib.data.pit import P, PRef  # pylint: disable=C0415

    Operators.reset()
    Operators.register(OpsList + [P, PRef])

    if getattr(C, "custom_ops", None) is not None:
        Operators.register(C.custom_ops)
        logger.debug("register custom operator {}".format(C.custom_ops))

# Adding Custom Operators to Qlib

This guide explains how to extend Qlib's expression engine with your own operators. It summarises the
expectations encoded in `qlib/data/ops.py` and gives a reference workflow for registering the new
operator so it can be used inside expression strings (for example `MyOp($close)`).

## 1. Understand the Operator Building Blocks

Operators in Qlib all inherit from `ExpressionOps`, which itself extends `Expression`. The concrete
mixins defined in `qlib/data/ops.py` provide most of the plumbing you need:

- `ElemOperator` (`qlib/data/ops.py:37`) - use for element-wise transforms that operate on a single
  upstream expression.
- `NpElemOperator` (`qlib/data/ops.py:97`) - subclass when the logic can be expressed through a NumPy
  unary function name such as `"abs"` or `"log"`.
- `PairOperator` (`qlib/data/ops.py:231`) - base class for operators that consume two expressions
  (or scalars). `NpPairOperator` builds on this for NumPy-powered pairwise functions.
- `Rolling` (`qlib/data/ops.py:713`) and `PairRolling` (`qlib/data/ops.py:1437`) - encapsulate the
  standard rolling or expanding window semantics, including window-size bookkeeping.

All operators must implement the `_load_internal` method defined by `Expression`. Many base classes
already supply sensible `get_longest_back_rolling` and `get_extended_window_size` implementations.
Override them only if your operator needs additional history or future context.

## 2. Sketch the Operator

1. Pick the base class that matches your use case.
2. Define any constructor arguments and store them on `self`.
3. Implement `_load_internal(self, instrument, start_index, end_index, *args)` to fetch upstream
   series via `self.feature.load(...)` (or `self.feature_left`/`self.feature_right`), apply your
   transformation, and return a `pd.Series` indexed by the calendar.
4. If the operator needs more historical data than its inputs advertise, override
   `get_longest_back_rolling` or `get_extended_window_size` to request the required padding.

### Minimal template

```python
from qlib.data.ops import ElemOperator

class Diff(ElemOperator):
    """First difference."""

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.diff()

    def get_extended_window_size(self):
        left, right = self.feature.get_extended_window_size()
        return left + 1, right
```

The `Diff` example above mirrors the version used in the unit test suite (`tests/test_register_ops.py`).

## 3. Register the Operator

Qlib discovers operators through the `Operators` registry (`qlib/data/ops.py:1670`) populated by
`register_all_ops` (`qlib/data/ops.py:1721`). You have several integration options:

- **Project code (no config changes).** Import `Operators` and register your class manually before
  evaluating expressions.

  ```python
  from qlib.data.ops import Operators
  Operators.register([Diff])
  ```

- **Via configuration.** Set the `custom_ops` entry when initialising Qlib (see
  `qlib/config.py:281`). Each entry can be either the operator class or a dict with
  `{"class": "MyOp", "module_path": "path.to.module"}` for lazy loading.

  ```python
  from qlib import init
  init(provider_uri=..., custom_ops=[Diff])
  ```

  When `register_all_ops` runs it will pull in everything from `OpsList` plus your `custom_ops`
  payload.

- **Contributing to core.** Add the operator class to `OpsList` (`qlib/data/ops.py:1616`) so it is
  bundled by default, and include accompanying tests or documentation updates.

## 4. Exercise the Operator

After registration the class name becomes available in expression strings:

```python
from qlib.data import D
fields = ["Diff($close)"]
res = D.features(["SH600000"], fields, start_time="2010-01-01", end_time="2017-12-31", freq="day")
```

Before shipping the change:

- Run `pytest tests/test_register_ops.py` (or a dedicated test you add under `tests/`).
- Execute any domain-specific validation that demonstrates the operator behaves as intended.

## 5. Troubleshooting Checklist

- **Registration errors** often stem from name clashes. The registry warns if you override an
  existing operator (`qlib/data/ops.py:1706`).
- **Shape mismatches** in pairwise ops trigger the explicit length check in `NpPairOperator`
  (`qlib/data/ops.py:301`). Make sure the upstream series align on index and length.
- **Infinite lookback** requirements should be signalled explicitly. Returning `np.inf` from
  `get_longest_back_rolling` (as seen in `Rolling` and `Ref`) tells the engine to keep arbitrarily
  long history.

Following the pattern above keeps custom operators consistent with Qlib's built-in implementations
and ensures they cooperate with the expression cache and backtest pipelines.

# Alpha101 Operators vs Qlib Implementation Analysis

This document compares the operators defined in `alpha101.txt` with their implementations in qlib's `qlib/data/ops.py`.

## Summary

| Category | Total | ✅ Correct | ⚠️ Needs Clarification | ❌ Missing |
|----------|-------|-----------|----------------------|-----------|
| Basic Math | 7 | 7 | 0 | 0 |
| Comparison | 5 | 5 | 0 | 0 |
| Time-Series | 8 | 8 | 0 | 0 |
| Rolling Stats | 4 | 4 | 0 | 0 |
| Special | 6 | 6 | 0 | 0 |
| **Total** | **30** | **30** | **0** | **0** |

---

## Detailed Analysis

### 1. Basic Mathematical Operators

#### ✅ `abs(x)` - Standard Definition
- **Alpha101**: Standard absolute value
- **Qlib**: `class Abs(NpElemOperator)` - Line 135
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Uses numpy's `np.abs()` function

#### ✅ `log(x)` - Standard Definition
- **Alpha101**: Standard natural logarithm
- **Qlib**: `class Log(NpElemOperator)` - Line 180
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Uses numpy's `np.log()` function

#### ✅ `sign(x)` - Standard Definition
- **Alpha101**: Returns -1, 0, or 1 based on sign
- **Qlib**: `class Sign(NpElemOperator)` - Line 153
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Uses numpy's `np.sign()` function, with float32 conversion for bool inputs

#### ✅ `+`, `-`, `*`, `/` - Standard Arithmetic
- **Alpha101**: Standard arithmetic operators
- **Qlib**: 
  - `class Add(NpPairOperator)` - Line 629
  - `class Sub(NpPairOperator)` - Line 649
  - `class Mul(NpPairOperator)` - Line 669
  - `class Div(NpPairOperator)` - Line 689
- **Status**: ✅ **Correctly Implemented**

---

### 2. Comparison Operators

#### ✅ `>`, `<`, `==` - Standard Comparisons
- **Alpha101**: Standard comparison operators that return boolean masks
- **Qlib**: 
  - `class Gt(NpPairOperator)` - Line 749 (boolean >)
  - `class Ge(NpPairOperator)` - Line 769 (boolean >=)
  - `class Lt(NpPairOperator)` - Line 789 (boolean <)
  - `class Le(NpPairOperator)` - Line 809 (boolean <=)
  - `class Eq(NpPairOperator)` - Line 829 (boolean ==)
  - `class Ne(NpPairOperator)` - Line 849 (boolean !=)
  - `class Greater(NpPairOperator)` - Line 709 (element-wise max, **not** the Alpha101 `>` operator)
  - `class Less(NpPairOperator)` - Line 729 (element-wise min, **not** the Alpha101 `<` operator)
- **Status**: ✅ **Correctly Implemented** for boolean comparators
- **Notes**: Use `Gt`/`Ge`/`Lt`/`Le`/`Eq`/`Ne` for Alpha101 semantics; `Greater`/`Less` are convenience helpers for max/min aggregation.

#### ✅ `||`, `x ? y : z` - Logical and Conditional
- **Alpha101**: OR operator and ternary conditional
- **Qlib**: 
  - `class Or(NpPairOperator)` - Line 889
  - `class And(NpPairOperator)` - Line 869
  - `class If(ExpressionOps)` - Line 914
- **Status**: ✅ **Correctly Implemented**

---

### 3. Cross-Sectional Operators

#### ✅ `rank(x)` - Cross-Sectional Rank
- **Alpha101**: "cross-sectional rank" - ranks across all stocks at same time
- **Qlib**: `class CSRank(ElemOperator)` - Line 386
- **Status**: ✅ **CORRECTLY IMPLEMENTED**
- **Details**: 
  - Alpha101's `rank(x)` is a **cross-sectional** operator (ranks across stocks)
  - Qlib's `CSRank` correctly implements this behavior
  - Returns percentile rank in the open interval (0, 1] across all instruments at each timestamp
  - Uses caching for performance optimization
- **Note**: The `Rank(Rolling)` class (Line 1415) is a **different operator** that implements `ts_rank(x, d)` (time-series rank)
- **Usage**: `CSRank(Feature("close"))` for cross-sectional ranking in expressions

#### ✅ `scale(x, a)` - Cross-Sectional Scale
- **Alpha101**: "rescaled x such that sum(abs(x)) = a (default a = 1)"
- **Qlib**: `class Scale(ElemOperator)` - Line 472
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Loads the full universe per timestamp, rescales each cross-section so `sum(abs(x)) = a`, caches results per window, and defaults to `a = 1.0`. Only numeric targets are supported, aligning with Alpha101 usage.

---

### 4. Time-Series Operators

#### ✅ `delay(x, d)` - Delay/Lag
- **Alpha101**: "value of x d days ago"
- **Qlib**: `class Ref(Rolling)` - Line 1058
- **Status**: ✅ **Correctly Implemented**
- **Notes**: `Ref(feature, N)` returns value N periods ago

#### ✅ `delta(x, d)` - Difference
- **Alpha101**: "today's value of x minus the value of x d days ago"
- **Qlib**: `class Delta(Rolling)` - Line 1467
- **Status**: ✅ **Correctly Implemented**
- **Formula**: `x(t) - x(t-d)`

#### ✅ `correlation(x, y, d)` - Time-Serial Correlation
- **Alpha101**: "time-serial correlation of x and y for the past d days"
- **Qlib**: `class Corr(PairRolling)` - Line 1801
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Computes rolling Pearson correlation with NaN handling for zero std dev

#### ✅ `covariance(x, y, d)` - Time-Serial Covariance
- **Alpha101**: "time-serial covariance of x and y for the past d days"
- **Qlib**: `class Cov(PairRolling)` - Line 1834
- **Status**: ✅ **Correctly Implemented**

#### ✅ `ts_min(x, d)` - Time-Series Min
- **Alpha101**: "time-series min over the past d days"
- **Qlib**: `class Min(Rolling)` - Line 1285
- **Status**: ✅ **Correctly Implemented**
- **Aliases**: `min(x, d)` = `ts_min(x, d)`

#### ✅ `ts_max(x, d)` - Time-Series Max
- **Alpha101**: "time-series max over the past d days"
- **Qlib**: `class Max(Rolling)` - Line 1237
- **Status**: ✅ **Correctly Implemented**
- **Aliases**: `max(x, d)` = `ts_max(x, d)`

#### ✅ `ts_argmax(x, d)` - Time-Series ArgMax
- **Alpha101**: "which day ts_max(x, d) occurred on"
- **Qlib**: `class IdxMax(Rolling)` - Line 1257
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Returns the index (1-based) of max value in window

#### ✅ `ts_argmin(x, d)` - Time-Series ArgMin
- **Alpha101**: "which day ts_min(x, d) occurred on"
- **Qlib**: `class IdxMin(Rolling)` - Line 1305
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Returns the index (1-based) of min value in window

#### ✅ `ts_rank(x, d)` - Time-Series Rank
- **Alpha101**: "time-series rank in the past d days"
- **Qlib**: `class Rank(Rolling)` - Line 1415
- **Status**: ✅ **CORRECTLY IMPLEMENTED**
- **Notes**: 
  - This IS the correct operator for `ts_rank` (time-series rank)
  - Returns percentile rank of current value within rolling window
  - Uses `percentileofscore` for compatibility with older pandas versions
  - **Different from** `CSRank` which implements cross-sectional `rank(x)`

---

### 5. Rolling Statistics

#### ✅ `sum(x, d)` - Time-Series Sum
- **Alpha101**: "time-series sum over the past d days"
- **Qlib**: `class Sum(Rolling)` - Line 1132
- **Status**: ✅ **Correctly Implemented**

#### ✅ `product(x, d)` - Time-Series Product
- **Alpha101**: "time-series product over the past d days"
- **Qlib**: `class Product(Rolling)` - Line 1195
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Uses rolling product with a safe fallback to `rolling.apply(np.prod)` for older pandas builds. Raises a `ValueError` when asked for exponential windows (`0 < d < 1`), matching Alpha101's integer window semantics.

#### ✅ `stddev(x, d)` - Moving Standard Deviation
- **Alpha101**: "moving time-series standard deviation over the past d days"
- **Qlib**: `class Std(Rolling)` - Line 1152
- **Status**: ✅ **Correctly Implemented**

#### ✅ Mean (Implied)
- **Alpha101**: Not explicitly defined but used in formulas
- **Qlib**: `class Mean(Rolling)` - Line 1113
- **Status**: ✅ **Correctly Implemented**

---

### 6. Special Operators

#### ✅ `signedpower(x, a)` - Signed Power
- **Alpha101**: "x^a" (preserving the original sign of `x`)
- **Qlib**: `class SignedPower(PairOperator)` - Line 624
- **Status**: ✅ **Correctly Implemented**
- **Notes**:
  - Computes `sign(x) * |x|^a`, supports scalar or expression exponents, and validates length mismatches for easier debugging.
  - Referenced in Alpha#1 and Alpha#84 and now directly supported without custom workarounds.

#### ✅ `decay_linear(x, d)` - Linear Decay Weighted Average
- **Alpha101**: "weighted moving average over the past d days with linearly decaying weights d, d-1, ..., 1 (rescaled to sum up to 1)"
- **Qlib**: `class WMA(Rolling)` - Line 1600
- **Status**: ✅ **CORRECTLY IMPLEMENTED**
- **Notes**: 
  - WMA implementation uses weights `[1, 2, 3, ..., N]` normalized to sum to 1
  - This matches Alpha101's decay_linear specification exactly
  - Formula: `w = np.arange(len(x)) + 1; w = w / w.sum()`

#### ✅ `indneutralize(x, g)` - Industry Neutralization
- **Alpha101**: "x cross-sectionally neutralized against groups g (2, 3, 4)"
- **Qlib**: `class IndNeutralize(ElemOperator)` - Line 235
- **Status**: ✅ **Implemented** (for Vietnamese stocks)
- **Notes**: 
  - Implementation exists but specific to Vietnamese HOSE symbols
  - Uses ICB industry codes (icb_code2, icb_code3, icb_code4)
  - For other markets, would need different implementation or custom mapping

#### ✅ `adv{d}` - Average Daily Dollar Volume
- **Alpha101**: "average daily dollar volume for the past d days"
- **Qlib**: `class Adv(ExpressionOps)` - Line 1671
- **Status**: ✅ **Correctly Implemented**
- **Formula**: Mean of (price × volume) over d days

---

## Additional Qlib Operators Not in Alpha101

Qlib provides several additional operators not defined in Alpha101:

1. **`class Var(Rolling)`** - Rolling variance
2. **`class Skew(Rolling)`** - Rolling skewness
3. **`class Kurt(Rolling)`** - Rolling kurtosis
4. **`class Med(Rolling)`** - Rolling median
5. **`class Mad(Rolling)`** - Rolling mean absolute deviation
6. **`class Quantile(Rolling)`** - Rolling quantile
7. **`class Count(Rolling)`** - Rolling count of non-NaN values
8. **`class Slope(Rolling)`** - Rolling linear regression slope
9. **`class Rsquare(Rolling)`** - Rolling R-squared
10. **`class Resi(Rolling)`** - Rolling regression residuals
11. **`class EMA(Rolling)`** - Exponential moving average
12. **`class WMA(Rolling)`** - Weighted moving average
13. **`class Not(NpElemOperator)`** - Bitwise NOT
14. **`class Mask(NpElemOperator)`** - Feature masking

---

## Critical Differences

### 1. Cross-Sectional vs Time-Series Operations

**Important Distinction**: Qlib now correctly implements both variants:

- **Alpha101 `rank(x)`**: Cross-sectional → **Qlib `CSRank(feature)`** (Line 386)
  - Ranks across all stocks at each timestamp
  - Returns percentile rank (0 to 1) across instruments
  
- **Alpha101 `ts_rank(x, d)`**: Time-series → **Qlib `Rank(feature, d)`** (Line 1415)
  - Ranks within rolling window for each stock
  - Returns percentile rank (0 to 1) of current value in past d days

**Correct Usage for Alpha101 Formulas**:
```python
# For cross-sectional rank(x)
from qlib.data.ops import CSRank, Feature
cross_sectional = CSRank(Feature("close"))

# For time-series ts_rank(x, d)
from qlib.data.ops import Rank, Feature
time_series = Rank(Feature("close"), 10)
```

### 2. Coverage Updates

The Alpha101 operators that were previously missing are now part of core Qlib:

- **`scale(x, a)`** → `Scale(x, a)` handles numeric targets with cross-sectional caching.
- **`signedpower(x, a)`** → `SignedPower(x, a)` keeps the sign while exponentiating magnitudes.
- **`product(x, d)`** → `Product(x, d)` computes rolling products with a backward-compatible pandas fallback.

`decay_linear` continues to map to the existing `WMA` operator.

### 3. Implementation Recommendations

**For Cross-Sectional Operations** (like `rank`, `indneutralize`):
```python
# Use CSRank operator in expressions
from qlib.data.ops import CSRank, IndNeutralize, Feature

# Cross-sectional rank
rank_feature = CSRank(Feature("close"))

# Industry neutralization (Vietnamese stocks)
neutral_feature = IndNeutralize(Feature("close"), level=3)
```

**For Time-Series Operations** (like `ts_rank`, `delay`, `correlation`):
```python
# Use standard rolling operators
from qlib.data.ops import Rank, Ref, Corr, Feature

# Time-series rank over 10 days
ts_rank_feature = Rank(Feature("close"), 10)

# Delay by 5 days
delayed = Ref(Feature("close"), 5)

# 20-day correlation
corr_feature = Corr(Feature("close"), Feature("volume"), 20)
```

---

## Compatibility Matrix

| Alpha101 Function | Qlib Operator | Usage Context | Line Number | Compatibility |
|-------------------|---------------|---------------|-------------|---------------|
| `rank(x)` | `CSRank` | Cross-sectional | 386 | ✅ Direct mapping |
| `ts_rank(x, d)` | `Rank(x, d)` | Time-series | 1415 | ✅ Direct mapping |
| `delay(x, d)` | `Ref(x, d)` | Time-series | 1058 | ✅ Direct mapping |
| `delta(x, d)` | `Delta(x, d)` | Time-series | 1467 | ✅ Direct mapping |
| `correlation(x, y, d)` | `Corr(x, y, d)` | Time-series | 1801 | ✅ Direct mapping |
| `covariance(x, y, d)` | `Cov(x, y, d)` | Time-series | 1834 | ✅ Direct mapping |
| `scale(x, a)` | `Scale(x, a)` | Cross-sectional | 472 | ✅ Direct mapping |
| `signedpower(x, a)` | `SignedPower(x, a)` | Element-wise | 624 | ✅ Direct mapping |
| `decay_linear(x, d)` | `WMA(x, d)` | Time-series | 1600 | ✅ Direct mapping |
| `product(x, d)` | `Product(x, d)` | Time-series | 1195 | ✅ Direct mapping |
| `indneutralize(x, g)` | `IndNeutralize(x, g)` | Cross-sectional | 235 | ✅ VN-specific impl |
| `adv{d}` | `Adv(d)` | Time-series | 1671 | ✅ Direct mapping |

---

## Conclusion

Qlib now provides a comprehensive set of operators that cover every Alpha101 requirement, delivering **100% compatibility** (30 out of 30 operators correctly implemented).

### ✅ Correctly Implemented (30/30)

1. **Cross-Sectional Operators**: `CSRank` (Line 386), `Scale` (Line 472), `IndNeutralize` (Line 235)
2. **Time-Series Operators**: Full suite including `Rank/ts_rank`, `Ref/delay`, `Delta`, `Corr`, `Cov`, `Product`, etc.
3. **Rolling Statistics**: `Sum`, `Product`, `Mean`, `Std`, `Min`, `Max`, `IdxMax`, `IdxMin`
4. **Special Operators**: `WMA/decay_linear`, `Adv`, `EMA`, `SignedPower`
5. **Basic Math & Logic**: All standard element-wise, arithmetic, and comparison operators

### Key Takeaways

1. **Naming Clarity**: 
   - `CSRank` → Alpha101's `rank(x)` (cross-sectional)
   - `Rank` → Alpha101's `ts_rank(x, d)` (time-series)
   - `Scale` → Alpha101's `scale(x, a)` cross-sectional normalisation

2. **Implementation Quality**: All operators match Alpha101 specifications with proper NaN handling, edge case management, caching, and performance optimisations.

3. **For Alpha101 Formulas**:
   - Use `CSRank(feature)` for cross-sectional ranking
   - Use `Rank(feature, N)` for time-series ranking
   - Use `WMA` for `decay_linear` operations
   - Use `Scale(feature, a)` for Alpha101's `scale(x, a)` normalisation
   - Use `SignedPower(left, right)` when you need `signedpower`

4. **Industry Neutralization**: Available but specific to Vietnamese stocks (ICB codes); adaptable for other markets.

---

**Last Updated**: 2025-10-07  
**Source Files**: 
- `/home/manh/Code/qlib/scripts/data_collector/vnstock/alpha101.txt`
- `/home/manh/Code/qlib/qlib/data/ops.py` (2017 lines)

**Key Line Numbers**:
- `CSRank`: Line 386
- `Rank` (ts_rank): Line 1415
- `IndNeutralize`: Line 235
- `WMA` (decay_linear): Line 1600
- `Adv`: Line 1671

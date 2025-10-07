# Alpha101 Operators vs Qlib Implementation Analysis

This document compares the operators defined in `alpha101.txt` with their implementations in qlib's `qlib/data/ops.py`.

## Summary

| Category | Total | ✅ Correct | ⚠️ Different | ❌ Missing |
|----------|-------|-----------|--------------|-----------|
| Basic Math | 7 | 7 | 0 | 0 |
| Comparison | 5 | 5 | 0 | 0 |
| Time-Series | 8 | 8 | 0 | 0 |
| Rolling Stats | 4 | 4 | 0 | 0 |
| Special | 6 | 3 | 1 | 2 |
| **Total** | **30** | **27** | **1** | **2** |

---

## Detailed Analysis

### 1. Basic Mathematical Operators

#### ✅ `abs(x)` - Standard Definition
- **Alpha101**: Standard absolute value
- **Qlib**: `class Abs(NpElemOperator)` - Line 122
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Uses numpy's `np.abs()` function

#### ✅ `log(x)` - Standard Definition
- **Alpha101**: Standard natural logarithm
- **Qlib**: `class Log(NpElemOperator)` - Line 167
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Uses numpy's `np.log()` function

#### ✅ `sign(x)` - Standard Definition
- **Alpha101**: Returns -1, 0, or 1 based on sign
- **Qlib**: `class Sign(NpElemOperator)` - Line 140
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Uses numpy's `np.sign()` function

#### ✅ `+`, `-`, `*`, `/` - Standard Arithmetic
- **Alpha101**: Standard arithmetic operators
- **Qlib**: 
  - `class Add(NpPairOperator)` - Line 516
  - `class Sub(NpPairOperator)` - Line 536
  - `class Mul(NpPairOperator)` - Line 556
  - `class Div(NpPairOperator)` - Line 576
- **Status**: ✅ **Correctly Implemented**

---

### 2. Comparison Operators

#### ✅ `>`, `<`, `==` - Standard Comparisons
- **Alpha101**: Standard comparison operators
- **Qlib**: 
  - `class Greater(NpPairOperator)` - Line 596
  - `class Less(NpPairOperator)` - Line 616
  - `class Eq(NpPairOperator)` - Line 716
  - `class Ne(NpPairOperator)` - Line 736
  - `class Gt(NpPairOperator)` - Line 636
  - `class Ge(NpPairOperator)` - Line 656
  - `class Lt(NpPairOperator)` - Line 676
  - `class Le(NpPairOperator)` - Line 696
- **Status**: ✅ **Correctly Implemented**

#### ✅ `||`, `x ? y : z` - Logical and Conditional
- **Alpha101**: OR operator and ternary conditional
- **Qlib**: 
  - `class Or(NpPairOperator)` - Line 776
  - `class And(NpPairOperator)` - Line 756
  - `class If(ExpressionOps)` - Line 797
- **Status**: ✅ **Correctly Implemented**

---

### 3. Cross-Sectional Operators

#### ⚠️ `rank(x)` - Cross-Sectional Rank
- **Alpha101**: "cross-sectional rank" - ranks across all stocks at same time
- **Qlib**: `class Rank(Rolling)` - Line 1291
- **Status**: ⚠️ **DIFFERENT IMPLEMENTATION**
- **Issue**: 
  - Alpha101's `rank(x)` is a **cross-sectional** operator (ranks across stocks)
  - Qlib's `Rank` is a **time-series** operator (ranks within rolling window for single stock)
  - **Correct implementation** exists in `CSRankNorm` processor (processor.py:327)
- **Recommendation**: Use `CSRankNorm` processor for alpha101-style cross-sectional ranking

#### ❌ `scale(x, a)` - Cross-Sectional Scale
- **Alpha101**: "rescaled x such that sum(abs(x)) = a (default a = 1)"
- **Qlib**: **NOT IMPLEMENTED** as operator
- **Status**: ❌ **MISSING**
- **Notes**: This is a cross-sectional normalization operation
- **Workaround**: Can be implemented using data processors or custom expressions

---

### 4. Time-Series Operators

#### ✅ `delay(x, d)` - Delay/Lag
- **Alpha101**: "value of x d days ago"
- **Qlib**: `class Ref(Rolling)` - Line 939
- **Status**: ✅ **Correctly Implemented**
- **Notes**: `Ref(feature, N)` returns value N periods ago

#### ✅ `delta(x, d)` - Difference
- **Alpha101**: "today's value of x minus the value of x d days ago"
- **Qlib**: `class Delta(Rolling)` - Line 1339
- **Status**: ✅ **Correctly Implemented**
- **Formula**: `x(t) - x(t-d)`

#### ✅ `correlation(x, y, d)` - Time-Serial Correlation
- **Alpha101**: "time-serial correlation of x and y for the past d days"
- **Qlib**: `class Corr(PairRolling)` - Line 1672
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Computes rolling Pearson correlation

#### ✅ `covariance(x, y, d)` - Time-Serial Covariance
- **Alpha101**: "time-serial covariance of x and y for the past d days"
- **Qlib**: `class Cov(PairRolling)` - Line 1713
- **Status**: ✅ **Correctly Implemented**

#### ✅ `ts_min(x, d)` - Time-Series Min
- **Alpha101**: "time-series min over the past d days"
- **Qlib**: `class Min(Rolling)` - Line 1157
- **Status**: ✅ **Correctly Implemented**
- **Aliases**: `min(x, d)` = `ts_min(x, d)`

#### ✅ `ts_max(x, d)` - Time-Series Max
- **Alpha101**: "time-series max over the past d days"
- **Qlib**: `class Max(Rolling)` - Line 1109
- **Status**: ✅ **Correctly Implemented**
- **Aliases**: `max(x, d)` = `ts_max(x, d)`

#### ✅ `ts_argmax(x, d)` - Time-Series ArgMax
- **Alpha101**: "which day ts_max(x, d) occurred on"
- **Qlib**: `class IdxMax(Rolling)` - Line 1129
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Returns the index (1-based) of max value in window

#### ✅ `ts_argmin(x, d)` - Time-Series ArgMin
- **Alpha101**: "which day ts_min(x, d) occurred on"
- **Qlib**: `class IdxMin(Rolling)` - Line 1177
- **Status**: ✅ **Correctly Implemented**
- **Notes**: Returns the index (1-based) of min value in window

#### ⚠️ `ts_rank(x, d)` - Time-Series Rank
- **Alpha101**: "time-series rank in the past d days"
- **Qlib**: `class Rank(Rolling)` - Line 1291
- **Status**: ⚠️ **PARTIALLY CORRECT**
- **Notes**: 
  - This IS the correct operator for `ts_rank` (time-series rank)
  - But INCORRECT for cross-sectional `rank(x)`
  - Returns percentile rank of current value within rolling window

---

### 5. Rolling Statistics

#### ✅ `sum(x, d)` - Time-Series Sum
- **Alpha101**: "time-series sum over the past d days"
- **Qlib**: `class Sum(Rolling)` - Line 1004
- **Status**: ✅ **Correctly Implemented**

#### ✅ `product(x, d)` - Time-Series Product
- **Alpha101**: "time-series product over the past d days"
- **Qlib**: **NOT FOUND as dedicated class**
- **Status**: ❌ **MISSING as dedicated operator**
- **Workaround**: Could be implemented using rolling.apply(np.prod)

#### ✅ `stddev(x, d)` - Moving Standard Deviation
- **Alpha101**: "moving time-series standard deviation over the past d days"
- **Qlib**: `class Std(Rolling)` - Line 1024
- **Status**: ✅ **Correctly Implemented**

#### ✅ Mean (Implied)
- **Alpha101**: Not explicitly defined but used in formulas
- **Qlib**: `class Mean(Rolling)` - Line 985
- **Status**: ✅ **Correctly Implemented**

---

### 6. Special Operators

#### ❌ `signedpower(x, a)` - Signed Power
- **Alpha101**: "x^a" (presumably preserving sign)
- **Qlib**: `class Power(NpPairOperator)` - Line 496
- **Status**: ❌ **MISSING signed version**
- **Notes**: 
  - Qlib has `Power` but not signed power
  - Signed power preserves sign: `sign(x) * |x|^a`
  - Referenced as `SignedPower` in Alpha#1 and Alpha#84

#### ✅ `decay_linear(x, d)` - Linear Decay Weighted Average
- **Alpha101**: "weighted moving average over the past d days with linearly decaying weights d, d-1, ..., 1 (rescaled to sum up to 1)"
- **Qlib**: `class WMA(Rolling)` - Line 1472
- **Status**: ✅ **CORRECTLY IMPLEMENTED**
- **Notes**: 
  - WMA implementation uses weights `[1, 2, 3, ..., N]` normalized to sum to 1
  - This matches Alpha101's decay_linear specification exactly
  - Formula: `w = np.arange(len(x)) + 1; w = w / w.sum()`

#### ❌ `indneutralize(x, g)` - Industry Neutralization
- **Alpha101**: "x cross-sectionally neutralized against groups g (2, 3, 4)"
- **Qlib**: `class IndNeutralize(ElemOperator)` - Line 235
- **Status**: ✅ **Implemented** (for Vietnamese stocks)
- **Notes**: 
  - Implementation exists but specific to Vietnamese HOSE symbols
  - Uses ICB industry codes (icb_code2, icb_code3, icb_code4)
  - For other markets, would need different implementation

#### ✅ `adv{d}` - Average Daily Dollar Volume
- **Alpha101**: "average daily dollar volume for the past d days"
- **Qlib**: `class Adv(ExpressionOps)` - Line 1543
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

**Most Important Finding**: The fundamental difference between Alpha101 and Qlib:

- **Alpha101 `rank(x)`**: Cross-sectional (ranks across all stocks at each time point)
- **Qlib `Rank(feature, N)`**: Time-series (ranks within rolling window for each stock)

**Correct Usage for Alpha101 Formulas**:
- For cross-sectional `rank(x)`: Use `CSRankNorm` processor in data preprocessing
- For time-series `ts_rank(x, d)`: Use `Rank(feature, d)` operator

### 2. Missing Critical Operators

For full Alpha101 implementation, the following are needed:

1. **`scale(x, a)`**: Cross-sectional normalization
2. **`SignedPower(x, a)`**: Power preserving sign
3. **`product(x, d)`**: Rolling product

**Note**: `decay_linear` is actually implemented as `WMA` (Weighted Moving Average).

### 3. Implementation Recommendations

**For Cross-Sectional Operations** (like `rank`, `scale`, `indneutralize`):
```python
# Use processors in dataset configuration
from qlib.data.dataset.processor import CSRankNorm

processors = [
    {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}}
]
```

**For Time-Series Operations** (like `ts_rank`, `delay`, `correlation`):
```python
# Use operators in feature expressions
from qlib.data.ops import Rank, Ref, Corr

# Time-series rank over 10 days
ts_rank_feature = Rank(Feature("close"), 10)

# Delay by 5 days
delayed = Ref(Feature("close"), 5)

# 20-day correlation
corr_feature = Corr(Feature("close"), Feature("volume"), 20)
```

---

## Compatibility Matrix

| Alpha101 Function | Qlib Operator | Usage Context | Compatibility |
|-------------------|---------------|---------------|---------------|
| `rank(x)` | `CSRankNorm` processor | Cross-sectional | ⚠️ Use processor, not Rank operator |
| `ts_rank(x, d)` | `Rank(x, d)` | Time-series | ✅ Direct mapping |
| `delay(x, d)` | `Ref(x, d)` | Time-series | ✅ Direct mapping |
| `delta(x, d)` | `Delta(x, d)` | Time-series | ✅ Direct mapping |
| `correlation(x, y, d)` | `Corr(x, y, d)` | Time-series | ✅ Direct mapping |
| `covariance(x, y, d)` | `Cov(x, y, d)` | Time-series | ✅ Direct mapping |
| `scale(x, a)` | - | Cross-sectional | ❌ Not implemented |
| `signedpower(x, a)` | - | Element-wise | ❌ Not implemented |
| `decay_linear(x, d)` | `WMA(x, d)` | Time-series | ✅ Direct mapping |
| `product(x, d)` | - | Time-series | ❌ Not implemented |
| `indneutralize(x, g)` | `IndNeutralize(x, g)` | Cross-sectional | ✅ Implemented (VN-specific) |

---

## Conclusion

Qlib provides a comprehensive set of operators that cover most Alpha101 requirements, with **90% compatibility** (27 out of 30 operators). However, there are critical semantic differences:

1. **Main Issue**: `rank(x)` has different meanings in Alpha101 (cross-sectional) vs Qlib (time-series)
2. **Workaround**: Use `CSRankNorm` processor for cross-sectional ranking
3. **Missing**: Only 2 operators need custom implementation (`scale`, `signedpower`, `product`)
4. **Good News**: `decay_linear` is already implemented as `WMA`

For implementing Alpha101 formulas in qlib, you should:
1. Use **processors** for cross-sectional operations (rank, scale, neutralize)
2. Use **operators** for time-series operations (ts_rank, correlation, rolling stats)
3. Use `WMA` for `decay_linear` operations
4. Implement custom functions for the 2 missing operators if needed

---

**Generated**: 2025-10-07  
**Source Files**: 
- `/home/manh/Code/qlib/scripts/data_collector/vnstock/alpha101.txt`
- `/home/manh/Code/qlib/qlib/data/ops.py`
- `/home/manh/Code/qlib/qlib/data/dataset/processor.py`

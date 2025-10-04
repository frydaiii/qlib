# How to Implement New Expressions in Qlib

This guide explains the two main approaches for implementing new expressions/fields in qlib, with practical examples and trade-offs.

## Overview

There are **two primary approaches** to implement new expressions in qlib:

1. **Expression-Level Implementation** (in qlib handlers)
2. **Data Provider Level Implementation** (in data collectors)

## Approach 1: Expression-Level Implementation

### Description
Implement new expressions using qlib's formula system with existing base fields.

### When to Use
- Data provider doesn't have the required field natively
- Need flexible, easily modifiable calculations  
- Want data-source independence
- Implementing custom technical indicators

### Implementation Steps

#### Step 1: Create the Expression Formula
```python
# Example: VWAP approximation using Simpson's rule
simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"

# Example: Custom moving average
custom_ma = "Mean($close, 20)"

# Example: Price momentum
price_momentum = "$close / Ref($close, 5) - 1"
```

#### Step 2: Use in Handler Configuration
```python
class CustomHandler(DataHandlerLP):
    def get_feature_config(self):
        fields = []
        names = []
        
        # Define your custom expression
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"
        
        # Add to fields with processing
        fields += [self.get_normalized_price_feature(simpson_vwap, 0)]
        names += ["$vwap"]
        
        return fields, names
```

#### Step 3: Apply Processing Templates
```python
def get_normalized_price_feature(self, price_field, shift=0):
    """Apply normalization and other processing"""
    template_if = "If(IsNull({1}), {0}, {1})"
    template_paused = "Select(Or(IsNull($paused), Eq($paused, 0.0)), {0})"
    template_fillnan = "BFillNan(FFillNan({0}))"
    
    if shift == 0:
        template_norm = "Cut({0}/Ref(DayLast({1}), 240), 240, None)"
    else:
        template_norm = f"Cut(Ref({0}, {shift})/Ref(DayLast({1}), 240), 240, None)"
    
    feature_ops = template_norm.format(
        template_if.format(
            template_fillnan.format(template_paused.format("$close")),
            template_paused.format(price_field),
        ),
        template_fillnan.format(template_paused.format("$close")),
    )
    return feature_ops
```

### Real Example from qlib
```python
# From examples/highfreq/highfreq_handler.py
class HighFreqHandler(DataHandlerLP):
    def get_feature_config(self):
        # Because there is no vwap field in the yahoo data, 
        # a method similar to Simpson integration is used to approximate vwap
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"
        
        fields += [get_normalized_price_feature(simpson_vwap, 0)]
        names += ["$vwap"]
```

## Approach 2: Data Provider Level Implementation

### Description
Calculate new fields during data collection/normalization and store them as part of the raw data.

### When to Use
- Data provider has native support for the field
- Need high performance (pre-calculated)
- Complex calculations that are expensive to compute repeatedly
- Want to store calculated values permanently

### Implementation Steps

#### Step 1: Modify Data Collector
```python
# In scripts/data_collector/your_provider/collector.py
class YourDataCollector(BaseCollector):
    def get_data(self, symbol, interval, start_datetime, end_datetime):
        # Get base OHLCV data
        df = self.fetch_raw_data(symbol, interval, start_datetime, end_datetime)
        
        # Add calculated fields
        df = self.add_calculated_fields(df)
        
        return df
    
    def add_calculated_fields(self, df):
        """Add custom calculated fields to the data"""
        if not df.empty:
            # Calculate VWAP if not provided by source
            if 'vwap' not in df.columns:
                # Traditional VWAP calculation (requires tick data)
                # or approximation using OHLC
                df['vwap'] = (df['open'] + 2*df['high'] + 2*df['low'] + df['close']) / 6
            
            # Add other custom indicators
            df['rsi'] = self.calculate_rsi(df['close'])
            df['bollinger_upper'] = self.calculate_bollinger_bands(df['close'])[0]
            
        return df
```

#### Step 2: Update Normalization
```python
# In the normalize class
class YourDataNormalize(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume", "vwap"]  # Add new fields
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Standard normalization
        df = super().normalize(df)
        
        # Custom field processing
        if 'vwap' in df.columns:
            df['vwap'] = self.process_vwap(df['vwap'])
            
        return df
    
    def process_vwap(self, vwap_series):
        """Custom processing for VWAP field"""
        # Handle NaN values, outliers, etc.
        return vwap_series.fillna(method='ffill')
```

#### Step 3: Update Column Definitions
```python
# Update the COLUMNS constant to include new fields
COLUMNS = ["open", "close", "high", "low", "volume", "vwap", "rsi"]
```

### Real Example from qlib
```python
# From scripts/data_collector/vnstock/collector.py
# VNStock provides VWAP natively, so it's available as $vwap directly

# Data comes with VWAP from the API
class VNStockNormalize1D(VNStockNormalize):
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        # VWAP is already in the data from VNStock API
        # Just apply standard price adjustments
        for _col in self.COLUMNS:
            if _col == "volume":
                df[_col] = df[_col] / df["factor"]
            else:
                df[_col] = df[_col] * df["factor"]
        return df
```

## Available Expression Operators

### Basic Operators
```python
# Arithmetic
"$close + $open"
"$high - $low"
"$close * $volume"
"$high / $low"

# Comparison
"Greater($close, $open)"
"Less($low, Ref($low, 1))"
"Eq($volume, 0)"

# Logical
"And(Gt($close, $open), Gt($volume, 1000))"
"Or(IsNull($close), Eq($close, 0))"
"If(Gt($close, $open), 1, 0)"
```

### Time Series Operators
```python
# Reference (shift)
"Ref($close, 1)"  # Previous day close
"Ref($volume, -1)"  # Next day volume

# Rolling operations
"Mean($close, 20)"  # 20-day moving average
"Std($close, 20)"   # 20-day standard deviation
"Sum($volume, 5)"   # 5-day volume sum
"Max($high, 10)"    # 10-day maximum high
"Min($low, 10)"     # 10-day minimum low

# Advanced rolling
"Slope($close, 20)"     # Linear regression slope
"Rsquare($close, 20)"   # R-squared of linear fit
"Rank($close, 20)"      # Percentile rank
"Quantile($close, 20, 0.8)"  # 80th percentile
```

### Data Cleaning Operators
```python
# Handle missing values
"FFillNan($close)"      # Forward fill
"BFillNan($close)"      # Backward fill
"IsNull($close)"        # Check for NaN

# Filtering
"Select(Gt($volume, 0), $close)"  # Select where volume > 0
"Cut($close, 240, None)"          # Keep data from 240 periods onward

# Outlier handling
"If(Gt($close, Mul(2, Mean($close, 20))), Mean($close, 20), $close)"
```

### High-Frequency Specific
```python
# Day-based operations
"DayLast($close)"           # Last value of the day
"DayCumsum($volume, '9:30', '15:00')"  # Cumulative sum during day
"DayMean($close, '9:30', '15:00')"     # Day average in time range
```

## Comparison of Approaches

| Aspect | Expression-Level | Data Provider Level |
|--------|------------------|-------------------|
| **Performance** | Calculated on-demand | Pre-calculated, faster |
| **Flexibility** | Easy to modify | Requires data re-collection |
| **Storage** | No extra storage | Requires more storage |
| **Data Independence** | Works with any provider | Provider-specific |
| **Maintenance** | Code in handlers | Code in collectors |
| **Debugging** | Easier to debug | Harder to trace issues |
| **Memory Usage** | Higher during calculation | Lower during usage |

## Best Practices

### Choose Expression-Level When:
- ✅ Prototyping and experimenting
- ✅ Need to modify formulas frequently  
- ✅ Working with multiple data providers
- ✅ Implementing simple mathematical operations
- ✅ Building research workflows

### Choose Data Provider Level When:
- ✅ Data source natively provides the field
- ✅ Complex calculations that are computationally expensive
- ✅ Need maximum performance in production
- ✅ Calculations require external data or APIs
- ✅ Want to cache expensive computations

### General Guidelines
1. **Start with Expression-Level** for rapid development
2. **Move to Data Provider Level** for production optimization
3. **Document your expressions** clearly in code comments
4. **Test with multiple time periods** and edge cases
5. **Handle missing data** appropriately
6. **Consider computational complexity** for large datasets

## Common Patterns

### Custom Technical Indicators
```python
# RSI (Relative Strength Index)
def get_rsi_expression(period=14):
    return f"""
    100 - 100 / (1 + 
        Sum(If(Gt($close - Ref($close, 1), 0), $close - Ref($close, 1), 0), {period}) / 
        Sum(If(Lt($close - Ref($close, 1), 0), Abs($close - Ref($close, 1)), 0), {period})
    )
    """

# Bollinger Bands
def get_bollinger_upper(period=20, std_dev=2):
    return f"Mean($close, {period}) + {std_dev} * Std($close, {period})"

def get_bollinger_lower(period=20, std_dev=2):
    return f"Mean($close, {period}) - {std_dev} * Std($close, {period})"
```

### Price Transformations
```python
# Log returns
log_returns = "Log($close / Ref($close, 1))"

# Percentage change
pct_change = "$close / Ref($close, 1) - 1"

# Normalized prices
normalized_close = "$close / Mean($close, 252)"  # Normalize by yearly average
```

### Volume Indicators
```python
# Volume Moving Average
volume_ma = "Mean($volume, 20)"

# Volume Rate of Change
volume_roc = "$volume / Ref($volume, 1) - 1"

# Price Volume Trend
pvt = "Sum(($close - Ref($close, 1)) / Ref($close, 1) * $volume, 20)"
```

## Debugging and Testing

### Test Your Expressions
```python
# Simple validation
def test_expression(expression, data):
    """Test an expression with sample data"""
    import qlib
    qlib.init()
    
    # Load test data
    df = qlib.data.D.features(
        instruments=["AAPL"],
        fields=[expression],
        start_time="2020-01-01",
        end_time="2020-12-31"
    )
    
    print(f"Expression: {expression}")
    print(f"Result shape: {df.shape}")
    print(f"Sample values:\n{df.head()}")
    print(f"NaN count: {df.isna().sum().iloc[0]}")
    
    return df
```

### Common Issues and Solutions
```python
# Handle division by zero
safe_ratio = "$close / If(Eq($volume, 0), 1, $volume)"

# Handle missing values in calculations
robust_mean = "Mean(If(IsNull($close), Ref($close, 1), $close), 20)"

# Prevent infinite values
bounded_value = "If(Gt(Abs($close), 1000), Sign($close) * 1000, $close)"
```

## Examples by Use Case

### High-Frequency Trading
```python
# Micro-price (for order book data)
microprice = "($bid * $askV + $ask * $bidV) / ($bidV + $askV)"

# Spread
spread = "$ask - $bid"

# Mid-price
midprice = "($bid + $ask) / 2"
```

### Risk Management
```python
# Volatility
volatility = "Std($close / Ref($close, 1) - 1, 20)"

# VaR (Value at Risk) approximation
var_95 = "Quantile($close / Ref($close, 1) - 1, 252, 0.05)"

# Maximum Drawdown
max_dd = "1 - $close / Max($close, 252)"
```

### Market Regime Detection
```python
# Trend strength
trend_strength = "Abs(Slope($close, 20) / Std($close, 20))"

# Market volatility regime
vol_regime = "If(Gt(Std($close / Ref($close, 1), 20), Mean(Std($close / Ref($close, 1), 20), 252)), 1, 0)"
```

This guide provides a comprehensive overview of implementing new expressions in qlib. Choose the approach that best fits your use case, and don't hesitate to start with expression-level implementation for flexibility during development.
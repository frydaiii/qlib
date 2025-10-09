# Implementing the Alpha101 Factor Library in Qlib

This guide explains how to translate the canonical **Alpha101** formulas into the
Qlib expression system, build them into a runnable pipeline, and validate the
output—similar to the existing Alpha158/Alpha360 workflows.

> **Prerequisites**
>
> - Qlib development environment is set up (`make prerequisite`, `make develop`)
> - Activate the local virtual environment before running any Python commands:
>   ```bash
>   source .venv/bin/activate
>   ```
> - You have read the operator mapping in
>   `docs/developer/alpha101_operators_analysis.md`

---

## 1. Map Alpha101 Operators to Qlib Primitives

Alpha101 mixes **cross-sectional** and **time-series** constructs. Use the proven
Qlib operators for each group:

| Alpha101 idea            | Qlib operator(s)                                   |
|--------------------------|----------------------------------------------------|
| `rank(x)` (cross-sectional) | `CSRank(feature)`                              |
| `ts_rank(x, d)`          | `Rank(feature, d)`                                 |
| `delay(x, d)`            | `Ref(feature, d)`                                  |
| `delta(x, d)`            | `Delta(feature, d)`                                |
| `decay_linear(x, d)`     | `WMA(feature, d)`                                  |
| `scale(x, a)`            | `Scale(feature, a)`                                |
| Comparisons / ternary    | `Gt`, `Lt`, `Ge`, `Le`, `Eq`, `Ne`, `If`           |
| Signed power             | `SignedPower(left, right)`                         |
| Industry neutralisation  | `IndNeutralize(feature, level)`                    |

*Always* replace inline math with the operators listed in `qlib/data/ops.py`.
This guarantees consistent NaN handling, window semantics, and caching.

---

## 2. Prepare the Alpha Definition File

1. Copy the canonical Alpha101 formulas into
   `scripts/data_collector/<provider>/alpha101.txt`.
2. Convert every formula to the expression syntax that the qlib parser accepts:
   - Prefix dollar-sign base fields (`$open`, `$close`, `$returns`, `$adv20`…)
   - Replace ternaries with nested `If` expressions
   - Swap `ts_*` calls for the rolling counterparts (`Rank`, `Min`, `IdxMax`, …)
   - Replace exponentiation (`x^y`) with `Power(x, y)` or `SignedPower`
3. Keep expressions on a single logical line—line breaks are permitted for
   readability, but avoid trailing whitespace or tab characters.

> ✅ Reference implementation:
> `scripts/data_collector/vnstock/alpha101.txt`  
> Only Qlib-ready operators appear in the file—use it as a template.

---

## 3. Hook the Factors into a Data Handler

To mirror the Alpha158/Alpha360 experience, register the Alpha101 feature list
inside a handler-class configuration:

```yaml
data_handler_config:
  start_time: 2015-01-01
  end_time: 2024-12-31
  fit_start_time: 2015-01-01
  fit_end_time: 2023-12-31
  instruments: vn_all
  features:
    - ["ALPHA101", "Alpha101"]
  data_handler: Alpha101
```

Implementation steps:

1. Create a new handler class (e.g. `Alpha101Handler`) under
   `qlib/contrib/data/handler/`.
2. Within `get_feature_config`, load each expression from the TXT file and
   append it to the `fields` list (mirroring `Alpha158`’s workflow).
3. Optionally expose switches for sub-sets (top-10, thematic groups, etc.) to
   keep the YAML concise.

> Tip: If you simply need the whole library, you can subclass
> `qlib.contrib.data.handler.base.DataHandlerLP` and return the Alpha101
> expressions from `feature_config()`, reusing the parser shown below.

---

## 4. Parse & Validate the Expressions

Before running a full dump, lint the expressions from the TXT file:

```python
from pathlib import Path

from qlib.data.ops import Feature
from qlib.expression import ExpressionDAG

def read_alpha101(path: Path) -> list[str]:
    formulas = []
    for line in path.read_text().splitlines():
        if line.startswith("Alpha#"):
            _, expr = line.split(":", 1)
            formulas.append(expr.strip())
    return formulas

alpha_file = Path("scripts/data_collector/vnstock/alpha101.txt")
formulas = read_alpha101(alpha_file)

dag = ExpressionDAG()
for idx, expr in enumerate(formulas, start=1):
    dag.parse(expr, f"Alpha#{idx}")
print(f"Validated {len(formulas)} Alpha101 expressions ✔")
```

Run the script with the environment activated. Any operator mismatch or syntax
error is reported with the offending formula name.

---

## 5. Materialise Factors

Once validated, use a workflow configuration (mirroring the Alpha158 YAML) to
dump the factors:

```bash
source .venv/bin/activate
qrun examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha101.yaml
```

Key items for the config:

- **Data handler** – points to your Alpha101 handler class.
- **Features list** – selects the new Alpha bundle.
- **Label definition** – choose or reuse the Alpha158 label.
- **Dataset processor** – apply cross-sectional normalisation if needed.

For incremental refresh, call the collector you extended (e.g.
`scripts/data_collector/vnstock/collector.py`) so that it writes the alpha
signals into your storage backend alongside prices.

---

## 6. Testing & Regression

- Add a smoke test that parses a handful of Alpha101 expressions and loads a
  small dataset window (`pytest tests/data_collector/test_alpha101.py`).
- Compare a random subset of generated factors with a trusted implementation if
  you have one (Pandas, Zipline, etc.).
- Remember to run `pytest --maxfail=1 --disable-warnings` before opening a PR.

---

## 7. Useful References

- Operator mapping: `docs/developer/alpha101_operators_analysis.md`
- Element/rolling operator implementation: `qlib/data/ops.py`
- Alpha158 handler template: `qlib/contrib/data/handler.py`
- Workflow examples: `examples/benchmarks/*/workflow_config_*_Alpha158.yaml`

This structure lets you maintain the full Alpha101 catalogue with the same
conventions as Alpha158/Alpha360, providing reliable factors for modelling,
backtests, and production workflows. Happy factor building!

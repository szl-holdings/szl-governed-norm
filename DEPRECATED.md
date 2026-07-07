# DEPRECATED — this repo has been consolidated into `szl-lambda-gate`

**Status: DEPRECATED (Wave D consolidation). Canonical home: [`szl-holdings/szl-lambda-gate`](https://github.com/szl-holdings/szl-lambda-gate).**

`szl-governed-norm` was a duplicate SZL *kernels* micro-repo. To keep ONE
canonical kernels package, its unique code has been **copied (folded) into the
canonical `szl-lambda-gate` repo** under the subpackage
[`szl_lambda_gate.governed_norm`](https://github.com/szl-holdings/szl-lambda-gate/tree/main/torch-ext/szl_lambda_gate/governed_norm).

Nothing here was deleted. This repository is **kept intact and reversible** —
**archiving it is a later founder step**, not part of this consolidation.

## What moved (and where)

| This repo (`szl_governed_norm`) | Canonical home (`szl_lambda_gate.governed_norm`) |
| --- | --- |
| `rms_norm`, `layer_norm`, `fused_add_rms_norm` (`_norm.py`) | `szl_lambda_gate.governed_norm.rms_norm` / `layer_norm` / `fused_add_rms_norm` (also re-exported at `szl_lambda_gate.rms_norm`, etc.) |
| `ReceiptChain`, `emit_receipt`, SHA3-256 governance receipts (`_receipt.py`) | `szl_lambda_gate.governed_norm.ReceiptChain` / `emit_receipt` |
| `receipt_head` / `receipt_count` / `receipt_tail` / `receipt_verify` / `selfcheck` | same names under `szl_lambda_gate.governed_norm` |
| Hub layers `RMSNorm` / `LayerNorm` / `FusedAddRMSNorm` (`layers.py`) | `szl_lambda_gate.governed_norm.layers` |
| Correctness + receipt tests | `szl-lambda-gate/tests/governed_norm_test_*.py` |

## Migration

```python
# Before
import szl_governed_norm as gn
y = gn.rms_norm(x, eps=1e-6, governed=True)

# After (canonical)
from szl_lambda_gate import governed_norm as gn
y = gn.rms_norm(x, eps=1e-6, governed=True)
# or the top-level convenience re-export:
import szl_lambda_gate as lg
y = lg.rms_norm(x, eps=1e-6)
```

## Honesty note (unchanged)

The doctrine labels are preserved verbatim in the folded copy: the normalization
kernels are a correctness reference (no fabricated benchmarks), governance
receipts are an **integrity/EVIDENCE trail, not a proof of correctness**, and
**Λ = Conjecture 1 (advisory, uniqueness OPEN)** — never upgraded to proven.

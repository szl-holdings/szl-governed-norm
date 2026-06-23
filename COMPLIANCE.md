# Kernel Hub Compliance Checklist — `szl_governed_norm`

**Repo:** `SZLHOLDINGS/szl-governed-norm` (repo_type=model)
**Kind:** Universal (pure-Python) kernel · **Version:** 0.1.0 · **License:** Apache-2.0
**Verified by:** Dev #4 (release/packaging) · Date: 2026-06-23
**Verification environment:** torch 2.12 (CPU), Python 3.12 (`py_compile` target `>=3.9`).

Requirements checked against the Hugging Face Kernel Hub
[kernel-requirements](https://huggingface.co/docs/kernels/kernel-requirements).
Every item below was checked mechanically (grep / `py_compile` / live import),
not by eyeballing. Results are reported HONESTLY.

---

## Summary: PASS (8/8)

| # | Requirement | Result |
|---|-------------|--------|
| 1 | Pure-Python / universal — no compiled artifacts | **PASS** |
| 2 | Only `torch` + stdlib (+ self) imports | **PASS** |
| 3 | Python 3.9+ compatible (`py_compile`) | **PASS** |
| 4 | Relative imports only (no absolute self-imports) | **PASS** |
| 5 | `layers` defined, importable, exported in `__init__` | **PASS** |
| 6 | Layers are pure `nn.Module` (no `__init__`, no class vars, only `forward`) | **PASS** |
| 7 | Universal build dir layout (`build/torch-universal/szl_governed_norm/`) | **PASS** |
| 8 | Unique ops namespace suffix (not version/git-tag based) | **PASS** |

---

## 1. Pure-Python / universal — no compiled artifacts — **PASS**

`build.toml` declares `[torch] universal = true`, i.e. a pure-Python package
with no compiled files. A filesystem scan for native binaries found **none**:

```
find . -type f \( -name '*.so' -o -name '*.pyd' -o -name '*.dll' \
  -o -name '*.dylib' -o -name '*.o' -o -name '*.a' \)
# → (no matches)
```

The only non-`.py` runtime content is `metadata.json` (informational). The
`__pycache__/*.pyc` files present locally are build-time caches and are
**excluded** from upload (see `PUBLISH_PLAN.md`).

## 2. Only `torch` + stdlib (+ self) imports — **PASS**

Every `.py` in `build/torch-universal/szl_governed_norm/` and
`torch-ext/szl_governed_norm/` was parsed with Python's `ast` module and each
import classified. Full enumeration (identical in both trees):

| File | Import | Class |
|------|--------|-------|
| `__init__.py` | `from typing import Any, Dict, List, Optional` | stdlib |
| `__init__.py` | `import torch` | torch |
| `__init__.py` | `from . import layers` | self (relative) |
| `__init__.py` | `from ._norm import layer_norm` | self (relative) |
| `__init__.py` | `from ._norm import rms_norm` | self (relative) |
| `__init__.py` | `from ._receipt import ReceiptChain, default_chain` | self (relative) |
| `_norm.py` | `from typing import Optional` | stdlib |
| `_norm.py` | `import torch` | torch |
| `_ops.py` | `import torch` | torch |
| `_receipt.py` | `import hashlib` | stdlib |
| `_receipt.py` | `import json` | stdlib |
| `_receipt.py` | `import threading` | stdlib |
| `_receipt.py` | `import time` | stdlib |
| `_receipt.py` | `from typing import Any, Dict, List, Optional` | stdlib |
| `_receipt.py` | `import torch` | torch |
| `layers.py` | `import torch` | torch |
| `layers.py` | `from torch import nn` | torch |
| `layers.py` | `from ._norm import layer_norm, rms_norm` | self (relative) |

**Imports outside {stdlib, torch, self}: NONE.** Note: the `import torch` /
`from kernels import get_kernel` lines visible in `__init__.py` near the top
are inside the module **docstring** (a usage example), not executed imports —
confirmed by AST parsing, which reports them nowhere.

## 3. Python 3.9+ compatible — **PASS**

All sources compile cleanly:

```
python3 -m py_compile build/torch-universal/szl_governed_norm/*.py   # → OK
python3 -m py_compile torch-ext/szl_governed_norm/*.py               # → OK
python3 -m py_compile tests/test_norm.py                             # → OK
```

Type hints use `typing.Optional` / `Dict` / `List` (not the 3.10+ `X | Y` or
bare-builtin-generic forms), so they are valid on 3.9. `requires-python` in
`pyproject.toml` is `>=3.9`.

## 4. Relative imports only — **PASS**

All intra-package imports use the leading-dot relative form
(`from . import layers`, `from ._norm import ...`, `from ._receipt import ...`).
There are **no** absolute self-imports of the form `from szl_governed_norm...`
or `import szl_governed_norm...`. Verified:

```
grep -rnE 'from szl_governed_norm|import szl_governed_norm' \
  build/torch-universal torch-ext   # → (no matches)
```

## 5. `layers` defined, importable, exported — **PASS**

`layers.py` defines `RMSNorm` and `LayerNorm`. `__init__.py` does
`from . import layers` and lists `"layers"` in `__all__`, as required for Hub
layer mapping. Confirmed by live import (Item 6).

## 6. Layers are pure `nn.Module` — **PASS**

`RMSNorm` and `LayerNorm` each subclass `torch.nn.Module`, define **only** a
`forward` method, declare **no** `__init__`, and use **no** class variables
(the only permitted class vars would be `has_backward` / `can_torch_compile`,
which are correctly absent). They read `weight`/`bias`/`eps` off the bound host
module via `getattr`, so they remain stateless/pure. Verified by live import:

```
sys.path.insert(0, 'build/torch-universal'); import szl_governed_norm as gn
issubclass(gn.layers.RMSNorm,  torch.nn.Module)  # → True
issubclass(gn.layers.LayerNorm, torch.nn.Module) # → True
gn.rms_norm(torch.randn(4,16), eps=1e-6).shape   # → (4, 16)
gn.rms_norm(x, eps=1e-6, governed=True); gn.receipt_count()   # → 1
gn.receipt_verify()  # → {'ok': True, 'depth': 1, 'first_break_seq': -1, ...}
```

All checks returned the expected values; the package imports and runs on CPU.

## 7. Universal build directory layout — **PASS**

The loader resolves `build/<variant>/<repo_name_underscored>/`. For a universal
kernel the variant directory is `torch-universal`, and it contains the package
dir `szl_governed_norm/` with `__init__.py` — present and correct. This is the
tree `get_kernel("SZLHOLDINGS/szl-governed-norm")` will import.

## 8. Unique ops namespace suffix — **PASS**

`_ops.py` registers under `torch.ops._szl_governed_norm_20260623075422` and the
prefix helper uses the same `_szl_governed_norm_20260623075422::` namespace. The
suffix is build-time "random material" (a timestamp), **not** a version number
or git tag — satisfying the uniqueness rule that lets multiple versions load in
one process. (For a pure-Python universal kernel this namespace is unused at
runtime since no native ops are registered, but it is present and compliant.)

---

## Notes / honesty caveats (non-blocking)

- **README.md / LICENSE are owned by other devs.** `pyproject.toml` references
  `readme = "README.md"` and `license = "Apache-2.0"`; the SPDX headers in the
  sources also declare Apache-2.0. Confirm `README.md` and a `LICENSE` file are
  actually present at the repo root before publishing (see `PUBLISH_PLAN.md`).
  These are outside Dev #4's ownership and are NOT verified here.
- **CPU-only verification.** `backends` lists `["cpu","cuda"]`; CUDA was not
  exercised (no GPU in this environment). The code is device-agnostic pure
  PyTorch, so CUDA support is by construction, but it was not run on a GPU here.
- **`torch-ext/` vs `build/torch-universal/`** are byte-for-byte equivalent
  Python sources; both pass every check above.

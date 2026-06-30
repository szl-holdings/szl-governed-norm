---
tags:
- kernel
license: apache-2.0
---

# szl-governed-norm

> **GitHub mirror** of the Kernel Hub kernel published at **[huggingface.co/SZLHOLDINGS/szl-governed-norm](https://huggingface.co/SZLHOLDINGS/szl-governed-norm)**. The Hugging Face repo is the canonical `get_kernel` source; this repository mirrors the same source of truth.

**The first *governed* kernel on the Hugging Face Kernel Hub.** Correctness-verified RMSNorm & LayerNorm with optional governance receipts that make every call auditable at the kernel layer.

> Most Kernel Hub kernels compete on raw speed. `szl-governed-norm` opens a different axis: **verifiable provenance**. Same clean `get_kernel` one-liner, plus a SHA3-256 hash-chained audit trail no other kernel ships.

A universal (pure-PyTorch) normalization kernel from [SZL Holdings](https://huggingface.co/SZLHOLDINGS). It gives you a trustworthy reference implementation of RMSNorm and LayerNorm that runs on CPU and CUDA and plays nicely with `torch.compile` — plus an opt-in *governed* mode that emits content-addressed, SHA3-256 hash-chained receipts of each normalization call.

---

## What it is

`szl-governed-norm` is a [Kernel Hub](https://huggingface.co/docs/kernels) kernel built for two things people actually need from a normalization layer:

1. **A correctness reference you can trust.** RMSNorm and LayerNorm are implemented in pure PyTorch, computed in float32 for numerical stability and cast back to the input dtype (the standard Llama-style convention). They are verified against PyTorch's own references in the test suite.
2. **Provenance you can verify.** Run any call with `governed=True` and the kernel records a small, deterministic receipt — input shape/dtype, `eps`, and a SHA3-256 digest of the (rounded) output — hash-chained to the previous receipt. The result is an independently re-walkable audit trail for a sequence of kernel calls.

This is a **universal kernel**: it ships no hand-tuned CUDA/Triton binary. Its differentiator is verifiable governance, not raw FLOPs. See [Correctness & honesty](#correctness--honesty) below — we are deliberately precise about what this does and does not claim.

---

## Quickstart

Load the kernel straight from the Hub with [`kernels`](https://huggingface.co/docs/kernels):

```python
import torch
from kernels import get_kernel

gn = get_kernel("SZLHOLDINGS/szl-governed-norm")

x = torch.randn(4, 1024, dtype=torch.float16, device="cuda")
w = torch.ones(1024, dtype=torch.float16, device="cuda")

# Plain path — drop-in normalization.
y = gn.rms_norm(x, weight=w, eps=1e-6)
z = gn.layer_norm(x, weight=w, eps=1e-5)
```

### Governed mode + receipts

```python
# Same math, plus an audit receipt appended to the in-process chain.
y = gn.rms_norm(x, weight=w, eps=1e-6, governed=True)
z = gn.layer_norm(x, weight=w, eps=1e-5, governed=True)

# Inspect the hash-chain.
print(gn.receipt_head())     # SHA3-256 head over all governed calls
print(gn.receipt_count())    # number of governed calls recorded
print(gn.receipt_tail(2))    # last N receipts (dicts)

# Re-walk and verify the chain end to end.
print(gn.receipt_verify())
# -> {'ok': True, 'depth': 2, 'first_break_seq': -1, 'head': '...'}
```

Governance is strictly opt-in: with `governed=False` (the default) nothing is recorded, and the kernel never writes to disk or the network.

### Per-call chain (no global-state contention)

For multi-threaded or multi-request callers, pass your own `ReceiptChain` so each caller records into its own isolated chain instead of the shared process-wide default:

```python
chain = gn.ReceiptChain()
y = gn.rms_norm(x, weight=w, eps=1e-6, chain=chain)   # records into YOUR chain only
print(chain.verify())   # (ok, depth, first_break_seq) — the default chain is untouched
```

Passing `chain=` implies governance even if `governed=` is left at its default `False`. The chain is thread-safe: concurrent calls into one chain stay consistent and still verify.

### One-shot self-check

```python
print(gn.__version__)    # e.g. "0.2.0"
print(gn.selfcheck())
# -> {'ok': True, 'version': '0.2.0',
#     'checks': {'rms_norm': True, 'layer_norm': True,
#                'fused_add_rms_norm': True, 'governance': True},
#     'receipt_ok': True, 'receipt_head': '...', 'error': None}
```

`selfcheck()` runs a tiny CPU-only smoke test against PyTorch references plus a governance round-trip (on a private throwaway chain, so the default chain is never touched) and never raises — safe to call from a health probe. Downstream code (and SZL's own a11oy / hatun-mcp) can call `get_kernel(...).selfcheck()` to confirm the loaded kernel is correct *and* its receipts verify, in one call.

---

## API reference

### Functional API

| Function | Signature | Notes |
|---|---|---|
| `rms_norm` | `rms_norm(x, weight=None, eps=1e-6, governed=False, chain=None)` | RMSNorm over the last dim. `weight` optional. Emits a receipt when `governed=True` or a `chain` is given. |
| `layer_norm` | `layer_norm(x, weight=None, bias=None, eps=1e-5, governed=False, chain=None)` | LayerNorm over the last dim. `weight`/`bias` optional. Emits a receipt when `governed=True` or a `chain` is given. |
| `fused_add_rms_norm` | `fused_add_rms_norm(x, residual, weight=None, eps=1e-6, governed=False, chain=None)` | Residual-add + RMSNorm (pre-norm transformer block pattern). Returns `(y, new_residual)`. Emits a receipt when `governed=True` or a `chain` is given. |
| `selfcheck` | `selfcheck()` | One-shot correctness + governance smoke test. Returns a JSON-able dict `{ok, version, checks, receipt_ok, receipt_head, error}`; never raises. |

Both compute in float32 and cast back to the input dtype. `rms_norm` matches a Llama-style RMSNorm reference; `layer_norm` matches `torch.nn.functional.layer_norm` for the last-dim case (both verified in `tests/`).

### Governance receipt API

These operate on the default in-process receipt chain.

| Function | Returns | Description |
|---|---|---|
| `receipt_head()` | `str` | SHA3-256 head of the receipt chain (`"0"*64` if empty). |
| `receipt_count()` | `int` | Number of governed calls recorded. |
| `receipt_tail(n=10)` | `list[dict]` | The last `n` receipts. |
| `receipt_verify()` | `dict` | Re-walks the chain; returns `{ok, depth, first_break_seq, head}`. |

You can also construct your own isolated chain with the `ReceiptChain` class (`emit`, `head`, `count`, `tail`, `verify`) and pass it per-call via `chain=` (see *Per-call chain* above) to avoid contention on the shared default chain. `__version__` exposes the kernel version string.

### `nn.Module` layers

For use with the `kernels` layer-mapping mechanism, the kernel exposes pure `torch.nn.Module` subclasses (only a `forward` method, no custom `__init__`, no class variables) so they can drop in over an existing module:

| Layer | Reads from host module | Description |
|---|---|---|
| `RMSNorm` | `self.weight` (optional), `self.variance_epsilon` or `self.eps` | Pure RMSNorm forward. |
| `LayerNorm` | `self.weight`/`self.bias` (optional), `self.eps` | Pure LayerNorm forward. |
| `FusedAddRMSNorm` | `self.weight` (optional), `self.variance_epsilon` or `self.eps` | Residual-add + RMSNorm forward; returns `(normalized, new_residual)`. |

---

## Governed mode — provenance at the kernel layer

Most normalization layers are a black box: you trust the output because you trust the code path. `szl-governed-norm` lets you go further and produce *evidence* about what ran.

When a call runs in governed mode, the kernel builds a receipt body:

```json
{
  "seq": 0,
  "op": "rms_norm",
  "in_shape": [4, 1024],
  "in_dtype": "float16",
  "eps": 1e-06,
  "out_digest": "<sha3-256 of the rounded output>",
  "prev": "<previous receipt digest or 64 zeros>"
}
```

It then takes a **SHA3-256 digest over the canonical JSON body** and links each receipt to the one before it via the `prev` field — a classic hash chain. `receipt_verify()` re-walks the chain and reports the first break (if any), so tampering with any receipt invalidates everything downstream.

The output digest is computed over the output tensor's contents rounded to a fixed decimal precision, which keeps it reproducible across devices and dtypes for the same logical values. This is the same **provenance doctrine** SZL Holdings applies across its [a11oy governed-AI platform](https://a-11-oy.com) — applied here at the lowest layer of the stack, the kernel itself. It sits alongside SZL Holdings' broader governance and observability work (governance MCP server, OTel governance exporters, security gates) published on the [SZL Holdings Hugging Face org](https://huggingface.co/SZLHOLDINGS).

---

## Correctness & honesty

We hold this kernel to a plain-spoken standard. Here is exactly what it is:

- **It is a universal, pure-Python kernel — a correctness reference.** RMSNorm and LayerNorm are implemented in pure PyTorch and verified against PyTorch's own references in the test suite.
- **It runs on CPU and CUDA**, and is `torch.compile`-friendly.
- **There are no fabricated benchmarks here.** Because this is a universal kernel and not a hand-tuned CUDA/Triton binary, we make **no speedup claims**. If you need a speed record, this is not it; if you need a trustworthy, auditable normalization, it is.
- **The receipt digest is an integrity fingerprint, NOT a cryptographic signature.** A SHA3-256 hash chain proves that a sequence of receipts is internally consistent and untampered. It does **not** prove authorship or identity — signing (e.g., DSSE) is a separate, out-of-band concern.
- **Governance is opt-in and side-effect-free by default.** Nothing is recorded unless you pass `governed=True`, and the kernel never touches disk or the network.

---

## Compatibility

| Requirement | Version |
|---|---|
| Python | 3.9+ |
| PyTorch | `torch>=2.5` |
| Dependencies | Python standard library + `torch` only |

The universal-kernel constraint (stdlib + torch only) is intentional: it keeps the kernel portable, easy to audit, and free of supply-chain surprises.

---

## About SZL Holdings

SZL Holdings, founded by **Stephen Lutar**, builds governed-AI infrastructure — provenance, observability, and security tooling for AI systems. Its work includes:

- The **[a11oy governed-AI platform](https://a-11-oy.com)** and **killinchu**.
- **45+ public repositories and datasets** on the [SZL Holdings Hugging Face org](https://huggingface.co/SZLHOLDINGS), spanning governance MCP servers, OTel governance exporters, security gates, and engineering-recipe tooling.
- Research published on **[Zenodo](https://zenodo.org/)**.

This kernel applies that same governance doctrine at the level of a single PyTorch operation.

---

## Citation

If you use this kernel, please cite it via the included [`CITATION.cff`](./CITATION.cff). Authored by Stephen Lutar (ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173)).

## License

Apache-2.0 — see [`LICENSE`](./LICENSE). Copyright 2026 SZL Holdings.

---

<sub>
<b>SZL Holdings</b> · governed normalization · provenance at the kernel layer ·
<a href="https://a-11-oy.com">a-11-oy.com</a> ·
<a href="https://github.com/szl-holdings/szl-governed-norm">github.com/szl-holdings/szl-governed-norm</a> ·
<a href="https://huggingface.co/SZLHOLDINGS/szl-governed-norm">huggingface.co/SZLHOLDINGS/szl-governed-norm</a>
</sub>

---

## Interactive showcase (roadmap)

A companion holographic Space (`governed-norm-holo`) and a live receipt-chain demo
Space (`receipt-chain-live`) are on the roadmap and **not yet deployed**. When
available, they will appear here. The kernel itself is fully functional today —
see [Quickstart](#quickstart) to run it locally.

The broader SZL governed-AI platform: [huggingface.co/SZLHOLDINGS](https://huggingface.co/SZLHOLDINGS).

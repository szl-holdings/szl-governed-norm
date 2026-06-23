# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Hardening / stress-test regression suite for szl_governed_norm (Dev1 upgrade).

Every test here corresponds to an edge case that was stress-tested during the
"harden norm kernel" upgrade, or to a bug that was found and fixed, or to a new
upgrade feature. Each fixed bug gets a named regression test so it cannot
silently come back.

Coverage:
  * empty tensors (zero-size normalized dim -> clear error; empty batch -> ok)
  * single-element last dim (no warning; matches F.layer_norm)
  * very large last dim
  * non-contiguous inputs and weights
  * NaN / Inf propagation (documented behavior, not silently sanitized)
  * requires_grad through governed mode
  * thread-safety of the receipt chain under concurrent calls
  * torch.compile(fullgraph=True) on every op, including governed mode
  * dtype matrix fp16 / bf16 / fp32 / fp64
  * UPGRADE 1: per-call (caller-owned) ReceiptChain
  * UPGRADE 2: __version__ exposure + selfcheck()

CPU-only so it runs anywhere.

Run:  python -m pytest tests/test_hardening.py -q
"""
import sys
import threading
import warnings
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build" / "torch-universal"))
import szl_governed_norm as gn  # noqa: E402
from szl_governed_norm import ReceiptChain  # noqa: E402

ALL_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


# --------------------------------------------------------------------------- #
# Empty tensors                                                               #
# --------------------------------------------------------------------------- #
def test_zero_size_last_dim_rejected_rms():
    """REGRESSION: a zero-size normalized last dim is undefined -> ValueError,
    not a silently-returned empty/NaN tensor."""
    with pytest.raises(ValueError, match="zero-size"):
        gn.rms_norm(torch.randn(4, 0))


def test_zero_size_last_dim_rejected_layer():
    with pytest.raises(ValueError, match="zero-size"):
        gn.layer_norm(torch.randn(4, 0))


def test_zero_size_last_dim_rejected_fused():
    with pytest.raises(ValueError, match="zero-size"):
        gn.fused_add_rms_norm(torch.randn(4, 0), torch.randn(4, 0))


def test_empty_batch_ok():
    """A zero-size *batch* (rows) with a valid last dim is fine: shape-preserving."""
    x = torch.randn(0, 16)
    assert gn.rms_norm(x).shape == (0, 16)
    assert gn.layer_norm(x).shape == (0, 16)
    y, r = gn.fused_add_rms_norm(x, torch.randn(0, 16))
    assert y.shape == (0, 16) and r.shape == (0, 16)


# --------------------------------------------------------------------------- #
# Single-element last dim                                                     #
# --------------------------------------------------------------------------- #
def test_single_element_layer_norm_no_warning_and_matches_torch():
    """REGRESSION: layer_norm over a 1-element last dim must not emit torch's
    'degrees of freedom <= 0' UserWarning, and must match F.layer_norm (==0)."""
    x = torch.randn(4, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        out = gn.layer_norm(x)
    ref = torch.nn.functional.layer_norm(x, (1,))
    torch.testing.assert_close(out, ref)


def test_single_element_rms_norm_is_sign_like():
    """RMSNorm over a 1-element last dim -> x/sqrt(x^2+eps) ~ sign(x)."""
    x = torch.tensor([[2.0], [-3.0], [0.5]])
    out = gn.rms_norm(x, eps=1e-12)
    torch.testing.assert_close(out, torch.sign(x), rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
# Very large last dim                                                         #
# --------------------------------------------------------------------------- #
def test_very_large_last_dim_finite_and_unit_rms():
    x = torch.randn(2, 131072)
    out = gn.rms_norm(x, eps=1e-8)
    assert torch.isfinite(out).all()
    rms = out.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(2), rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# Non-contiguous inputs / weights                                             #
# --------------------------------------------------------------------------- #
def test_non_contiguous_input_matches_contiguous():
    base = torch.randn(8, 16)
    nc = base.t()  # (16, 8), non-contiguous
    assert not nc.is_contiguous()
    torch.testing.assert_close(gn.rms_norm(nc), gn.rms_norm(nc.contiguous()))
    torch.testing.assert_close(gn.layer_norm(nc), gn.layer_norm(nc.contiguous()))


def test_non_contiguous_weight_matches_contiguous():
    x = torch.randn(4, 16)
    w_full = torch.randn(16, 2)
    w_nc = w_full[:, 0]  # non-contiguous 1-D view
    assert not w_nc.is_contiguous()
    torch.testing.assert_close(
        gn.rms_norm(x, weight=w_nc), gn.rms_norm(x, weight=w_nc.contiguous())
    )


# --------------------------------------------------------------------------- #
# NaN / Inf propagation (documented behavior)                                 #
# --------------------------------------------------------------------------- #
def test_nan_input_propagates_not_sanitized():
    """Documented: NaN inputs propagate (we do not silently hide them)."""
    x = torch.randn(4, 16)
    x[0, 0] = float("nan")
    out = gn.rms_norm(x)
    assert torch.isnan(out[0]).any()       # the NaN row is contaminated
    assert torch.isfinite(out[1:]).all()   # other rows unaffected


def test_inf_input_propagates_not_sanitized():
    x = torch.randn(4, 16)
    x[0, 0] = float("inf")
    out = gn.layer_norm(x)
    assert not torch.isfinite(out[0]).all()  # inf contaminates its row
    assert torch.isfinite(out[1:]).all()


# --------------------------------------------------------------------------- #
# requires_grad through governed mode                                         #
# --------------------------------------------------------------------------- #
def test_requires_grad_through_governed_default_chain():
    x = torch.randn(4, 16, requires_grad=True)
    w = torch.randn(16, requires_grad=True)
    out = gn.rms_norm(x, weight=w, eps=1e-6, governed=True)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_requires_grad_through_governed_per_call_chain():
    chain = ReceiptChain()
    x = torch.randn(4, 16, requires_grad=True)
    out = gn.layer_norm(x, eps=1e-5, chain=chain)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert chain.count() == 1
    # The receipt must not capture the graph (digest is over a detached tensor).
    assert chain.verify()[0] is True


# --------------------------------------------------------------------------- #
# Thread-safety of the receipt chain                                          #
# --------------------------------------------------------------------------- #
def test_receipt_chain_thread_safe_per_call():
    """Concurrent emits into one shared per-call chain stay consistent and the
    chain still verifies (no lost/torn records)."""
    chain = ReceiptChain()
    n_threads, per_thread = 8, 64
    errors = []

    def worker():
        try:
            for _ in range(per_thread):
                x = torch.randn(2, 8)
                gn.rms_norm(x, chain=chain)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    ok, depth, brk = chain.verify()
    assert depth == n_threads * per_thread
    assert ok is True
    assert brk == -1


def test_receipt_chain_thread_safe_default_chain():
    start = gn.receipt_count()
    n_threads, per_thread = 8, 32

    def worker():
        for _ in range(per_thread):
            gn.rms_norm(torch.randn(2, 8), governed=True)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert gn.receipt_count() == start + n_threads * per_thread
    assert gn.receipt_verify()["ok"] is True


# --------------------------------------------------------------------------- #
# torch.compile(fullgraph=True) on every op                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("governed", [False, True])
def test_rms_norm_fullgraph_compile(governed):
    x = torch.randn(4, 64)
    w = torch.randn(64)
    c = torch.compile(
        lambda a, ww: gn.rms_norm(a, weight=ww, eps=1e-6, governed=governed),
        fullgraph=True,
    )
    torch.testing.assert_close(c(x, w), gn.rms_norm(x, weight=w, eps=1e-6))


def test_layer_norm_fullgraph_compile():
    x = torch.randn(4, 64)
    w = torch.randn(64)
    b = torch.randn(64)
    c = torch.compile(
        lambda a, ww, bb: gn.layer_norm(a, weight=ww, bias=bb, eps=1e-5), fullgraph=True
    )
    torch.testing.assert_close(c(x, w, b), gn.layer_norm(x, weight=w, bias=b, eps=1e-5))


def test_fused_fullgraph_compile():
    x = torch.randn(4, 64)
    res = torch.randn(4, 64)
    w = torch.randn(64)
    c = torch.compile(
        lambda a, r, ww: gn.fused_add_rms_norm(a, r, weight=ww, eps=1e-6), fullgraph=True
    )
    yc, rc = c(x, res, w)
    y0, r0 = gn.fused_add_rms_norm(x, res, weight=w, eps=1e-6)
    torch.testing.assert_close(yc, y0)
    torch.testing.assert_close(rc, r0)


def test_governed_compile_does_not_record_but_numerics_match():
    """REGRESSION: governed=True used to break fullgraph compile (lock context
    manager). Now it compiles; receipts are simply not recorded while tracing,
    and the numerics are identical to the eager governed path."""
    x = torch.randn(4, 64)
    before = gn.receipt_count()
    c = torch.compile(lambda a: gn.rms_norm(a, governed=True), fullgraph=True)
    out = c(x)
    # No receipt recorded during compiled execution (documented behavior).
    assert gn.receipt_count() == before
    torch.testing.assert_close(out, gn.rms_norm(x))
    # And the eager governed call still does record.
    gn.rms_norm(x, governed=True)
    assert gn.receipt_count() == before + 1


# --------------------------------------------------------------------------- #
# dtype matrix fp16 / bf16 / fp32 / fp64                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", ALL_DTYPES,
                         ids=[str(d).replace("torch.", "") for d in ALL_DTYPES])
def test_dtype_matrix_rms_layer_fused(dtype):
    torch.manual_seed(0)
    x = torch.randn(4, 128).to(dtype)
    w = torch.randn(128).to(dtype)
    b = torch.randn(128).to(dtype)
    res = torch.randn(4, 128).to(dtype)

    yr = gn.rms_norm(x, weight=w, eps=1e-6)
    yl = gn.layer_norm(x, weight=w, bias=b, eps=1e-5)
    yf, nr = gn.fused_add_rms_norm(x, res, weight=w, eps=1e-6)

    for out in (yr, yl, yf, nr):
        assert out.dtype == dtype
        assert torch.isfinite(out).all()


def test_fp64_gradcheck_layer_norm():
    """fp64 is preserved (not downcast) so analytic == numerical Jacobian."""
    x = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    w = torch.randn(16, dtype=torch.float64, requires_grad=True)
    b = torch.randn(16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, ww, bb: gn.layer_norm(a, weight=ww, bias=bb, eps=1e-6),
        (x, w, b), eps=1e-6,
    )


# --------------------------------------------------------------------------- #
# UPGRADE 1: per-call (caller-owned) ReceiptChain                             #
# --------------------------------------------------------------------------- #
def test_per_call_chain_isolates_from_default():
    chain = ReceiptChain()
    before_default = gn.receipt_count()
    gn.rms_norm(torch.randn(2, 8), chain=chain)
    gn.layer_norm(torch.randn(2, 8), chain=chain)
    gn.fused_add_rms_norm(torch.randn(2, 8), torch.randn(2, 8), chain=chain)
    assert chain.count() == 3
    # The process-default chain must be completely untouched.
    assert gn.receipt_count() == before_default


def test_per_call_chain_implies_governance():
    """Passing a chain records even when governed is left at its default False."""
    chain = ReceiptChain()
    gn.rms_norm(torch.randn(2, 8), chain=chain)  # no governed=True
    assert chain.count() == 1
    assert chain.tail(1)[0]["op"] == "rms_norm"


def test_two_per_call_chains_are_independent():
    c1, c2 = ReceiptChain(), ReceiptChain()
    gn.rms_norm(torch.randn(2, 8), chain=c1)
    gn.rms_norm(torch.randn(2, 8), chain=c1)
    gn.rms_norm(torch.randn(2, 8), chain=c2)
    assert c1.count() == 2 and c2.count() == 1
    assert c1.verify()[0] and c2.verify()[0]


# --------------------------------------------------------------------------- #
# UPGRADE 2: __version__ + selfcheck()                                        #
# --------------------------------------------------------------------------- #
def test_version_exposed():
    assert isinstance(gn.__version__, str)
    # semantic-version-ish: at least major.minor
    parts = gn.__version__.split(".")
    assert len(parts) >= 2 and all(p.isdigit() for p in parts[:2])


def test_selfcheck_passes_and_is_jsonable():
    import json
    result = gn.selfcheck()
    # Round-trips through JSON (downstream a11oy/hatun-mcp consume it as JSON).
    json.dumps(result)
    assert result["ok"] is True
    assert result["error"] is None
    assert result["version"] == gn.__version__
    assert set(result["checks"]) >= {
        "rms_norm", "layer_norm", "fused_add_rms_norm", "governance"
    }
    assert all(result["checks"].values())
    assert result["receipt_ok"] is True
    assert len(result["receipt_head"]) == 64  # SHA3-256 hex


def test_selfcheck_does_not_touch_default_chain():
    before = gn.receipt_count()
    gn.selfcheck()
    assert gn.receipt_count() == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

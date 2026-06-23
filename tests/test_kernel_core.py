# SPDX-License-Identifier: Apache-2.0
"""Dev1 kernel-core tests: validation guards, fused op, autograd, dtypes, governance.

Run: python -m pytest tests/test_kernel_core.py -q  (or: python tests/test_kernel_core.py)
CPU-only so it runs anywhere.
"""
import sys
from pathlib import Path

import pytest
import torch

# Import the built universal kernel package directly (same as test_norm.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build" / "torch-universal"))
import szl_governed_norm as gn  # noqa: E402


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #
def test_rms_norm_rejects_non_tensor():
    with pytest.raises(TypeError):
        gn.rms_norm([1.0, 2.0, 3.0])


def test_rms_norm_rejects_integer_dtype():
    with pytest.raises(TypeError):
        gn.rms_norm(torch.ones(4, 8, dtype=torch.int64))


def test_rms_norm_rejects_zero_dim():
    with pytest.raises(ValueError):
        gn.rms_norm(torch.tensor(3.0))


def test_rms_norm_rejects_nonpositive_eps():
    with pytest.raises(ValueError):
        gn.rms_norm(torch.randn(2, 8), eps=0.0)
    with pytest.raises(ValueError):
        gn.rms_norm(torch.randn(2, 8), eps=-1e-6)


def test_rms_norm_rejects_mismatched_weight_shape():
    with pytest.raises(ValueError):
        gn.rms_norm(torch.randn(2, 8), weight=torch.randn(7))


def test_rms_norm_rejects_2d_weight():
    with pytest.raises(ValueError):
        gn.rms_norm(torch.randn(2, 8), weight=torch.randn(2, 8))


def test_layer_norm_rejects_mismatched_bias():
    with pytest.raises(ValueError):
        gn.layer_norm(torch.randn(2, 8), bias=torch.randn(7))


def test_fused_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        gn.fused_add_rms_norm(torch.randn(2, 8), torch.randn(2, 7))


def test_valid_calls_still_pass_after_guards():
    x = torch.randn(4, 16)
    w = torch.randn(16)
    gn.rms_norm(x, weight=w)
    gn.layer_norm(x, weight=w, bias=torch.randn(16))
    gn.fused_add_rms_norm(x, torch.randn(4, 16), weight=w)


# --------------------------------------------------------------------------- #
# fused_add_rms_norm correctness                                              #
# --------------------------------------------------------------------------- #
def test_fused_add_rms_norm_matches_unfused():
    torch.manual_seed(7)
    x = torch.randn(8, 512, dtype=torch.float32)
    residual = torch.randn(8, 512, dtype=torch.float32)
    w = torch.randn(512, dtype=torch.float32)
    eps = 1e-6

    y, new_res = gn.fused_add_rms_norm(x, residual, weight=w, eps=eps)

    # Reference: add then rms_norm (the two-step path), compute add in fp32.
    h = (x.to(torch.float32) + residual.to(torch.float32))
    ref = gn.rms_norm(h.to(x.dtype), weight=w, eps=eps)

    torch.testing.assert_close(new_res, h.to(x.dtype), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# dtype coverage: fp32 compute, cast back; fp16/bf16/fp32 all correct         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_dtype_correct(dtype):
    torch.manual_seed(3)
    x = torch.randn(4, 256)
    w = torch.randn(256)
    out = gn.rms_norm(x.to(dtype), weight=w.to(dtype), eps=1e-6)
    assert out.dtype == dtype
    # Compare against a clean fp32 reference, with dtype-appropriate tolerance.
    ref = gn.rms_norm(x, weight=w, eps=1e-6).to(dtype)
    tol = {torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 4e-2}[dtype]
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32),
                               rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_layer_norm_dtype_matches_torch(dtype):
    torch.manual_seed(4)
    x = torch.randn(4, 128).to(dtype)
    w = torch.randn(128).to(dtype)
    b = torch.randn(128).to(dtype)
    out = gn.layer_norm(x, weight=w, bias=b, eps=1e-5)
    ref = torch.nn.functional.layer_norm(x, (128,), weight=w, bias=b, eps=1e-5)
    assert out.dtype == dtype
    tol = {torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 4e-2}[dtype]
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32),
                               rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_dtype_correct(dtype):
    torch.manual_seed(5)
    x = torch.randn(4, 128).to(dtype)
    res = torch.randn(4, 128).to(dtype)
    w = torch.randn(128).to(dtype)
    y, new_res = gn.fused_add_rms_norm(x, res, weight=w, eps=1e-6)
    assert y.dtype == dtype and new_res.dtype == dtype
    assert torch.isfinite(y).all() and torch.isfinite(new_res).all()


# --------------------------------------------------------------------------- #
# Autograd: backward runs and grads are finite                                #
# --------------------------------------------------------------------------- #
def test_rms_norm_backward_finite():
    x = torch.randn(8, 64, dtype=torch.float32, requires_grad=True)
    w = torch.randn(64, dtype=torch.float32, requires_grad=True)
    out = gn.rms_norm(x, weight=w, eps=1e-6)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_layer_norm_backward_finite():
    x = torch.randn(8, 64, dtype=torch.float32, requires_grad=True)
    w = torch.randn(64, dtype=torch.float32, requires_grad=True)
    b = torch.randn(64, dtype=torch.float32, requires_grad=True)
    out = gn.layer_norm(x, weight=w, bias=b, eps=1e-5)
    out.sum().backward()
    for g in (x.grad, w.grad, b.grad):
        assert g is not None and torch.isfinite(g).all()


def test_fused_add_rms_norm_backward_finite():
    x = torch.randn(8, 64, dtype=torch.float32, requires_grad=True)
    res = torch.randn(8, 64, dtype=torch.float32, requires_grad=True)
    w = torch.randn(64, dtype=torch.float32, requires_grad=True)
    y, new_res = gn.fused_add_rms_norm(x, res, weight=w, eps=1e-6)
    # Use both outputs so gradients flow through the residual path too.
    loss = y.pow(2).mean() + new_res.sum()
    loss.backward()
    for g in (x.grad, res.grad, w.grad):
        assert g is not None and torch.isfinite(g).all()


def test_rms_norm_gradcheck_double():
    # Strong correctness signal: analytic vs numerical Jacobian (float64).
    x = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    w = torch.randn(16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: gn.rms_norm(a, weight=b, eps=1e-6), (x, w), eps=1e-6
    )


# --------------------------------------------------------------------------- #
# torch.compile friendliness (guards must not break tracing)                  #
# --------------------------------------------------------------------------- #
def test_ops_torch_compile():
    x = torch.randn(4, 128)
    res = torch.randn(4, 128)
    w = torch.randn(128)
    rc = torch.compile(lambda a, b: gn.rms_norm(a, weight=b, eps=1e-6))
    lc = torch.compile(lambda a, b: gn.layer_norm(a, weight=b, eps=1e-5))
    fc = torch.compile(lambda a, r, b: gn.fused_add_rms_norm(a, r, weight=b, eps=1e-6))
    torch.testing.assert_close(rc(x, w), gn.rms_norm(x, weight=w, eps=1e-6))
    torch.testing.assert_close(lc(x, w), gn.layer_norm(x, weight=w, eps=1e-5))
    yc, rcc = fc(x, res, w)
    y0, r0 = gn.fused_add_rms_norm(x, res, weight=w, eps=1e-6)
    torch.testing.assert_close(yc, y0)
    torch.testing.assert_close(rcc, r0)


# --------------------------------------------------------------------------- #
# Governance hook for the new fused op                                        #
# --------------------------------------------------------------------------- #
def test_fused_governed_emits_receipt_and_verifies():
    x = torch.randn(4, 32)
    res = torch.randn(4, 32)
    start = gn.receipt_count()
    gn.fused_add_rms_norm(x, res, eps=1e-6, governed=True)
    assert gn.receipt_count() == start + 1
    tail = gn.receipt_tail(1)[0]
    assert tail["op"] == "fused_add_rms_norm"
    v = gn.receipt_verify()
    assert v["ok"] is True
    assert v["depth"] == gn.receipt_count()


def test_fused_governed_off_by_default():
    before = gn.receipt_count()
    gn.fused_add_rms_norm(torch.randn(2, 16), torch.randn(2, 16))
    assert gn.receipt_count() == before


# --------------------------------------------------------------------------- #
# Pure-layer purity + behavior                                                #
# --------------------------------------------------------------------------- #
def test_layers_are_pure_modules():
    from torch import nn
    for cls in (gn.layers.RMSNorm, gn.layers.LayerNorm, gn.layers.FusedAddRMSNorm):
        assert issubclass(cls, nn.Module)
        # No custom __init__ (inherits nn.Module's), only forward defined in body.
        assert "__init__" not in cls.__dict__, f"{cls.__name__} must not define __init__"
        own = {k for k in cls.__dict__ if not k.startswith("__")}
        own.discard("forward")
        # docstring lives in __doc__ (dunder); only 'forward' should remain.
        assert own == set(), f"{cls.__name__} has non-forward members: {own}"


def test_fused_layer_forward_matches_functional():
    layer = gn.layers.FusedAddRMSNorm()
    layer.weight = torch.randn(32)
    layer.eps = 1e-6
    x = torch.randn(2, 32)
    res = torch.randn(2, 32)
    y, new_res = layer(x, res)
    y0, r0 = gn.fused_add_rms_norm(x, res, weight=layer.weight, eps=1e-6)
    torch.testing.assert_close(y, y0)
    torch.testing.assert_close(new_res, r0)


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))

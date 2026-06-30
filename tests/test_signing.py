# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""ADDITIVE szl-receipt signing layer for szl_governed_norm.

Proves the doctrine contract on the governed-norm receipt path:
  * With a generated ECDSA-P256 key, the emitted receipt carries a DSSE
    signature envelope that verifies via ``szl_receipt.verify_receipt``.
  * Keyless => UNSIGNED-honest (signed=False, honest note). No fake pass.
  * The SHA3-256 hash chain stays intact and tamper-evident alongside it.

CPU-only.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build" / "torch-universal"))
import szl_governed_norm as gn  # noqa: E402

szl_receipt = pytest.importorskip("szl_receipt")
from szl_receipt import generate_keypair, verify_receipt  # noqa: E402


def test_signed_receipt_verifies():
    priv, pub = generate_keypair()
    chain = gn.ReceiptChain()
    x = torch.randn(2, 64, dtype=torch.float32)
    gn.rms_norm(x, eps=1e-6, chain=chain, sign_key=priv, organ="norm")
    rec = chain.tail(1)[0]
    env = rec["signature"]
    assert env["signed"] is True
    assert env["organ"] == "norm"
    ok, why = verify_receipt(env, pub)
    assert ok and why == "ok", (ok, why)

    # Wrong key must NOT verify.
    _, other_pub = generate_keypair()
    bad_ok, _ = verify_receipt(env, other_pub)
    assert bad_ok is False

    # Additive signature must not disturb the SHA3-256 chain.
    chain_ok, _, brk = chain.verify()
    assert chain_ok and brk == -1


def test_keyless_is_unsigned_honest():
    chain = gn.ReceiptChain()
    x = torch.randn(2, 32, dtype=torch.float32)
    gn.layer_norm(x, eps=1e-5, chain=chain)  # no sign_key
    rec = chain.tail(1)[0]
    env = rec["signature"]
    assert env["signed"] is False
    assert "UNSIGNED-honest" in env["note"]
    ok, why = verify_receipt(env)
    assert (ok, why) == (False, "unsigned-honest")

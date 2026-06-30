# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Content-addressed governance receipts for normalization calls.

SZL Holdings' provenance doctrine applied at the kernel layer: when a
normalization runs in *governed* mode, it emits a small, deterministic
receipt describing the call — input shape/dtype, eps, and a SHA3-256 digest
of the (quantized) output tensor — and hash-chains it to the previous
receipt. This makes a sequence of kernel calls independently auditable
without trusting the caller.

HONESTY:
- The digest is a real SHA3-256 over the output bytes (rounded to a fixed
  decimal precision so it is reproducible across runs/devices). It is an
  integrity fingerprint, NOT a cryptographic signature — we never claim
  it proves authorship. DSSE signing is a separate, out-of-band concern.
- Receipts are kept in an in-process, append-only chain. Nothing is written
  to disk or the network from inside the kernel.
- Stdlib + torch only (Kernel Hub universal-kernel requirement).
"""
import hashlib
import json
import threading
import time
from typing import Any, Dict, List, Optional, Union

import torch

_GENESIS = "0" * 64

# Logical signing-authority label stamped onto signature envelopes.
_ORGAN = "szl-governed-norm"


def _maybe_sign(
    body: Dict[str, Any],
    sign_key: Optional[Union[str, bytes]],
    organ: str,
) -> Optional[Dict[str, Any]]:
    """ADDITIVE szl-receipt signature layer over the receipt *body*.

    Returns a DSSE envelope (from ``szl_receipt.sign_receipt``) covering the
    exact canonical body, or ``None`` when szl-receipt is not installed (the
    kernel then behaves exactly as before). Doctrine: with no *sign_key* the
    envelope is UNSIGNED-honest (``signed=False``); a signature is NEVER
    fabricated. This is distinct from and additive to the SHA3-256 chain
    integrity hash (``digest``) — szl-receipt's envelope carries its own
    SHA-256 ``digest``/``algo`` so the two integrity hashes are explicit.
    """
    try:
        from szl_receipt import Receipt, sign_receipt
    except Exception:  # noqa: BLE001 - signing is optional; absence is honest
        return None
    env = sign_receipt(Receipt(kind="governed-norm", body=body),
                       sign_key, organ=organ)
    return env


def _tensor_digest(t: torch.Tensor, decimals: int = 6) -> str:
    """Deterministic SHA3-256 over a tensor's rounded float32 contents.

    Rounding to a fixed number of decimals makes the digest stable across
    devices/dtypes for the same logical values (tiny FP noise won't change
    it). This is an integrity fingerprint, not a signature.
    """
    flat = t.detach().to(torch.float32).reshape(-1)
    # Round to `decimals` places, integerize, hash the raw bytes. CPU move is
    # required to read bytes; kept O(n) and allocation-light.
    scaled = torch.round(flat * (10 ** decimals)).to(torch.int64).cpu().numpy().tobytes()
    h = hashlib.sha3_256()
    h.update(scaled)
    return h.hexdigest()


class ReceiptChain:
    """Append-only, SHA3-256 hash-chained log of normalization receipts.

    Each receipt: {seq, op, in_shape, in_dtype, eps, out_digest, prev, digest, ts}
    digest = SHA3-256 over the canonical JSON body (excluding digest/ts).
    verify() re-walks the chain and returns (ok, depth, first_break_seq).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: List[Dict[str, Any]] = []

    @staticmethod
    def _digest_body(body: Dict[str, Any]) -> str:
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha3_256(raw).hexdigest()

    def emit(
        self,
        op: str,
        x: torch.Tensor,
        out: torch.Tensor,
        eps: float,
        sign_key: Optional[Union[str, bytes]] = None,
        organ: str = _ORGAN,
    ) -> Dict[str, Any]:
        with self._lock:
            prev = self._records[-1]["digest"] if self._records else _GENESIS
            seq = len(self._records)
            body = {
                "seq": seq,
                "op": op,
                "in_shape": list(x.shape),
                "in_dtype": str(x.dtype).replace("torch.", ""),
                "eps": float(eps),
                "out_digest": _tensor_digest(out),
                "prev": prev,
            }
            digest = self._digest_body(body)
            rec = dict(body, digest=digest, ts=time.time())
            sig = _maybe_sign(body, sign_key, organ)
            if sig is not None:
                rec["signature"] = sig
            self._records.append(rec)
            return rec

    def head(self) -> str:
        with self._lock:
            return self._records[-1]["digest"] if self._records else _GENESIS

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def tail(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._records[-n:])

    def verify(self):
        """Re-walk the chain. Returns (ok: bool, depth: int, first_break: int)."""
        with self._lock:
            prev = _GENESIS
            for i, rec in enumerate(self._records):
                body = {k: rec[k] for k in
                        ("seq", "op", "in_shape", "in_dtype", "eps", "out_digest", "prev")}
                if rec["prev"] != prev or rec["digest"] != self._digest_body(body):
                    return (False, len(self._records), i)
                prev = rec["digest"]
            return (True, len(self._records), -1)


# Module-level default chain (opt-in: only written when governed=True is used).
_DEFAULT_CHAIN: Optional[ReceiptChain] = None
_chain_lock = threading.Lock()


def default_chain() -> ReceiptChain:
    global _DEFAULT_CHAIN
    with _chain_lock:
        if _DEFAULT_CHAIN is None:
            _DEFAULT_CHAIN = ReceiptChain()
        return _DEFAULT_CHAIN

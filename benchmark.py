# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings
"""HONEST micro-benchmark for szl_governed_norm.

============================================================================
 DISCLAIMER — READ FIRST
 ----------------------------------------------------------------------------
 szl_governed_norm is a UNIVERSAL (pure-Python / pure-PyTorch) kernel. This
 script measures wall-clock latency of its rms_norm / layer_norm on the
 CURRENT machine. These numbers are NOT a tuned-CUDA / Triton speed claim and
 should NOT be read as a performance record or compared against hand-written
 GPU kernels. They exist for REPRODUCIBILITY and CORRECTNESS regression
 tracking only.

 No number printed here is hardcoded or fabricated — every value is produced
 by actually running the kernel in this process. Re-run it yourself; results
 will vary with hardware, load, and torch build. The kernel's differentiator
 is verifiable governance (hash-chained receipts), not raw FLOPs.
============================================================================

Run:  python benchmark.py
      python benchmark.py --iters 200 --warmup 20
"""
import argparse
import platform
import sys
import time
from pathlib import Path

import torch

# Import the built universal kernel package directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "build" / "torch-universal"))
import szl_governed_norm as gn  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_op(fn, x, *, warmup, iters, device, **kwargs):
    """Return median ms/iter over `iters` timed calls after `warmup` calls."""
    for _ in range(warmup):
        fn(x, **kwargs)
    _sync(device)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x, **kwargs)
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    samples.sort()
    return samples[len(samples) // 2]  # median


def _dtypes():
    ds = [("fp32", torch.float32)]
    if hasattr(torch, "float16"):
        ds.append(("fp16", torch.float16))
    if hasattr(torch, "bfloat16"):
        ds.append(("bf16", torch.bfloat16))
    return ds


def _devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


# Representative transformer-ish shapes (batch*seq, hidden).
_SHAPES = [
    (32, 768),
    (128, 1024),
    (512, 2048),
    (1024, 4096),
]


def main():
    ap = argparse.ArgumentParser(description="Honest micro-benchmark for szl_governed_norm")
    ap.add_argument("--iters", type=int, default=100, help="timed iterations per cell")
    ap.add_argument("--warmup", type=int, default=10, help="warmup iterations per cell")
    args = ap.parse_args()

    print(__doc__.split("Run:")[0].rstrip())  # reprint the disclaimer block
    print(f"szl_governed_norm v{getattr(gn, '__version__', '?')}")
    print(f"torch {torch.__version__} | python {platform.python_version()} | "
          f"{platform.system()} {platform.machine()}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"timing: median of {args.iters} iters, {args.warmup} warmup, per cell\n")

    ops = [("rms_norm", gn.rms_norm, {"eps": 1e-6})]
    if hasattr(gn, "layer_norm"):
        ops.append(("layer_norm", gn.layer_norm, {"eps": 1e-5}))

    header = f"{'op':<11}{'device':<7}{'shape':<14}{'dtype':<7}{'ms/iter':>10}{'iters':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for device in _devices():
        for op_name, op_fn, op_kwargs in ops:
            for shape in _SHAPES:
                for dlabel, dtype in _dtypes():
                    try:
                        x = torch.randn(*shape, dtype=dtype, device=device)
                    except Exception as e:  # dtype unsupported on this device
                        print(f"{op_name:<11}{device.type:<7}{str(shape):<14}"
                              f"{dlabel:<7}{'skip: ' + type(e).__name__:>18}")
                        continue
                    ms = _time_op(op_fn, x, warmup=args.warmup, iters=args.iters,
                                  device=device, **op_kwargs)
                    print(f"{op_name:<11}{device.type:<7}{str(shape):<14}"
                          f"{dlabel:<7}{ms:>10.4f}{args.iters:>8}")
                    rows.append((op_name, device.type, str(shape), dlabel, ms, args.iters))

    # Quick governed-overhead sample (honest: shows receipt cost is real).
    print("\ngoverned-mode overhead sample (rms_norm, cpu, (512, 2048), fp32):")
    x = torch.randn(512, 2048, dtype=torch.float32)
    cpu = torch.device("cpu")
    plain = _time_op(gn.rms_norm, x, warmup=args.warmup, iters=args.iters,
                     device=cpu, eps=1e-6, governed=False)
    gov = _time_op(gn.rms_norm, x, warmup=args.warmup, iters=args.iters,
                   device=cpu, eps=1e-6, governed=True)
    print(f"  plain        : {plain:.4f} ms/iter")
    print(f"  governed=True: {gov:.4f} ms/iter  (receipts add the difference)")

    print("\nReminder: universal pure-Python kernel on this host — "
          "reproducibility/correctness numbers, NOT a CUDA speed claim.")
    return rows


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# Auto-style ops namespace shim for the universal kernel. Unique suffix lets
# multiple versions load in the same process (Kernel Hub requirement).
import torch

ops = torch.ops._szl_governed_norm_20260623075422


def add_op_namespace_prefix(op_name: str) -> str:
    return f"_szl_governed_norm_20260623075422::{op_name}"

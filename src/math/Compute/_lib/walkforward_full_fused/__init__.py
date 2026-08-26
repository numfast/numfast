"""WalkForward Full Fused kernel package (Stage1 v3 MEGA-FUSION, checkpointing)."""
from Compute._lib.walkforward_full_fused.wgsl import WGSL, SEED_WGSL, build_tuples
from Compute._lib.walkforward_full_fused.descriptor import describe, describe_seed, register
from Compute._lib.walkforward_full_fused.cpu import evaluate, mean_ret, cpu

__all__ = ["WGSL", "SEED_WGSL", "build_tuples", "describe", "describe_seed",
           "register", "evaluate", "mean_ret", "cpu"]

"""WalkForward Full Fused descriptor -- validation before compile.

Two kernels (state checkpointing, opt="checkpointing-B"):
- WalkForwardFullFusedSeed: grid (1, folds), out f32[folds*dx*10] snapshots.
- WalkForwardFullFused:     grid (dx, folds), inputs [close, snaps], out u32[2*20*folds].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Runtime._lib.mod_iface import InputSlot, OutputSlot, ExecutionPlan

MAX_N = 4_194_240  # WebGPU dispatch limit (65535 x 64)
SEED_OUT = "wf_seed_snap"  # snapshot buffer name (job chain seed -> main)


def describe(params: dict) -> ExecutionPlan:
    n = int(params["n"])
    fold_sz = int(params["fold_sz"])
    folds = int(params.get("folds", 4))
    chunk = int(params.get("chunk", 4096))
    if n < 1:
        raise ValueError(f"WalkForwardFullFused n must be >= 1; got {n} (0-byte buffer impossible on frozen driver)")
    if fold_sz < 1:
        raise ValueError(f"WalkForwardFullFused fold_sz must be >= 1; got {fold_sz}")
    if folds < 1 or folds > 65535:
        raise ValueError(f"WalkForwardFullFused folds must be in [1,65535]; got {folds}")
    if chunk < 1:
        raise ValueError(f"WalkForwardFullFused chunk must be >= 1; got {chunk}")
    dx = (fold_sz + chunk - 1) // chunk
    return ExecutionPlan(
        inputs=[InputSlot(name="close", dtype="float")],
        outputs=[OutputSlot(dtype="uint", template=f"wf_full_fused_{n}")],
        workspace=[],
        uniforms={"n": n, "fold_sz": fold_sz, "folds": folds, "chunk": chunk},
        dispatch=(dx, folds, 1),
    )


def describe_seed(params: dict) -> ExecutionPlan:
    """Seed pass plan: 1 WG per fold writes f32 state snapshots.

    Output: folds*dx*10 f32 (dx = ceil(fold_sz/chunk) boundaries per fold,
    10 serial-state vars each). Fixed-size -> output_size_fn mandatory.
    """
    n = int(params["n"])
    fold_sz = int(params["fold_sz"])
    folds = int(params.get("folds", 4))
    chunk = int(params.get("chunk", 4096))
    if n < 1:
        raise ValueError(f"WalkForwardFullFusedSeed n must be >= 1; got {n}")
    if fold_sz < 1:
        raise ValueError(f"WalkForwardFullFusedSeed fold_sz must be >= 1; got {fold_sz}")
    if folds < 1 or folds > 65535:
        raise ValueError(f"WalkForwardFullFusedSeed folds must be in [1,65535]; got {folds}")
    if chunk < 1:
        raise ValueError(f"WalkForwardFullFusedSeed chunk must be >= 1; got {chunk}")
    dx = (fold_sz + chunk - 1) // chunk
    from Compute._lib.walkforward_full_fused.wgsl import NSNAP, SEED_REPLICAS
    plan = ExecutionPlan(
        inputs=[InputSlot(name="close", dtype="float")],
        outputs=[OutputSlot(dtype="float", template=f"wf_seed_snap_{n}")],
        workspace=[],
        uniforms={"n": n, "fold_sz": fold_sz, "folds": folds, "chunk": chunk},
        dispatch=(SEED_REPLICAS, folds, 1),
    )
    plan.output_size_fn = lambda input_sizes: [folds * dx * NSNAP]
    return plan


def _cpu_seed(ctx):
    """CPU adapter for the seed pass: snapshot buffer is a GPU-resident
    intermediate; the main kernel's CPU oracle ignores it."""


def register(runtime):
    """Register WalkForwardFullFused + Seed into Runtime kernel_table."""
    from Compute._lib.walkforward_full_fused.wgsl import WGSL, SEED_WGSL
    from Compute._lib.walkforward_full_fused.cpu import cpu
    runtime.register_kernel(
        "WalkForwardFullFused",
        describe=describe,
        cpu=cpu,
        wgsl=WGSL,
        abi_version=1,
        capabilities={
            "input_dtype": "float",
            "output_dtype": "uint",
            "zero_init": True,
            "multi_input": True,
            "multi_output": False,
            "chunkable": False,
            "streaming": False,
            "workspace": False,
        },
    )
    runtime.register_kernel(
        "WalkForwardFullFusedSeed",
        describe=describe_seed,
        cpu=_cpu_seed,
        wgsl=SEED_WGSL,
        abi_version=1,
        capabilities={
            "input_dtype": "float",
            "output_dtype": "float",
            "zero_init": False,
            "multi_input": False,
            "multi_output": False,
            "chunkable": False,
            "streaming": False,
            "workspace": False,
        },
    )


__all__ = ["describe", "register"]

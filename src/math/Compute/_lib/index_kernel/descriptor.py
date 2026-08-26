"""IndexKernel descriptor — array creation; pattern (mode 3/4) = обычный вход.

Modes: 0=const(p0), 1=arange(start=p0, step=p1), 2=linspace(a=p0, b=p1, denom=p2),
3=tile(pat, k), 4=repeat(pat, rep).

dtype-семантика (решение владельца, int32 index):
params.dtype = "float32" (default) | "int32". int32 -> OutputSlot dtype="int32",
выходной буфер array<i32>, значение mode=1 arange = p0 + p1*i ТОЧНО до 2^31
(лимит f32 2^24 не применяется). Uniform-флаг int_mode (0|1) выбирает вариант
WGSL (wgsl(uniforms)); строковый ключ "dtype" удаляется из params — uniforms
обязаны быть numeric (pack_uniforms). Descriptor idempotent: повторный вызов
(planner/builder пере-describe) читает уже проставленный int_mode.

Публичный контракт tile/repeat: pattern — GPU-resident ВХОДНОЙ буфер
(InputSlot "pat"), ЛЮБОЙ k. Лимита 64 нет. В int32-режиме значения паттерна
транкуются к i32 (та же семантика, что прежний host-cast astype(int32)).

Uniform layout: скалярные ключи строго в порядке WGSL-структуры Params
(n, mode, p0, p1, p2, k, rep, int_mode). Массивов в uniform НЕТ (правила
выравнивания uniform address space: stride кратен 16); прежний fast-path
pat_0..63 удалён — modes 3/4 читают pattern из storage-binding 2.
"""

import math

from Runtime._lib.mod_iface import ExecutionPlan, InputSlot, OutputSlot

WORKGROUP = 64


def describe(params: dict) -> ExecutionPlan:
    """IndexKernel plan.

    params.dtype: "float32" | "int32" (default "float32"); преобразуется в
    numeric uniform-флаг int_mode и удаляется из params (idempotent).

    Modes 0/1/2: inputs=[], uniforms {n, mode, p0, p1, p2, k=1, rep=1,
    int_mode}.
    Modes 3/4:   inputs=[InputSlot("pat")], uniforms {n, mode, p0, p1, p2,
    k=len(pattern), rep, int_mode}; k >= 1, rep >= 1 (mode 4).

    int32 mode: p0/p1/p2 трактуются как целые (trunc), арифметика i32 WRAP.
    """
    n = int(params.get("n", 0))
    mode = int(params.get("mode", 0))

    if n < 1:
        raise ValueError(f"IndexKernel: n must be >= 1, got {n}")
    if mode not in (0, 1, 2, 3, 4):
        raise ValueError(f"IndexKernel: unknown mode {mode}, expected 0|1|2|3|4")

    # --- dtype -> numeric uniform flag (idempotent; describe вызывается
    # повторно в compile/planner/builder на уже мутациированных params) ---
    if "int_mode" not in params:
        dtype = params.pop("dtype", "float32")
        if dtype not in ("float32", "int32"):
            raise ValueError(
                f"IndexKernel: dtype {dtype!r} not supported "
                f"(allowed: float32|int32)")
        params["int_mode"] = 1 if dtype == "int32" else 0
    int_mode = int(params["int_mode"])

    if int_mode:
        p0 = int(params.get("p0", 0))
        p1 = int(params.get("p1", 1))
        p2 = int(params.get("p2", max(n - 1, 1)))
    else:
        p0 = float(params.get("p0", 0.0))
        p1 = float(params.get("p1", 1.0))
        p2 = float(params.get("p2", float(max(n - 1, 1))))
    params["p0"], params["p1"], params["p2"] = p0, p1, p2

    if mode == 2 and p2 == 0:
        raise ValueError("IndexKernel: linspace denominator p2 must be != 0")

    k = int(params.get("k", 1))
    rep = int(params.get("rep", 1))
    if mode in (3, 4):
        # Pattern — входной буфер "pat" (любой k), лимита 64 нет.
        if k < 1:
            raise ValueError("IndexKernel: pattern must contain at least 1 element")
    else:
        k = 1
    if mode == 4 and rep < 1:
        raise ValueError(f"IndexKernel: repeat requires rep >= 1, got {rep}")
    if mode not in (3, 4):
        rep = 1

    def _sizes(_input_sizes):
        return [n]

    uniforms = {"n": n, "mode": mode, "p0": p0, "p1": p1, "p2": p2,
                "k": k, "rep": rep, "int_mode": int_mode}

    return ExecutionPlan(
        inputs=[InputSlot(name="pat",
                          dtype="int" if int_mode else "float")]
        if mode in (3, 4) else [],
        outputs=[OutputSlot(dtype="int32" if int_mode else "float",
                            template=f"idx_{n}")],
        uniforms=uniforms,
        output_size_fn=_sizes,
        dispatch=(math.ceil(n / WORKGROUP), 1, 1),
    )

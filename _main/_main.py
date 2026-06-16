# Copyright (c) 2026 NumFast
# SPDX-License-Identifier: AGPL-3.0-only

from _main._lib.main_lib import (
    _get_version,
    _get_status,
    _check_cuda,
    _check_rocm,
    _check_webgpu,
    _check_cpu,
    _recommend_backend,
)


def info() -> str:
    version = _get_version()
    return f"NumFast v{version}"


def status() -> dict:
    ext_status = _get_status()
    return ext_status


def diagnostics() -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console = Console()

    console.print()
    console.print("[bold cyan]=== NumFast Diagnostics ===[/bold cyan]")
    console.print()

    cuda = _check_cuda()
    rocm = _check_rocm()
    webgpu = _check_webgpu()
    cpu = _check_cpu()

    # CUDA
    cuda_status = "[green]YES[/green]" if cuda["available"] else "[red]NO[/red]"
    console.print(f"[bold]CUDA:[/bold]  {cuda_status}")
    if cuda["available"]:
        device_parts = []
        if cuda["device"]:
            device_parts.append(cuda["device"])
        if cuda["memory"]:
            device_parts.append(cuda["memory"])
        console.print(f"  [dim]Device:[/dim] {', '.join(device_parts)}")
    console.print()

    # ROCm
    rocm_status = "[green]YES[/green]" if rocm["available"] else "[red]NO[/red]"
    console.print(f"[bold]ROCm:[/bold]  {rocm_status}")
    if rocm["available"] and rocm["device"]:
        console.print(f"  [dim]Device:[/dim] {rocm['device']}")
    if not rocm["available"] and rocm.get("reason"):
        console.print(f"  [dim]Reason:[/dim] {rocm['reason']}")
    console.print()

    # WebGPU
    wg_status = "[green]YES[/green]" if webgpu["available"] else "[red]NO[/red]"
    console.print(f"[bold]WebGPU:[/bold]  {wg_status}")
    if webgpu["available"]:
        if webgpu["device"]:
            console.print(f"  [dim]Adapter:[/dim] {webgpu['device']}")
        if webgpu["backend_type"]:
            console.print(f"  [dim]Type:[/dim] {webgpu['backend_type']}")
    console.print()

    # CPU
    console.print(f"[bold]CPU:[/bold]")
    console.print(f"  [dim]Cores:[/dim] {cpu['cores']}")
    console.print()

    # Recommendation
    rec = _recommend_backend(cuda, rocm, webgpu)
    console.print(f"[bold yellow]Recommended backend:[/bold yellow]  [green]{rec}[/green]")
    console.print()

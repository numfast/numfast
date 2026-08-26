"""StateKernel — CPU reference.

Mode 0: out[i] = a*in[i] + b*out[i-1]           (EMA single)
Mode 1: out0[i] = a*in0[i] + b*out0[i-1]         (Dual EMA for RSI)
         out1[i] = a*in1[i] + b*out1[i-1]
Mode 2: SuperTrend state machine with trailing stop

cpu_scan_local / cpu_scan_totals / cpu_scan_final — numpy-реализации той же
математики для параллельного 3-проходного scan (chunk-декомпозиция).
"""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    mode = int(ctx.uniforms.get("mode", 0))
    
    if mode == 0:
        src = ctx.inputs[0].view
        dst = ctx.outputs[0].view
        state = ctx.workspace[0].view
        a = ctx.uniforms.get("a", 1.0)
        b = ctx.uniforms.get("b", 0.0)
        n = src.length()
        if n == 0:
            return
        prev = src.read(0)
        dst.write(0, prev)
        state.write(0, prev)
        for i in range(1, n):
            val = a * src.read(i) + b * prev
            dst.write(i, val)
            state.write(i, val)
            prev = val
            
    elif mode == 1:
        in0 = ctx.inputs[0].view
        in1 = ctx.inputs[1].view
        out0 = ctx.outputs[0].view
        out1 = ctx.outputs[1].view
        st0 = ctx.workspace[0].view
        st1 = ctx.workspace[1].view
        a = ctx.uniforms.get("a", 1.0)
        b = ctx.uniforms.get("b", 0.0)
        n = in0.length()
        p0 = 0.0
        p1 = 0.0
        for i in range(n):
            v0 = a * in0.read(i) + b * p0
            v1 = a * in1.read(i) + b * p1
            out0.write(i, v0)
            out1.write(i, v1)
            st0.write(i, v0)
            st1.write(i, v1)
            p0 = v0
            p1 = v1
            
    elif mode == 2:
        close = ctx.inputs[0].view
        upper_band = ctx.inputs[1].view
        lower_band = ctx.inputs[2].view
        dst = ctx.outputs[0].view
        st_upper = ctx.workspace[0].view
        st_lower = ctx.workspace[1].view
        st_dir = ctx.workspace[2].view
        n = close.length()
        
        prev_upper = 0.0
        prev_lower = 0.0
        prev_dir = 1.0
        
        for i in range(n):
            if i == 0:
                direction = 1.0
                final_upper = upper_band.read(i)
                final_lower = lower_band.read(i)
            else:
                if prev_dir == 1.0:
                    if close.read(i) < prev_lower:
                        direction = -1.0
                    else:
                        direction = 1.0
                else:
                    if close.read(i) > prev_upper:
                        direction = 1.0
                    else:
                        direction = -1.0
                
                if direction == 1.0:
                    final_upper = max(upper_band.read(i), prev_upper)
                    final_lower = lower_band.read(i)
                else:
                    final_upper = upper_band.read(i)
                    final_lower = min(lower_band.read(i), prev_lower)
            
            dst.write(i, direction)
            st_upper.write(i, final_upper)
            st_lower.write(i, final_lower)
            st_dir.write(i, direction)
            prev_upper = final_upper
            prev_lower = final_lower
            prev_dir = direction

    else:
        # S87 defense-in-depth: mode not in {0,1,2} on direct cpu() call
        # was a silent no-op; raise the SAME message as descriptor.describe.
        raise ValueError(f"Unknown StateKernel mode: {mode}")


# ---------------------------------------------------------------------------
# Parallel 3-pass scan (mode 0 SINGLE / mode 1 DUAL)
# ---------------------------------------------------------------------------

def cpu_scan_local(ctx: ExecutionContext):
    mode = int(ctx.uniforms.get("mode", 0))
    a = ctx.uniforms.get("a", 1.0)
    b = ctx.uniforms.get("b", 0.0)
    chunk = int(ctx.uniforms.get("chunk", 256))
    if chunk < 1:
        chunk = 1

    if mode == 0:
        src = ctx.inputs[0].view
        dst = ctx.outputs[0].view
        last = ctx.outputs[1].view
        n = src.length()
        nblocks = (n + chunk - 1) // chunk
        for tid in range(nblocks):
            start = tid * chunk
            end = min(start + chunk, n)
            p = 0.0
            i0 = start
            if tid == 0:
                p = src.read(start)
                dst.write(start, p)
                i0 = start + 1
            for i in range(i0, end):
                val = a * src.read(i) + b * p
                dst.write(i, val)
                p = val
            last.write(tid, p)
    elif mode == 1:
        in0 = ctx.inputs[0].view
        in1 = ctx.inputs[1].view
        out0 = ctx.outputs[0].view
        out1 = ctx.outputs[1].view
        last0 = ctx.outputs[2].view
        last1 = ctx.outputs[3].view
        n = in0.length()
        nblocks = (n + chunk - 1) // chunk
        for tid in range(nblocks):
            start = tid * chunk
            end = min(start + chunk, n)
            p0 = 0.0
            p1 = 0.0
            for i in range(start, end):
                v0 = a * in0.read(i) + b * p0
                v1 = a * in1.read(i) + b * p1
                out0.write(i, v0)
                out1.write(i, v1)
                p0 = v0
                p1 = v1
            last0.write(tid, p0)
            last1.write(tid, p1)
    else:
        raise ValueError(f"Unknown StateKernelScanLocal mode: {mode}")


def cpu_scan_totals(ctx: ExecutionContext):
    mode = int(ctx.uniforms.get("mode", 0))
    b = ctx.uniforms.get("b", 0.0)
    chunk = int(ctx.uniforms.get("chunk", 256))
    if chunk < 1:
        chunk = 1

    if mode == 0:
        last = ctx.inputs[0].view
        sp = ctx.outputs[0].view
        nblocks = last.length()
        pw = 1.0
        for _ in range(chunk):
            pw = pw * b
        if nblocks > 0:
            sp.write(0, 0.0)
        for t in range(1, nblocks):
            sp.write(t, last.read(t - 1) + pw * sp.read(t - 1))
    elif mode == 1:
        last0 = ctx.inputs[0].view
        last1 = ctx.inputs[1].view
        sp0 = ctx.outputs[0].view
        sp1 = ctx.outputs[1].view
        nblocks = last0.length()
        pw = 1.0
        for _ in range(chunk):
            pw = pw * b
        if nblocks > 0:
            sp0.write(0, 0.0)
            sp1.write(0, 0.0)
        for t in range(1, nblocks):
            sp0.write(t, last0.read(t - 1) + pw * sp0.read(t - 1))
            sp1.write(t, last1.read(t - 1) + pw * sp1.read(t - 1))
    else:
        raise ValueError(f"Unknown StateKernelScanTotals mode: {mode}")


def cpu_scan_final(ctx: ExecutionContext):
    mode = int(ctx.uniforms.get("mode", 0))
    b = ctx.uniforms.get("b", 0.0)
    chunk = int(ctx.uniforms.get("chunk", 256))
    if chunk < 1:
        chunk = 1

    if mode == 0:
        tmp = ctx.inputs[0].view
        sp = ctx.inputs[1].view
        out = ctx.outputs[0].view
        n = tmp.length()
        nblocks = (n + chunk - 1) // chunk
        for tid in range(nblocks):
            start = tid * chunk
            end = min(start + chunk, n)
            c = 0.0 if tid == 0 else sp.read(tid)
            for i in range(start, end):
                c = c * b
                out.write(i, tmp.read(i) + c)
    elif mode == 1:
        tmp0 = ctx.inputs[0].view
        tmp1 = ctx.inputs[1].view
        sp0 = ctx.inputs[2].view
        sp1 = ctx.inputs[3].view
        out0 = ctx.outputs[0].view
        out1 = ctx.outputs[1].view
        n = tmp0.length()
        nblocks = (n + chunk - 1) // chunk
        for tid in range(nblocks):
            start = tid * chunk
            end = min(start + chunk, n)
            c0 = 0.0 if tid == 0 else sp0.read(tid)
            c1 = 0.0 if tid == 0 else sp1.read(tid)
            for i in range(start, end):
                c0 = c0 * b
                c1 = c1 * b
                out0.write(i, tmp0.read(i) + c0)
                out1.write(i, tmp1.read(i) + c1)
    else:
        raise ValueError(f"Unknown StateKernelScanFinal mode: {mode}")
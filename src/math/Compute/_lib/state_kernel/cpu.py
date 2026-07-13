"""StateKernel — CPU reference.

Mode 0: out[i] = a*in[i] + b*out[i-1]           (EMA single)
Mode 1: out0[i] = a*in0[i] + b*out0[i-1]         (Dual EMA for RSI)
         out1[i] = a*in1[i] + b*out1[i-1]
Mode 2: SuperTrend state machine with trailing stop
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
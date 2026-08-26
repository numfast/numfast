"""ArgMin/ArgMax — CPU reference: single pass, returns (value, index)."""

from Runtime._lib.mod_iface import ExecutionContext


def cpu(ctx: ExecutionContext):
    op = ctx.uniforms.get("op", "min")
    src = ctx.inputs[0].view
    val_out = ctx.outputs[0].view
    idx_out = ctx.outputs[1].view
    n = src.length()
    out_n = val_out.length()
    
    if n == 0 or out_n == 0:
        return
    
    # Simple chunked approach: divide work into out_n chunks
    chunk = max(1, n // out_n)
    for j in range(out_n):
        start = j * chunk
        end = min(start + chunk, n)
        if start >= n:
            val_out.write(j, 0.0)
            idx_out.write(j, 0.0)
            continue
        
        best_val = src.read(start)
        best_idx = float(start)
        
        for i in range(start + 1, end):
            v = src.read(i)
            if (op == "min" and v < best_val) or (op == "max" and v > best_val):
                best_val = v
                best_idx = float(i)
        
        val_out.write(j, best_val)
        idx_out.write(j, best_idx)

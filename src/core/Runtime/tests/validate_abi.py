"""ABI-hardening validation script."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from Runtime import Runtime
from Trading import register_all as register_trading_kernels
from Runtime.tests.register_test_kernels import register_test_kernels

runtime = Runtime()
register_trading_kernels(runtime)
register_test_kernels(runtime)

print(f"Kernels: {list(runtime.kernel_table.keys())}")

# 1. EMA
data = np.array([1.,2.,3.,4.,5.], dtype=np.float64)
tasks = runtime.compile([{"op":"EMA","inputs":["Close"],"params":{"period":3}}])
runtime.execute(tasks,{"Close":data})
ema = runtime.driver.memory.resolve_output("ema_3")
assert ema is not None
print(f"1. EMA(3)={ema[-1]:.4f}")

# 2. MACD
tasks2 = runtime.compile([{"op":"MACD","inputs":["Close"],"params":{"fast":12,"slow":26}}])
runtime.execute(tasks2,{"Close":data})
assert runtime.driver.memory.resolve_output("macd_12_26") is not None
print("2. MACD OK")

# 3. FFT
sig = np.sin(np.linspace(0, 2*np.pi, 16))
tasks3 = runtime.compile([{"op":"FFT","inputs":["signal"],"params":{"name":"test"}}])
runtime.execute(tasks3,{"signal":sig})
assert runtime.driver.memory.resolve_output("fft_test") is not None
print("3. FFT OK")

# 4. Blur
img = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.], dtype=np.float64)
tasks4 = runtime.compile([{"op":"BLUR","inputs":["image"],"params":{"size":3,"width":4}}])
runtime.execute(tasks4,{"image":img})
blur = runtime.driver.memory.resolve_output("blur_3")
assert blur is not None
blur.reshape(3,4)
print("4. Blur OK")

# 5. Stress
data5 = np.array([1.,2.,3.,4.,5.], dtype=np.float64)
tasks5 = runtime.compile([{
    "op":"STRESS","inputs":["a","b","c","d","e"],
    "params":{"name":"test","scale":2.0,"offset":1.0,"clamp":True}
}])
runtime.execute(tasks5, {
    "a": data5, "b": data5*2, "c": data5*3,
    "d": np.array([1,2,3,4,5], dtype=np.int64),
    "e": data5*0.5,
})
for name in ["out_test_1","out_test_2","out_test_7"]:
    val = runtime.driver.memory.resolve_output(name)
    assert val is not None, f"{name} missing"
    print(f"5. {name} last={val[-1]:.4f}")
print("5. Stress OK")

# 6. Uniforms validation
from Runtime._lib.mod_iface import validate_uniforms
try:
    validate_uniforms({"bad": [1,2,3]})
    assert False, "Should have raised TypeError"
except TypeError:
    print("6. Uniforms validation OK")

# 7. Chain
tasks7 = runtime.compile([
    {"op":"EMA","inputs":["Close"],"params":{"period":3}},
    {"op":"MACD","inputs":["ema_3"],"params":{"fast":12,"slow":26}},
])
runtime.execute(tasks7,{"Close":data})
assert runtime.driver.memory.resolve_output("macd_12_26") is not None
print("7. Chain OK")

print("\n=== ALL VALIDATION PASSED ===")

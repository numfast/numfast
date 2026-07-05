import sys, os
import numpy as np

from numfast import AgentSDK, Result
from numfast.agent import Session, Workspace

NUMPATH = r'C:\App\numfast\numfast'

print('=== 1. AgentSDK stateless ===')
sdk1 = AgentSDK()
sdk2 = AgentSDK()
assert type(sdk1) is type(sdk2)
print('OK: AgentSDK is stateless')

print()
print('=== 2. Result dataclass ===')
r = Result(status='ok', data=[1,2,3], metadata={'driver': 'cpu', 'version': '1.0'})
assert r.status == 'ok'
assert r.data == [1, 2, 3]
assert r.metadata['driver'] == 'cpu'
assert r.metadata['version'] == '1.0'
print(f'OK: Result(status={r.status}, metadata={r.metadata})')

r2 = Result(status='error', error='test error')
assert r2.status == 'error'
assert r2.error == 'test error'
print('OK: Result with error')

print()
print('=== 3. Session lifecycle ===')
with sdk1.session() as session:
    assert session.runtime is not None
    assert session.driver_name == 'cpu'
    print(f'Session open: runtime={session.runtime is not None}, driver={session.driver_name}')

assert session.runtime is None
print(f'Session closed: runtime={session.runtime}')

print()
print('=== 4. session.diff() ===')
with sdk1.session() as s:
    r = s.diff(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert r.status == 'ok'
    assert r.data['max_err'] == 0.0
    print(f'diff(equal): status={r.status}, max_err={r.data["max_err"]}')

    r = s.diff(np.array([1.0, 2.0]), np.array([1.0, 5.0]))
    assert r.status == 'error'
    print(f'diff(different): status={r.status}, max_err={r.data["max_err"]}')

print()
print('=== 5. session.compare() ===')
with sdk1.session() as s:
    cpu = np.array([1.0, 2.0, 3.0])
    gpu = np.array([1.0, 2.0, 3.0001])
    r = s.compare(cpu, gpu, tolerance=0.01)
    assert r.status == 'ok'
    print(f'compare(within tol): status={r.status}, max_diff={r.data["max_diff"]:.2e}')

    r = s.compare(cpu, gpu, tolerance=1e-6)
    assert r.status == 'error'
    print(f'compare(exceeds tol): status={r.status}, max_diff={r.data["max_diff"]:.2e}')

print()
print('=== 6. session.run() ===')
with sdk1.session() as s:
    code = 'import numpy as np; result = np.array([1.0, 2.0, 3.0]).sum()'
    r = s.run(code)
    assert r.status == 'ok'
    result_value = r.data.get('result')
    print(f'run(sum): status={r.status}, result={result_value}')

    r = s.run('raise ValueError("test error")')
    assert r.status == 'error'
    assert 'test error' in r.error
    print(f'run(error): status={r.status}, error={r.error[:50]}')

print()
print('=== 7. session.benchmark() ===')
with sdk1.session() as s:
    code = '[i**2 for i in range(1000)]'
    r = s.benchmark('list_comp', script=code, n_iter=3)
    assert r.status == 'ok'
    assert len(r.data['times']) == 3
    print(f'benchmark: status={r.status}, avg={r.data["avg"]:.4f}s, min={r.data["min"]:.4f}s')

print()
print('=== 8. Workspace ===')
ws = sdk1.workspace(NUMPATH)
assert ws.path == NUMPATH
labs = ws.labs()
print(f'workspace.labs: {len(labs)} found')
for lab in labs:
    print(f'  Lab {lab["number"]}: {lab["name"]}')

exs = ws.examples()
print(f'workspace.examples: {len(exs)} found')
for ex in exs:
    print(f'  {ex["name"]}')

print()
print('=== 9. Project ===')
proj = ws.open('examples')
assert proj.name == 'examples'
assert proj.exists
print(f'Project: {proj}')

proj2 = ws.open('nonexistent_project_xyz')
assert not proj2.exists
print(f'Non-existent project: {proj2}, exists={proj2.exists}')

print()
print('=== 10. Integration with operations ===')
from numfast import scan, matmul

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = scan(x)
expected = np.cumsum(x)
assert np.allclose(y, expected)
print(f'scan([1,2,3,4,5]) = {y} OK')

A = np.ones((2, 2), dtype=np.float64)
B = np.ones((2, 2), dtype=np.float64)
C = matmul(A.flatten(), B.flatten(), M=2, N=2, K=2)
assert np.allclose(C, np.array([[2, 2], [2, 2]]).flatten())
print(f'matmul 2x2 ones = {C} OK')

print()
print('=== 11. AgentSDK + operations in session ===')
with sdk1.session() as session:
    x = np.array([1.0, 2.0, 3.0])
    y = scan(x)
    expected = np.array([1.0, 3.0, 6.0])
    r = session.diff(y, expected)
    assert r.status == 'ok'
    print(f'session.diff(scan, expected): status={r.status}, max_err={r.data["max_err"]:.0e}')

print()
print('ALL INTEGRATION TESTS PASSED')

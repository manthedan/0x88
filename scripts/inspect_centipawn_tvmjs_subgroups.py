#!/usr/bin/env python3
"""Inspect shader strings actually embedded in staged wasm; no WebGPU execution."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path('public/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32')
STEM = 'bt4-soap-rem-c19000-final.batch16.f32.webgpu.tvmjs-wasm.probe'

def shaders(path):
    text = path.read_bytes().decode('latin1')
    out = {}
    for match in re.finditer(r'fn (\w+_kernel)\(', text):
        header = text.rfind('// Function:', 0, match.start())
        start = text.index('@group', header, match.start())
        enable = text.find('enable subgroups;', header, start)
        if enable >= 0:
            start = enable
        body = text.index('{', match.end())
        depth = 1
        end = body + 1
        while depth:
            depth += (text[end] == '{') - (text[end] == '}')
            end += 1
        out[match[1]] = text[start:end] + '\n'
    assert len(out) == 46, f'Expected 46 unique shaders, got {len(out)}'
    return out

def normalized(code):
    # This compiler emits unused CSE lets in nondeterministic map order. All
    # arithmetic uses expanded expressions. Assert no live reference is removed.
    code = code.replace('enable subgroups;\n\n', '')
    code = re.sub(r'^\s*let cse_v\d+ : [^\n]+\n', '', code, flags=re.M)
    assert not re.search(r'\bcse_v\d+\b', code), 'Live CSE reference: do not normalize'
    return code

sha = lambda b: hashlib.sha256(b).hexdigest()
a = shaders(ROOT/'v2-shape-k16'/f'{STEM}.tvmjs.wasm')
b = shaders(ROOT/'v3-subgroups'/f'{STEM}.tvmjs.wasm')
control = shaders(Path('artifacts/centipawn-subgroups/control')/f'{STEM}.tvmjs.wasm')
assert a.keys() == b.keys() == control.keys()
rows = []
for name, code in b.items():
    ops = re.findall(r'\b(subgroup\w+)\(', code)
    rows.append({
        'kernel': name,
        'workgroup': re.search(r'@workgroup_size\(([^)]+)\)', code)[1],
        'subgroupOps': {op: ops.count(op) for op in sorted(set(ops))},
        'v2Sha256': sha(a[name].encode()),
        'v3Sha256': sha(code.encode()),
        'controlMatchesV2Normalized': normalized(a[name]) == normalized(control[name]),
        'bodyMatchesV2Normalized': normalized(a[name]) == normalized(code),
    })
assert all(r['controlMatchesV2Normalized'] for r in rows), 'Control does not reproduce v2 shader arithmetic'
assert [r['kernel'] for r in rows if not r['bodyMatchesV2Normalized']] == [r['kernel'] for r in rows if r['subgroupOps']], 'Unexpected non-reduction change'
result = {'schema':'centipawn-subgroups-wgsl.v1', 'normalization':'Drop enable subgroups and unused CSE let declarations only; reject any live CSE reference.', 'kernels':rows}
out = Path('lab/centipawn-tvmjs-subgroups-results/wgsl-inspection.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2)+'\n')
changed_dir = out.parent/'wgsl'
changed_dir.mkdir(exist_ok=True)
for row in rows:
    if row['subgroupOps']:
        (changed_dir/f"{row['kernel']}.wgsl").write_text(b[row['kernel']])
print(json.dumps({'kernels':len(rows),'subgroupKernels':[r['kernel'] for r in rows if r['subgroupOps']], 'controlReproducesV2Arithmetic':True}))

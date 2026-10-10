#!/usr/bin/env python3
"""Read a -g2 SPIR-V disassembly back as source functions.

Slang/glslc builds with full debug info keep the original function names, source line
numbers and variable names even though OpName is stripped, so a shipped .spv can be read
back close to how it was written. Everything is inlined into one `main`, so the only
record of the original function boundaries is `OpExtInst ... DebugScope %callee %inlinedAt`
-- the *first* operand is the callee. (`DebugInlinedAt` alone only gives the call site.)

    spirv-dis world.rchit.spv > world.rchit.spvasm
    python spirv-funcs.py world.rchit.spvasm parallaxTrace snapUvToTexelCentre

Prints the function table first, then every instruction in encounter order, prefixed with
the owning function and the source line it came from. Omit the function names to dump all.

Two dead ends worth knowing, so they are not retried:
  * spirv-cross cannot read these modules -- the GLSL/HLSL backends reject the buffer
    device address OpPtrAccessChain, and the MSL backend rejects PrimitiveId in a ray
    tracing execution model.
  * spirv-cross's argument order is `[SPIR-V file] [options]`, and the version is set with
    `--version 460 --no-es` (there is no `--target-env`).
"""
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else 'up.spvasm'
WANT = set(sys.argv[2:])

SKIP = ('DebugScope', 'DebugNoLine', 'DebugValue', 'DebugDeclare', 'DebugFunction',
        'DebugLocalVariable', 'DebugType', 'OpString')

lines = open(PATH, encoding='utf-8', errors='replace').read().splitlines()

strings = {}
for line in lines:
    m = re.search(r'^\s*(\S+) = OpString "(.*)"$', line)
    if m:
        strings[m.group(1)] = m.group(2)

# DebugFunction: %id -> (name, file, line)
dfun = {}
for line in lines:
    m = re.search(r'^\s*(\S+) = OpExtInst \S+ \S+ DebugFunction (\S+) \S+ (\S+) %uint_(\d+)', line)
    if m:
        dfun[m.group(1)] = (strings.get(m.group(2), '?'),
                            strings.get(m.group(3), '?').split('/')[-1],
                            int(m.group(4)))

print('functions declared in debug info: %d   (file:line in the shader that was compiled)'
      % len(dfun))
for _, (name, _, ln) in sorted(dfun.items(), key=lambda kv: kv[1][2]):
    print('  %-30s :%d' % (name, ln))
print()

last = None
cur_fn = '<toplevel>'
cur_line = None
for line in lines:
    s = line.strip()
    if not s:
        continue
    m = re.search(r'OpExtInst \S+ \S+ DebugScope (\S+)(?: (\S+))?$', s)
    if m:
        cur_fn = dfun.get(m.group(1), (m.group(1), '?', 0))[0]
        continue
    m = re.search(r'OpExtInst \S+ \S+ DebugLine \S+ %uint_(\d+)', s)
    if m:
        cur_line = int(m.group(1))
        continue
    if any(k in s for k in SKIP) or s.startswith(';'):
        continue
    if WANT and cur_fn not in WANT:
        continue
    key = (cur_fn, cur_line)
    if key != last:
        print('-- %s :%s' % key)
        last = key
    print('     %s' % s)

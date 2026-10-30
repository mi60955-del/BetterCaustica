#!/usr/bin/env bash
# Measures the lens chain at its most expensive settings, to put a ceiling on what the pass can cost.
set -euo pipefail
MCP=/c/Users/mi609/Documents/JAVAMCMOD/mc-mcp
GAME=/e/Minecraft/26.2

cd "$MCP"
python tools/mcbox.py stop | tail -1 || true
sleep 4

python - "$GAME/config/bettercaustica.toml" <<'PY'
import io, sys
path = sys.argv[1]
text = io.open(path, encoding='utf-8').read()
text = text.replace('\n[lens]\n\tsamples = 32\n\thalo-taps = 48\n\tspikes = 12\n', '')
text += '\n[lens]\n\tsamples = 32\n\thalo-taps = 48\n\tspikes = 12\n'
io.open(path, 'w', encoding='utf-8', newline='\n').write(text)
PY

rm -f "$GAME/rt-frame-stats/gpu.csv"
python tools/mcbox.py start --wait --world "RT测试" --jvm=-Dcaustica.rt.gpuTiming=true --width 1920 --height 1080 | tail -2
python tools/mcbox.py focus | tail -1
MSYS_NO_PATHCONV=1 python tools/mccall.py call mc_command command="/time set 1000" | head -1
python tools/mccall.py call mc_teleport position=10,73,107 yaw=-90 pitch=10 | head -1
sleep 12

python - "$GAME/rt-frame-stats/gpu.csv" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tail = rows[-240:]
print(len(rows), 'rows')
for key in rows[0]:
    if key == 'frame':
        continue
    vals = sorted(float(r[key]) for r in tail if r[key])
    if vals:
        print(f"{key:16s} median {statistics.median(vals):7.3f} ms  p95 {vals[int(len(vals) * 0.95)]:7.3f}")
PY

# Leave the shipped config clean: the defaults are the tuned look.
python - "$GAME/config/bettercaustica.toml" <<'PY'
import io, sys
path = sys.argv[1]
text = io.open(path, encoding='utf-8').read()
text = text.replace('\n[lens]\n\tsamples = 32\n\thalo-taps = 48\n\tspikes = 12\n', '')
io.open(path, 'w', encoding='utf-8', newline='\n').write(text)
PY
echo done

# Stock-control upstream vs current benchmark

Date: 2026-06-29
GPU: NVIDIA GeForce RTX 3090
Workload: `./mfaktc --no-startup-selftest -tf 81144083 78 79`
Timed window: `timeout 240s`

Hardware controls:
- Power limit: stock/default 420 W
- Graphics clock lock: reset/none
- Memory clock lock: reset/none
- GPU target temperature: stock/default 83 C

Upstream baseline:
- Source: `origin/main` at `52c091f`
- Build: upstream default, `THREADS_PER_BLOCK=256`
- Runtime config: `GPUSieveSize=2047`, `GPUSieveProcessSize=32`, `Checkpoints=0`
- Self-test: passed
- Final timed endpoint: class 100, `6348.55 GHz-days/day`
- Telemetry sample: 415.20 W / 420.00 W, 1845 MHz graphics, 9501 MHz memory, 67 C, no hardware slowdown

Current optimized build:
- Source: `main` at `53d2548`
- Build: fixed Barrett87 GPU sieve path, `THREADS_PER_BLOCK=512`
- Runtime config: `GPUSieveSize=2046`, `GPUSieveProcessSize=48`, `Checkpoints=0`
- Fixed kernel resource check: `REG:40 STACK:0 SHARED:80 LOCAL:0`, 3520 SASS instructions
- Self-test: passed
- Final timed endpoint: class 112, `7199.42 GHz-days/day`
- Telemetry sample: 415.79 W / 420.00 W, 1815 MHz graphics, 9501 MHz memory, 70 C, no hardware slowdown

Result:
- Absolute gain: `+850.87 GHz-days/day`
- Relative gain: `+13.40%`

Artifacts:
- `upstream/run.txt`
- `upstream/telemetry-during.csv`
- `current/run.txt`
- `current/telemetry-during.csv`

---
name: motor-control
description: Control EPICS motors at APS beamlines. Use when the user asks to move a motor, read motor position, jog, tweak, home, stop, set velocity or limits, or mentions caget/caput/EPICS/IOC/PV names.
compatibility: Requires EPICS Channel Access tools (caget/caput) on PATH and a running IOC (e.g. 20idMotSim)
metadata:
  author: pawan-tripathi
  version: "1.0"
  epics-motor-record: "https://epics-modules.github.io/motor/motorRecord.html"
---

## EPICS Motor Control Workflow

All tools are on the `motor` MCP server (`epics_motor_server.py`).
They wrap `caget` / `caput` commands for standard EPICS motorRecord PVs.

### PV naming convention

```
{prefix}:{motor}         → VAL (setpoint / desired position)
{prefix}:{motor}.RBV     → readback (actual position)
{prefix}:{motor}.DMOV    → done-moving flag (1=done, 0=moving)
{prefix}:{motor}.STOP    → stop motor (set to 1)
{prefix}:{motor}.HLM     → user high soft limit
{prefix}:{motor}.LLM     → user low soft limit
{prefix}:{motor}.HLS     → high limit switch (1=tripped)
{prefix}:{motor}.LLS     → low limit switch (1=tripped)
{prefix}:{motor}.VELO    → velocity (EGU/s)
{prefix}:{motor}.EGU     → engineering units (mm, deg, etc.)
{prefix}:{motor}.DESC    → description string
```

### Test IOC

- Prefix: `20idMotSim`
- Start command (run manually in a separate terminal):
  ```
  /net/s20iddata/xorApps/epics/synApps_6_3/ioc/20idMotSim/iocBoot/ioc20idMotSim/softioc/20idMotSim.pl run
  ```
- Do NOT start the IOC via the AI agent.
- Motors: `m1` through `m8` (standard 8-motor crate)

### Available tools (12)

| Tool | Purpose | Key fields used |
|---|---|---|
| `get_motor_position` | Read current RBV, VAL, EGU | `.RBV`, `.VAL`, `.EGU` |
| `get_motor_status` | Full status (position, limits, velocity, motion state) | `.RBV`, `.DMOV`, `.HLS`, `.LLS`, `.HLM`, `.LLM`, `.VELO`, `.EGU`, `.DESC` |
| `move_motor_absolute` | Move to target position (with limit checks) | `.VAL`, `.RBV`, `.DMOV`, `.HLM`, `.LLM` |
| `move_motor_relative` | Move by delta from current RBV | `.RBV` + `.VAL` |
| `stop_motor` | Emergency stop | `.STOP = 1` |
| `set_motor_velocity` | Change speed | `.VELO` |
| `jog_motor` | Jog forward/reverse for a duration | `.JOGF` / `.JOGR` |
| `tweak_motor` | Small step forward/reverse | `.TWV`, `.TWF` / `.TWR` |
| `get_motor_limits` | Read all limit fields | `.HLM`, `.LLM`, `.DHLM`, `.DLLM`, `.HLS`, `.LLS` |
| `set_motor_limits` | Set user soft limits | `.HLM`, `.LLM` |
| `list_motors` | Read positions for multiple motors | bulk `.RBV`, `.DMOV`, `.EGU`, `.DESC` |
| `home_motor` | Home a motor | `.HOMF` / `.HOMR` |

### Step 1 — Check status first

Before any move, always call `get_motor_status` to see current position, limits, and whether the motor is already moving.

```
get_motor_status(prefix="20idMotSim", motor="m1")
```

### Step 2 — Move

Absolute move:
```
move_motor_absolute(
    prefix="20idMotSim",
    motor="m1",
    position=5.0,
    wait=True,
    timeout=60
)
```

Relative move:
```
move_motor_relative(
    prefix="20idMotSim",
    motor="m1",
    delta=0.5,
    wait=True
)
```

### Step 3 — Verify

After the move, the tool returns the final RBV. If `wait=True` (default), the tool polls DMOV until motion completes.

### Safety rules — never violate

- **Soft limit check**: `move_motor_absolute` reads HLM/LLM and rejects out-of-range targets before issuing caput.
- **Large move guard**: Moves >50% of travel range require `confirm_large_move=True`.
- **Limit switch guard**: If HLS=1 or LLS=1, the motor cannot move further in that direction.
- **STOP is always allowed**: `stop_motor` executes immediately with no checks.
- **Never set STOP=0** (arming) — only set STOP=1 (stopping).
- **Never home without explicit user instruction**.
- **Never start the IOC** — the user starts it manually in a separate session.

### Shell equivalents (for reference)

```bash
# Read position
caget 20idMotSim:m1.RBV

# Move to 5.0
caput 20idMotSim:m1 5.0

# Stop
caput 20idMotSim:m1.STOP 1

# Read status
caget 20idMotSim:m1.RBV 20idMotSim:m1.DMOV 20idMotSim:m1.HLS 20idMotSim:m1.LLS
```

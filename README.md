# aioflexiv

<div align="center">
  <img width="340" src="assets/aioflexiv_logo.png" alt="aioflexiv logo">
</div>
<p align="center">
  <a href="https://pypi.org/project/aioflexiv/">
    <img src="https://img.shields.io/pypi/v/aioflexiv" alt="PyPI version">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
</p>

**aioflexiv** is an experimental asyncio-based Python library for controlling
Flexiv robots from a Python-owned real-time torque loop. It combines
**`flexivrdk`** for Flexiv robot access, a small **`pybind11`** shim for RT
joint-torque mode, and **`Ruckig`** for smooth joint-space trajectory
generation.

The library is designed for research workflows that need direct torque control,
joint impedance control, operational-space control, and minimal ceremony around
async Python experiments.

If you want the same kind of Python control workflow for a Franka FR3 robot,
check out [aiofranka](https://github.com/Improbable-AI/aiofranka).

This is a hardware-control experiment. Keep clear of the robot, verify your
limits, and be ready to stop the robot before running any command or script.

## Installation

Install from this repository:

```bash
pip install .
```

Or for development:

```bash
git clone https://github.com/Improbable-AI/aioflexiv.git
cd aioflexiv
pip install -e .
```

The install pulls `flexivrdk==1.9.1`, `numpy`, and `ruckig`, builds the
`pybind11` RT shim, and installs the `aioflexiv` console command.

## Quick Start

Run the torque loop in-process with asyncio:

- **Single script**: no separate server process
- **Direct access**: controller state and commands stay in Python
- **Async discipline**: avoid blocking calls after `controller.start()` because
  the background torque loop must keep stepping

```python
import asyncio
import numpy as np
from aioflexiv import FlexivController

async def main():
    controller = FlexivController("Rizon4s-063533")

    await controller.start()
    try:
        await controller.move()

        controller.switch("impedance")
        controller.kp = np.ones(controller.dof) * 80.0
        controller.kd = np.ones(controller.dof) * 4.0
        controller.set_freq(50)

        q0 = controller.initial_qpos.copy()
        for cnt in range(250):
            target = q0.copy()
            target[0] += 0.05 * np.sin(cnt / 50.0)
            await controller.set("q_desired", target)
    finally:
        await controller.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## CLI Reference

The CLI currently focuses on status inspection.

```text
aioflexiv status [ROBOT_SN] [--network-interface IP] [--events N] [--verbose]
```

### `status`

Show Flexiv robot connection, system, state, tool, device, and recent event
information.

```bash
aioflexiv status Rizon4s-063533
```

Use `--network-interface` to whitelist one or more local IPv4 interfaces for RDK
discovery:

```bash
aioflexiv status Rizon4s-063533 --network-interface 192.168.2.10
```

## Core Concepts

### Torque Loop

`FlexivController.start()` starts the Flexiv RT joint-torque bridge and launches
an asyncio task that repeatedly computes and sends torque commands. Stop with
`await controller.stop()` in a `finally` block so the robot exits torque mode
cleanly.

### Safety Limits

Torque clipping is enabled by default and uses Flexiv RDK `RobotInfo::tau_max`.
No torque-rate limit is guessed by default. Set
`controller.torque_diff_limit = <Nm/s>` if you want an additional user-space
slew limit.

By default, `FlexivController(ext_offset=True)` records the initial `tau_ext`
when the torque loop starts and sends `commanded_torque - initial_tau_ext` to the
robot. Pass `ext_offset=False` to disable this compensation.

### Rate Limiting

Use `set_freq()` to pace command updates:

```python
controller.set_freq(50)

for target in trajectory:
    await controller.set("q_desired", target)
```

### Force/Torque Sensing

For compatible robots with an F/T sensor, `controller.robot.info["has_FT_sensor"]`
reports availability and `controller.state` exposes 6D wrench readings as NumPy
arrays:

```python
state = controller.state
raw_flange_wrench = state["ft_sensor_raw"]          # [fx, fy, fz, mx, my, mz]
tcp_wrench = state["ext_wrench_in_tcp"]             # [N, N, N, Nm, Nm, Nm]
world_wrench = state["ext_wrench_in_world"]
```

## Controllers

### 1. Joint Impedance

Joint-space spring-damper control:

```python
controller.switch("impedance")
controller.kp = np.ones(controller.dof) * 80.0
controller.kd = np.ones(controller.dof) * 4.0
await controller.set("q_desired", target_qpos)
```

**Use case**: compliant joint-space holding and trajectory tracking.

### 2. Operational Space

End-effector pose control using Flexiv RDK model data:

```python
controller.switch("osc")
controller.ee_kp = np.array([150, 150, 150, 20, 20, 20], dtype=float)
controller.ee_kd = np.array([20, 20, 20, 3, 3, 3], dtype=float)

target = controller.initial_ee.copy()
target[2, 3] += 0.03
await controller.set("ee_desired", target)
```

**Use case**: Cartesian holding and small task-space motions.

### 3. Direct Torque

Direct user torque commands:

```python
controller.switch("torque")
controller.set_freq(10)
await controller.set("torque", np.zeros(controller.dof))
```

**Use case**: low-level experiments where you want to own the torque command.

### 4. Ruckig Moves

Point-to-point joint motions are generated with Ruckig and tracked through the
same Python torque loop:

```python
await controller.move()
await controller.move([0.0, -0.7, 0.0, 1.57, 0.0, 0.7, 0.0])
```

## Examples

```bash
python examples/01_joint_impedance.py Rizon4s-063533
python examples/02_osc_hold.py Rizon4s-063533
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

If you use this library in your research, please cite:

```bibtex
@software{aioflexiv,
  author = {Park, Younghyo},
  title = {aioflexiv: Asyncio-based Flexiv Robot Control},
  year = {2026},
  url = {https://github.com/Improbable-AI/aioflexiv}
}
```

## Acknowledgments

- Built on [Flexiv RDK](https://www.flexiv.com/software/rdk)
- Trajectory generation with [Ruckig](https://github.com/pantor/ruckig)

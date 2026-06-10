# aioflexiv

Experimental Python-owned Flexiv RT torque control shim.

This package currently targets `flexivrdk==1.9.1` and exposes a small
`aiofranka`-style control surface:

- `FlexivRobotInterface`: low-level Python-owned RT torque loop bridge
- `FlexivController`: async joint impedance, operational-space, and direct
  torque control
- `aioflexiv-zero-torque`: foreground zero-torque smoke test

```bash
aioflexiv-zero-torque Rizon4s-063533 --i-am-clear
```

The command switches the robot to `RT_JOINT_TORQUE` and repeatedly sends zero
user-commanded torque while leaving Flexiv gravity compensation and soft joint
limits enabled.

This is a hardware-control experiment. Keep clear of the robot and be ready to
stop it before running the command.

## Install

From this repository:

```bash
pip install .
```

That single install pulls `flexivrdk==1.9.1`, builds the tiny pybind11 shim, and
installs the `aioflexiv-zero-torque` console command.

## Joint Impedance

```python
import asyncio
import numpy as np
from aioflexiv import FlexivController

async def main():
    controller = FlexivController("Rizon4s-063533")
    await controller.start()
    controller.switch("impedance")
    controller.kp = np.ones(7) * 80.0
    controller.kd = np.ones(7) * 4.0
    controller.set_freq(50)

    q0 = controller.initial_qpos.copy()
    for i in range(250):
        target = q0.copy()
        target[0] += 0.05 * np.sin(i / 50.0)
        await controller.set("q_desired", target)

    await controller.stop()

asyncio.run(main())
```

## Operational Space

```python
controller.switch("osc")
controller.ee_kp = np.array([250, 250, 250, 30, 30, 30], dtype=float)
controller.ee_kd = np.array([20, 20, 20, 4, 4, 4], dtype=float)

target = controller.initial_ee.copy()
target[2, 3] += 0.03
await controller.set("ee_desired", target)
```

# aioflexiv

Experimental Python-owned Flexiv RT torque control shim.

This package currently targets `flexivrdk==1.9.1` and exposes one command:

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


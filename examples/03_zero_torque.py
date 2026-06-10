#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from aioflexiv import FlexivController


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_sn", nargs="?", default="Rizon4s-063533")
    args = parser.parse_args()

    controller = FlexivController(args.robot_sn)
    await controller.start()
    try:
        base = np.ones(controller.dof)
        with controller.state_lock:
            controller.kp = base * 80.0
            controller.kd = base * 4.0
            print("Moving to initial position...")
        await controller.move([0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853])

        controller.switch("torque")
        controller.set_freq(50)
        zero_torque = np.zeros(controller.dof)
        while True:
            await controller.set("torque", zero_torque)
    finally:
        await controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

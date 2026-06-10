#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from aioflexiv import FlexivController


async def run(robot_sn: str, clear: bool) -> None:
    if not clear:
        raise SystemExit("Refusing to run without --i-am-clear")

    controller = FlexivController(robot_sn)
    await controller.start()
    try:
        controller.switch("osc")
        controller.ee_kp = np.array([120.0, 120.0, 120.0, 15.0, 15.0, 15.0])
        controller.ee_kd = np.array([18.0, 18.0, 18.0, 2.5, 2.5, 2.5])
        controller.null_kp = np.ones(controller.dof) * 3.0
        controller.null_kd = np.ones(controller.dof) * 1.0
        controller.set_freq(50)

        target = controller.initial_ee.copy()
        while True:
            await controller.set("ee_desired", target)
    finally:
        await controller.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_sn", nargs="?", default="Rizon4s-063533")
    parser.add_argument("--i-am-clear", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.robot_sn, args.i_am_clear))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


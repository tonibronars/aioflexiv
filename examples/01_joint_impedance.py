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
        controller.switch("impedance")
        controller.kp = np.ones(controller.dof) * 60.0
        controller.kd = np.ones(controller.dof) * 4.0
        controller.set_freq(50)

        q0 = controller.initial_qpos.copy()
        while True:
            await controller.set("q_desired", q0)
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


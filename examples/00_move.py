#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

from aioflexiv import FlexivController


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_sn", nargs="?", default="Rizon4s-063533")
    parser.add_argument("--joint", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()

    controller = FlexivController(args.robot_sn)
    await controller.start()
    try:
        if not 0 <= args.joint < controller.dof:
            raise ValueError(
                f"joint must be in [0, {controller.dof - 1}], got {args.joint}"
            )
        target = controller.initial_qpos.copy()
        target[args.joint] += args.delta
        await controller.move(target)
    finally:
        await controller.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

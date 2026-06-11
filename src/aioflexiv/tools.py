from __future__ import annotations

from typing import Any, Iterable


FLANGE_TCP_LOCATION = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    return text.split(".", 1)[-1] if "." in text else text


def call_any_method(obj: Any, names: Iterable[str], *args):
    for name in names:
        method = getattr(obj, name, None)
        if method is not None:
            return method(*args)
    joined = ", ".join(names)
    raise AttributeError(f"{type(obj).__name__} has none of these methods: {joined}")


def switch_robot_to_idle(robot: Any, flexivrdk: Any) -> None:
    mode = getattr(robot, "mode", None)
    if mode is not None and _enum_name(mode()).upper() == "IDLE":
        return

    switch_mode = getattr(robot, "SwitchMode", None)
    idle_mode = getattr(getattr(flexivrdk, "Mode", None), "IDLE", None)
    if switch_mode is not None and idle_mode is not None:
        switch_mode(idle_mode)
        return

    stop = getattr(robot, "Stop", None)
    if stop is not None:
        stop()


def switch_active_tool(
    robot_sn: str,
    tool_name: str,
    *,
    network_interface_whitelist: list[str] | None = None,
    verbose: bool = False,
) -> str:
    """Switch the robot's active Flexiv tool using the official Python RDK."""
    if not str(tool_name).strip():
        raise ValueError("tool_name must not be empty")

    import flexivrdk

    robot = flexivrdk.Robot(
        robot_sn,
        network_interface_whitelist or [],
        verbose,
        False,
    )
    tool = flexivrdk.Tool(robot)
    current = str(tool.name())
    if current == tool_name:
        return current

    switch_robot_to_idle(robot, flexivrdk)
    call_any_method(tool, ("Switch", "switch"), tool_name)
    return str(tool.name())

from __future__ import annotations

import argparse
import math
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable, Sequence


DEFAULT_ROBOT_SN = "Rizon4s-063533"
_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _IS_TTY else ""


BOLD = _c("\033[1m")
DIM = _c("\033[2m")
GREEN = _c("\033[32m")
YELLOW = _c("\033[33m")
RED = _c("\033[31m")
RST = _c("\033[0m")


def _get_version() -> str:
    try:
        return version("aioflexiv")
    except PackageNotFoundError:
        return "?"


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    return text.split(".", 1)[-1] if "." in text else text


def _human_enum(value: Any) -> str:
    return _enum_name(value).replace("_", " ").lower()


def _yes_no(value: bool, *, good_when: bool = True) -> str:
    color = GREEN if bool(value) is good_when else RED
    return f"{color}{'yes' if value else 'no'}{RST}"


def _ok_bad(ok: bool, ok_text: str, bad_text: str) -> str:
    return f"{GREEN}{ok_text}{RST}" if ok else f"{RED}{bad_text}{RST}"


def _line(label: str, value: Any) -> None:
    dots = "." * max(2, 18 - len(label))
    print(f"    {label} {dots} {value}")


def _section(name: str) -> None:
    print()
    print(f"  {BOLD}{name}{RST}")


def _safe_call(label: str, func, default: Any = None) -> Any:
    try:
        return func()
    except Exception as exc:
        return default if default is not None else f"unavailable ({exc})"


def _vec(values: Iterable[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def _max_abs_line(values: Sequence[float], limits: Sequence[float] | None = None) -> str:
    if not values:
        return "unavailable"
    idx, val = max(enumerate(values), key=lambda item: abs(float(item[1])))
    joint = f"A{idx + 1}"
    text = f"{abs(float(val)):.4f} @ {joint}"
    if limits is not None and idx < len(limits):
        limit = float(limits[idx])
        color = GREEN if abs(float(val)) <= limit else RED
        text = f"{color}{text}{RST} {DIM}(limit {limit:.4f}){RST}"
    return text


def _quat_to_rpy(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _print_tool(robot: Any, flexivrdk: Any) -> None:
    _section("Tool")
    try:
        tool = flexivrdk.Tool(robot)
        name = tool.name()
        params = tool.params()
    except Exception as exc:
        _line("Status", f"{DIM}unavailable ({exc}){RST}")
        return

    _line("Name", name)
    _line("Mass", f"{float(params.mass):.3f} kg")
    _line("CoM", f"{_vec(params.CoM)} m")

    inertia = list(params.inertia)
    if len(inertia) >= 6:
        ixx, iyy, izz, ixy, ixz, iyz = [float(v) for v in inertia[:6]]
        _line("Inertia", f"[{ixx:.4f}, {ixy:.4f}, {ixz:.4f}] kg*m^2")
        print(f"                       [{ixy:.4f}, {iyy:.4f}, {iyz:.4f}]")
        print(f"                       [{ixz:.4f}, {iyz:.4f}, {izz:.4f}]")

    tcp = list(params.tcp_location)
    if len(tcp) >= 7:
        _line("TCP offset", f"{_vec(tcp[:3])} m")
        _line("TCP quat", _vec(tcp[3:7]))
        _line("TCP RPY", f"{_vec(_quat_to_rpy(*[float(v) for v in tcp[3:7]]))} rad")


def _print_devices(robot: Any, flexivrdk: Any) -> None:
    try:
        device = flexivrdk.Device(robot)
        names = list(device.list())
    except Exception:
        return
    if not names:
        return

    _section("Devices")
    for name in names:
        enabled = _safe_call("enabled", lambda n=name: device.enabled(n), default=None)
        connected = _safe_call("connected", lambda n=name: device.connected(n), default=None)
        bits = []
        if connected is not None:
            bits.append(_ok_bad(bool(connected), "connected", "disconnected"))
        if enabled is not None:
            bits.append(_ok_bad(bool(enabled), "enabled", "disabled"))
        _line(name, ", ".join(bits) if bits else "present")


def _print_events(robot: Any, count: int) -> None:
    if count <= 0:
        return
    events = _safe_call("event_log", robot.event_log, default=[])
    if not events:
        return

    _section(f"Recent Events")
    for event in list(events)[-count:]:
        level = _enum_name(getattr(event, "level", "UNKNOWN")).lower()
        color = RED if level in {"error", "critical"} else YELLOW if level == "warning" else DIM
        event_id = getattr(event, "id", "?")
        desc = getattr(event, "description", "")
        _line(f"#{event_id}", f"{color}{level}{RST} {desc}")
        action = getattr(event, "recommended_actions", "")
        if action:
            print(f"                       {DIM}{action}{RST}")


def cmd_status(args: argparse.Namespace) -> int:
    import flexivrdk

    ver = _get_version()
    print(f"\n  {BOLD}aioflexiv{RST} {DIM}v{ver}{RST}  {DIM}|{RST}  {args.robot_sn}")

    try:
        robot = flexivrdk.Robot(
            args.robot_sn, args.network_interface or [], args.verbose, False
        )
    except Exception as exc:
        _section("Connection")
        _line("Status", f"{RED}failed{RST}")
        _line("Error", exc)
        print()
        return 1

    info = robot.info()
    states = _safe_call("states", robot.states, default=None)

    op_status = robot.operational_status()
    op_name = _enum_name(op_status)
    operational = robot.operational()
    op_color = GREEN if operational else RED if "FAULT" in op_name else YELLOW

    _section("System")
    _line("Connected", _yes_no(robot.connected()))
    _line("Serial", getattr(info, "serial_num", args.robot_sn) or args.robot_sn)
    _line("Model", getattr(info, "model_name", "?") or "?")
    _line("Software", getattr(info, "software_ver", "?") or "?")
    _line("License", getattr(info, "license_type", "?") or "?")
    _line("Operational", f"{op_color}{_human_enum(op_status)}{RST}")
    _line("Mode", _human_enum(robot.mode()))

    _section("Robot")
    _line("Fault", _yes_no(robot.fault(), good_when=False))
    _line("Recovery", _yes_no(robot.recovery(), good_when=False))
    _line("Reduced", _yes_no(robot.reduced(), good_when=False))
    _line("Stopped", _yes_no(robot.stopped()))
    _line("Busy", _yes_no(robot.busy(), good_when=False))
    _line("E-Stop released", _yes_no(robot.estop_released()))
    _line("Enable button", _ok_bad(robot.enabling_button_pressed(), "pressed", "released"))
    _line("Timeliness limit", _yes_no(robot.reached_timeliness_failure_limit(), good_when=False))
    _line("DoF", f"{info.DoF} total, {info.DoF_m} manipulator, {info.DoF_e} external")
    _line("FT sensor", _yes_no(bool(info.has_FT_sensor)))

    q_min = list(info.q_min)
    q_max = list(info.q_max)
    dq_max = list(info.dq_max)
    tau_max = list(info.tau_max)
    if q_min and q_max:
        _line("q min", f"{_vec(q_min, 3)} rad")
        _line("q max", f"{_vec(q_max, 3)} rad")
    if dq_max:
        _line("dq max", f"{_vec(dq_max, 3)} rad/s")
    if tau_max:
        _line("tau max", f"{_vec(tau_max, 2)} Nm")

    if states is not None:
        q = list(states.q)
        dq = list(states.dq)
        dtheta = list(states.dtheta)
        tau = list(states.tau)
        tau_ext = list(states.tau_ext)
        _section("State")
        _line("Timestamp", tuple(states.timestamp))
        _line("q", f"{_vec(q)} rad")
        _line("dq", f"{_vec(dq)} rad/s")
        _line("dtheta", f"{_vec(dtheta)} rad/s")
        _line("Max |dq|", _max_abs_line(dq, dq_max))
        _line("Max |dtheta|", _max_abs_line(dtheta, dq_max))
        _line("tau", f"{_vec(tau, 3)} Nm")
        _line("tau_ext", f"{_vec(tau_ext, 3)} Nm")
        if hasattr(states, "temperature"):
            _line("Temperature", f"{_vec(states.temperature, 1)} C")

        _section("TCP")
        _line("Pose", f"{_vec(states.tcp_pose)} [m, quat]")
        _line("Velocity", f"{_vec(states.tcp_vel)} [m/s, rad/s]")
        _line("Flange pose", f"{_vec(states.flange_pose)} [m, quat]")
        _line("Ext wrench TCP", f"{_vec(states.ext_wrench_in_tcp, 3)} [N, Nm]")
        _line("Ext wrench world", f"{_vec(states.ext_wrench_in_world, 3)} [N, Nm]")

    _print_tool(robot, flexivrdk)
    _print_devices(robot, flexivrdk)
    _print_events(robot, args.events)
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="aioflexiv command line tools")
    parser.add_argument("--version", action="version", version=f"aioflexiv {_get_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Show Flexiv robot status")
    status.add_argument("robot_sn", nargs="?", default=DEFAULT_ROBOT_SN)
    status.add_argument(
        "--network-interface",
        action="append",
        help="Whitelist a local IPv4 interface for RDK discovery; may be repeated.",
    )
    status.add_argument("--events", type=int, default=3, help="Number of recent events to show.")
    status.add_argument("--verbose", action="store_true", help="Enable Flexiv RDK verbose output.")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

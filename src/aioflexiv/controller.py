from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

from .robot import FlexivRobotInterface


def _as_vector(value, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return np.ones(size) * float(arr)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {arr.shape}")
    return arr


def _rotation_error_vector(R_goal: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    R_err = R_goal @ R_current.T
    cos_angle = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    skew_vec = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )
    if angle < 1e-6:
        return 0.5 * skew_vec
    return skew_vec * (angle / (2.0 * np.sin(angle)))


class FlexivController:
    """Async Python-owned torque controller for Flexiv robots.

    Supported modes:
    - `impedance`: joint-space spring damper
    - `osc`: operational-space control using Flexiv RDK `Model` data
    - `torque`: direct user torque
    """

    def __init__(
        self,
        robot: str | FlexivRobotInterface,
        *,
        link_name: str = "flange",
        enable_gravity_comp: bool = True,
        enable_soft_limits: bool = True,
        friction_comp_scale: float = 100.0,
    ) -> None:
        if isinstance(robot, FlexivRobotInterface):
            self.robot = robot
        else:
            self.robot = FlexivRobotInterface(
                robot,
                link_name=link_name,
                enable_gravity_comp=enable_gravity_comp,
                enable_soft_limits=enable_soft_limits,
                friction_comp_scale=friction_comp_scale,
            )

        self.state_lock = threading.Lock()
        self.type = "impedance"
        self.running = False
        self.task: asyncio.Task | None = None
        self.state: dict[str, np.ndarray] | None = None

        self.kp = np.ones(7) * 80.0
        self.kd = np.ones(7) * 4.0
        self.ee_kp = np.array([150.0, 150.0, 150.0, 20.0, 20.0, 20.0])
        self.ee_kd = np.array([20.0, 20.0, 20.0, 3.0, 3.0, 3.0])
        self.null_kp = np.ones(7) * 5.0
        self.null_kd = np.ones(7) * 1.0
        self.torque = np.zeros(7)

        self.clip = True
        self.torque_diff_limit = 990.0
        self.torque_limit = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

        self.initial_qpos: np.ndarray | None = None
        self.initial_ee: np.ndarray | None = None
        self.q_desired: np.ndarray | None = None
        self.ee_desired: np.ndarray | None = None

        self._update_freq = 50.0
        self._last_update_time: dict[str, float] = {}
        self._last_loop_time: float | None = None
        self._last_commanded_torque: np.ndarray | None = None

    @property
    def dof(self) -> int:
        return self.robot.dof

    def _sync_shapes(self) -> None:
        dof = self.dof
        self.kp = _as_vector(self.kp, dof, "kp")
        self.kd = _as_vector(self.kd, dof, "kd")
        self.null_kp = _as_vector(self.null_kp, dof, "null_kp")
        self.null_kd = _as_vector(self.null_kd, dof, "null_kd")
        self.torque = _as_vector(self.torque, dof, "torque")
        self.torque_limit = self.robot.torque_limit
        self._last_commanded_torque = np.zeros(dof)

    def initialize(self) -> None:
        state = self.robot.state
        self.state = state
        self.initial_qpos = state["qpos"].copy()
        self.initial_ee = state["ee"].copy()
        self.q_desired = self.initial_qpos.copy()
        self.ee_desired = self.initial_ee.copy()

    async def start(self) -> asyncio.Task:
        self.robot.start()
        self._sync_shapes()
        self.initialize()
        self.running = True
        self._last_loop_time = time.perf_counter()
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())
        await asyncio.sleep(0.1)
        return self.task

    async def stop(self) -> None:
        self.running = False
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.robot.stop()

    async def _run(self) -> None:
        try:
            while self.running:
                self.step()
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.running = False
            try:
                self.robot.stop()
            finally:
                raise

    def switch(self, controller_type: str) -> None:
        if controller_type not in {"impedance", "osc", "torque"}:
            raise ValueError(f"Unknown controller type: {controller_type}")
        self.type = controller_type
        if self.robot.started:
            self.initialize()
        self._last_update_time.clear()

    def set_freq(self, freq: float) -> None:
        self._update_freq = float(freq)

    async def set(self, attr: str, value) -> None:
        current_time = time.perf_counter()
        dt = 1.0 / self._update_freq

        if attr not in self._last_update_time:
            with self.state_lock:
                setattr(self, attr, np.asarray(value, dtype=float))
            self._last_update_time[attr] = current_time + dt
            await asyncio.sleep(dt)
            return

        target_time = self._last_update_time[attr] + dt
        with self.state_lock:
            setattr(self, attr, np.asarray(value, dtype=float))

        sleep_time = target_time - current_time
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        self._last_update_time[attr] = target_time

    def set_now(self, attr: str, value) -> None:
        with self.state_lock:
            setattr(self, attr, np.asarray(value, dtype=float))

    def step(self) -> None:
        now = time.perf_counter()
        dt = 1e-3 if self._last_loop_time is None else max(now - self._last_loop_time, 1e-4)
        self._last_loop_time = now

        if self.type == "impedance":
            state = self.robot.state_minimal()
            tau = self._impedance_torque(state)
        elif self.type == "osc":
            state = self.robot.state
            tau = self._osc_torque(state)
        elif self.type == "torque":
            state = self.robot.state_minimal()
            with self.state_lock:
                tau = np.asarray(self.torque, dtype=float).copy()
        else:
            raise ValueError(f"Unknown controller type: {self.type}")

        tau = self._clip_torque(tau, dt)
        self.robot.step(tau)
        state["last_torque"] = tau.copy()
        self.state = state

    def _clip_torque(self, tau: np.ndarray, dt: float) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        if not self.clip:
            self._last_commanded_torque = tau.copy()
            return tau

        last = (
            np.zeros_like(tau)
            if self._last_commanded_torque is None
            else self._last_commanded_torque
        )
        max_delta = self.torque_diff_limit * dt
        tau = last + np.clip(tau - last, -max_delta, max_delta)
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        self._last_commanded_torque = tau.copy()
        return tau

    def _impedance_torque(self, state: dict[str, np.ndarray]) -> np.ndarray:
        q = state["qpos"]
        dq = state["qvel"]
        with self.state_lock:
            q_goal = np.asarray(self.q_desired, dtype=float).copy()
            kp = np.asarray(self.kp, dtype=float).copy()
            kd = np.asarray(self.kd, dtype=float).copy()
        return kp * (q_goal - q) - kd * dq

    def _osc_torque(self, state: dict[str, np.ndarray]) -> np.ndarray:
        q = state["qpos"]
        dq = state["qvel"]
        ee = state["ee"]
        jac = state["jac"]
        mm = state["mm"]
        ee_vel = state.get("tcp_vel", jac @ dq)

        with self.state_lock:
            ee_goal = np.asarray(self.ee_desired, dtype=float).copy()
            ee_kp = np.asarray(self.ee_kp, dtype=float).copy()
            ee_kd = np.asarray(self.ee_kd, dtype=float).copy()
            null_kp = np.asarray(self.null_kp, dtype=float).copy()
            null_kd = np.asarray(self.null_kd, dtype=float).copy()
            q0 = np.asarray(self.initial_qpos, dtype=float).copy()

        twist_error = np.zeros(6)
        twist_error[:3] = ee_goal[:3, 3] - ee[:3, 3]
        twist_error[3:] = _rotation_error_vector(ee_goal[:3, :3], ee[:3, :3])

        minv = np.linalg.pinv(mm)
        mx_inv = jac @ minv @ jac.T
        mx = np.linalg.pinv(mx_inv)

        wrench = mx @ (ee_kp * twist_error - ee_kd * ee_vel)
        feedback = jac.T @ wrench

        jbar = minv @ jac.T @ mx
        null_projector = np.eye(self.dof) - jac.T @ jbar.T
        null_cmd = null_kp * (q0 - q) - null_kd * dq
        null = null_projector @ null_cmd

        return feedback + null

    async def move(self, qpos, *, duration: float = 3.0, freq: float = 50.0) -> None:
        self.switch("impedance")
        self.set_freq(freq)
        start = np.asarray(self.state["qpos"] if self.state is not None else self.robot.state_minimal()["qpos"])
        goal = np.asarray(qpos, dtype=float)
        steps = max(int(duration * freq), 1)
        for i in range(steps):
            alpha = (i + 1) / steps
            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
            await self.set("q_desired", (1.0 - alpha) * start + alpha * goal)


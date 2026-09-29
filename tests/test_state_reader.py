"""The read-only reader is tested against an SDK fake that rejects control calls."""

import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from aioflexiv.robot import RDK_MEASURED_STATE_FIELDS
from aioflexiv.state_reader import FlexivStateReader

READ_ONLY_CALLS = {"info", "connected", "states", "mode", "stopped", "fault",
                   "operational", "estop_released"}


class ReadOnlyRobot:
    def __init__(self):
        self.calls = []
        self.is_connected = True
        self.is_fault = False
        sizes = {"tcp_pose": 7, "tcp_vel": 6, "flange_pose": 7, "ft_sensor_raw": 6,
                 "ext_wrench_in_tcp": 6, "ext_wrench_in_world": 6,
                 "ext_wrench_in_tcp_raw": 6, "ext_wrench_in_world_raw": 6}
        self.state = SimpleNamespace(timestamp=(100, 123), q=[0.1] * 7, **{
            name: [float(index)] * sizes.get(name, 7)
            for index, name in enumerate(RDK_MEASURED_STATE_FIELDS)
        })
        self.info_value = SimpleNamespace(
            serial_num="Rizon4s-123456", model_name="Rizon4s", software_ver="3.9",
            license_type="research", has_FT_sensor=True, DoF=7, DoF_e=0, DoF_m=7,
            q_min=[-2.0] * 7, q_max=[2.0] * 7, dq_max=[1.0] * 7, tau_max=[50.0] * 7,
            K_q_nom=[1000.0] * 7, K_x_nom=[3000.0] * 6,
        )

    def __getattr__(self, name):
        if name not in READ_ONLY_CALLS:
            raise AssertionError(f"Unexpected SDK access: {name}")
        results = {
            "info": lambda: self.info_value,
            "connected": lambda: self.is_connected,
            "states": lambda: self.state,
            "mode": lambda: SimpleNamespace(name="???"),
            "stopped": lambda: False,
            "fault": lambda: self.is_fault,
            "operational": lambda: not self.is_fault,
            "estop_released": lambda: True,
        }

        def call():
            self.calls.append(name)
            return results[name]()

        return call


def open_reader(robot):
    constructor = mock.Mock(return_value=robot)
    with mock.patch.dict(sys.modules, flexivrdk=SimpleNamespace(Robot=constructor)):
        reader = FlexivStateReader("Rizon4s-123456", network_interface_whitelist=["10.0.0.1"])
    constructor.assert_called_once_with("Rizon4s-123456", ["10.0.0.1"], False, False)
    return reader


def test_read_returns_every_measured_channel_and_receipt_time():
    robot = ReadOnlyRobot()
    reader = open_reader(robot)
    with mock.patch("aioflexiv.state_reader.time.monotonic", side_effect=[10.0, 10.2]):
        state = reader.read()
    assert robot.calls == ["connected", "states"]
    assert state["timestamp"] == (100, 123)
    assert state["timestamp_monotonic"] == pytest.approx(10.1)
    np.testing.assert_array_equal(state["qpos"], robot.state.q)
    np.testing.assert_array_equal(state["qvel"], robot.state.dtheta)
    for name in RDK_MEASURED_STATE_FIELDS:
        assert isinstance(state[name], np.ndarray), name
        np.testing.assert_array_equal(state[name], getattr(robot.state, name))


def test_read_rejects_unset_timestamp_disconnect_and_closed_reader():
    robot = ReadOnlyRobot()
    reader = open_reader(robot)
    robot.state.timestamp = (0, 0)
    with pytest.raises(RuntimeError, match="has not returned a timestamped robot state"):
        reader.read()
    robot.state.timestamp = (100, 123)
    robot.is_connected = False
    with pytest.raises(RuntimeError, match="disconnected"):
        reader.read()
    reader.close()
    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.read()


def test_status_and_info_are_read_only_even_in_fault():
    robot = ReadOnlyRobot()
    robot.is_fault = True
    with open_reader(robot) as reader:
        status = reader.status()
        info = reader.info
    assert status == {"mode": "???", "stopped": False, "connected": True, "fault": True,
                      "operational": False, "estop_released": True}
    assert info["q_min"] == [-2.0] * 7
    assert info["K_x_nom"] == [3000.0] * 6
    assert info["serial_num"] == "Rizon4s-123456"
    assert set(robot.calls) <= READ_ONLY_CALLS

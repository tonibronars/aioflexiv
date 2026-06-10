#include <flexiv/rdk/data.hpp>
#include <flexiv/rdk/mode.hpp>
#include <flexiv/rdk/robot.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
namespace rdk = flexiv::rdk;

namespace {

bool TimestampChanged(
    const std::pair<int, int>& lhs, const std::pair<int, int>& rhs)
{
    return lhs.first != rhs.first || lhs.second != rhs.second;
}

class ActiveTorqueControl {
public:
    ActiveTorqueControl(const std::string& robot_sn,
        const std::vector<std::string>& network_interface_whitelist = {}, bool verbose = true)
        : robot_(robot_sn, network_interface_whitelist, verbose)
    {
        robot_.SwitchMode(rdk::Mode::RT_JOINT_TORQUE);
        last_timestamp_ = robot_.states().timestamp;
    }

    rdk::RobotStates read_once(int timeout_ms = 1000)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

        while (std::chrono::steady_clock::now() < deadline) {
            auto state = robot_.states();
            if (TimestampChanged(state.timestamp, last_timestamp_)) {
                last_timestamp_ = state.timestamp;
                return state;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }

        throw std::runtime_error("Timed out waiting for a fresh Flexiv robot state");
    }

    void write_once(const std::vector<double>& torques, bool enable_gravity_comp = true,
        bool enable_soft_limits = true, double friction_comp_scale = 100.0)
    {
        robot_.StreamJointTorque(
            torques, enable_gravity_comp, enable_soft_limits, friction_comp_scale);
    }

    rdk::RobotStates states() const { return robot_.states(); }
    void stop() { robot_.Stop(); }

private:
    rdk::Robot robot_;
    std::pair<int, int> last_timestamp_ {};
};

py::dict compile_time_rt_probe()
{
    auto switch_mode = &rdk::Robot::SwitchMode;
    auto states = &rdk::Robot::states;
    auto stream_torque = &rdk::Robot::StreamJointTorque;

    py::dict result;
    result["rt_joint_torque_mode_value"] = static_cast<int>(rdk::Mode::RT_JOINT_TORQUE);
    result["has_switch_mode_symbol"] = switch_mode != nullptr;
    result["has_states_symbol"] = states != nullptr;
    result["has_stream_joint_torque_symbol"] = stream_torque != nullptr;
    return result;
}

} // namespace

PYBIND11_MODULE(_aioflexiv_rt, m)
{
    m.doc() = "Tiny Flexiv RT torque-control bridge";

    py::class_<rdk::RobotStates>(m, "RobotStates")
        .def_readonly("timestamp", &rdk::RobotStates::timestamp)
        .def_readonly("q", &rdk::RobotStates::q)
        .def_readonly("theta", &rdk::RobotStates::theta)
        .def_readonly("dq", &rdk::RobotStates::dq)
        .def_readonly("dtheta", &rdk::RobotStates::dtheta)
        .def_readonly("tau", &rdk::RobotStates::tau)
        .def_readonly("tau_des", &rdk::RobotStates::tau_des)
        .def_readonly("tau_ext", &rdk::RobotStates::tau_ext);

    py::class_<ActiveTorqueControl>(m, "ActiveTorqueControl")
        .def(py::init<const std::string&, const std::vector<std::string>&, bool>(),
            py::arg("robot_sn"), py::arg("network_interface_whitelist") = std::vector<std::string> {},
            py::arg("verbose") = true)
        .def("read_once", &ActiveTorqueControl::read_once, py::arg("timeout_ms") = 1000,
            py::call_guard<py::gil_scoped_release>())
        .def("write_once", &ActiveTorqueControl::write_once, py::arg("torques"),
            py::arg("enable_gravity_comp") = true, py::arg("enable_soft_limits") = true,
            py::arg("friction_comp_scale") = 100.0, py::call_guard<py::gil_scoped_release>())
        .def("states", &ActiveTorqueControl::states)
        .def("stop", &ActiveTorqueControl::stop);

    m.def("compile_time_rt_probe", &compile_time_rt_probe);
}


#include <Eigen/Core>
#include <iomanip>
#include <map>
#include <memory>
#include <utility>
#include <vector>

/* Pinocchio !!!! NEED TO BE INCLUDED BEFORE BOOST*/
#include <pinocchio/algorithm/joint-configuration.hpp> // integrate
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <tsid/solvers/solver-HQP-base.hpp>
#include <tsid/solvers/solver-HQP-eiquadprog.hpp>
#include <tsid/solvers/solver-HQP-factory.hxx>
#include <tsid/solvers/utils.hpp>
#include <tsid/utils/statistics.hpp>
#include <tsid/utils/stop-watch.hpp>

#include <boost/filesystem.hpp>

#include "inria_wbc/controllers/damage_controller.hpp"
#include "inria_wbc/controllers/tasks.hpp"
#include "inria_wbc/trajs/utils.hpp"

using namespace tsid;
using namespace tsid::math;

namespace inria_wbc {

    namespace tasks {
        ////// Contacts //////
        /// this looks like a task, but this does not derive from tsid::task::TaskBase
        std::shared_ptr<tsid::contacts::Contact6dExt> make_contact_task(
            const std::shared_ptr<robots::RobotWrapper>& robot,
            const std::shared_ptr<InverseDynamicsFormulationAccForce>& tsid,
            const std::string& task_name, const YAML::Node& node, const YAML::Node& controller_node)
        {
            assert(tsid);
            assert(robot);

            // parse yaml
            auto kp = IWBC_CHECK(node["kp"].as<double>());
            auto joint_name = IWBC_CHECK(node["joint"].as<std::string>());
            auto lxn = IWBC_CHECK(node["lxn"].as<double>());
            auto lyn = IWBC_CHECK(node["lyn"].as<double>());
            auto lxp = IWBC_CHECK(node["lxp"].as<double>()); 
            auto lyp = IWBC_CHECK(node["lyp"].as<double>());
            auto lz = IWBC_CHECK(node["lz"].as<double>());
            auto mu = IWBC_CHECK(node["mu"].as<double>());
            auto normal = IWBC_CHECK(node["normal"].as<std::vector<double>>());
            auto fmin = IWBC_CHECK(node["fmin"].as<double>());
            auto fmax = IWBC_CHECK(node["fmax"].as<double>());
            IWBC_ASSERT(normal.size() == 3, "normal size:", normal.size());
            IWBC_ASSERT(robot->model().existFrame(joint_name), joint_name, " does not exist!");
            auto activate = IWBC_CHECK(node["activate"].as<bool>());
            auto x = IWBC_CHECK(node["x"].as<double>());
            auto y = IWBC_CHECK(node["y"].as<double>());
            auto z = IWBC_CHECK(node["z"].as<double>());

            Matrix3x contact_points(3, 4);
            Eigen::Vector3d contact_normal(normal.data());
            bool horizontal;
            if (normal.at(0) == 0. && normal.at(1) == 0){
                // Horizontal plan (default for the foot contacts)
                horizontal = true;
                contact_points <<   -lxn, -lxn, lxp, lxp,
                                -lyn, lyp, -lyn, lyp,
                                lz, lz, lz, lz;  // z constant = horizontal
            } else {
                horizontal = false;
                Eigen::Vector3d vertical = {0., 0., 1.};
                Eigen::Vector3d u = vertical.cross(contact_normal);
                u.normalize();
                Eigen::Vector3d v = contact_normal.cross(u);
                Eigen::Vector3d tl = -lxn * u + lyp * v;
                Eigen::Vector3d tr = lxp * u + lyp * v;
                Eigen::Vector3d br = lxp * u - lyn * v;
                Eigen::Vector3d bl = -lxn * u - lyn * v;
                contact_points << bl[0],  tl[0],  br[0],   bl[0], 
                                bl[1],  tl[1],  br[1],   bl[1],
                                bl[2],  tl[2],  br[2],   bl[2];
            }

            auto contact_task = std::make_shared<tsid::contacts::Contact6dExt>(task_name, *robot, joint_name, contact_points, contact_normal, mu, fmin, fmax);
            contact_task->Kp(kp * Vector::Ones(6));
            contact_task->Kd(2.0 * contact_task->Kp().cwiseSqrt());
            auto contact_ref = robot->framePosition(tsid->data(), robot->model().getFrameId(joint_name));
            if (!horizontal){
                contact_ref.translation()[0] = x;
                contact_ref.translation()[1] = y;
                contact_ref.translation()[2] = z;
            }
            // hand orientation
            Eigen::Quaterniond q;
            if (!horizontal) {
                auto mask = contact_task->getMotionTask().getMask();
                mask[0] = 1.;
                mask[1] = 1.;
                mask[2] = 1.;
                mask[3] = 0.;
                mask[4] = 0.;
                mask[5] = 0.;
                contact_task->setMask(mask);
            }

            contact_task->Contact6d::setReference(contact_ref);
            // add the task
            if (activate)
                tsid->addRigidContact(*contact_task, cst::w_force_feet);

            return contact_task;
        }
    }

    namespace controllers {

        static Register<DamageController> __damage_controller("damage-controller");

        DamageController::DamageController(const YAML::Node& config) : TalosPosTracker(config)
        {
            if (verbose_)
                std::cout << "Talos Damage Controller initialized" << std::endl;
        }

        Eigen::VectorXd DamageController::tau(bool filter_mimics) const
        {
            return filter_mimics ? utils::slice_vec(tau_, non_mimic_indexes_) : tau_;
        }

        void DamageController::update(const SensorData& sensor_data)
        {
            //Do what you want with your parameters and solve
            //solve everything
            TalosPosTracker::update(sensor_data);
        }
    } // namespace controllers
} // namespace inria_wbc

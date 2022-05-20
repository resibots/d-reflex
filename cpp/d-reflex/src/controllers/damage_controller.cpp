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
            auto horizontal = IWBC_CHECK(node["horizontal"].as<bool>());
            auto x_cst =  IWBC_CHECK(node["x_cst"].as<bool>());
            auto activate = IWBC_CHECK(node["activate"].as<bool>());
            auto x = IWBC_CHECK(node["x"].as<double>());
            auto y = IWBC_CHECK(node["y"].as<double>());
            auto z = IWBC_CHECK(node["z"].as<double>());
            auto roll = IWBC_CHECK(node["roll"].as<double>());
            auto pitch = IWBC_CHECK(node["pitch"].as<double>());
            auto yaw = IWBC_CHECK(node["yaw"].as<double>());

            // create the task
            Matrix3x contact_points(3, 4);
            if (horizontal){
                contact_points <<   -lxn, -lxn, lxp, lxp,
                                    -lyn, lyp, -lyn, lyp,
                                    lz, lz, lz, lz;  // z constant = horizontal
            } else {
                if (x_cst){
                    contact_points << lz,  lz,  lz,   lz, // x constant = verticale
                                    -lyn,  lyp, -lyn, lyp,
                                    -lxn, -lxn, lxp,  lxp;
                } else {
                    contact_points << -lxn, -lxn, lxp,  lxp,
                                    lz, lz,  lz,  lz,  // y constant = verticale
                                    -lyn,  lyp, -lyn, lyp;
                }

            }

            Eigen::Vector3d contact_normal(normal.data());
            auto contact_task = std::make_shared<tsid::contacts::Contact6dExt>(task_name, *robot, joint_name, contact_points, contact_normal, mu, fmin, fmax);
            contact_task->Kp(kp * Vector::Ones(6));
            contact_task->Kd(2.0 * contact_task->Kp().cwiseSqrt());
            auto contact_ref = robot->framePosition(tsid->data(), robot->model().getFrameId(joint_name));
            if (!horizontal){
                contact_ref.translation()[0] = x;
                contact_ref.translation()[1] = y;
                contact_ref.translation()[2] = z;
            }
            // try changing rotation 
            auto euler = contact_ref.rotation().eulerAngles(0, 1, 2);
            euler[0] = roll;
            euler[1] = pitch;
            euler[2] = yaw;
            auto q = Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ());
            contact_ref.rotation() = q.toRotationMatrix();
            // apply
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
   //         parse_configuration(config["CONTROLLER"]);
            if (verbose_)
                std::cout << "Talos Damage Controller initialized" << std::endl;
        }

        Eigen::VectorXd DamageController::tau(bool filter_mimics) const
        {
            return filter_mimics ? utils::slice_vec(tau_, non_mimic_indexes_) : tau_;
        }
        
    //    void DamageController::parse_configuration(const YAML::Node& config)
    //    {
    //        TalosPosTracker::parse_configuration(config);
     //   }

        void DamageController::update(const SensorData& sensor_data)
        {
            //Do what you want with your parameters and solve
            //solve everything
            TalosPosTracker::update(sensor_data);
        }
    } // namespace controllers
} // namespace inria_wbc

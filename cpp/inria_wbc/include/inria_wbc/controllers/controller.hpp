#ifndef IWBC_TALOS_BASE_CONTROLLER_HPP
#define IWBC_TALOS_BASE_CONTROLLER_HPP

#include <Eigen/Core>
#include <iostream>
#include <string>
#include <unordered_map>

#include <pinocchio/spatial/se3.hpp>

#include <tsid/contacts/contact-6d-ext.hpp>
#include <tsid/contacts/contact-point.hpp>
#include <tsid/formulations/inverse-dynamics-formulation-acc-force.hpp>
#include <tsid/math/fwd.hpp>
#include <tsid/math/utils.hpp>
#include <tsid/robots/fwd.hpp>
#include <tsid/robots/robot-wrapper.hpp>
#include <tsid/tasks/task-actuation-bounds.hpp>
#include <tsid/tasks/task-angular-momentum-equality.hpp>
#include <tsid/tasks/task-com-equality.hpp>
#include <tsid/tasks/task-joint-bounds.hpp>
#include <tsid/tasks/task-joint-posVelAcc-bounds.hpp>
#include <tsid/tasks/task-joint-posture.hpp>
#include <tsid/tasks/task-momentum-equality.hpp>
#include <tsid/tasks/task-se3-equality.hpp>
#include <tsid/trajectories/trajectory-base.hpp>

#include <inria_wbc/safety/collision_check.hpp>
#include <inria_wbc/utils/factory.hpp>
#include <inria_wbc/utils/utils.hpp>

#include <boost/optional.hpp>
#include <boost/variant.hpp>

namespace inria_wbc {

    namespace controllers {

        using SensorData = std::unordered_map<std::string, Eigen::MatrixXd>;

        struct behavior_types {
            static const std::string FIXED_BASE;
            static const std::string SINGLE_SUPPORT;
            static const std::string DOUBLE_SUPPORT;
        };
        class Controller {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            Controller(const YAML::Node& config);
            Controller(const Controller&) = delete;
            Controller& operator=(const Controller& o) = delete;
            virtual ~Controller(){};

            // path where the files are stored (everything should be relative to this)
            const std::string& base_path() const { return base_path_; }

            virtual void update(const SensorData& sensor_data = {});

            // Removes the universe and root (floating base) joint names
            std::vector<std::string> controllable_dofs(bool filter_mimics = true) const;
            // Order of the floating base in q_ according to dart naming convention
            std::vector<std::string> floating_base_dofs() const;
            const std::vector<std::string>& mimic_names() const { return mimic_dof_names_; }
            std::vector<std::string> all_dofs(bool filter_mimics = true) const;
            std::vector<std::string> activated_contacts() { return activated_contacts_; };
            std::unordered_map<std::string, tsid::math::Vector> activated_contacts_forces() { return activated_contacts_forces_; };
            virtual const boost::optional<Eigen::Vector2d>& cop() const
            {
                static boost::optional<Eigen::Vector2d> tmp = boost::none;
                IWBC_ERROR("No COP estimator in controller.");
                return tmp;
            }

            virtual const boost::optional<Eigen::Vector2d>& cop_raw() const
            {
                static boost::optional<Eigen::Vector2d> tmp = boost::none;
                IWBC_ERROR("No COP estimator in controller.");
                return tmp;
            }

            virtual const boost::optional<Eigen::Vector2d>& lcop() const
            {
                static boost::optional<Eigen::Vector2d> tmp = boost::none;
                IWBC_ERROR("No COP estimator in controller.");
                return tmp;
            }

            virtual const boost::optional<Eigen::Vector2d>& rcop() const
            {
                static boost::optional<Eigen::Vector2d> tmp = boost::none;
                IWBC_ERROR("No COP estimator in controller.");
                return tmp;
            }

            virtual const Eigen::Vector3d& lf_force_filtered() const
            {
                static Eigen::Vector3d tmp;
                IWBC_ERROR("No COP estimator in controller.");
                return tmp;
            }

            virtual const Eigen::Vector3d& rf_force_filtered() const
            {
                static Eigen::Vector3d tmp;
                IWBC_ERROR("No COP estimator in controller.");
                return tmp;
            }

            // this could call a CoM estimator
            virtual const tsid::math::Vector3& com() const { return robot_->com(tsid_->data()); }

            virtual const tsid::math::Vector3 momentum() const { return momentum_; }

            const std::vector<int>& non_mimic_indexes() const { return non_mimic_indexes_; }
            Eigen::VectorXd filter_cmd(const Eigen::VectorXd& cmd) const { return utils::slice_vec(cmd, non_mimic_indexes_); }

            Eigen::VectorXd tau(bool filter_mimics = true) const;
            Eigen::VectorXd ddq(bool filter_mimics = true) const;
            Eigen::VectorXd dq(bool filter_mimics = true) const;
            Eigen::VectorXd q0(bool filter_mimics = true) const;
            Eigen::VectorXd q(bool filter_mimics = true) const;
            tsid::math::Vector q_tsid() const { return q_tsid_; };

            double t() const { return t_; };
            double dt() const { return dt_; };

            std::shared_ptr<tsid::robots::RobotWrapper> robot() { return robot_; };
            std::shared_ptr<tsid::InverseDynamicsFormulationAccForce> tsid() { return tsid_; };
            std::vector<double> pinocchio_model_masses() const;
            double pinocchio_total_model_mass() const;

            const std::vector<double>& pinocchio_model_cumulated_masses() { return tsid_->data().mass; };
            const std::vector<std::string>& pinocchio_joint_names() const { return robot_->model().names; }
            const pinocchio::SE3& model_joint_pos(const std::string& joint_name) const
            {
                assert(tsid_);
                assert(robot_);
                IWBC_ASSERT(robot_->model().existJointName(joint_name), "[", joint_name, "] (joint) does not exist!");
                return robot_->position(tsid_->data(), robot_->model().getJointId(joint_name));
            }
            pinocchio::SE3 model_frame_pos(const std::string& frame_name) const
            {
                assert(tsid_);
                assert(robot_);
                IWBC_ASSERT(robot_->model().existFrame(frame_name), "[", frame_name, "] (frame) does not exist!");
                return robot_->framePosition(tsid_->data(), robot_->model().getFrameId(frame_name));
            }
            double cost(const std::shared_ptr<tsid::tasks::TaskBase>& task) const
            {
                assert(task);
                return (task->getConstraint().matrix() * ddq_ - task->getConstraint().vector()).norm();
            }
            virtual double cost(const std::string& task_name) const = 0;

            void set_verbose(bool b) { verbose_ = b; }
            bool verbose() const { return verbose_; }

            void save_configuration(const std::string& config_name, const std::string& robot_name = "robot") const;
            virtual void set_behavior_type(const std::string& bt);
            const std::string& behavior_type() const { return behavior_type_; }
            const std::string& urdf() const { return urdf_; }
            const std::string& floating_base_joint_name() const { return floating_base_joint_name_; }

            //check if pinocchio model is colliding
            inria_wbc::utils::CollisionCheck collision_check() { return collision_check_; }
            bool is_model_colliding() { return is_model_colliding_; }
            void set_send_cmd(const bool send_cmd) { send_cmd_ = send_cmd; };

            Eigen::VectorXd q_solver(bool filter_mimics = true) const;
            const void qp_step_back(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const pinocchio::Data& data);
            const void qp_step_back() { qp_step_back(q_tsid_prev_, v_tsid_prev_, *data_prev_); };

        private:
            std::vector<int> get_non_mimics_indexes() const;

        protected:
            void _reset();
            // you can use q_tsid_ and v_tsid_ for open-loop control
            void _solve(const Eigen::VectorXd& q, const Eigen::VectorXd& dq);
            void _solve() { _solve(q_tsid_, v_tsid_); }

            bool verbose_ = false;
            double t_;
            double dt_;
            bool floating_base_;
            std::string base_path_;
            std::string urdf_;
            std::string floating_base_joint_name_;

            // true if we close the loop with actuator position/vel
            // and floating base position
            bool _closed_loop = false;

            std::string behavior_type_;

            std::string fb_joint_name_; //name of the floating base joint
            std::vector<std::string> mimic_dof_names_;
            std::vector<std::string> tsid_joint_names_; //contain floating base and mimics
            std::vector<int> non_mimic_indexes_;
            std::vector<std::string> activated_contacts_;
            std::vector<std::string> all_contacts_;

            //---- TSID conventions for the floating base: quaternion
            tsid::math::Vector q_tsid_; // tsid joint positions
            tsid::math::Vector v_tsid_; // tsid joint velocities
            tsid::math::Vector a_tsid_; // tsid joint accelerations
            tsid::math::Vector tau_tsid_; // tsid joint torques
            tsid::math::Vector momentum_; // momentum
            std::unordered_map<std::string, tsid::math::Vector> activated_contacts_forces_; //tsid contact forces of the activated contacts

            tsid::math::Vector q_tsid_prev_; // latest sent tsid joint positions 
            tsid::math::Vector v_tsid_prev_; // latest sent tsid joint positions 
            std::shared_ptr<pinocchio::Data> data_prev_; // latest sent  pinocchio data 

            //---- Dart conventions for the floating base: axis-angle
            Eigen::VectorXd q0_; // tsid joint positions resized for dart
            Eigen::VectorXd q_; // tsid joint positions resized for dart
            Eigen::VectorXd dq_; // tsid joint velocities resized for dart
            Eigen::VectorXd ddq_; // tsid joint accelerations resized for dart
            Eigen::VectorXd tau_; // tsid joint torques resized for dart

            std::shared_ptr<tsid::robots::RobotWrapper> robot_;
            std::shared_ptr<tsid::InverseDynamicsFormulationAccForce> tsid_;
            std::shared_ptr<tsid::solvers::SolverHQPBase> solver_;

            inria_wbc::utils::CollisionCheck collision_check_;
            bool check_model_collisions_;
            bool is_model_colliding_ = false;
            bool send_cmd_ = true;
            Eigen::VectorXd q_solver_; //q computed by the qp solver even when not sent (with send_cmd_ == false)

            std::string solver_to_use_;
        };

        using Factory = utils::Factory<Controller, YAML::Node>;
        template <typename T>
        using Register = Factory::AutoRegister<T>;
    } // namespace controllers
} // namespace inria_wbc
#endif
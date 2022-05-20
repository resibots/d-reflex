#include "inria_wbc/behaviors/hands.hpp"

namespace inria_wbc {
    namespace behaviors {
        static Register<Hands> __ex_behavior("hands");

        Hands::Hands(const controller_ptr_t& controller, const YAML::Node& config) : Behavior(controller, config)
        {
            //////////////////// DEFINE COM TRAJECTORIES  //////////////////////////////////////
            traj_selector_ = 0;
            auto tracker = std::dynamic_pointer_cast<inria_wbc::controllers::PosTracker>(controller_);
            IWBC_ASSERT(tracker, "we need a pos tracker here");
            behavior_type_ = this->behavior_type();
            controller_->set_behavior_type(behavior_type_);

            auto lh_init = tracker->get_se3_ref("lh");
            auto rh_init = tracker->get_se3_ref("rh");

            YAML::Node c = IWBC_CHECK(config["BEHAVIOR"]);
            auto trajectory_duration = IWBC_CHECK(c["trajectory_duration"].as<float>());
            auto rh_shift = IWBC_CHECK(c["rh_shift"].as<std::vector<float>>());
            auto lh_shift = IWBC_CHECK(c["lh_shift"].as<std::vector<float>>());
            IWBC_ASSERT(lh_shift.size() == 6, "the left hand motion should be pos rel and rot abs [x, y, z, roll, pitch, yaw]");
            IWBC_ASSERT(rh_shift.size() == 6, "the right hand motion should be abs [x, y, z, roll, pitch, yaw]");
            reflex_contact_pos_ = IWBC_CHECK(c["reflex_contact_pos"].as<std::vector<float>>());
            reflex_speed_ = IWBC_CHECK(c["reflex_speed"].as<float>());
            IWBC_ASSERT(reflex_contact_pos_.size() == 3, "the reflex contact pos should be abs [x, y, z]");

            // left hand 
            auto lh_final = lh_init;
            lh_final.translation()(0) += lh_shift[0];
            lh_final.translation()(1) += lh_shift[1];
            lh_final.translation()(2) += lh_shift[2];
            auto euler = lh_final.rotation().eulerAngles(0, 1, 2);
            euler[0] += lh_shift[3];
            euler[1] += lh_shift[4];
            euler[2] += lh_shift[5];
            auto q = Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ());
            lh_final.rotation() = q.toRotationMatrix();
            lh_trajectories_.push_back(trajectory_handler::compute_traj(lh_init, lh_final, controller_->dt(), trajectory_duration));
            lh_trajectories_.push_back(trajectory_handler::compute_traj(lh_final, lh_init, controller_->dt(), trajectory_duration));
            lh_current_trajectory_ = lh_trajectories_[traj_selector_];

            // right hand 
            auto rh_final = rh_init;
            rh_final.translation()(0) += rh_shift[0];
            rh_final.translation()(1) += rh_shift[1];
            rh_final.translation()(2) += rh_shift[2];

            auto euler2 = rh_final.rotation().eulerAngles(0, 1, 2);
            euler2[0] += rh_shift[3];
            euler2[1] += rh_shift[4];
            euler2[2] += rh_shift[5];
            auto q2 = Eigen::AngleAxisd(euler2[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(euler2[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(euler2[2], Eigen::Vector3d::UnitZ());
            rh_final.rotation() = q2.toRotationMatrix();

            rh_trajectories_.push_back(trajectory_handler::compute_traj(rh_init, rh_final, controller_->dt(), trajectory_duration));
            rh_trajectories_.push_back(trajectory_handler::compute_traj(rh_final, rh_init, controller_->dt(), trajectory_duration));
            rh_current_trajectory_ = rh_trajectories_[traj_selector_];

            // reflex_trajectory

        }

        void Hands::activate_reflex(){
            use_reflex_ = true; 
            /*
            auto rh_init = controller_->model_frame_pos("gripper_right_base_link");
            auto rh_final = rh_init;

            rh_final.translation()(0) = reflex_contact_pos_[0];
            rh_final.translation()(1) = reflex_contact_pos_[1];
            rh_final.translation()(2) = reflex_contact_pos_[2];

            auto reflex_duration = (rh_final.translation()-rh_init.translation()).norm()/reflex_speed_;

            auto euler2 = rh_final.rotation().eulerAngles(0, 1, 2);
            euler2[0] = -1.57;
            euler2[1] = 0.;
            euler2[2] = 0.;
            auto q2 = Eigen::AngleAxisd(euler2[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(euler2[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(euler2[2], Eigen::Vector3d::UnitZ());
            rh_final.rotation() = q2.toRotationMatrix();

            reflex_trajectory_ = trajectory_handler::compute_traj(rh_init, rh_final, controller_->dt(), reflex_duration);
            */
        }

        void Hands::update(const controllers::SensorData& sensor_data)
        {
            
            if (!use_reflex_){
                auto rh_ref = rh_current_trajectory_[time_];
                auto lh_ref = lh_current_trajectory_[time_];
                std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller_)->set_se3_ref(lh_ref, "lh");
                std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller_)->set_se3_ref(rh_ref, "rh");
            }

            controller_->update(sensor_data);
            time_++;
            if (time_ == lh_current_trajectory_.size()) {
                time_ = 0;
                traj_selector_ = ++traj_selector_ % lh_trajectories_.size();
                lh_current_trajectory_ = lh_trajectories_[traj_selector_];
                rh_current_trajectory_ = rh_trajectories_[traj_selector_];

            }
        }

    } // namespace behaviors
} // namespace inria_wbc
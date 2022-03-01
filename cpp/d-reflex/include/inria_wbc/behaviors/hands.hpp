#ifndef IWBC_EX_BEHAVIOR_HPP
#define IWBC_EX_BEHAVIOR_HPP
#include <chrono>
#include <iostream>
#include <signal.h>

#include <inria_wbc/behaviors/behavior.hpp>
#include <inria_wbc/controllers/pos_tracker.hpp>
#include <inria_wbc/utils/trajectory_handler.hpp>

namespace inria_wbc {
    namespace behaviors {
        class Hands : public Behavior {
        public:
            Hands(const controller_ptr_t& controller, const YAML::Node& config);
            Hands() = delete;
            Hands(const Hands&) = delete;

            void update(const controllers::SensorData& sensor_data) override;
            void activate_reflex();
            virtual ~Hands() {}
            std::string behavior_type() const override { return controllers::behavior_types::DOUBLE_SUPPORT; };
        private:
            int time_ = 0;
            int reflex_time_ = 0;
            int traj_selector_ = 0;
            bool use_reflex_ = false; 
            float reflex_speed_;
            std::vector<float> reflex_contact_pos_;
            std::vector<std::vector<pinocchio::SE3>> lh_trajectories_;
            std::vector<pinocchio::SE3> lh_current_trajectory_;
            std::vector<std::vector<pinocchio::SE3>> rh_trajectories_;
            std::vector<pinocchio::SE3> rh_current_trajectory_;
            std::vector<pinocchio::SE3> reflex_trajectory_;
        };
    } // namespace behaviors
} // namespace inria_wbc
#endif
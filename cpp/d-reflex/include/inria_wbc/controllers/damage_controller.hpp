#ifndef IWBC_DAMAGE_CONTROLLER_HPP
#define IWBC_DAMAGE_CONTROLLER_HPP

#include <inria_wbc/controllers/talos_pos_tracker.hpp>

namespace inria_wbc {
    namespace tasks {
        // contacts cannot be in the same factory
        std::shared_ptr<tsid::contacts::Contact6dExt> make_contact_task(
            const std::shared_ptr<tsid::robots::RobotWrapper>& robot,
            const std::shared_ptr<tsid::InverseDynamicsFormulationAccForce>& tsid,
            const std::string& task_name, const YAML::Node& node, const YAML::Node& controller_node);
    }

    namespace controllers {

        class DamageController : public TalosPosTracker {
        public:
            typedef tsid::math::Vector Vector;
            DamageController(const YAML::Node& config);
            DamageController(const DamageController& other) = delete;
            DamageController& operator=(const DamageController& o) const = delete;
            virtual ~DamageController(){};
            Eigen::VectorXd tau(bool filter_mimics) const;
            virtual void update(const SensorData& sensor_data = {}) override;
            // joint weight
            void set_joint_weight_weights(const Vector& weights); 
            std::shared_ptr<tsid::contacts::Contact6dExt> contact_task(const std::string& str) { return contact(str); }
        protected:
            virtual void parse_configuration(const YAML::Node& config);
        };

    } // namespace controllers
} // namespace inria_wbc
#endif
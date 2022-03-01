#include "inria_wbc/behaviors/talos_reaching_low_online.hpp"
#include "inria_wbc/controllers/avoidance_controller.hpp"
#define LOG_REACHING_LOW_ONLINE

Eigen::MatrixXd load(const std::string& path, size_t cols)
                    
{
    std::ifstream fstr(path.c_str());
    std::vector<double> data = std::vector<double>{
        std::istream_iterator<double>(fstr),
        std::istream_iterator<double>()};
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(&data.data()[0], data.size() / cols, cols);
}

namespace inria_wbc {
    namespace behaviors {
        namespace humanoid {
            static Register<TalosReachingLowOnline> __talos_reaching_low_online("talos-reaching-low-avoidance-online");

            TalosReachingLowOnline::TalosReachingLowOnline(const controller_ptr_t& controller, const YAML::Node& config) : Behavior(controller, config)
            {
                //////////////////// DEFINE COM TRAJECTORIES  //////////////////////////////////////
                traj_selector_ = 0;
                robot_ = controller_->robot();
                tsid_ = controller_->tsid();
            
                behavior_type_ = this->behavior_type();
                controller_->set_behavior_type(behavior_type_);

                auto tasks = config["CONTROLLER"]["tasks"].as<std::string>();
                trajectory_duration_ = config["BEHAVIOR"]["trajectory_duration"].as<float>();
                std::vector<float> rh_motion = config["BEHAVIOR"]["rh_motion"].as<std::vector<float>>();
                std::vector<float> lh_motion = config["BEHAVIOR"]["lh_motion"].as<std::vector<float>>();
                use_gp_cartesian_avoidance_ = config["PARAMS"]["use_gp_cartesian_avoidance"].as<bool>();
                use_gp_joint_avoidance_ = config["PARAMS"]["use_gp_joint_avoidance"].as<bool>();
                use_gp_com_avoidance_ = config["PARAMS"]["use_gp_com_avoidance"].as<bool>();
                add_sample_frequency_ =  config["PARAMS"]["add_sample_frequency"].as<int>();
                log_level_ =  config["PARAMS"]["log_level"].as<int>();
                t_damage_ = config["PARAMS"]["time_damage_application"].as<float>();

                // right gripper trajectory 
                auto rh_init = std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller_)->get_se3_ref("rh");
                auto rh_final = rh_init;
                rh_final.translation()(0) += rh_motion[0];
                rh_final.translation()(1) += rh_motion[1];
                rh_final.translation()(2) += rh_motion[2];
                rh_trajectories_.push_back(trajectory_handler::compute_traj(rh_init, rh_final, controller_->dt(), trajectory_duration_));
                rh_trajectories_.push_back(trajectory_handler::compute_traj(rh_final, rh_init, controller_->dt(), trajectory_duration_));
                rh_current_trajectory_ = rh_trajectories_[traj_selector_];
                
                // left gripper trajectory 
                auto lh_init = std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller_)->get_se3_ref("lh");
                auto lh_final = lh_init;
                lh_final.translation()(0) += lh_motion[0];
                lh_final.translation()(1) += lh_motion[1];
                lh_final.translation()(2) += lh_motion[2];
                lh_trajectories_.push_back(trajectory_handler::compute_traj(lh_init, lh_final, controller_->dt(), trajectory_duration_));
                lh_trajectories_.push_back(trajectory_handler::compute_traj(lh_final, lh_final, controller_->dt(), 10.));
                lh_current_trajectory_ = lh_trajectories_[traj_selector_];
                std::cout << "lh_init:  " << lh_init.translation().transpose() << std::endl; 
                std::cout << "lh_final:  " << lh_final.translation().transpose() << std::endl; 

                // CoM trajectory
                //auto lf_low = controller_->model_joint_pos("leg_left_6_joint");
                auto rf_low = controller_->model_joint_pos("leg_right_6_joint");
                float alpha = config["BEHAVIOR"]["com_ratio"].as<float>();
                float z = config["BEHAVIOR"]["z"].as<float>();
                time_before_moving_com_ = (int)(config["BEHAVIOR"]["time_before_moving_com"].as<float>()*1000);
                com_init_ = controller_->com();
                //Eigen::VectorXd com_lf = lf_low.translation();
                Eigen::VectorXd com_rf = rf_low.translation();
                std::cout << "init: " << com_init_.transpose() << std::endl; 
                //std::cout << "com_lf: " << com_lf.transpose() << std::endl; 
                std::cout << "com_rf: " << com_rf.transpose() << std::endl; 
                //com_lf(2) = com_init_(2) + z;
                com_rf(2) = com_rf(2) + z;
                com_final_ = com_init_; 
                if (alpha>=0){
                    com_final_ = (1-alpha) * com_init_ + alpha * com_rf; 
                } /*else {
                    com_final_ = (1+alpha) * com_init_ - alpha * com_lf; 
                }*/
                float com_trajectory_duration = config["BEHAVIOR"]["com_trajectory_duration"].as<float>();
                auto com_trajectory = trajectory_handler::compute_traj(com_init_, com_final_, controller_->dt(), com_trajectory_duration);
                
                
                /* too add velocity and acceleration */
                Eigen::VectorXd pos0 = com_trajectory.at(0);
                Eigen::VectorXd vel0 = Eigen::VectorXd::Zero(3);

                for (uint i=1; i<com_trajectory.size(); i++){
                    tsid::trajectories::TrajectorySample ts(3);
                    ts.pos = com_trajectory.at(i);
                    ts.vel = (com_trajectory.at(i) - pos0) / controller_->dt() ;
                    ts.acc = (ts.vel - vel0) / controller_->dt();
                    com_trajectory_.push_back(ts);
                    pos0 = ts.pos;
                    vel0 = ts.vel; 
                }/**/

                if (use_gp_com_avoidance_){
                    Eigen::VectorXd sample(3);
                    sample(2) = com_init_(2);
                    for (double dx = -0.1; dx < 0.1; dx += 0.01){
                        for (double dy = -0.1; dy < 0.1; dy += 0.01){
                            sample(0) = com_init_(0) + dx;
                            sample(1) = com_init_(1) + dy;
                            com_samples_.push_back(sample);
                        }
                    }
                }

                if (use_gp_joint_avoidance_){
                    // Initialized the mask (size 32)
                    mask_ = Eigen::VectorXd::Zero(32);
                    
                    std::vector<bool> body_parts = config["PARAMS"]["body_parts"].as<std::vector<bool>>();
                    assert( body_parts.size() == 6);  // 0:L-Leg, 1:R-Leg, 2:Torso, 3:L-Arm, 4:R-Arm, 5:Head 
                    if (body_parts.at(0)) {
                        printf("use Left Leg\n");
                        for(uint i=0; i<6; i++) mask_(i) = 1;
                    } else printf("ignore Left Leg\n");
                    if (body_parts.at(1)) {
                        printf("use Right Leg\n");
                        for(uint i=6; i<12; i++) mask_(i) = 1;
                    } else printf("ignore Right Leg\n");
                    if (body_parts.at(2)) {
                        printf("use Torso\n");
                        for(uint i=12; i<14; i++) mask_(i) = 1;
                    } else printf("ignore Torso\n");
                    if (body_parts.at(3)) {
                        printf("use Left Arm\n");
                        for(uint i=14; i<22; i++) mask_(i) = 1;
                    } else printf("ignore Left Arm\n");
                    if (body_parts.at(4)) {
                        printf("use Right Arm\n");
                        for(uint i=22; i<30; i++) mask_(i) = 1;
                    } else printf("ignore Right Arm\n");
                    if (body_parts.at(5)) {
                        printf("use Head\n");
                        for(uint i=30; i<32; i++) mask_(i) = 1;
                    } else printf("ignore Head\n");
                    // Load default joint trajectory and evaluate the initial gp prediction 
                    /*
                    Eigen::MatrixXd baseline_default_data = load("/home/pal/humanoid_adaptation/data/baseline/default/normalized_32joint.dat", 32);
                    baseline_default_.empty();
                    for (uint i=0; i< baseline_default_data.rows(); i++ ){
                        baseline_default_.push_back(baseline_default_data.row(i));
                    }
                    std::vector<Eigen::VectorXd> samples;
                    samples.empty();
                    std::vector<Eigen::VectorXd> observations;
                    observations.empty();
                    set_gp_data(samples, observations);
                    Eigen::VectorXd mean_predictions;
                    Eigen::VectorXd variance_predictions;
                    std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->get_gp_avoidance_predictions(baseline_default_, mean_predictions, variance_predictions);
                    static std::ofstream ofs_mean_prediction(xp_folder_ +"/mean_prediction.dat");
                    static std::ofstream ofs_variance_prediction(xp_folder_ +"/variance_prediction.dat");
                    ofs_mean_prediction << mean_predictions.transpose() << std::endl;  
                    ofs_variance_prediction << variance_predictions.transpose() << std::endl;  
                    */
                }
            }

            void TalosReachingLowOnline::set_gp_joint_data(const std::vector<Eigen::VectorXd> &samples, const std::vector<Eigen::VectorXd> &observations){
                auto avoidance_controller = std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_);
                assert(samples.size() == observations.size());
                avoidance_controller->set_gp_avoidance_mask(mask_);
                avoidance_controller->set_gp_avoidance_points(samples, observations);
            }

            void TalosReachingLowOnline::set_gp_cartesian_data(const std::vector<Eigen::VectorXd> &samples, const std::vector<Eigen::VectorXd> &observations){
                auto avoidance_controller = std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_);
                assert(samples.size() == observations.size());
                avoidance_controller->set_gp_cartesian_avoidance_mask(mask_);
                avoidance_controller->set_gp_cartesian_avoidance_points(samples, observations);
            }

            void TalosReachingLowOnline::set_gp_com_data(const std::vector<Eigen::VectorXd> &samples, const std::vector<Eigen::VectorXd> &observations){
                auto avoidance_controller = std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_);
                assert(samples.size() == observations.size());
                avoidance_controller->set_gp_com_avoidance_mask(mask_);
                avoidance_controller->set_gp_com_avoidance_points(samples, observations);
            }

            void TalosReachingLowOnline::add_gp_point(const Eigen::VectorXd &sample, const Eigen::VectorXd& observation){
                assert(sample.size() == 32);
                assert(observation.size() == 1);
                std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->add_gp_avoidance_point(sample, observation);
            }

            void TalosReachingLowOnline::set_xp_folder(const std::string& xp_folder){
                xp_folder_ = xp_folder;
                if (use_gp_cartesian_avoidance_){
                    std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->set_gp_cartesian_avoidance_xp_folder(xp_folder);
                } else if (use_gp_com_avoidance_){
                    std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->set_gp_com_avoidance_xp_folder(xp_folder);
                } else if (use_gp_joint_avoidance_){
                    std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->set_gp_avoidance_xp_folder(xp_folder);
                }
            }

            void TalosReachingLowOnline::set_gp_avoidance_joint_limits(const Eigen::VectorXd& lower_limits, const Eigen::VectorXd& upper_limits){
                /*static std::ofstream ofs_limits_behavior(xp_folder_ +"/limits_behavior.dat");
                ofs_limits_behavior << lower_limits.transpose() << std::endl;   
                ofs_limits_behavior << upper_limits.transpose() << std::endl; */
                std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->set_gp_avoidance_joint_limits(lower_limits, upper_limits);
            }

            void TalosReachingLowOnline::set_frozen(bool frozen){
                frozen_ = frozen;
            }
            
            void TalosReachingLowOnline::set_started(bool started){
                started_ = started;
            }

            void TalosReachingLowOnline::set_backward(bool backward){
                backward_ = backward;
                if (use_gp_cartesian_avoidance_){
                    if (backward){
                        std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->set_gp_cartesian_avoidance_weight(0.);
                        std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->reset_gp_cartesian_avoidance();
                    } else {
                        std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->set_gp_cartesian_avoidance_weight(1.);
                    }
                } else {
                    if (backward){
                        std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->set_gp_avoidance_weight(0.);
                        std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->reset_gp_avoidance();
                    } else {
                        std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->set_gp_avoidance_weight(1.);
                    }
                }
            }

            Eigen::VectorXd TalosReachingLowOnline::get_pos(std::string name){
                return robot_->position(tsid_->data(), robot_->model().getJointId(name)).translation();
            }

            void TalosReachingLowOnline::get_gp_com_prediction(){
                static bool saved_samples = false;
                static std::ofstream ofs_com_samples(xp_folder_ +"/com_samples.dat");
                static std::ofstream ofs_com_prediction(xp_folder_ +"/com_prediction.dat");
                for (uint i =0; i< com_samples_.size(); i++){
                    ofs_com_prediction <<  std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->get_gp_com_avoidance_C(com_samples_.at(i)) << " ";  

                    if (!saved_samples){
                        ofs_com_samples << com_samples_.at(i).transpose() << std::endl;
                    }
                }
                ofs_com_prediction << std::endl;
                if (!saved_samples){
                    saved_samples = true;
                }
            }

            void TalosReachingLowOnline::update(const controllers::SensorData& sensor_data)
            {
                static int counter = 0;
                auto last_rh = rh_current_trajectory_[time_];
                auto last_lh = lh_current_trajectory_[time_];
                
                // move the com 
                static bool placed_com = false;
                
                if ((time_ < com_trajectory_.size()+time_before_moving_com_) & !placed_com){
                    if (time_ < time_before_moving_com_){
                        std::static_pointer_cast<inria_wbc::controllers::TalosPosTracker>(controller_)->set_com_ref(com_init_);
                    } else {   
                        std::static_pointer_cast<inria_wbc::controllers::TalosPosTracker>(controller_)->set_com_ref(com_trajectory_.at(time_-time_before_moving_com_));
                    }
                } else {
                    if (!placed_com){
                        placed_com = true;
                    }
                    std::static_pointer_cast<inria_wbc::controllers::TalosPosTracker>(controller_)->set_com_ref(com_final_);
                }
                
                //std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller_)->set_se3_ref(last_rh, "rh");
                std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller_)->set_se3_ref(last_lh, "lh");

    #ifdef LOG_REACHING_LOW_ONLINE
                {
                    if (started_ && log_level_>0){
                        //static std::ofstream ofs_com(xp_folder_ +"/com.dat");
                        static std::ofstream ofs_rh_tsid(xp_folder_ +"/rh_tsid.dat");
                        static std::ofstream ofs_rh_ref(xp_folder_ +"/rh_ref.dat");
                        static std::ofstream ofs_lh_tsid(xp_folder_ +"/lh_tsid.dat");
                        static std::ofstream ofs_lh_ref(xp_folder_ +"/lh_ref.dat");
                        //static std::ofstream ofs_q(xp_folder_ +"/q_tsid.dat");
                        //ofs_com << controller_->com().transpose() << std::endl;
                        ofs_rh_tsid << robot_->position(tsid_->data(), robot_->model().getJointId("gripper_right_joint")).translation().transpose() << std::endl;
                        ofs_rh_ref << last_rh.translation().transpose() << std::endl;
                        ofs_lh_tsid << robot_->position(tsid_->data(), robot_->model().getJointId("gripper_left_joint")).translation().transpose() << std::endl;
                        ofs_lh_ref << last_lh.translation().transpose() << std::endl;
                        //ofs_q << controller_->q(true).transpose() << std::endl;      
                    }
                }
    #endif
                static bool detach = false;
                if (!detach && (time_ > t_damage_*1000)) {
                    detach = true;
                    //std::static_pointer_cast<inria_wbc::controllers::TalosPosTracker>(controller_)->remove_contact("contact_lfoot");
                }
                controller_->update(sensor_data);

    #ifdef LOG_REACHING_LOW_ONLINE
                {
                    auto avoidanceController = std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_);
                    if (started_ && log_level_>0){
                        static std::ofstream ofs_C(xp_folder_ +"/C.dat");
                        if (use_gp_cartesian_avoidance_){
                            auto C = avoidanceController->get_gp_cartesian_avoidance_C();
                            ofs_C << C << std::endl;
                        } else if (use_gp_com_avoidance_){
                            auto C = avoidanceController->get_gp_com_avoidance_C();
                            if ((time_ % 10) == 0){
                                get_gp_com_prediction();
                            }
                        } else if (use_gp_joint_avoidance_){
                            auto C = avoidanceController->get_gp_avoidance_C();
                            ofs_C << C << std::endl;
                            /*
                            if (counter++ % add_sample_frequency_ == 0){
                                Eigen::VectorXd mean_predictions;
                                Eigen::VectorXd variance_predictions;
                                std::static_pointer_cast<inria_wbc::controllers::AvoidanceController>(controller_)->get_gp_avoidance_predictions(baseline_default_, mean_predictions, variance_predictions);
                                static std::ofstream ofs_mean_prediction(xp_folder_ +"/mean_prediction.dat");
                                static std::ofstream ofs_variance_prediction(xp_folder_ +"/variance_prediction.dat");
                                ofs_mean_prediction << mean_predictions.transpose() << std::endl;  
                                ofs_variance_prediction << variance_predictions.transpose() << std::endl;  
                            }*/
                        }
                    }
                }
    #endif
                if (!frozen_){
                    if (!backward_){
                        time_++;
                    } else {
                        if (time_>0){
                            time_--;
                        }
                    }
                }

                if (!backward_ && time_ == lh_current_trajectory_.size()) {
                    time_ = 0;
                    traj_selector_ = ++traj_selector_ % lh_trajectories_.size();
                    //rh_current_trajectory_ = rh_trajectories_[traj_selector_];
                    lh_current_trajectory_ = lh_trajectories_[traj_selector_];
                    std::static_pointer_cast<inria_wbc::controllers::TalosPosTracker>(controller_)->add_contact("contact_lh");
                }
                if (backward_ && time_ == -1){
                    traj_selector_ = --traj_selector_ % rh_trajectories_.size();
                    rh_current_trajectory_ = rh_trajectories_[traj_selector_];
                    lh_current_trajectory_ = lh_trajectories_[traj_selector_];
                    time_ = rh_current_trajectory_.size()-1;
                }
            }
        } //namespace humanoid 
    } // namespace behaviors
} // namespace inria_wbc
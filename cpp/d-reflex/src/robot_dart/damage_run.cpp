#include <algorithm>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <signal.h>
#include <ctime> 

#include <boost/filesystem.hpp>
#include <sys/stat.h> 
#include <dart/dynamics/BodyNode.hpp>

#include <robot_dart/control/pd_control.hpp>
#include <robot_dart/robot.hpp>
#include <robot_dart/robot_dart_simu.hpp>
#include <robot_dart/sensor/force_torque.hpp>
#include <robot_dart/sensor/imu.hpp>
#include <robot_dart/sensor/torque.hpp>

#ifdef GRAPHIC
#include <robot_dart/gui/magnum/graphics.hpp>
#include <robot_dart/gui/magnum/windowless_graphics.hpp>
#endif

#include "inria_wbc/behaviors/behavior.hpp"
#include "inria_wbc/behaviors/hands.hpp"
#include "inria_wbc/controllers/pos_tracker.hpp"
#include "inria_wbc/controllers/talos_pos_tracker.hpp"
#include "inria_wbc/controllers/damage_controller.hpp"
#include "inria_wbc/exceptions.hpp"
#include "inria_wbc/robot_dart/cmd.hpp"
#include "inria_wbc/robot_dart/damages.hpp"
#include "inria_wbc/robot_dart/external_collision_detector.hpp"
#include "inria_wbc/robot_dart/self_collision_detector.hpp"
#include "inria_wbc/robot_dart/utils.hpp"
#include "inria_wbc/trajs/saver.hpp"
#include "inria_wbc/utils/timer.hpp"
#include "tsid/tasks/task-self-collision.hpp"

#include <dart/dynamics/BodyNode.hpp>
#include <dart/constraint/ConstraintSolver.hpp>
#include <dart/collision/CollisionObject.hpp>

static const std::string red = "\x1B[31m";
static const std::string rst = "\x1B[0m";
static const std::string bold = "\x1B[1m";

enum State {Failure, Success, Fallen};

// header 
void write_timestate(State state, std::string xp_folder, double time);
bool create_folder(const char *path);
std::string fixed_decimal(int value);
std::string create_xp_folder();
double median(std::vector<double> &v);

double median(std::vector<double> &v)
{
  if(v.empty()) {
    return 0.0;
  }
  auto n = v.size() / 2;
  nth_element(v.begin(), v.begin()+n, v.end());
  auto med = v[n];
  if(!(v.size() & 1)) { //If the set size is even
    auto max_it = max_element(v.begin(), v.begin()+n);
    med = (*max_it + med) / 2.0;
  }
  return med;    
}

void write_timestate(State state, std::string xp_folder, double time){
    static std::ofstream ofs_detection(xp_folder +"/end.dat");
    ofs_detection << time << " " << static_cast<int>(state) << std::endl;
}

bool create_folder(const char *path){
    if( boost::filesystem::exists(path)){
        if (boost::filesystem::is_directory(path)){
            return true;
        } else {
            std::cerr << "Error: exists but isn't a directory" << std::endl;
            return false;
        }
    } else {
        if (mkdir(path, 0777) == -1) {
        std::cerr << "Error :  " << strerror(errno) << std::endl; 
        return false;
        } else {
            return true;
        }   
    }
}

std::string fixed_decimal(int value){
    if (value<10){
        return "0"+std::to_string(value);
    } else {
        return std::to_string(value);
    }
}

std::string create_xp_folder(){
    std::string data_path = "../data";
    if (!create_folder(data_path.c_str())){
        std::cerr << "Error creating folder " << data_path << std::endl;
        return "error";
    }
    time_t now = time(0);
    tm *ltm = localtime(&now);
    // create year folder
    std::string year = std::to_string(1900 + ltm->tm_year);
    data_path = data_path + "/"+ year; 
    if (!create_folder(data_path.c_str())){
        std::cerr << "Error creating folder " << data_path << std::endl;
        return "error";
    }
    // create month folder
    data_path = data_path + "/"+ fixed_decimal(1 + ltm->tm_mon); 
    if (!create_folder(data_path.c_str())){
        std::cerr << "Error creating folder " << data_path << std::endl;
        return "error";
    }
    // create day folder
    data_path = data_path + "/"+ fixed_decimal(ltm->tm_mday); 
    if (!create_folder(data_path.c_str())){
        std::cerr << "Error creating folder " << data_path << std::endl;
        return "error";
    }
    // create exp folder
    std::string timestamp = fixed_decimal(2 + ltm->tm_hour) + ":" + fixed_decimal(ltm->tm_min) + ":" + fixed_decimal(ltm->tm_sec);
    data_path = data_path + "/"+ timestamp; 
    if (!create_folder(data_path.c_str())){
        std::cerr << "Error creating folder " << data_path << std::endl;
        return "error";
    }
    return data_path;
}

inline Eigen::VectorXd compute_spd(const std::shared_ptr<::robot_dart::Robot>& robot, const Eigen::VectorXd& targetpos, double dt, const std::vector<std::string>& joints, bool floating_base = true, float arm_stiffness = 10000)
{
    Eigen::VectorXd q = robot->positions(joints);
    Eigen::VectorXd dq = robot->velocities(joints);
    float stiffness = 10000;
    float damping = 100;
    int ndofs = joints.size();
    Eigen::MatrixXd Kp = Eigen::MatrixXd::Identity(ndofs, ndofs);
    Eigen::MatrixXd Kd = Eigen::MatrixXd::Identity(ndofs, ndofs);
    
    for (std::size_t i = 0; i < ndofs; ++i) {
        if (joints.at(i).substr(0,9) == "arm_right") {
            Kp(i, i) = arm_stiffness;
        } else {
            Kp(i, i) = stiffness;
        } 
        Kd(i, i) = damping;
    }

    if (robot->free() && floating_base) // floating base
        for (std::size_t i = 0; i < 6; ++i) {
            Kp(i, i) = 0;
            Kd(i, i) = 0;
        }

    Eigen::MatrixXd invM = (robot->mass_matrix(joints) + Kd * dt).inverse();
    Eigen::VectorXd p = -Kp * (q + dq * dt - targetpos);
    Eigen::VectorXd d = -Kd * dq;
    Eigen::VectorXd qddot = invM * (-robot->coriolis_gravity_forces(joints) + p + d + robot->constraint_forces(joints));
    Eigen::VectorXd commands = p + d - Kd * qddot * dt;
    return commands;
}

int main(int argc, char* argv[])
{
    try {
        // program options
        namespace po = boost::program_options;
        po::options_description desc("Test_controller options");
        // clang-format off
        desc.add_options()
        ("actuators,a", po::value<std::string>()->default_value("spd"), "actuator model torque/velocity/servo/spd  [default:spd]")
        ("behavior,b", po::value<std::string>()->default_value("/home/pal/humanoid_adaptation/etc/behaviors/hands.yaml"), "Configuration file of the tasks (yaml) [default: ../etc/talos/talos_squat.yaml]")
        ("check_self_collisions", "check the self collisions (print if a collision)")
        ("check_fall", "check if the robot has fallen (print if a collision)")
        ("collision,k", po::value<std::string>()->default_value("fcl"), "collision engine [default:fcl]")
        ("collisions", po::value<std::string>(), "display the collision shapes for task [name]")
        ("controller,c", po::value<std::string>()->default_value("/home/pal/humanoid_adaptation/etc/damage_controller.yaml"), "Configuration file of the tasks (yaml) [default: ../etc/talos/talos_pos_tracker.yaml]")
        ("damage", po::value<bool>()->default_value(false), "damage talos")
        ("enforce_position,e", po::value<bool>()->default_value(true), "enforce the positions of the URDF [default:true]")
        ("fast,f", "fast (simplified) Talos [default: false]")
        ("srdf,s", po::value<float>()->default_value(0.0), "save the configuration at the specified time")
        ("ghost,g", "display the ghost (Pinocchio model)")
        ("closed_loop", "Close the loop with floating base position and joint positions; required for torque control [default: from YAML file]")
        ("help,h", "produce help message")
        ("height", po::value<bool>()->default_value(false), "print total feet force data to adjust height in config")
        ("mp4,m", po::value<std::string>(), "save the display to a mp4 video [filename]")
        ("push,p", po::value<std::vector<float>>(), "push the robot at t=x1 0.25 s")
        ("norm_force,n", po::value<float>()->default_value(-150) , "push norm force value")
        ("save_traj,S", po::value<std::vector<std::string>>()->multitoken(), "save the trajectory in dir <dir> for references <refs>: -S traj1 rh lh com")
        ("windowless,w", po::value<bool>()->default_value(false), "activate windowless graphics")
        ("log,l", po::value<std::vector<std::string>>()->default_value(std::vector<std::string>(),""), 
            "log the trajectory of a dart body [with urdf names] or timing or CoM or cost, example: -l timing -l com -l lf -l cost_com -l cost_lf")
        ;
        // clang-format on
        po::variables_map vm;
        try {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            po::notify(vm);
        }
        catch (po::too_many_positional_options_error& e) {
            // A positional argument like `opt2=option_value_2` was given
            std::cerr << e.what() << std::endl;
            std::cerr << desc << std::endl;
            return 1;
        }
        catch (po::error_with_option_name& e) {
            // Another usage error occurred
            std::cerr << e.what() << std::endl;
            std::cerr << desc << std::endl;
            return 1;
        }

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        // clang-format off
        std::cout<< "------ CONFIGURATION ------" << std::endl;
        std::ostringstream oss_conf;
        for (const auto& kv : vm){
            oss_conf << kv.first << " ";
            try { oss_conf << kv.second.as<std::string>();
            } catch(...) {/* do nothing */ }
            try { oss_conf << kv.second.as<bool>();
            } catch(...) {/* do nothing */ }
            try { oss_conf << kv.second.as<int>();
            } catch(...) {/* do nothing */ }
            oss_conf << std::endl;
        }
        std::cout << oss_conf.str();
        std::cout << "--------------------------" << std::endl;
        // clang-format on

        std::map<std::string, std::shared_ptr<std::ofstream>> log_files;
        for (auto& x : vm["log"].as<std::vector<std::string>>())
            log_files[x] = std::make_shared<std::ofstream>((x + ".dat").c_str());

        //////////////////// INIT DART ROBOT //////////////////////////////////////
        ///// CONTROLLER
        auto controller_path = vm["controller"].as<std::string>();
        auto controller_config = IWBC_CHECK(YAML::LoadFile(controller_path));

        std::srand(std::time(NULL));
        std::vector<std::pair<std::string, std::string>> packages = {{"talos_description", "talos/talos_description"}};
        std::string urdf = vm.count("fast") ? "talos/talos_fast.urdf" : "talos/" + IWBC_CHECK(controller_config["CONTROLLER"]["urdf"].as<std::string>()); 
        auto robot = std::make_shared<robot_dart::Robot>(urdf, packages);
        robot->set_position_enforced(vm["enforce_position"].as<bool>());
        if (vm["actuators"].as<std::string>() == "spd")
            robot->set_actuator_types("torque");
        else
            robot->set_actuator_types(vm["actuators"].as<std::string>());
        //////////////////// INIT DART SIMULATION WORLD //////////////////////////////////////
 

        // Create experience folder for saving data
        std::string xp_folder = IWBC_CHECK(controller_config["CONTROLLER"]["xp_folder"].as<std::string>());
        if (xp_folder == "")
            xp_folder = create_xp_folder();
        if (xp_folder == "error"){
            return 0;
        }

        double dt = IWBC_CHECK(controller_config["CONTROLLER"]["dt"].as<double>());
        auto simu = std::make_shared<robot_dart::RobotDARTSimu>(dt);
        simu->set_collision_detector(vm["collision"].as<std::string>());

        auto cam_pos = IWBC_CHECK(controller_config["CONTROLLER"]["cam_pos"].as<std::vector<double>>());


#ifdef GRAPHIC
    robot_dart::gui::magnum::GraphicsConfiguration configuration;

    configuration.width = 1280;  // 1280
    configuration.height = 720;  // 960 

    configuration.bg_color = Eigen::Vector4d{1.0, 1.0, 1.0, 1.0};
    
    if (vm["windowless"].as<bool>()){
        auto graphics = std::make_shared<robot_dart::gui::magnum::WindowlessGraphics>(configuration);
        simu->set_graphics(graphics);
        graphics->look_at({cam_pos[0], cam_pos[1], cam_pos[2]}, {cam_pos[3], cam_pos[4], cam_pos[5]});
        if (vm.count("mp4"))
            graphics->record_video(xp_folder + "/" + vm["mp4"].as<std::string>());
    } else{
        auto graphics = std::make_shared<robot_dart::gui::magnum::Graphics>(configuration);
        simu->set_graphics(graphics);
        //graphics->look_at({3.5, 2, 2.2}, {0., 0., 1.4});
        graphics->look_at({cam_pos[0], cam_pos[1], cam_pos[2]}, {cam_pos[3], cam_pos[4], cam_pos[5]});
        if (vm.count("mp4"))
            graphics->record_video(xp_folder + "/" + vm["mp4"].as<std::string>());
    }
    // for slide video 
    //graphics->look_at({4.15, 0, 0.4}, {0., 0., 1.57});
    simu->enable_status_bar(false);
    
#endif
        //robot->set_cast_shadows(false);
        simu->add_robot(robot);
        auto floor = simu->add_checkerboard_floor();
        
        // do some modifications according to command-line options
        controller_config["CONTROLLER"]["urdf"] = robot->model_filename();
        controller_config["CONTROLLER"]["mimic_dof_names"] = robot->mimic_dof_names();
        int control_freq = 1 / dt; 

        auto controller_name = IWBC_CHECK(controller_config["CONTROLLER"]["name"].as<std::string>());
        auto closed_loop = IWBC_CHECK(controller_config["CONTROLLER"]["closed_loop"].as<bool>());
        if (vm.count("closed_loop")) {
            closed_loop = true;
            controller_config["CONTROLLER"]["closed_loop"] = true;
        }

        if (vm["actuators"].as<std::string>() == "torque" && !closed_loop)
            std::cout << "WARNING (iwbc): you should activate the closed loop if you are using torque control! (--closed_loop or yaml)" << std::endl;

        auto controller = inria_wbc::controllers::Factory::instance().create(controller_name, controller_config);
        auto controller_pos = std::dynamic_pointer_cast<inria_wbc::controllers::PosTracker>(controller);
        IWBC_ASSERT(controller_pos, "we expect a PosTracker here");

        ///// BEHAVIOR
        auto behavior_path = vm["behavior"].as<std::string>();
        auto behavior_config = IWBC_CHECK(YAML::LoadFile(behavior_path));
        auto behavior_name = IWBC_CHECK(behavior_config["BEHAVIOR"]["name"].as<std::string>());
        auto behavior = inria_wbc::behaviors::Factory::instance().create(behavior_name, controller, behavior_config);
        IWBC_ASSERT(behavior, "invalid behavior");

        auto all_dofs = controller->all_dofs();
        auto floating_base = all_dofs;
        floating_base.resize(6);

        auto controllable_dofs = controller->controllable_dofs();
        robot->set_positions(controller->q0(), all_dofs);

        uint ncontrollable = controllable_dofs.size();

        // add sensors to the robot
        auto ft_sensor_left = simu->add_sensor<robot_dart::sensor::ForceTorque>(robot, "leg_left_6_joint", control_freq);
        // tim
        //auto ft_sensor_right = simu->add_sensor<robot_dart::sensor::ForceTorque>(robot, "leg_right_6_joint", control_freq);
        robot_dart::sensor::IMUConfig imu_config;
        imu_config.body = robot->body_node("imu_link"); // choose which body the sensor is attached to
        imu_config.frequency = control_freq; // update rate of the sensor
        auto imu = simu->add_sensor<robot_dart::sensor::IMU>(imu_config);

        //////////////////// Load parameters //////////////////////////////////////
        // duration
        double duration = IWBC_CHECK(controller_config["CONTROLLER"]["duration"].as<double>());
        // fall early stopping
        bool use_falling_early_stopping = IWBC_CHECK(controller_config["CONTROLLER"]["use_falling_early_stopping"].as<bool>());
        double fallen_treshold = IWBC_CHECK(controller_config["CONTROLLER"]["fallen_treshold"].as<double>());
        // damages
        double damage_time = IWBC_CHECK(controller_config["CONTROLLER"]["damage_time"].as<double>());
        auto locked = IWBC_CHECK(controller_config["CONTROLLER"]["locked"].as<std::vector<std::string>>());
        auto passive = IWBC_CHECK(controller_config["CONTROLLER"]["passive"].as<std::vector<std::string>>());
        auto amputated = IWBC_CHECK(controller_config["CONTROLLER"]["amputated"].as<std::vector<std::string>>());
        bool use_damage = locked.size()>0 || passive.size()>0 || amputated.size()>0;
        // reflex
        auto reflex_time = IWBC_CHECK(controller_config["CONTROLLER"]["reflex_time"].as<double>());
        auto use_baseline = IWBC_CHECK(controller_config["CONTROLLER"]["use_baseline"].as<bool>());
        auto use_reflex_trajectory = IWBC_CHECK(controller_config["CONTROLLER"]["use_reflex_trajectory"].as<bool>());
        auto update_contact_when_detected = IWBC_CHECK(controller_config["CONTROLLER"]["update_contact_when_detected"].as<bool>());
        auto reflex_arm_stiffness = IWBC_CHECK(controller_config["CONTROLLER"]["reflex_arm_stiffness"].as<float>());
        auto reflex_stiffness_alpha = IWBC_CHECK(controller_config["CONTROLLER"]["reflex_stiffness_alpha"].as<float>());

        // logging
        auto log_level = IWBC_CHECK(controller_config["CONTROLLER"]["log_level"].as<int>());

        // vizual 
        auto wall_color = IWBC_CHECK(controller_config["CONTROLLER"]["wall_opacity"].as<std::vector<double>>());
        //////////////////// START SIMULATION //////////////////////////////////////
        std::cout << "dt:" << dt << " control freq:" << control_freq <<std::endl;
        simu->set_control_freq(control_freq); // default = 1000 Hz

        std::shared_ptr<robot_dart::Robot> ghost;
        if (vm.count("ghost") || vm.count("collisions")) {
            ghost = robot->clone_ghost();
            ghost->skeleton()->setPosition(4, -1.57);
            ghost->skeleton()->setPosition(5, 1.1);
            simu->add_robot(ghost);
        }

        // Add collision shapes
        double wall_distance = 0.;
        double wall_angle = 0.;
        auto colision_shapes = controller_config["CONTROLLER"]["colision_shapes"].as<std::vector<std::pair<bool,std::vector<double>>>>();
        std::vector<std::shared_ptr<robot_dart::Robot>> colision_shapes_robots; 
        for (auto colision_shape: colision_shapes ){
            assert(colision_shape.second.size() == 9);
            const Eigen::Vector3d dims = {colision_shape.second.at(6), colision_shape.second.at(7), colision_shape.second.at(8)};
            Eigen::Vector6d pose = Eigen::Vector6d::Zero(6);
            for (int i=3; i<6; i++){
                pose(i) = colision_shape.second.at(i-3);
                pose(i-3) = colision_shape.second.at(i);
            } 
            wall_distance = pose[4];
            wall_angle = pose[2];
            const std::string type = "not free"; 
            double mass = 1.;
            const Eigen::Vector4d color= {wall_color[0], wall_color[1], wall_color[2], wall_color[3]};
            const std::string ellipsoid_name = "collision_shape";
            std::shared_ptr<robot_dart::Robot> shape;
            if (colision_shape.first){
                shape =  robot_dart::Robot::create_ellipsoid(
                    dims, pose, type, mass,color, ellipsoid_name);
            } else {
                shape =  robot_dart::Robot::create_box(
                    dims, pose, type, mass,color, ellipsoid_name);
                shape->set_cast_shadows(false);
            }
            simu->add_robot(shape);
            colision_shapes_robots.push_back(shape);
        }
        if (false){
            auto base_path = IWBC_CHECK(controller_config["CONTROLLER"]["base_path"].as<std::string>());
            auto tasks_file = IWBC_CHECK(controller_config["CONTROLLER"]["tasks"].as<std::string>());
            auto tasks_config = IWBC_CHECK(YAML::LoadFile(base_path+"/"+tasks_file));
            Eigen::Vector6d pose = Eigen::Vector6d::Zero(6);
            pose(3) = IWBC_CHECK(tasks_config["contact_rhand"]['x'].as<double>());  // x
            pose(4) = IWBC_CHECK(tasks_config["contact_rhand"]['y'].as<double>()); // y
            pose(5) = IWBC_CHECK(tasks_config["contact_rhand"]['z'].as<double>());  // z
            const std::string type = "not free"; 
            double mass = 1.;
            const Eigen::Vector4d color= {0.3, 0.8, 0.23, 1.};
            const Eigen::Vector3d dims = {0.1, 0.1, 0.1};
            std::shared_ptr<robot_dart::Robot> shape =  robot_dart::Robot::create_ellipsoid(dims, pose, type, mass, color, "sphere");
            simu->add_visual_robot(shape);
        }

        // add CoM visualization 
        double mass = 1.;
        const std::string type = "not free"; 
        const Eigen::Vector3d dims = {0.02, 0.02, 10.};
        Eigen::Vector6d pose = Eigen::Vector6d::Zero(6);
        const Eigen::Vector4d color= {1., 0., 0., 1};
        std::shared_ptr<robot_dart::Robot> tsid_com =  robot_dart::Robot::create_box(dims, pose, type, mass, color, "tsid_com");
        

        const Eigen::Vector4d color2 = {1., 0.75, 0., 1};
        std::shared_ptr<robot_dart::Robot> simu_com =  robot_dart::Robot::create_box(dims, pose, type, mass, color2, "simu_com");
        bool visualize_com = vm.count("ghost");
        if (visualize_com) {
            simu->add_visual_robot(tsid_com);
            simu->add_visual_robot(simu_com);
        }

        if(vm.count("ghost")){
            //simu->add_visual_robot(contact_vizu);
            robot->set_draw_axis("gripper_right_base_link", 1.);
        }
        
        // self-collision shapes
        std::vector<std::shared_ptr<robot_dart::Robot>> self_collision_spheres;
        if (vm.count("collisions")) {
            auto task_self_collision = controller_pos->task<tsid::tasks::TaskSelfCollision>(vm["collisions"].as<std::string>());
            for (size_t i = 0; i < task_self_collision->avoided_frames_positions().size(); ++i) {
                auto pos = task_self_collision->avoided_frames_positions()[i];
                auto tf = Eigen::Isometry3d(Eigen::Translation3d(pos[0], pos[1], pos[2]));
                double r0 = task_self_collision->avoided_frames_r0s()[i];
                auto sphere = robot_dart::Robot::create_ellipsoid(Eigen::Vector3d(r0 * 2, r0 * 2, r0 * 2), tf, "fixed", 1, Eigen::Vector4d(0, 1, 0, 0.5), "self-collision-" + std::to_string(i));
                sphere->set_color_mode("aspect");
                self_collision_spheres.push_back(sphere);
                simu->add_visual_robot(self_collision_spheres.back());
            }
        }
        std::vector<std::shared_ptr<robot_dart::sensor::Torque>> torque_sensors;
  
        auto talos_tracker_controller = std::static_pointer_cast<inria_wbc::controllers::TalosPosTracker>(controller);
        for (const auto& joint : talos_tracker_controller->torque_sensor_joints()) {
            torque_sensors.push_back(simu->add_sensor<robot_dart::sensor::Torque>(robot, joint, control_freq));
            //std::cerr << "Add joint torque sensor:  " << joint << std::endl;
        }

        // reading from sensors
        Eigen::VectorXd tq_sensors = Eigen::VectorXd::Zero(torque_sensors.size());

        using namespace std::chrono;
        // to save trajectories
        std::shared_ptr<inria_wbc::trajs::Saver> traj_saver;
        if (!vm["save_traj"].empty()) {
            auto args = vm["save_traj"].as<std::vector<std::string>>();
            auto name = args[0];
            auto refs = std::vector<std::string>(args.begin() + 1, args.end());
            traj_saver = std::make_shared<inria_wbc::trajs::Saver>(controller_pos, args[0], refs);
        }

        // the main loop
        Eigen::VectorXd cmd;
        inria_wbc::controllers::SensorData sensor_data;
        inria_wbc::utils::Timer timer;
        auto active_dofs_controllable = controllable_dofs; // undamaged case
        auto active_dofs = controller->all_dofs(false); // false here: no filter at all
        inria_wbc::robot_dart::RobotDamages robot_damages(robot, simu, active_dofs_controllable, active_dofs);
        std::vector<double> imuz;
        std::vector<double> filtered_imuz;
        State state = State::Failure; 
        bool has_collide_with_the_wall = false; 
        bool first_contact = true;
        Eigen::VectorXd activated_joints = Eigen::VectorXd::Zero(active_dofs.size());
        float arm_stiffness = 10000;
        bool in_contact = false;

        while (simu->scheduler().current_time() < duration && !simu->graphics()->done() && (state != State::Fallen)) {
            // check all collisions 
            bool still_in_contact = false;
            auto col = simu->world()->getConstraintSolver()->getLastCollisionResult();
            size_t nc = col.getNumContacts();
            for (size_t i = 0; i < nc; i++) {
                auto& ct = col.getContact(i);
                auto f1 = ct.collisionObject1->getShapeFrame();
                auto f2 = ct.collisionObject2->getShapeFrame();
                std::string name1, name2;
                if (f1->isShapeNode())
                    name1 = f1->asShapeNode()->getBodyNodePtr()->getName();
                if (f2->isShapeNode())
                    name2 = f2->asShapeNode()->getBodyNodePtr()->getName();
                
                // distinguish the contacts
                bool has_floor_contact = false;
                bool has_wall_contact = false; 
                std::string floor_contact = "";
                std::string wall_contact = "";
                if (name1 == "BodyNode"){
                    has_floor_contact = true;
                    floor_contact = name2;
                }
                if (name2 == "BodyNode"){
                    has_floor_contact = true;
                    floor_contact = name1;
                } 
                if (name1 == "collision_shape"){
                    has_wall_contact = true;
                    wall_contact = name2;
                } 
                if (name2 == "collision_shape") {
                    has_wall_contact = true;
                    wall_contact = name1;
                } 

                //contact with the floor 
                if (has_floor_contact && floor_contact != "leg_left_6_link" && floor_contact != "leg_right_6_link"){
                    state = State::Fallen;
                    std::cout << "Fallen at " << simu->scheduler().current_time() << " for touching the floor with " << floor_contact << std::endl;
                    write_timestate(state, xp_folder, simu->scheduler().current_time());
                    break;
                }

                // contact with the wall
                if (has_wall_contact) {
                    // shutdown if initial wall prenetration (sampling rejection)
                    if (simu->scheduler().current_time() < damage_time){
                        std::cout << "pre-damage wall contact" << std::endl;
                        state = State::Failure;
                        write_timestate(state, xp_folder, simu->scheduler().current_time());
                        return 0; 
                    }
                    // good contact (handball or right arm)
                    if ((wall_contact == "gripper_right_base_link") || (wall_contact.substr(0,9) == "arm_right")){
                        // if conserve contact 
                        if (in_contact){
                            still_in_contact = true;
                            arm_stiffness = reflex_stiffness_alpha * arm_stiffness + (1-reflex_stiffness_alpha ) * 10000.; 
                        } // else (if new contact)
                        else {
                            std::cout << "Made contact at " << simu->scheduler().current_time() << " the wall with " << wall_contact << std::endl;
                            static std::ofstream ofs_contact_pos(xp_folder +"/contact_pos.dat");
                            auto contact_pos = controller->robot()->framePosition(controller->tsid()->data(), controller->robot()->model().getFrameId("gripper_right_base_link_joint"));
                            ofs_contact_pos << simu->scheduler().current_time() << " " << contact_pos.translation().transpose() <<  std::endl; 
                            in_contact = true;
                        } 
                        // if first contact on the wall 
                        if (!has_collide_with_the_wall){
                            auto contact_pos = controller->robot()->framePosition(controller->tsid()->data(), controller->robot()->model().getFrameId("gripper_right_base_link_joint"));
                            has_collide_with_the_wall = true; 
                            if (first_contact && update_contact_when_detected){
                                first_contact = false;
                                // update contact
                                std::cout << "update contact to " << contact_pos.translation().transpose() << std::endl; 
                                auto contact_task = std::static_pointer_cast<inria_wbc::controllers::DamageController>(controller)->contact_task("contact_rhand");
                                tsid::contacts::Contact6d::SE3 contact_ref;
                                contact_ref.translation() = contact_pos.translation();
                                contact_ref.rotation() = contact_pos.rotation();
                                contact_task->Contact6d::setReference(contact_ref);

                                // update com
                                /*
                                auto lf_ref = controller->model_frame_pos("leg_left_6_joint");
                                auto contact_task = std::static_pointer_cast<inria_wbc::controllers::DamageController>(controller)->contact_task("contact_rhand");
                                auto contact_motion_task = contact_task->getMotionTask();
                                auto contact_ref = contact_motion_task.getReference();
                                auto ref = (lf_ref.translation() + contact_ref.pos.head(3))/2;
                                std::cout << "new CoM ref: " <<  lf_ref.translation().transpose() << std::endl; 
                                std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller)->set_com_ref(lf_ref.translation());
                                */

                                // update compliance 
                                arm_stiffness = reflex_arm_stiffness; 

                                if (use_reflex_trajectory){
                                    std::static_pointer_cast<inria_wbc::controllers::TalosPosTracker>(controller)->add_contact("contact_rhand");
                                    //remove right hand task
                                    auto zero = Eigen::VectorXd::Zero(6);
                                    auto rh_task = std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller)->se3_task("rh"); 
                                    rh_task->Kp(zero); 
                                    rh_task->Kd(zero);   
                                }
                            }
                        }
                    } // bad contact 
                    else if(use_falling_early_stopping) {  
                        state = State::Fallen;
                        std::cout << "Fallen at " << simu->scheduler().current_time() << " for touching the wall with " << wall_contact << std::endl;
                        write_timestate(state, xp_folder, simu->scheduler().current_time());
                        has_collide_with_the_wall = true;
                        break;
                    }
                }
            }
            // after all contacts are treated we check if we still get a good contact on the wall
            if (in_contact && !still_in_contact){
                in_contact = false;
                //arm_stiffness = 10000.;
                std::cout << "Loose contact with the wall at " << simu->scheduler().current_time() << std::endl;
                static std::ofstream ofs_loose_contact(xp_folder +"/lose_contact.dat");
                ofs_loose_contact << simu->scheduler().current_time() <<  std::endl; 
            }

            if (use_damage) {
                try {
                    if (simu->scheduler().current_time() >= damage_time && simu->scheduler().current_time() < damage_time + dt) {
                        std::cout << "Apply damage." << std::endl; 
                        for (auto const link: amputated){
                            robot_damages.cut(link);
                        }
                        for (auto const  joint: passive){
                            robot_damages.motor_damage(joint, inria_wbc::robot_dart::PASSIVE);
                            
                            for (auto& bd : robot->skeleton()->getBodyNodes()) {
                                auto& visual_shapes = bd->getShapeNodesWith<dart::dynamics::VisualAspect>(); 
                                if (bd->getName().substr(0,11) == joint.substr(0,11)) {
                                    for (auto& shape : visual_shapes) {
                                        shape->getShape()->setDataVariance(dart::dynamics::Shape::DYNAMIC_COLOR);
                                        shape->getVisualAspect()->setRGBA(Eigen::Vector4d{1.0, 0.0, 0.0, 1.0});
                                    }
                                }
                            }
                        }
                        for (auto const joint: locked){
                            robot_damages.motor_damage(joint, inria_wbc::robot_dart::LOCKED);
                        }
                        active_dofs_controllable = robot_damages.active_dofs_controllable();
                        active_dofs = robot_damages.active_dofs();
                    }
                }
                catch (std::exception& e) {
                    std::cout << red << bold << "Error (exception): " << rst << e.what() << std::endl;
                }
            }

            // get actual torque from sensors
            for (size_t i = 0; i < torque_sensors.size(); ++i)
                if (torque_sensors[i]->active())
                    tq_sensors(i) = torque_sensors[i]->torques()(0, 0);
                else
                    tq_sensors(i) = 0;

            // step the command
            if (simu->schedule(simu->control_freq())) {

                // update the sensors
                // left foot
                if (ft_sensor_left->active()) {
                    sensor_data["lf_torque"] = ft_sensor_left->torque();
                    sensor_data["lf_force"] = ft_sensor_left->force();
                }
                else {
                    sensor_data["lf_torque"] = Eigen::VectorXd::Constant(6, 1e-8);
                    sensor_data["lf_force"] = Eigen::VectorXd::Constant(3, 1e-8);
                }
                // right foot
                /*if ft_sensor_right->active()) {
                    sensor_data["rf_torque"] = ft_sensor_right->torque();
                    sensor_data["rf_force"] = ft_sensor_right->force();
                }
                else */
                {
                    sensor_data["rf_torque"] = Eigen::VectorXd::Constant(6, 1e-8);
                    sensor_data["rf_force"] = Eigen::VectorXd::Constant(3, 1e-8);
                }
                // accelerometer
                sensor_data["acceleration"] = imu->linear_acceleration();
                sensor_data["velocity"] = robot->com_velocity().tail<3>();
                // joint positions / velocities (excluding floating base)
                // 0 for joints that are not in active_dofs_controllable
                Eigen::VectorXd positions = Eigen::VectorXd::Zero(controller->controllable_dofs(false).size());
                Eigen::VectorXd velocities = Eigen::VectorXd::Zero(controller->controllable_dofs(false).size());
                for (size_t i = 0; i < controllable_dofs.size(); ++i) {
                    auto name = controllable_dofs[i];
                    if (std::count(active_dofs_controllable.begin(), active_dofs_controllable.end(), name) > 0) {
                        positions(i) = robot->positions({name})[0];
                        velocities(i) = robot->velocities({name})[0];
                    }
                }

                sensor_data["positions"] = positions;
                sensor_data["joints_torque"] = tq_sensors;
                sensor_data["joint_velocities"] = velocities;
                // floating base (perfect: no noise in the estimate)
                sensor_data["floating_base_position"] = inria_wbc::robot_dart::floating_base_pos(robot->positions());
                sensor_data["floating_base_velocity"] = inria_wbc::robot_dart::floating_base_vel(robot->velocities());

                timer.begin("solver");
                behavior->update(sensor_data);
                auto q = controller->q(false);
                timer.end("solver");

                auto q_no_mimic = controller->filter_cmd(q).tail(ncontrollable); //no fb
                auto q_damaged = inria_wbc::robot_dart::filter_cmd(q_no_mimic, controllable_dofs, active_dofs_controllable);
                timer.begin("cmd");
                if (vm["actuators"].as<std::string>() == "velocity" || vm["actuators"].as<std::string>() == "servo")
                    cmd = inria_wbc::robot_dart::compute_velocities(robot, q_damaged, 1. / control_freq, active_dofs_controllable);
                else if (vm["actuators"].as<std::string>() == "spd") {
                    //cmd = inria_wbc::robot_dart::compute_spd(robot, q_damaged, 1. / control_freq, active_dofs_controllable, false);
                    cmd = compute_spd(robot, q_damaged, 1. / control_freq, active_dofs_controllable, false, arm_stiffness);
                }
                else // torque
                    cmd = controller->tau(false);
                
                timer.end("cmd");

                if (ghost) {
                    Eigen::VectorXd translate_ghost = Eigen::VectorXd::Zero(6);
                    translate_ghost(0) -= 1;
                    ghost->set_positions(controller->filter_cmd(q).tail(ncontrollable), controllable_dofs);
                    ghost->set_positions(q.head(6) + translate_ghost, floating_base);
                }
            }

            if (simu->schedule(simu->graphics_freq()) && vm.count("collisions")) {
                auto controller_pos = std::dynamic_pointer_cast<inria_wbc::controllers::PosTracker>(controller);
                auto task_self_collision = controller_pos->task<tsid::tasks::TaskSelfCollision>(vm["collisions"].as<std::string>());
                for (size_t i = 0; i < task_self_collision->avoided_frames_positions().size(); ++i) {
                    auto cp = self_collision_spheres[i]->base_pose();
                    cp.translation() = task_self_collision->avoided_frames_positions()[i];
                    cp.translation()[0] -= 1; // move to the ghost
                    self_collision_spheres[i]->set_base_pose(cp);
                    auto bd = self_collision_spheres[i]->skeleton()->getBodyNodes()[0];
                    auto visual = bd->getShapeNodesWith<dart::dynamics::VisualAspect>()[0];
                    visual->getShape()->setDataVariance(dart::dynamics::Shape::DYNAMIC_COLOR);
                    bool c = task_self_collision->collision(i);
                    if (c) {
                        visual->getVisualAspect()->setRGBA(dart::Color::Red(1.0));
                    }
                    else {
                        visual->getVisualAspect()->setRGBA(dart::Color::Green(1.0));
                    }
                }
            }

            // push the robot
            if (vm.count("push")) {
                auto pv = vm["push"].as<std::vector<float>>();
                auto pforce = vm["norm_force"].as<float>();
                for (auto& p : pv) {
                    if (simu->scheduler().current_time() > p && simu->scheduler().current_time() < p + 0.5) {
                        robot->set_external_force("base_link", Eigen::Vector3d(0, pforce, 0));
                        // robot->set_external_force("base_link", Eigen::Vector3d(pforce, 0, 0));
                    }
                    if (simu->scheduler().current_time() > p + 0.25)
                        robot->clear_external_forces();
                }
            }

            // step the simulation
            {
                timer.begin("sim");
                robot->set_commands(cmd, active_dofs_controllable);
                simu->step_world();
                timer.end("sim");
            }



            // log 
            if (log_level > 0){ 
                //static std::ofstream ofs_com(xp_folder +"/com_vel.dat");
                //ofs_com << robot->com_velocity().transpose() << std::endl;
                static std::ofstream ofs_lf_force(xp_folder +"/lf_force.dat");
                ofs_lf_force << (ft_sensor_left->force()).transpose() << std::endl;
                    
                static std::ofstream ofs_tau(xp_folder +"/tau.dat");
                ofs_tau << tq_sensors.transpose() << std::endl;

                static std::ofstream ofs_rh_trajectory(xp_folder +"/rh_traj.dat");
                ofs_rh_trajectory << robot->body_pose("gripper_right_base_link").translation().transpose()  << std::endl;

                static std::ofstream ofs_rh_velocity(xp_folder +"/rh_vel.dat");
                ofs_rh_velocity << robot->body_velocity("gripper_right_base_link").transpose()  << std::endl;
            }

            //update com visualization position 
            if (visualize_com){
                Eigen::Vector6d pose = Eigen::Vector6d::Zero(6);
                auto com = controller->com();
                pose[3] = com[0]-1;
                pose[4] = com[1];
                tsid_com->set_base_pose(pose);
                Eigen::Vector6d pose2 = Eigen::Vector6d::Zero(6);
                auto com2 = robot->com();
                pose2[3] = com2[0];
                pose2[4] = com2[1];
                simu_com->set_base_pose(pose2);
            }

            // Stop if fallen : code 2
            auto head_z_diff = std::abs(controller->model_frame_pos("head_1_link").translation()(2) - robot->body_pose("head_1_link").translation()(2));
            auto base_z_diff = std::abs(inria_wbc::robot_dart::floating_base_pos(robot->positions())[2] - controller->q(false)[2]);
            if ((state != State::Fallen) && use_falling_early_stopping && (head_z_diff > fallen_treshold || base_z_diff > fallen_treshold)){
                state = State::Fallen;
                std::cout << "Fallen at " << simu->scheduler().current_time() << std::endl;
                write_timestate(state, xp_folder, simu->scheduler().current_time());
            }
   
            // Damage Reflex
            if (simu->scheduler().current_time() >= reflex_time && simu->scheduler().current_time() < reflex_time + dt) {
                std::cout << simu->scheduler().current_time() << " reflex" << std::endl; 
                // log
                if (log_level > 0){
                    static std::ofstream ofs_tsid_q(xp_folder +"/tsid_q.dat");
                    ofs_tsid_q << controller->q(true).transpose() << std::endl;

                    static std::ofstream ofs_real_q(xp_folder +"/real_q.dat");
                    ofs_real_q << robot->positions() << std::endl;

                    static std::ofstream ofs_dof_names(xp_folder +"/dof_names.dat");
                    for (auto name: robot->dof_names() ){
                        ofs_dof_names << name << std::endl;
                    }   
                    

                    static std::ofstream ofs_lh(xp_folder +"/lh.dat");
                    ofs_lh << controller->model_frame_pos("gripper_left_joint").translation().transpose() << std::endl;

                    static std::ofstream ofs_rh(xp_folder +"/rh.dat");
                    ofs_rh << controller->model_frame_pos("arm_right_7_joint").translation().transpose() << std::endl;

                    static std::ofstream ofs_base(xp_folder +"/base.dat");
                    ofs_base << inria_wbc::robot_dart::floating_base_pos(robot->positions()).transpose() << std::endl;

                    static std::ofstream ofs_lh_real(xp_folder +"/lh_real.dat");
                    ofs_lh_real << robot->body_pose("gripper_left_base_link").translation().transpose() << std::endl;

                    static std::ofstream ofs_rh_real(xp_folder +"/rh_real.dat");
                    ofs_rh_real << robot->body_pose("arm_right_7_link").translation().transpose()  << std::endl;

                    static std::ofstream ofs_com(xp_folder +"/com.dat");
                    ofs_com << robot->com().transpose() << std::endl;
                }

                // right hand contact task
                //// change ref
                if (use_baseline) {
                    auto contact_rhand = std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller)->contact("contact_rhand");
                    auto contact_ref = controller->model_frame_pos("gripper_right_base_link_joint");
    
                    //contact_ref.translation()[0] = 0.3;
                    contact_ref.translation()[1] = wall_distance + 0.1;
                    //contact_ref.translation()[2] = 0.7;
                    auto euler = contact_ref.rotation().eulerAngles(0, 1, 2);
                    euler[0] = -1.57;
                    euler[1] = 0.;
                    euler[2] = 0.;
                    auto q = Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ());
                    contact_ref.rotation() = q.toRotationMatrix();
                    std::cout << "wall contact at: " << contact_ref.translation().transpose() << std::endl; 
                    // apply
                    contact_rhand->Contact6d::setReference(contact_ref);

                }
                
                // TODO ne faite q'une fois le cast 
                auto zero = Eigen::VectorXd::Zero(6);

                if (use_reflex_trajectory){
                    //std::static_pointer_cast<inria_wbc::behaviors::Hands>(behavior)->activate_reflex();
                    auto rh_task = std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller)->se3_task("rh"); 
                    //rh_task->Kp(30 * Eigen::VectorXd::Ones(6)); 
                    //rh_task->Kd(2.0 * rh_task->Kp().cwiseSqrt()); 
                } else {
                    std::static_pointer_cast<inria_wbc::controllers::TalosPosTracker>(controller)->add_contact("contact_rhand");

                    // com
                    /*
                    auto lf_ref = controller->model_frame_pos("leg_left_6_joint");
                    auto contact_task = std::static_pointer_cast<inria_wbc::controllers::DamageController>(controller)->contact_task("contact_rhand");
                    auto contact_motion_task = contact_task->getMotionTask();
                    auto contact_ref = contact_motion_task.getReference();
                    auto ref = (lf_ref.translation() + contact_ref.pos.head(3))/2;
                    std::cout << "new CoM ref: " <<  ref.transpose() << std::endl; 
                    std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller)->set_com_ref(ref);
                    */
                    /*
                    auto com_task = std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller)->com_task();
                    com_task->Kp(Eigen::VectorXd::Zero(3)); 
                    com_task->Kd(Eigen::VectorXd::Zero(3)); 
                    */

                    //remove right hand task
                    auto rh_task = std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller)->se3_task("rh"); 
                    rh_task->Kp(zero); 
                    rh_task->Kd(zero);   
                }

                
                //remove left hand task
                auto lh_task = std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller)->se3_task("lh"); 
                lh_task->Kp(zero); 
                lh_task->Kd(zero);   
                
                // remove right foot tasks if we use the undamaged talos
                if (IWBC_CHECK(controller_config["CONTROLLER"]["urdf"].as<std::string>()) == "talos.urdf"){
                    std::static_pointer_cast<inria_wbc::controllers::TalosPosTracker>(controller)->remove_contact("contact_rfoot");
                    auto rf_task = std::static_pointer_cast<inria_wbc::controllers::PosTracker>(controller)->se3_task("rf"); 
                    rf_task->Kp(zero); 
                    rf_task->Kd(zero); 
                }

            }
            

            if (traj_saver)
                traj_saver->update();
            
            if (vm.count("srdf")) {
                auto conf = vm["srdf"].as<float>();
                if (controller->t() >= conf && controller->t() < conf + controller->dt())
                    controller->save_configuration("configuration.srdf");
            }
            if (timer.iteration() == 100) {
                
                std::ostringstream oss;
#ifdef GRAPHIC // to avoid the warning
                oss.precision(3);
                timer.report(oss, simu->scheduler().current_time(), -1, '\n');
                if (!vm.count("mp4"))
                    simu->set_text_panel(oss.str());
#endif
            }
            timer.report(simu->scheduler().current_time(), 100);
        }
        if (state != State::Fallen){
            if (simu->scheduler().current_time() >= duration){
                std::cout << "Successs" << std::endl;
                state = State::Success;
            } else {
                std::cout << "Failure: timeout" << std::endl;
                state = State::Failure;
            }
            write_timestate(state, xp_folder, simu->scheduler().current_time());
        }
    }
    catch (YAML::RepresentationException& e) {
        std::cout << red << bold << "YAML Parse error (missing key in YAML file?): " << rst << e.what() << std::endl;
    }
    catch (YAML::ParserException& e) {
        std::cout << red << bold << "YAML Parse error: " << rst << e.what() << std::endl;
    }
    catch (std::exception& e) {
        std::cout << red << bold << "Error (exception): " << rst << e.what() << std::endl;
    }
    return 0;
}

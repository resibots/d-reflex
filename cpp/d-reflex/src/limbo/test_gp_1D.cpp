#include <fstream>
#include <limbo/kernel/exp.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/tools.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/serialize/text_archive.hpp>

#include <algorithm>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <signal.h>
#include <ctime> 
#include <boost/filesystem.hpp>
#include <sys/stat.h> 
#include <robot_dart/control/pd_control.hpp>
#include <robot_dart/robot.hpp>
#include <robot_dart/robot_dart_simu.hpp>
#include <robot_dart/sensor/force_torque.hpp>
#include <robot_dart/sensor/imu.hpp>
#include <robot_dart/sensor/torque.hpp>
#include <robot_dart/gui/magnum/windowless_graphics.hpp>

#include "inria_wbc/behaviors/behavior.hpp"
#include "inria_wbc/exceptions.hpp"
#include "inria_wbc/robot_dart/cmd.hpp"

static const std::string red = "\x1B[31m";
static const std::string rst = "\x1B[0m";
static const std::string bold = "\x1B[1m";

using namespace limbo;

struct Params {
    struct kernel_exp {
        BO_PARAM(double, sigma_sq, 1.0);
        BO_PARAM(double, l, 0.2);
    };
    struct kernel : public defaults::kernel {
    };
    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    };
    struct opt_rprop : public defaults::opt_rprop {
    };
};

Eigen::MatrixXd load(const std::string& path, size_t cols)
                    
{
    std::ifstream fstr(path.c_str());
    std::vector<double> data = std::vector<double>{
        std::istream_iterator<double>(fstr),
        std::istream_iterator<double>()};
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(&data.data()[0], data.size() / cols, cols);
}

int main(int argc, char** argv)
{
   try {
        // program options
        namespace po = boost::program_options;
        po::options_description desc("Test_controller options");
        // clang-format off
        desc.add_options()
        ("help,h", "produce help message")
        ("conf,c", po::value<std::string>()->default_value("../etc/learn_discrepancy.yaml"), "Configuration file (yaml) [default: ../etc/learn_discrepancy.yaml]")
        ("verbose,v", "verbose mode (controller)")
        ("xp_folder,x", po::value<std::string>()->default_value(""), "xp_folder name")
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
        std::string sot_config_path = vm["conf"].as<std::string>();
        YAML::Node config = YAML::LoadFile(sot_config_path);


        ////////////////////////////////////////////////////////////////
        ///                          start                           ///
        ////////////////////////////////////////////////////////////////
        std::string xp_folder = vm["xp_folder"].as<std::string>();
        std::vector<Eigen::VectorXd> samples;
        std::vector<Eigen::VectorXd> observations;
        
        size_t N = 20;
        for (size_t i = 0; i < N; i++) {
            Eigen::VectorXd s = tools::random_vector(1).array();
            samples.push_back(s);
            observations.push_back(tools::make_vector(std::sin(2*3.1415*s(0))));
        }

        // the type of the GP
        using Kernel_t = kernel::Exp<Params>;
        using Mean_t = mean::Data<Params>;
        using GP_t = model::GP<Params, Kernel_t, Mean_t>;

        // 1-D inputs, 1-D outputs
        GP_t gp(1, 1);

        // compute the GP
        gp.compute(samples, observations);

        // write the predicted data in a file (e.g. to be plotted)
        std::ofstream ofs(xp_folder+"/gp.dat");
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd v = tools::make_vector(i / 100.0).array();
            Eigen::VectorXd mu;
            double sigma;
            std::tie(mu, sigma) = gp.query(v);
            // an alternative (slower) is to query mu and sigma separately:
            //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
            //  double s2 = gp.sigma(v);
            ofs << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
        }

        // an alternative is to optimize the hyper-parameters
        // in that case, we need a kernel with hyper-parameters that are designed to be optimized
        using Kernel2_t = kernel::SquaredExpARD<Params>;
        using Mean_t = mean::Data<Params>;
        using GP2_t = model::GP<Params, Kernel2_t, Mean_t, model::gp::KernelLFOpt<Params>>;

        GP2_t gp_ard(1, 1);
        // do not forget to call the optimization!
        gp_ard.compute(samples, observations, false);
        gp_ard.optimize_hyperparams();

        // Get alpha, kernel for computing gradient
        auto alpha = gp_ard.alpha();
        auto kernel = gp_ard.kernel_function();
        auto log_params = kernel.h_params();
        double l = std::exp(log_params(0));

        // write the predicted data in a file (e.g. to be plotted)
        std::ofstream ofs_ard(xp_folder+"/gp_ard.dat");
        std::ofstream ofs_grad(xp_folder+"/grad_mu_gp.dat");
        std::ofstream ofs_hessian(xp_folder+"/hessian_mu_gp.dat");
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd v = tools::make_vector(i / 100.0).array();
            Eigen::VectorXd mu;
            double sigma;
            std::tie(mu, sigma) = gp_ard.query(v);
            ofs_ard << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(1);
            Eigen::VectorXd hessian = Eigen::VectorXd::Zero(1);
            for (size_t j = 0; j < N; j++) {
                grad = grad + kernel(v, samples.at(j))*alpha(j)*(samples.at(j)-v)/std::pow(l,2);
                hessian = hessian + kernel(v, samples.at(j))*alpha(j)*((samples.at(j)-v)*(samples.at(j)-v)/std::pow(l,4)-Eigen::VectorXd::Ones(1)/std::pow(l,2));
            }
            ofs_grad << grad << std::endl;
            ofs_hessian << hessian << std::endl;
        }

        // write the data to a file (useful for plotting)
        std::ofstream ofs_data(xp_folder+"/data.dat");
        for (size_t i = 0; i < samples.size(); ++i){
            ofs_data << samples[i].transpose() << " " << observations[i].transpose() << std::endl;
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
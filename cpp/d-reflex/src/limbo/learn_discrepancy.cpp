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
    struct kernel : public defaults::kernel {BO_PARAM(bool, optimize_noise, true)};
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
        ///                       load DATA                          ///
        ////////////////////////////////////////////////////////////////
        using namespace std::chrono;
        std::string xp_folder = vm["xp_folder"].as<std::string>();
        std::vector<Eigen::VectorXd> samples;
        std::vector<Eigen::VectorXd> observations;

        std::vector<Eigen::VectorXd> samples_test;
        std::vector<Eigen::VectorXd> observations_test;

        size_t inputDim = 50;
        // load training data
        //auto X = load(xp_folder+"/X0.dat", inputDim);
        //auto Y = load(xp_folder+"/Y0.dat", 1); 
        std::vector<std::vector<double>> X = {{0.00112157, 6.93309e-07, 1.06011, -0.000209743, 0.00661476, 0.00186544, -0.00186379, 0.000357071, -0.266857, 0.587836, -0.327593, -0.000141163, -0.0018638, 0.00035662, -0.266217, 0.587463, -0.32786, -0.000140712, 0.00400853, 0.0547649, 0.400048, 0.241028, -0.600427, -1.44812, 2.70038e-05, -8.01933e-05, 0.00035684, -2.13121e-06, 2.13824e-06, 1.4291e-06, -1.2152e-06, 8.49659e-08, -4.35424e-07, -7.5622e-07, -0.399631, -0.240054, 0.595207, -1.45096, 9.35502e-05, -0.00127377, -0.000408091, -3.23314e-06, 1.16028e-06, 1.80517e-06, -1.08061e-06, 9.45883e-08, -7.59727e-07, -6.48468e-07, 0.000471532, -6.40683e-07}, 
                        {0.00177306, 0.000181999, 1.06011, 0.00209823, 0.011325, 0.0102861, -0.0103066, -0.00281626, -0.272211, 0.584869, -0.323993, 0.00077643, -0.0103068, -0.00283057, -0.271201, 0.588327, -0.328462, 0.000790741, 0.0219506, 0.0560691, 0.402562, 0.243648, -0.600094, -1.44745, 3.68239e-05, -3.19654e-05, 0.000448898, 1.59082e-07, 1.41266e-06, 9.152e-07, -1.62154e-06, 2.48656e-07, 2.97108e-07, -1.21799e-06, -0.398322, -0.244693, 0.580853, -1.43782, -0.000688608, -0.00434885, 0.0043105, -8.87858e-07, 7.17974e-07, 1.31639e-06, -1.42094e-06, 3.14587e-07, 1.62729e-07, -1.09017e-06, 0.000254723, 1.07851e-05},
                        {0.00184534, 0.000197916, 1.06011, 0.00176721, 0.0116083, 0.00838989, -0.00840881, -0.00244706, -0.272382, 0.585139, -0.324373, 0.00072923, -0.00840894, -0.00245694, -0.271603, 0.588057, -0.32807, 0.000739115, 0.0181755, 0.055511, 0.401815, 0.243477, -0.599906, -1.44738, 3.69258e-05, -9.57718e-06, 0.000476507, -2.67674e-05, 8.83747e-08, 1.31837e-07, 0.000102312, 1.85564e-07, -4.86154e-05, 2.76567e-05, -0.396941, -0.246145, 0.584189, -1.44217, -0.000346419, -0.00358633, 0.00246977, -4.69409e-06, 2.35365e-09, -3.63851e-08, 5.92351e-05, 2.72899e-07, -1.8558e-06, 5.39904e-05, 0.000422575, 6.41659e-06}
                            };
        std::vector<double> Y = {1, 0.3, 0.8};
        // load testing data
        auto X_test = load(xp_folder+"/X1.dat", inputDim);
        auto Y_test = load(xp_folder+"/Y1.dat", 1); 

        // convert for gp learning
  /*       size_t N = config["PARAMS"]["N"].as<int>();
        float step = X.rows()/N;

        for (size_t i = 0; i < N; i++) {
            auto index = int(step*i);
            Eigen::VectorXd s = X.row(index);
            samples.push_back(s);
            observations.push_back(Y.row(index));
        }  */


        size_t N = config["PARAMS"]["N"].as<int>();

        for (size_t i = 0; i < 3; i++) {
            Eigen::VectorXd s(50);
            for (int j=0; j<50; j++){
                s[j] = X.at(i).at(j);
            }
            samples.push_back(s);
            Eigen::VectorXd o(1);
            o[0]=Y.at(i);
            observations.push_back(o);
        } 

        auto step = X_test.rows()/N;
        for (size_t i = 0; i < N; i++) {
            auto index = int(step*i);
            Eigen::VectorXd s = X_test.row(index);
            samples_test.push_back(s);
            observations_test.push_back(Y_test.row(index));
        } 

        // write the training data to a file 
        std::ofstream ofs_data(xp_folder+"/data.dat");
        for (size_t i = 0; i < samples.size(); ++i){
            ofs_data << samples[i].transpose() << " " << observations[i].transpose() << std::endl;
        }
        // write the testing data to a file 
        std::ofstream ofs_data_test(xp_folder+"/data_test.dat");
        for (size_t i = 0; i < samples_test.size(); ++i){
            ofs_data_test << samples_test[i].transpose() << " " << observations_test[i].transpose() << std::endl;
        }
        ////////////////////////////////////////////////////////////////
        ///                       default GP                         ///
        ////////////////////////////////////////////////////////////////
        // the type of the GP
        //using Kernel_t = kernel::Exp<Params>;
        using Kernel_t = kernel::SquaredExpARD<Params>;
        using Mean_t = mean::Data<Params>;
        //using GP_t = model::GP<Params, Kernel_t, Mean_t>;
        using GP_t = model::GP<Params, Kernel_t, Mean_t, model::gp::KernelLFOpt<Params>>;

        // 50-D inputs, 1-D outputs
        GP_t gp(inputDim, 1);

        // compute the GP
        auto t1_gp = high_resolution_clock::now();
        gp.compute(samples, observations);
        gp.optimize_hyperparams();
        auto t2_gp = high_resolution_clock::now();
        auto gp_alpha = gp.alpha();
        auto gp_kernel = gp.kernel_function();
        auto gp_log_params = gp_kernel.h_params();
        
        //std::cout << "gp: " << gp_log_params.transpose() << std::endl; 
        //double gp_l = std::exp(gp_log_params[0]);
        //Eigen::VectorXd gp_l = Eigen::VectorXd::Ones(inputDim);
        // write the predicted training data in a file 
        std::ofstream ofs(xp_folder+"/gp.dat");
        for (int i = 0; i < 3; ++i) {
            Eigen::VectorXd v = samples.at(i);
            Eigen::VectorXd mu;
            double sigma;
            std::tie(mu, sigma) = gp.query(v);
            ofs << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
        }
        auto t3_gp = high_resolution_clock::now();
  
        // write the predicted testing data in a file
        std::ofstream ofs_gp_test(xp_folder+"/gp_test.dat");
        std::ofstream ofs_gp_grad(xp_folder+"/gp_grad.dat");
        std::ofstream ofs_gp_limbo_grad(xp_folder+"/gp_limbo_grad.dat");
        std::ofstream ofs_gp_num_grad(xp_folder+"/gp_num_grad.dat");
        std::ofstream ofs_gp_hessian(xp_folder+"/gp_hessian.dat");
        std::ofstream ofs_gp_num_hessian(xp_folder+"/gp_num_hessian.dat");
        // for finite difference grad
        double eps = 0.001;

        bool num_diff = true;

        for (int i = 0; i < N; ++i) {
            Eigen::VectorXd v = samples_test.at(i);
            Eigen::VectorXd mu;
            double sigma;
            std::tie(mu, sigma) = gp.query(v);
            ofs_gp_test << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(inputDim);
            Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(inputDim, inputDim);
            for (size_t j = 0; j < N; j++) {
                if (gp_log_params.size() == 3){
                    auto gp_l = std::exp(gp_log_params[0]);
                    grad = grad + gp_kernel(v, samples.at(j))*gp_alpha(j)*(samples.at(j)-v) / (gp_l*gp_l);
                    hessian = hessian + gp_kernel(v, samples.at(j))*gp_alpha(j)*(Eigen::MatrixXd(samples.at(j)-v)*Eigen::MatrixXd(samples.at(j)-v).transpose()/std::pow(gp_l,4)-Eigen::MatrixXd::Identity(inputDim, inputDim)/std::pow(gp_l,2));
                } else {
                    Eigen::VectorXd aux = gp_kernel(v, samples.at(j))*gp_alpha(j)*(samples.at(j)-v);
                    for (size_t k = 0; k < inputDim; k++) {
                         aux[k] = aux[k] / std::pow(std::exp(gp_log_params[k]), 2);
                    }
                    grad = grad + aux;
                }
            }
            ofs_gp_grad << grad.transpose() << std::endl;
            ofs_gp_limbo_grad << gp.gradient(v).transpose() << std::endl;
            ofs_gp_hessian << hessian.norm() << std::endl;
            if (num_diff){
                Eigen::VectorXd num_grad = Eigen::VectorXd::Zero(inputDim);
                for (int j =0; j <inputDim; j++){
                    auto v_plus = v;
                    v_plus[j] = v_plus[j] + eps;
                    auto v_minus = v;
                    v_minus[j] = v_minus[j] - eps;
                    auto mu_plus = gp.mu(v_plus);
                    auto mu_minus = gp.mu(v_minus);
                    num_grad[j] = (mu_plus[0]-mu_minus[0])/(2*eps);
                }
                Eigen::MatrixXd num_hessian = Eigen::MatrixXd::Zero(inputDim, inputDim);
                for (int k =0; k <inputDim; k++){
                    auto v1 = v;
                    v1[k] = v1[k] + eps;
                    auto v2 = v;
                    v2[k] = v2[k] - eps;
                    for (int j =0; j <inputDim; j++){
                        // compute gradient of v1
                        auto v1_plus = v1;
                        v1_plus[j] = v1_plus[j] + eps;
                        auto v1_minus = v1;
                        v1_minus[j] = v1_minus[j] - eps;
                        auto mu1_plus = gp.mu(v1_plus);
                        auto mu1_minus = gp.mu(v1_minus);
                        // compute gradient of v2
                        auto v2_plus = v2;
                        v2_plus[j] = v2_plus[j] + eps;
                        auto v2_minus = v2;
                        v2_minus[j] = v2_minus[j] - eps;
                        auto mu2_plus = gp.mu(v2_plus);
                        auto mu2_minus = gp.mu(v2_minus);
                        // compute hessian of gradient
                        num_hessian(k, j) = ((mu1_plus[0]-mu1_minus[0])- (mu2_plus[0]-mu2_minus[0]))/(4*eps*eps);
                    }
                }
                ofs_gp_num_grad << num_grad.transpose() << std::endl;
                ofs_gp_num_hessian << num_hessian.norm() << std::endl;
            }
        }
        auto t4_gp = high_resolution_clock::now();

        ////////////////////////////////////////////////////////////////
        ///                     optimized GP                         ///
        ////////////////////////////////////////////////////////////////
        /*
        // an alternative is to optimize the hyper-parameters
        // in that case, we need a kernel with hyper-parameters that are designed to be optimized
        using Kernel2_t = kernel::SquaredExpARD<Params>;
        using Mean_t = mean::Data<Params>;
        using GP2_t = model::GP<Params, Kernel2_t, Mean_t, model::gp::KernelLFOpt<Params>>;

        GP2_t gp_ard(inputDim, 1);
        // do not forget to call the optimization!
        auto t1_gp_ard = high_resolution_clock::now();
        gp_ard.compute(samples, observations, false);
        gp_ard.optimize_hyperparams();
        auto t2_gp_ard = high_resolution_clock::now();
        // Get alpha, kernel for computing gradient
        auto alpha = gp_ard.alpha();
        auto kernel = gp_ard.kernel_function();
        auto log_params = kernel.h_params();
        double l = std::exp(log_params(0));

        std::cout << "l: " << log_params << std::endl;
        // write the predicted data in a file (e.g. to be plotted)
        std::ofstream ofs_ard(xp_folder+"/gp_ard.dat");
        std::ofstream ofs_grad(xp_folder+"/grad_mu_gp.dat");
        std::ofstream ofs_hessian(xp_folder+"/hessian_mu_gp.dat");
    
        for (int i = 0; i < N; ++i) {
            Eigen::VectorXd v = samples.at(i);
            Eigen::VectorXd mu;
            double sigma;
            std::tie(mu, sigma) = gp_ard.query(v);
            ofs_ard << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(inputDim);
            //Eigen::VectorXd hessian = Eigen::VectorXd::Zero(inputDim);
            for (size_t j = 0; j < N; j++) {
                grad = grad + kernel(v, samples.at(j))*alpha(j)*(samples.at(j)-v)/std::pow(l,2);
                //hessian = hessian + kernel(v, samples.at(j))*alpha(j)*((samples.at(j)-v)*(samples.at(j)-v)/std::pow(l,4)-Eigen::VectorXd::Ones(1)/std::pow(l,2));
            }
            //std::cout << "i: " << i <<  grad.transpose() << std::endl;
            ofs_grad << grad.transpose() << std::endl;
            //ofs_hessian << hessian << std::endl;
        }
        auto t3_gp_ard = high_resolution_clock::now();
        std::ofstream ofs_ard_test(xp_folder+"/gp_ard_test.dat");
        for (int i = 0; i < N; ++i) {
                Eigen::VectorXd v = samples_test.at(i);
                Eigen::VectorXd mu;
                double sigma;
                std::tie(mu, sigma) = gp_ard.query(v);
                ofs_ard_test << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
            }
        auto t4_gp_ard = high_resolution_clock::now();
        */
        std::ofstream ofs_times(xp_folder+"/times.dat");
        auto gp_train = duration_cast<microseconds>(t2_gp - t1_gp).count();
        auto gp_eval = duration_cast<microseconds>(t3_gp - t2_gp).count();
        auto gp_test = duration_cast<microseconds>(t4_gp - t3_gp).count();
/*         auto gp_ard_train = duration_cast<microseconds>(t2_gp_ard - t1_gp_ard).count();
        auto gp_ard_eval = duration_cast<microseconds>(t3_gp_ard - t2_gp_ard).count();
        auto gp_ard_test = duration_cast<microseconds>(t4_gp_ard - t3_gp_ard).count(); 
        ofs_times << gp_train << " " << gp_eval << " " << gp_test << " " <<  gp_ard_train << " " << gp_ard_eval << " " << gp_ard_test <<  std::endl;*/
        ofs_times << gp_train << " " << gp_eval << " " << gp_test << " " <<  std::endl;
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
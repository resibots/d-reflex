#include <fstream>
#include <limbo/kernel/exp.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/tools.hpp>
#include <limbo/tools/macros.hpp>

#include <limbo/serialize/text_archive.hpp>

// this tutorials shows how to use a Gaussian process for regression

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

int main(int argc, char** argv)
{

    // our data (1-D inputs, 1-D outputs)
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
    std::ofstream ofs("gp.dat");
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
    std::ofstream ofs_ard("gp_ard.dat");
    std::ofstream ofs_grad("grad_mu_gp.dat");
    std::ofstream ofs_hessian("hessian_mu_gp.dat");
    for (int i = 0; i < 100; ++i) {
        Eigen::VectorXd v = tools::make_vector(i / 100.0).array();
        Eigen::VectorXd mu;
        double sigma;
        std::tie(mu, sigma) = gp_ard.query(v);
        ofs_ard << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(1);
        Eigen::VectorXd hessian = Eigen::VectorXd::Zero(1);
        for (size_t i = 0; i < N; i++) {
            grad = grad + kernel(v, samples.at(i))*alpha(i)*(samples.at(i)-v)/std::pow(l,2);
            hessian = hessian + kernel(v, samples.at(i))*alpha(i)*((samples.at(i)-v)*(samples.at(i)-v)/std::pow(l,4)-Eigen::VectorXd::Ones(1)/std::pow(l,2));
        }
        ofs_grad << grad << std::endl;
        ofs_hessian << hessian << std::endl;
    }

    // write the data to a file (useful for plotting)
    std::ofstream ofs_data("data.dat");
    for (size_t i = 0; i < samples.size(); ++i)
        ofs_data << samples[i].transpose() << " " << observations[i].transpose() << std::endl;

/*     // Sometimes is useful to save an optimized GP
    gp_ard.save<serialize::TextArchive>("myGP");

    // Later we can load -- we need to make sure that the type is identical to the one saved
    gp_ard.load<serialize::TextArchive>("myGP"); */
    return 0;
}
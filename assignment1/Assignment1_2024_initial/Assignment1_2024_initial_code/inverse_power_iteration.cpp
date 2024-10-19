#include "inverse_power_iteration.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double inverse_power_iteration::solve(const linear_algebra::square_matrix& A, const std::vector<double>& x0) const
    {
        linear_algebra::square_matrix L, U;
        lu(A, L, U);

        double residual = {};
        double increment = {};

        std:: vector <double> max_eig;
        max_eig.push_back(linear_algebra::scalar(x0,A*x0));

        std::vector < std::vector <double> > x;
        x.push_back(x0);

        for ( int k = 0 ; k< max_it; k++ ){

            std::vector<double> y = linear_algebra::forwardsolve(L, x.back());
            std::vector<double> z = linear_algebra::backsolve(U, y);

            std::vector<double> x_old = x.back();
            linear_algebra::normalize(z);
            x.push_back(z);

            max_eig.push_back(linear_algebra::scalar(x.back(),A*x.back()));

            residual = linear_algebra::norm(A*x_old - max_eig.back() * x_old);
            increment = std::abs(max_eig.back()-max_eig[max_eig.size()-2]) / std::abs(max_eig.back());

            // std::cout << "increment :" <<increment << std::endl;
            // std::cout << "residual :" <<residual << std::endl;
            // std::cout << "iter :" <<k << std::endl;

            if (increment < tolerance && residual < tolerance) {
                x.clear();
                return (max_eig.back());
            }
        }
        return (max_eig.back());
    }

    bool inverse_power_iteration::converged(const double& residual, const double& increment) const
    {
        bool conv;

        switch(termination) {
            case(RESIDUAL):
                conv = residual < tolerance;
                break;
            case(INCREMENT):
                conv = increment < tolerance;
                break;
            case(BOTH):
                conv = residual < tolerance && increment < tolerance;
                break;
            default:
                conv = false;
        }
        return conv;
    }

} // eigenvalue
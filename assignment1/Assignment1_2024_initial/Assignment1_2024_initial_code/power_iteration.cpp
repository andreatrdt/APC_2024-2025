#include "power_iteration.h"
#include <vector>


namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double power_iteration::solve(const linear_algebra::square_matrix& A, const std::vector<double>& x0) const
    {
        // residual and increment initialization
        double residual = {};
        double increment = {};

        // check norm x0
        if ( linear_algebra::norm(x0) != 1) {
            std::cout<< " || x0 || != 1 " << std::endl;
            return 0;
        }

        // eigenvalue vector initialization
        std:: vector <double> max_eig;
        max_eig.push_back(linear_algebra::scalar(x0,A*x0));

        // vector of vectors x initialization
        std::vector < std::vector <double> > x;
        x.push_back(x0);

        for ( int k = 0 ; k< max_it ; k++ ){

            std::vector <double> z;
            // update z
            z = A*x.back();

            std::vector<double> x_old = x.back();
            linear_algebra::normalize(z);
            // update x
            x.push_back(z);

            // update eigenvalue
            max_eig.push_back(linear_algebra::scalar(x.back(),A*x.back()));

            // res = || A * x^k - v^k * x^k ||
            residual = linear_algebra::norm(A*x_old - max_eig.back() * x_old);
            // increment = | v^(k+1) - v^k | / | v^k |
            increment = std::abs(max_eig.back()-max_eig[max_eig.size()-2]) / std::abs(max_eig.back());

            bool conv = converged(residual,increment);
            z.clear();
            if (conv) {
                // if conditions are met return eigenvalue else repeat
                return max_eig.back();
            }
        }
        // return eigenvalue it converged to if reached max iterations
        return max_eig.back();
    }

    bool power_iteration::converged(const double& residual, const double& increment) const
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
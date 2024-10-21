#include "inverse_power_iteration.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double inverse_power_iteration::solve(const linear_algebra::square_matrix& A, const std::vector<double>& x0) const
    {
        // check norm of x0
        if ( linear_algebra::norm(x0) != 1) {
            std::cout<< " || x0 || != 1 " << std::endl;
            return 0;
        }
        // initialize matrices for LU factorization
        linear_algebra::square_matrix L, U;
        lu(A, L, U);

        //  increment and residual initialization
        double residual = {};
        double increment = {};

        // solution for initial condition of eigenvalue
        std::vector<double> temp = linear_algebra::forwardsolve(L, x0);
        std::vector<double> Ax0 = linear_algebra::backsolve(U, temp);

        // initialize eigenvalue
        std:: vector <double> max_eig;
        max_eig.push_back(linear_algebra::scalar(x0,Ax0));

        // vector of vectors initialization
        std::vector < std::vector <double> > x;
        x.push_back(x0);

        for ( int k = 0 ; k< max_it; k++ ){

            // solving LU system for z
            std::vector<double> y = linear_algebra::forwardsolve(L, x.back());
            std::vector<double> z = linear_algebra::backsolve(U, y);

            std::vector<double> x_old = x.back();
            linear_algebra::normalize(z);
            // update x
            x.push_back(z);

            // update eigenvalue
            std::vector<double> temp_eig = linear_algebra::forwardsolve(L, x.back());
            max_eig.push_back(linear_algebra::scalar(x.back(), linear_algebra::backsolve(U, temp_eig)));

            std::vector<double> temp_res = linear_algebra::forwardsolve(L, x_old);
            // res = || A * x^k - v^k * x^k ||
            residual = linear_algebra::norm( linear_algebra::backsolve(U, temp_res)- max_eig.back() * x_old);
            // increment = | v^(k+1) - v^k | / | v^k |
            increment = std::abs(max_eig.back()-max_eig[max_eig.size()-2]) / std::abs(max_eig.back());

            if (increment < tolerance && residual < tolerance) {
                // if conditions are met return eigenvalue else repeat
                return 1/max_eig.back();
            }
        }
        // return eigenvalue it converged to if reached max iterations
        return 1/max_eig.back();
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
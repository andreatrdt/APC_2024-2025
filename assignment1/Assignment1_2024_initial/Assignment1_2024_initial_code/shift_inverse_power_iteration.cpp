#include "shift_inverse_power_iteration.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double shift_inverse_power_iteration::solve(const linear_algebra::square_matrix& A, const double& mu, const std::vector<double>& x0) const
    {

        linear_algebra::square_matrix A_tilde = A;
        for (int i=0 ; i<x0.size() ; i++) {
            A_tilde(i,i) = A_tilde(i,i)- mu;
        }

        inverse_power_iteration method;
        return method.solve(A_tilde,x0);

    }

} // eigenvalue
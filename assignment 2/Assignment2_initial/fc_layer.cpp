#include "fc_layer.hpp"

namespace convnet {

    fc_layer::fc_layer(std::size_t s_in, std::size_t s_out) : size_in(s_in), size_out(s_out) {
        initialize();
    };

    void fc_layer::initialize() {
        weights = matrix(size_out, size_in);
        weights.initialize_with_random_normal(0.0, 2.0 / (size_in + size_out));
    };

    std::vector<double> fc_layer::apply_activation(const std::vector<double> &z) const {
        return act_function.apply(z);
    };

    std::vector<double> fc_layer::compute(const std::vector<double> &inputs) const {
        // Check if the size of the input vector matches the expected size for the fully connected layer
        if (inputs.size() != size_in) {
            throw std::invalid_argument("Input size mismatch");
        }
        // Check if the number of rows in the weight matrix matches the expected size of the output
        if (weights.get_n_rows() != size_out) {
           throw std::invalid_argument("Weights size mismatch");
        }

        // Perform the dot product of the weight matrix and the input vector
        return weights.dot(inputs);
    };

    std::vector<double> fc_layer::forward_pass(const std::vector<double> &inputs) const {
        // apply activation function to fully connected layer
        return apply_activation(compute(inputs));
    };

    std::vector<double> fc_layer::get_parameters() const {
        return weights.get_values();
    }

    void fc_layer::set_parameters(const std::vector<double> parameters) {
        weights.set_values(parameters);
    }

    const matrix &fc_layer::get_weights() const {
        return weights;
    }

    void fc_layer::set_weights(const matrix &ws) {
        weights = ws;
    }

    std::size_t fc_layer::get_size_in() const {
        return size_in;
    }

    std::size_t fc_layer::get_size_out() const {
        return size_out;
    }


} // namespace
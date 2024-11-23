#include "max_pooling_layer.hpp"

namespace convnet {

    max_pooling_layer::max_pooling_layer(std::size_t s_filter, std::size_t strd) {
        size_filter = s_filter;
        stride = strd;
    };

     tensor_3d max_pooling_layer::evaluate(const tensor_3d &inputs) const {

        // Ensure filter size is valid
        if (size_filter <= 0) {
            throw std::invalid_argument("Filter size must be greater than zero");
        }

        // Ensure stride is valid
        if (stride <= 0) {
            throw std::invalid_argument("Stride must be greater than zero");
        }

        // Ensure input dimensions are sufficient for the pooling operation
        if (inputs.get_height() < size_filter || inputs.get_width() < size_filter) {
            throw std::invalid_argument("Input tensor dimensions must be greater than or equal to the filter size");
        }

        // Calculate the output dimensions based on input size, filter size, and stride
        std::size_t const H_out = (inputs.get_height() - size_filter) / stride + 1;
        std::size_t const W_out = (inputs.get_width() - size_filter) / stride + 1;

        // Ensure output dimensions are valid
        if (H_out <= 0 || W_out <= 0) {
            throw std::invalid_argument("Invalid output dimensions; check input size, filter size, and stride");
        }

        // Initialize the output tensor with zeros
        tensor_3d evaluate(H_out, W_out, inputs.get_depth());
        evaluate.initialize_with_zeros();

        // Perform the max-pooling operation for each depth slice
        for (std::size_t d = 0; d < inputs.get_depth(); ++d) { // Loop over the depth dimension
            for (std::size_t i = 0; i < H_out; ++i) {          // Loop over the output rows
                for (std::size_t j = 0; j < W_out; ++j) {      // Loop over the output columns

                    // Initialize the maximum value for the current pooling region
                    double max_val = -std::numeric_limits<double>::infinity();

                    // Find the maximum value
                    for (std::size_t h = 0; h < size_filter; ++h) { // Loop over filter height
                        for (std::size_t w = 0; w < size_filter; ++w) { // Loop over filter width

                            // Calculate input indices
                            std::size_t input_i = i * stride + h;
                            std::size_t input_j = j * stride + w;

                            // Ensure input indices are within bounds
                            if (input_i < inputs.get_height() && input_j < inputs.get_width()) {
                                // Update the maximum value for the pooling region
                                max_val = std::max(max_val, inputs(input_i, input_j, d));
                            }
                        }
                    }

                    // Assign the maximum value to the output tensor
                    evaluate(i, j, d) = max_val;
                }
            }
        }

        // Return the resulting tensor after max pooling
        return evaluate;
    }




    tensor_3d max_pooling_layer::apply_activation(const tensor_3d &Z) const {
        return Z;
    };

    tensor_3d max_pooling_layer::forward_pass(const tensor_3d &inputs) const {

        // apply the activation function (sigmoid) to the evaluated inputs
        return apply_activation(evaluate(inputs));
    };

    // Do nothing since max pooling has no learnable parameter
    void max_pooling_layer::set_parameters(const std::vector<std::vector<double>> parameters) {}

} // namespace
#include "max_pooling_layer.hpp"

namespace convnet {

    max_pooling_layer::max_pooling_layer(std::size_t s_filter, std::size_t strd) {
        size_filter = s_filter;
        stride = strd;
    };

     tensor_3d max_pooling_layer::evaluate(const tensor_3d &inputs) const {

        // Calculate output dimensions
        std::size_t const H_out = (inputs.get_height() - size_filter) / stride + 1;
        std::size_t const W_out = (inputs.get_width() - size_filter) / stride + 1;

        // Initialize output tensor with zeros
        tensor_3d evaluate(H_out, W_out, inputs.get_depth());
        evaluate.initialize_with_zeros();

        // Perform max-pooling operation for each depth
        for (std::size_t d = 0; d < inputs.get_depth(); ++d) {
            for (std::size_t i = 0; i < H_out; ++i) {
                for (std::size_t j = 0; j < W_out; ++j) {

                    // Initialize maximum value
                    double max_val = -std::numeric_limits<double>::infinity();

                    // Find maximum value
                    for (std::size_t h = 0; h < size_filter; ++h) {
                        for (std::size_t w = 0; w < size_filter; ++w) {

                            // Input indices
                            std::size_t input_i = i * stride + h;
                            std::size_t input_j = j * stride + w;

                            // Update maximum value
                            max_val = std::max(max_val, inputs(input_i, input_j, d));
                        }
                    }
                    // Assign maximum value to the output tensor
                    evaluate(i, j, d) = max_val;
                }
            }
        }

        // Return resulting tensor after max pooling
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
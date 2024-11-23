#include "convolutional_layer.hpp"

namespace convnet {

    convolutional_layer::convolutional_layer(std::size_t _s_filter, std::size_t _prev_depth, std::size_t _n_filters,
                                             std::size_t _s_stride, std::size_t _s_padding)
            : s_filter(_s_filter), prev_depth(_prev_depth), n_filters(_n_filters), s_stride(_s_stride),
              s_padding(_s_padding) {
        initialize();
    }

    void convolutional_layer::initialize() {
        for (std::size_t it = 0; it < n_filters; ++it) {
            tensor_3d filter(s_filter, s_filter, prev_depth);
            filter.initialize_with_random_normal(0.0, 3.0 / (2 * s_filter + prev_depth));
            filters.push_back(filter);
        }
    }

    tensor_3d convolutional_layer::evaluate(const tensor_3d &inputs) const {

    // Ensure there are filters in the layer
    if (filters.empty()) {
        throw std::invalid_argument("No filters provided for convolutional layer");
    }

    // Ensure the depth of input tensor matches the depth of each filter
    for (const auto &filter : filters) {
        if (filter.get_depth() != inputs.get_depth()) {
            throw std::invalid_argument("Depth of input tensor must match filter depth");
        }
    }

    // Calculate output dimensions
    std::size_t const H_out = (inputs.get_height() - filters[0].get_height() + 2 * s_padding) / s_stride + 1;
    std::size_t const W_out = (inputs.get_width() - filters[0].get_width() + 2 * s_padding) / s_stride + 1;

    // Ensure output dimensions are valid
    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("Invalid output dimensions; check input size, filter size, stride, or padding");
    }

    // Initialize the output tensor with zeros
    tensor_3d evaluate(H_out, W_out, n_filters);
    evaluate.initialize_with_zeros();

        // Ensure input dimensions are greater than or equal to filter dimensions
        if (inputs.get_height() < filters[0].get_height() || inputs.get_width() < filters[0].get_width()) {
            throw std::invalid_argument("Input tensor dimensions must be greater than or equal to filter dimensions");
        }

        // Perform the convolution operation
        for (std::size_t i = 0; i < H_out; ++i) {           // Loop over output rows
            for (std::size_t j = 0; j < W_out; ++j) {       // Loop over output columns
                for (std::size_t k = 0; k < n_filters; ++k) {  // Loop over each filter
                    for (std::size_t h = 0; h < filters[k].get_height(); ++h) { // Loop over filter height
                        for (std::size_t w = 0; w < filters[k].get_width(); ++w) { // Loop over filter width
                            for (std::size_t d = 0; d < filters[k].get_depth(); ++d) { // Loop over filter depth
                                // Calculate input indices
                                std::size_t input_i = i * s_stride + h;
                                std::size_t input_j = j * s_stride + w;

                                // Ensure input indices are within bounds
                                if (input_i < inputs.get_height() && input_j < inputs.get_width()) {
                                    // Perform the element-wise multiplication and accumulate
                                    evaluate(i, j, k) += inputs(input_i, input_j, d) * filters[k](h, w, d);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Return tensor
        return evaluate;
    }


    tensor_3d convolutional_layer::apply_activation(const tensor_3d &Z) const {
        return act_function.apply(Z);
    }

    tensor_3d convolutional_layer::forward_pass(const tensor_3d &inputs) const {

        // apply activation function (relu) to the evaluated input
        return apply_activation(evaluate(inputs));
    }


    std::vector<std::vector<double>> convolutional_layer::get_parameters() const {
        std::vector<std::vector<double> > parameters;
        for (tensor_3d filter: filters) {
            parameters.push_back(filter.get_values());
        }
        return parameters;
    }

    void convolutional_layer::set_parameters(const std::vector<std::vector<double>> parameters) {
        for (std::size_t i = 0; i < n_filters; ++i) {
            filters[i].set_values(parameters[i]);
        }
    }

} // namespace
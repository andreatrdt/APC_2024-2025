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
        std::size_t H_out = (inputs.get_height() - filters[0].get_height() + 2*s_padding) / s_stride +1;
        std::size_t W_out = (inputs.get_width() - filters[0].get_width() + 2*s_padding) / s_stride +1;

        tensor_3d evaluate(H_out,W_out,n_filters);

        for (std::size_t i = 0 ; i < H_out; ++i) {
            for ( std::size_t j = 0; j < W_out; ++j) {
                for ( std::size_t k = 0 ; k < n_filters; ++k) {
                    for (std::size_t h = 0 ; h < filters[k].get_height(); ++h) {
                        for ( std::size_t w = 0; w < filters[k].get_width(); ++w) {
                            for ( std::size_t d = 0 ; d < filters[k].get_depth(); ++d) {

                                evaluate(i,j,k) = inputs(i* s_stride + h,j*s_stride + w,d) * filters[k].operator()(h,w,d) ;
                            }
                        }
                    }
                }
            }
        }

        return evaluate;
    }

    tensor_3d convolutional_layer::apply_activation(const tensor_3d &Z) const {
        return act_function.apply(Z);
    }

    tensor_3d convolutional_layer::forward_pass(const tensor_3d &inputs) const {


        // TO BE CHECKED !!!!
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
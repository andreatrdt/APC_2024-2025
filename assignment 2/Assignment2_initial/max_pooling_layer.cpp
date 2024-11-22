#include "max_pooling_layer.hpp"

namespace convnet {

    max_pooling_layer::max_pooling_layer(std::size_t s_filter, std::size_t strd) {
        size_filter = s_filter;
        stride = strd;
    };

    tensor_3d max_pooling_layer::evaluate(const tensor_3d &inputs) const {

        std::size_t H_out = (inputs.get_height() - size_filter ) / stride +1;
        std::size_t W_out = (inputs.get_width() - size_filter ) / stride +1;

        tensor_3d evaluate(H_out,W_out,inputs.get_depth());

        for (std::size_t i = 0 ; i < H_out; ++i) {
            for ( std::size_t j = 0; j < W_out; ++j) {
                for ( std::size_t k = 0 ; k < inputs.get_depth(); ++k) {

                    double max_val = std::numeric_limits<double>::lowest();
                    for (std::size_t h = 0 ; h < size_filter; ++h) {
                        for ( std::size_t w = 0; w < size_filter; ++w) {
                            for ( std::size_t d = 0 ; d < inputs.get_depth(); ++d) {
                                max_val = std::max( max_val, inputs(i * stride + h ,j * stride + w,d) );
                            }
                        }
                    }
                    evaluate(i,j,k) = max_val;
                }
            }
        }

        return evaluate;
    };


    tensor_3d max_pooling_layer::apply_activation(const tensor_3d &Z) const {
        return Z;
    };

    tensor_3d max_pooling_layer::forward_pass(const tensor_3d &inputs) const {

        /* YOUR CODE SHOULD GO HERE */

    };

    // Do nothing since max pooling has no learnable parameter
    void max_pooling_layer::set_parameters(const std::vector<std::vector<double>> parameters) {}

} // namespace
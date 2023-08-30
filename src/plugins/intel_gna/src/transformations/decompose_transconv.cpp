// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decompose_transconv.hpp"

#include "openvino/core/node_vector.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/opsets/opset11.hpp"
#include "utils/transformation_helper.hpp"

#define PAD_VALUE ((size_t)-1)

using namespace ov::intel_gna;
using namespace ov::intel_gna::pass;

// using ov::op::v0::Constant;
using namespace ov::opset11;

// TODO: find openvino coutnerpart
static void BuildKernelMap1D(size_t dim_new,
                             size_t pad_outer,
                             size_t pad_additional,
                             size_t stride,
                             size_t n_weights,
                             bool do_folding,
                             std::vector<std::vector<std::vector<size_t>>>& kernel_list,  // list of kernel layouts
                             std::vector<std::vector<int32_t>>& input_list)               // list of input endpoints
{
    // Rather than calculating the indices of the kernels and inputs operations on-the-fly
    // we simply construct vectors indicating the position of each kernel row/column and non-zero input.
    std::vector<int32_t> input_index(dim_new, -1);  // holds index of input row/column or -1 if padding
    // Find input layout
    int32_t j = 0;
    for (size_t i = pad_outer; i < dim_new - pad_outer - pad_additional; i += stride) {
        input_index[i] = j++;
    }
    std::vector<std::vector<size_t>> kernel_pos;  // offsets for convolution kernel
    // Find kernel layout required for each virtual convolution output
    int32_t input_start_prev = -1;
    int32_t input_end_prev = -1;
    for (size_t i = 0; i < dim_new - n_weights + 1; i++) {
        int32_t input_start = -1;
        int32_t input_end = -1;
        kernel_pos.resize(kernel_pos.size() + 1);
        for (size_t j = 0; j < n_weights; j++) {
            if (input_index[i + j] >= 0) {  // do not include multiply by zero (indicated by index == -1)
                if (input_start < 0) {
                    input_start = input_index[i + j];
                    input_end = input_start;
                } else {
                    input_end = input_index[i + j];
                }
                kernel_pos[kernel_pos.size() - 1].push_back(j);
            }
        }
        if (input_start_prev < 0) {
            input_start_prev = input_start;
            input_end_prev = input_end;
        } else {
            if ((input_start != input_start_prev) || (input_end != input_end_prev)) {  // need a new layer
                std::vector<size_t> new_row = kernel_pos[kernel_pos.size() - 1];
                kernel_list.push_back(kernel_pos);  // save kernel layout
                kernel_list[kernel_list.size() - 1]
                    .pop_back();  // remove last row since it is really part of the next kernel
                std::vector<int32_t> input_range;
                input_range.push_back(input_start_prev);
                input_range.push_back(input_end_prev);
                input_list.push_back(input_range);
                kernel_pos.clear();
                kernel_pos.push_back(new_row);
                input_start_prev = input_start;
                input_end_prev = input_end;
            }
        }
    }

    // save final kernel and input layout
    std::vector<int32_t> input_range;
    input_range.push_back(input_start_prev);
    input_range.push_back(input_end_prev);
    input_list.push_back(input_range);
    std::vector<std::vector<size_t>> new_kernel;
    for (uint32_t m = 0; m < kernel_pos.size(); m++) {
        std::vector<size_t> new_kernel_element;
        for (uint32_t n = 0; n < kernel_pos[m].size(); n++) {
            new_kernel_element.push_back(kernel_pos[m][n]);
        }
        new_kernel.push_back(new_kernel_element);
    }
    kernel_list.push_back(new_kernel);  // save kernel layout

    // extend kernels by folding into output channels if possible
    if (do_folding) {
        if (input_list.size() > 2) {
            size_t i = 2;
            while (i < input_list.size()) {
                auto curr_low = input_list[i][0];
                auto prev_low = input_list[i - 1][0];
                auto curr_high = input_list[i][1];
                auto prev_high = input_list[i - 1][1];
                if ((curr_low >= prev_low) && (curr_high <= prev_high)) {  // time periods overlap
                    std::vector<std::vector<size_t>> append_kernel;
                    for (size_t j = 0; j < kernel_list[i].size(); j++) {
                        std::vector<size_t> new_kernel(kernel_list[i - 1][0].size(), PAD_VALUE);
                        for (size_t k = 0; k < kernel_list[i - 1][0].size(); k++) {
                            for (size_t m = 0; m < kernel_list[i][j].size(); m++) {
                                if (input_list[i][0] + m == input_list[i - 1][0] + k) {
                                    new_kernel[k] = kernel_list[i][j][m];
                                }
                            }
                        }
                        append_kernel.push_back(new_kernel);
                    }
                    for (size_t j = 0; j < append_kernel.size(); j++) {
                        kernel_list[i - 1].push_back(append_kernel[j]);
                    }
                    kernel_list.erase(kernel_list.begin() + i);  // remove redundant kernel
                    input_list.erase(input_list.begin() + i);    // remove redundant input
                } else {
                    i++;
                }
            }
        }
    }

    // consolidate convolutions when possible
    size_t i = 1;
    while (i < input_list.size()) {
        if (kernel_list[i].size() == kernel_list[i - 1].size()) {
            bool kernel_match = true;
            for (size_t j = 0; j < kernel_list[i].size(); j++) {
                if (kernel_list[i][j].size() == kernel_list[i - 1][j].size()) {
                    for (size_t j = 0; j < kernel_list[i].size(); j++) {
                        if (kernel_list[i][j] != kernel_list[i - 1][j]) {
                            kernel_match = false;
                        }
                    }
                } else {
                    kernel_match = false;
                }
            }

            if (kernel_match) {
                input_list[i - 1][1] = input_list[i][1];     // increase input length
                kernel_list.erase(kernel_list.begin() + i);  // remove redundant kernel
                input_list.erase(input_list.begin() + i);    // remove redundant input
            } else {
                i++;
            }
        } else {
            i++;
        }
    }

    {
        size_t output_index = 0;
        for (size_t i = 0; i < input_list.size(); i++) {
            printf("position %zu:\nx%d .. x%d\n", output_index, input_list[i][0], input_list[i][1]);
            for (uint32_t m = 0; m < kernel_list[i].size(); m++) {
                for (uint32_t n = 0; n < kernel_list[i][m].size(); n++) {
                    if (kernel_list[i][m][n] == PAD_VALUE) {
                        printf("00 ");
                    } else {
                        printf("k%lu ", kernel_list[i][m][n]);
                    }
                }
                printf("\n");
            }
            output_index +=
                kernel_list[i].size() * ((input_list[i][1] - input_list[i][0] + 1) - kernel_list[i][0].size() + 1);
            printf("\n");
        }
    }

    // reverse kernel -- OpenVINO stores them backwards
    for (size_t i = 0; i < input_list.size(); i++) {
        for (uint32_t m = 0; m < kernel_list[i].size(); m++) {
            for (uint32_t n = 0; n < kernel_list[i][m].size(); n++) {
                if (kernel_list[i][m][n] != PAD_VALUE) {
                    kernel_list[i][m][n] = n_weights - kernel_list[i][m][n] - 1;
                }
            }
        }
    }
}

// TODO: find openvino coutnerpart
static void InsertActivation(ov::OutputVector& upstream,
                             std::shared_ptr<PRelu> prelu,
                             std::shared_ptr<Relu> relu,
                             std::shared_ptr<Sigmoid> sigmoid,
                             std::shared_ptr<Tanh> tanh) {
    if (prelu) {
        auto slope_const = std::dynamic_pointer_cast<Constant>(prelu->input_value(1).get_node_shared_ptr());
        const float* slope_ptr = slope_const->get_data_ptr<float>();
        std::vector<float> new_slope(1, 0.0f);
        float* new_slope_ptr = new_slope.data();
        *new_slope_ptr = *slope_ptr;
        auto new_prelu_slope = Constant::create(ngraph::element::f32, ov::Shape{1ull}, new_slope);
        auto new_prelu = std::make_shared<PRelu>(upstream[0], new_prelu_slope->output(0));
        upstream[0] = new_prelu->output(0);
    } else if (relu) {
        auto new_relu = std::make_shared<Relu>(upstream[0]);
        upstream[0] = new_relu->output(0);
    } else if (sigmoid) {
        auto new_sigmoid = std::make_shared<Sigmoid>(upstream[0]);
        upstream[0] = new_sigmoid->output(0);
    } else if (tanh) {
        auto new_tanh = std::make_shared<Tanh>(upstream[0]);
        upstream[0] = new_tanh->output(0);
    }
}

bool DecomposeTransConv::run_on_model(const std::shared_ptr<ov::Model>& m) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : m->get_ordered_ops()) {
        auto conv = std::dynamic_pointer_cast<ov::op::v1::ConvolutionBackpropData>(node);
        if (nullptr == conv) {
            continue;
        }

        const Output<Node>& input = conv->input_value(0);
        const Output<Node>& weights = conv->input_value(1);
        auto conv_name = conv->get_friendly_name();
        auto input_shape = input.get_shape();
        auto weights_shape = weights.get_shape();
        auto output_shape = conv->get_output_shape();
        auto dilations = conv->get_dilations();
        auto output_padding = conv->get_output_padding();
        auto pads_begin = conv->get_pads_begin();
        auto pads_end = conv->get_pads_end();
        auto strides = conv->get_strides();
        auto weights_const = std::dynamic_pointer_cast<Constant>(conv->input_value(1).get_node_shared_ptr());

        if (weights_shape.size() ==
            4) {  // ConvTranspose2d: 4D input with N=1, 4D filters, 2D stride, 2D dilation, 2D padding
            if (input_shape.size() != 4 || pads_begin.size() != 2 || pads_end.size() != 2 || dilations.size() != 2 ||
                strides.size() != 2 || input_shape[0] != 1 ||
                pads_begin[0] != pads_end[0] ||  // only symmetric padding is supported
                pads_begin[1] != pads_end[1] ||
                strides[0] > weights_shape[2] ||  // only support cases where kernel overlaps input
                strides[1] > weights_shape[3]) {
                continue;
            }
        } else if (weights_shape.size() ==
                   3) {  // ConvTranspose1d: 3D input with N=1, 3D filters, 1D stride, 1D dilation, 1D padding
            if (input_shape.size() != 3 || pads_begin.size() != 1 || pads_end.size() != 1 || dilations.size() != 1 ||
                strides.size() != 1 || input_shape[0] != 1 ||
                pads_begin[0] != pads_end[0] ||   // only symmetric padding is supported
                strides[0] > weights_shape[2]) {  // only support cases where kernel overlaps input
                continue;
            }
        } else {
            continue;
        }

        // find Transpose-->Convolution--><Add>--><Activation>-->Transpose pattern else skip
        const Output<Node>& parent = conv->input_value(0);
        auto children = conv->output(0).get_target_inputs();
        if (children.size() != 1) {
            continue;
        }
        auto transpose_before = std::dynamic_pointer_cast<Transpose>(parent.get_node()->shared_from_this());
        if (transpose_before == nullptr) {
            continue;
        }
        auto add_after = std::dynamic_pointer_cast<Add>(children.begin()->get_node()->shared_from_this());
        auto fq_after = std::dynamic_pointer_cast<FakeQuantize>(children.begin()->get_node()->shared_from_this());
        auto prelu_after = std::dynamic_pointer_cast<PRelu>(children.begin()->get_node()->shared_from_this());
        auto relu_after = std::dynamic_pointer_cast<Relu>(children.begin()->get_node()->shared_from_this());
        auto sigmoid_after = std::dynamic_pointer_cast<Sigmoid>(children.begin()->get_node()->shared_from_this());
        auto tanh_after = std::dynamic_pointer_cast<Tanh>(children.begin()->get_node()->shared_from_this());
        auto transpose_after = std::dynamic_pointer_cast<Transpose>(children.begin()->get_node()->shared_from_this());
        auto reshape_after = std::dynamic_pointer_cast<Reshape>(children.begin()->get_node()->shared_from_this());
        if (add_after != nullptr) {
            auto add_children = add_after->output(0).get_target_inputs();
            if (add_children.size() != 1) {
                continue;
            }
            fq_after = std::dynamic_pointer_cast<FakeQuantize>(add_children.begin()->get_node()->shared_from_this());
            prelu_after = std::dynamic_pointer_cast<PRelu>(add_children.begin()->get_node()->shared_from_this());
            relu_after = std::dynamic_pointer_cast<Relu>(add_children.begin()->get_node()->shared_from_this());
            sigmoid_after = std::dynamic_pointer_cast<Sigmoid>(add_children.begin()->get_node()->shared_from_this());
            tanh_after = std::dynamic_pointer_cast<Tanh>(add_children.begin()->get_node()->shared_from_this());
            transpose_after =
                std::dynamic_pointer_cast<Transpose>(add_children.begin()->get_node()->shared_from_this());
            reshape_after = std::dynamic_pointer_cast<Reshape>(add_children.begin()->get_node()->shared_from_this());
        }
        if (fq_after != nullptr) {
            auto fq_children = fq_after->output(0).get_target_inputs();
            if (fq_children.size() != 1) {
                continue;
            }
            prelu_after = std::dynamic_pointer_cast<PRelu>(fq_children.begin()->get_node()->shared_from_this());
            relu_after = std::dynamic_pointer_cast<Relu>(fq_children.begin()->get_node()->shared_from_this());
            sigmoid_after = std::dynamic_pointer_cast<Sigmoid>(fq_children.begin()->get_node()->shared_from_this());
            tanh_after = std::dynamic_pointer_cast<Tanh>(fq_children.begin()->get_node()->shared_from_this());
            transpose_after = std::dynamic_pointer_cast<Transpose>(fq_children.begin()->get_node()->shared_from_this());
            reshape_after = std::dynamic_pointer_cast<Reshape>(fq_children.begin()->get_node()->shared_from_this());
        }
        if ((transpose_after == nullptr) && (reshape_after == nullptr)) {
            OutputVector upstream;
            if (prelu_after) {
                upstream.push_back(prelu_after->output(0));
            } else if (relu_after) {
                upstream.push_back(relu_after->output(0));
            } else if (sigmoid_after) {
                upstream.push_back(sigmoid_after->output(0));
            } else if (tanh_after) {
                upstream.push_back(tanh_after->output(0));
            }
            if (upstream.size() > 0) {
                auto act_children = upstream[0].get_target_inputs();
                if (act_children.size() != 1) {
                    continue;
                }
                transpose_after =
                    std::dynamic_pointer_cast<Transpose>(act_children.begin()->get_node()->shared_from_this());
                reshape_after =
                    std::dynamic_pointer_cast<Reshape>(act_children.begin()->get_node()->shared_from_this());
            } else {
                continue;
            }
        }
        if ((transpose_after == nullptr) && (reshape_after == nullptr)) {
            continue;
        }

        // find padded dimensions of virtual equivalent convolution

        bool is_1D = (weights_shape.size() == 3);
        size_t N = input_shape[0];
        size_t C = input_shape[1];
        size_t H = input_shape[2];
        size_t W = is_1D ? 1 : input_shape[3];
        size_t H_pad_inner = strides[0] - 1;              // number of zero rows to insert between rows of input
        size_t W_pad_inner = is_1D ? 0 : strides[1] - 1;  // number of zero columns to insert between columns of input
        size_t H_pad_outer = weights_shape[2] - pads_begin[0] - 1;              // new amount of outer padding in H
        size_t W_pad_outer = is_1D ? 0 : weights_shape[3] - pads_begin[1] - 1;  // new amount of outer padding in W
        size_t H_pad_additional = output_padding[0];  // additional padding in H to get the proper number of outputs
        size_t W_pad_additional =
            is_1D ? 0 : output_padding[1];  // additional padding in W to get the proper number of outputs
        size_t H_new = H + (H - 1) * H_pad_inner + 2 * H_pad_outer + H_pad_additional;
        size_t W_new = is_1D ? 1 : W + (W - 1) * W_pad_inner + 2 * W_pad_outer + W_pad_additional;

        // ========== Stride 1 cases ===========
        if (((strides.size() == 2) && (strides[0] == 1) && (strides[1] == 1)) ||
            ((strides.size() == 1) && (strides[0] == 1))) {
            const float* weight_ptr = weights_const->get_data_ptr<float>();
            std::vector<float> new_weights(weights_shape[0] * weights_shape[1] * weights_shape[2] * weights_shape[3],
                                           0.0f);
            float* new_weight_ptr = new_weights.data();
            ov::Shape new_weights_shape;
            ov::CoordinateDiff new_pads_begin;
            ov::CoordinateDiff new_pads_end;
            new_weights_shape.push_back(weights_shape[1]);
            new_weights_shape.push_back(weights_shape[0]);
            new_weights_shape.push_back(weights_shape[2]);
            new_weights_shape.push_back(weights_shape[3]);
            for (size_t i = 0; i < new_weights_shape[0]; i++) {      // Co
                for (size_t j = 0; j < new_weights_shape[1]; j++) {  // Ci
                    auto Kh = weights_shape[2];
                    auto Kw = weights_shape[3];
                    auto kernel_size = (weights_shape.size() == 4) ? new_weights_shape[2] * new_weights_shape[3]
                                                                   : new_weights_shape[2];
                    auto kernel = weight_ptr + j * weights_shape[1] * kernel_size + i * kernel_size;
                    auto new_kernel = new_weight_ptr + i * new_weights_shape[1] * kernel_size + j * kernel_size;
                    for (size_t k = 0; k < Kh; k++) {  // store kernels in weird OpenVINO backwards order
                        for (size_t m = 0; m < Kw; m++) {
                            new_kernel[k * Kw + m] = kernel[(Kh - k - 1) * Kw + (Kw - m - 1)];
                            // printf("%e ", new_kernel[k * Kw + m]);
                        }
                        // printf("\n");
                    }
                }
            }
            auto new_weights_const = Constant::create(ngraph::element::f32, new_weights_shape, new_weights);
            new_pads_begin.push_back(H_pad_outer);
            new_pads_end.push_back(H_pad_outer);
            if (strides.size() == 2) {
                new_pads_begin.push_back(W_pad_outer);
                new_pads_end.push_back(W_pad_outer);
            }
            auto new_conv = std::make_shared<Convolution>(transpose_before->output(0),
                                                          new_weights_const->output(0),
                                                          strides,
                                                          new_pads_begin,
                                                          new_pads_end,
                                                          dilations,
                                                          op::PadType::EXPLICIT);
            ngraph::replace_node(conv, new_conv);
            is_graph_modfied = true;
            continue;

        }
        // ========== 1D cases ===========
        else if (is_1D || (W_new == input_shape[3])) {  // 1D in H direction case

            Output<Node> source = transpose_before->input_value(0);
            std::vector<std::vector<std::vector<size_t>>> kernel_list;  // list of kernel layouts
            std::vector<std::vector<int32_t>> input_list;               // list of input endpoints

            BuildKernelMap1D(H_new,
                             H_pad_outer,
                             H_pad_additional,
                             strides[0],
                             weights_shape[2],
                             true,
                             kernel_list,
                             input_list);

            if (is_1D) {  // if 3D input tensor then reshape to 4D
                auto new_reshape =
                    std::make_shared<Reshape>(source,
                                              Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),
                                              false);
                source = new_reshape->output(0);
            }
            // Insert convolutions
            OutputVector parts;
            for (size_t n = 0; n < input_list.size(); n++) {
                auto new_reshape = std::make_shared<Reshape>(
                    source,
                    Constant::create(ngraph::element::i64, Shape{2}, {N * H, W * C})->output(0),
                    false);
                size_t H_start = input_list[n][0];
                size_t H_stop = input_list[n][1] + 1;
                auto slice_start = Constant::create(ngraph::element::i64,
                                                    Shape{2},
                                                    std::initializer_list<decltype(H_start)>{H_start, 0ull});
                auto slice_stop = Constant::create(ngraph::element::i64, Shape{2}, {N * H_stop, W * C});
                auto slice_step = Constant::create(ngraph::element::i64, Shape{2}, {1ull, 1ull});
                auto new_slice =
                    std::make_shared<op::v8::Slice>(new_reshape->output(0), slice_start, slice_stop, slice_step);
                new_reshape = std::make_shared<Reshape>(
                    new_slice->output(0),
                    Constant::create(ngraph::element::i64, Shape{4}, {N, H_stop - H_start, W, C})->output(0),
                    false);
                auto new_transpose =
                    std::make_shared<op::v1::Transpose>(new_reshape->output(0),
                                                        Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                // Extract partial kernels
                //   The trick here is to artificially increase the output channels to produce more outputs per step
                //
                size_t C_out = kernel_list[n].size() *
                               weights_shape[1];  // increase output channels (for TransConv 0 dim is input)
                size_t C_in = C;                  // preserve input channels
                size_t K_h = kernel_list[n][0]
                                 .size();  // include only kernel elements that would not be multiplied by zero padding
                size_t K_w = 1;            // 1D case
                const float* weight_ptr = weights_const->get_data_ptr<float>();
                std::vector<float> new_weights(C_out * C_in * K_h * K_w, 0.0f);
                float* new_weight_ptr = new_weights.data();
                size_t i_step = weights_shape[1] * weights_shape[2] * weights_shape[3];
                size_t j_step = weights_shape[2] * weights_shape[3];
                bool print_stuff = false;
                if (print_stuff) {
                    printf("prev weights\n");
                    for (size_t i = 0; i < weights_shape[0]; i++) {
                        for (size_t j = 0; j < weights_shape[1]; j++) {
                            for (size_t k = 0; k < weights_shape[2]; k++) {
                                if ((j == 2) && (k == 1)) {
                                    printf("%e\n", *(weight_ptr + i * i_step + j * j_step + k));
                                }
                                // printf("%d %d %d: %f\n", (int)i, (int)j, (int)k, *(weight_ptr + i * i_step + j *
                                // j_step + k));
                            }
                        }
                    }
                    printf("bias\n");
                    auto bias_const =
                        std::dynamic_pointer_cast<Constant>(add_after->input_value(1).get_node_shared_ptr());
                    const float* bias_ptr = bias_const->get_data_ptr<float>();
                    for (size_t j = 0; j < weights_shape[1]; j++) {
                        printf("%e\n", *(bias_ptr + j));
                    }
                    printf("weights\n");
                }
                for (size_t i = 0; i < C_out; i++) {
                    for (size_t j = 0; j < C_in; j++) {
                        size_t k = i / weights_shape[1];
                        for (size_t m = 0; m < kernel_list[n][k].size(); m++) {
                            size_t i_prev = j;                     // input channel
                            size_t j_prev = i % weights_shape[1];  // output channel
                            size_t k_prev = kernel_list[n][k][m];  // kernel element
                            if (k_prev == PAD_VALUE) {
                                *(new_weight_ptr + i * C_in * K_h + j * K_h + m) = 0;
                                if (print_stuff) {
                                    // printf("%d %d %d: %f\n", (int)i, (int)j, (int)m, *(new_weight_ptr + i * C_in *
                                    // K_h + j * K_h + m));
                                }
                            } else {
                                *(new_weight_ptr + i * C_in * K_h + j * K_h + m) =
                                    *(weight_ptr + i_prev * i_step + j_prev * j_step + k_prev);
                                if (print_stuff) {
                                    // printf("%d %d %d: %f\n", (int)i, (int)j, (int)m, *(new_weight_ptr + i * C_in *
                                    // K_h + j * K_h + m));
                                }
                            }
                        }
                    }
                }
                auto new_weights_const =
                    Constant::create(ngraph::element::f32, Shape{C_out, C_in, K_h, K_w}, new_weights);

                auto conv_weights_fq = std::make_shared<FakeQuantize>(
                    new_weights_const,
                    Constant::create(ov::element::f32, ov::Shape{1}, {-1.6015920639038086}),
                    Constant::create(ov::element::f32, ov::Shape{1}, {1.6015920639038086}),
                    Constant::create(ov::element::f32, ov::Shape{1}, {-1.6015920639038086}),
                    Constant::create(ov::element::f32, ov::Shape{1}, {1.6015920639038086}),
                    65535);  // levels

                auto new_conv = std::make_shared<Convolution>(new_transpose->output(0),
                                                              conv_weights_fq->output(0),
                                                              Strides{1, 1},
                                                              CoordinateDiff{0, 0},
                                                              CoordinateDiff{0, 0},
                                                              Strides{1, 1},
                                                              op::PadType::VALID);
                if (add_after != nullptr) {  // need to repeat bias vector to match new convolution
                    auto bias_const =
                        std::dynamic_pointer_cast<Constant>(add_after->input_value(1).get_node_shared_ptr());
                    if (bias_const != nullptr) {
                        const float* bias_ptr = bias_const->get_data_ptr<float>();
                        std::vector<float> new_bias(C_out, 0.0f);
                        float* new_bias_ptr = new_bias.data();
                        for (size_t i = 0; i < C_out; i++) {
                            size_t j = i % weights_shape[1];
                            *(new_bias_ptr + i) = *(bias_ptr + j);
                        }
                        auto new_bias_const =
                            Constant::create(ngraph::element::f32, Shape{1ull, C_out, 1ull, 1ull}, new_bias);
                        auto new_add = std::make_shared<Add>(new_conv->output(0), new_bias_const->output(0));
                        auto add_shape = new_add->get_output_shape(0);
                        auto H_out = add_shape[1] * add_shape[2] / weights_shape[1];
                        OutputVector upstream;
                        upstream.push_back(new_add->output(0));
                        InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                        if (fq_after) {
                            upstream.push_back(helper::InsertFQLayer(fq_after, new_add)->output(0));
                        }
                        new_transpose = std::make_shared<ov::op::v1::Transpose>(
                            upstream[0],
                            Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                        auto new_reshape = std::make_shared<Reshape>(
                            new_transpose->output(0),
                            Constant::create(ngraph::element::i64,
                                             Shape{4},
                                             std::initializer_list<decltype(N)>{N, H_out, 1ull, weights_shape[1]})
                                ->output(0),
                            false);
                        parts.push_back(new_reshape->output(0));
                    }
                } else {
                    auto conv_shape = new_conv->get_output_shape(0);
                    auto H_out = conv_shape[1] * conv_shape[2] / weights_shape[1];
                    OutputVector upstream;
                    upstream.push_back(new_conv->output(0));
                    InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                    if (fq_after) {
                        upstream[0] = helper::InsertFQLayer(fq_after, new_conv)->output(0);
                    }
                    new_transpose = std::make_shared<ov::op::v1::Transpose>(
                        upstream[0],
                        Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                    auto new_reshape = std::make_shared<Reshape>(
                        new_transpose->output(0),
                        Constant::create(ngraph::element::i64,
                                         Shape{4},
                                         std::initializer_list<decltype(N)>{N, H_out, 1ull, weights_shape[1]})
                            ->output(0),
                        false);
                    parts.push_back(new_reshape->output(0));
                }
            }
            auto new_concat = std::make_shared<Concat>(parts, 1);
            if (is_1D) {  // if 3D original input tensor then reshape back to 3D
                auto new_reshape = std::make_shared<Reshape>(
                    new_concat->output(0),
                    Constant::create(ngraph::element::i64,
                                     Shape{3},
                                     std::initializer_list<decltype(N)>{N, H_new, 1ull, weights_shape[1]})
                        ->output(0),
                    false);
            }
            if (transpose_after != nullptr) {
                ngraph::replace_node(transpose_after, new_concat);
            } else {
                ngraph::replace_node(reshape_after, new_concat);
            }

            is_graph_modfied = true;

        } else if (H_new == input_shape[2]) {
            Output<Node> source = transpose_before->input_value(0);
            std::vector<std::vector<std::vector<size_t>>> kernel_list;  // list of kernel layouts
            std::vector<std::vector<int32_t>> input_list;               // list of input endpoints

            BuildKernelMap1D(W_new,
                             W_pad_outer,
                             W_pad_additional,
                             strides[1],
                             weights_shape[3],
                             true,
                             kernel_list,
                             input_list);

            // Insert convolutions
            OutputVector parts;
            for (size_t n = 0; n < input_list.size(); n++) {
                size_t W_start = input_list[n][0];
                size_t W_stop = input_list[n][1] + 1;
                auto slice_start =
                    Constant::create(ngraph::element::i64,
                                     Shape{4},
                                     std::initializer_list<decltype(W_start)>{0ull, 0ull, W_start, 0ull});
                auto slice_stop = Constant::create(ngraph::element::i64, Shape{4}, {N, H, W_stop, C});
                auto slice_step = Constant::create(ngraph::element::i64, Shape{4}, {1ull, 1ull, 1ull, 1ull});
                auto new_slice = std::make_shared<op::v8::Slice>(source, slice_start, slice_stop, slice_step);
                auto new_transpose = std::make_shared<ov::op::v1::Transpose>(
                    new_slice->output(0),
                    Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                // Extract partial kernels
                //   The trick here is to artificially increase the output channels to produce more outputs per step
                //
                size_t C_out = kernel_list[n].size() *
                               weights_shape[1];  // increase output channels (for TransConv 0 dim is input)
                size_t C_in = C;                  // preserve input channels
                size_t K_h = 1;                   // 1D case
                size_t K_w = kernel_list[n][0]
                                 .size();  // include only kernel elements that would not be multiplied by zero padding
                const float* weight_ptr = weights_const->get_data_ptr<float>();
                std::vector<float> new_weights(C_out * C_in * K_h * K_w, 0.0f);
                float* new_weight_ptr = new_weights.data();
                size_t i_step = weights_shape[1] * weights_shape[2] * weights_shape[3];
                size_t j_step = weights_shape[2] * weights_shape[3];
                bool print_stuff = false;
                if (print_stuff) {
                    printf("prev weights\n");
                    for (size_t i = 0; i < weights_shape[0]; i++) {
                        for (size_t j = 0; j < weights_shape[1]; j++) {
                            for (size_t k = 0; k < weights_shape[3]; k++) {  // 1D in W
                                if ((j == 2) && (k == 1)) {
                                    printf("%e\n", *(weight_ptr + i * i_step + j * j_step + k));
                                }
                                // printf("%d %d %d: %f\n", (int)i, (int)j, (int)k, *(weight_ptr + i * i_step + j *
                                // j_step + k));
                            }
                        }
                    }
                    printf("bias\n");
                    auto bias_const =
                        std::dynamic_pointer_cast<Constant>(add_after->input_value(1).get_node_shared_ptr());
                    const float* bias_ptr = bias_const->get_data_ptr<float>();
                    for (size_t j = 0; j < weights_shape[1]; j++) {
                        printf("%e\n", *(bias_ptr + j));
                    }
                    printf("weights\n");
                }
                for (size_t i = 0; i < C_out; i++) {
                    for (size_t j = 0; j < C_in; j++) {
                        size_t k = i / weights_shape[1];
                        for (size_t m = 0; m < kernel_list[n][k].size(); m++) {
                            size_t i_prev = j;                     // input channel
                            size_t j_prev = i % weights_shape[1];  // output channel
                            size_t k_prev = kernel_list[n][k][m];  // kernel element
                            if (k_prev == PAD_VALUE) {
                                *(new_weight_ptr + i * C_in * K_w + j * K_w + m) = 0;
                                if (print_stuff) {
                                    // printf("%d %d %d: %f\n", (int)i, (int)j, (int)m, *(new_weight_ptr + i * C_in *
                                    // K_h + j * K_h + m));
                                }
                            } else {
                                *(new_weight_ptr + i * C_in * K_w + j * K_w + m) =
                                    *(weight_ptr + i_prev * i_step + j_prev * j_step + k_prev);
                                if (print_stuff) {
                                    // printf("%d %d %d: %f\n", (int)i, (int)j, (int)m, *(new_weight_ptr + i * C_in *
                                    // K_h + j * K_h + m));
                                }
                            }
                        }
                    }
                }
                auto new_weights_const =
                    Constant::create(ngraph::element::f32, Shape{C_out, C_in, K_h, K_w}, new_weights);
                auto new_conv = std::make_shared<Convolution>(new_transpose->output(0),
                                                              new_weights_const->output(0),
                                                              Strides{1, 1},
                                                              CoordinateDiff{0, 0},
                                                              CoordinateDiff{0, 0},
                                                              Strides{1, 1},
                                                              op::PadType::VALID);
                if (add_after != nullptr) {  // need to repeat bias vector to match new convolution
                    auto bias_const =
                        std::dynamic_pointer_cast<Constant>(add_after->input_value(1).get_node_shared_ptr());
                    if (bias_const != nullptr) {
                        const float* bias_ptr = bias_const->get_data_ptr<float>();
                        std::vector<float> new_bias(C_out, 0.0f);
                        float* new_bias_ptr = new_bias.data();
                        for (size_t i = 0; i < C_out; i++) {
                            size_t j = i % weights_shape[1];
                            *(new_bias_ptr + i) = *(bias_ptr + j);
                        }
                        auto new_bias_const =
                            Constant::create(ngraph::element::f32, Shape{1ull, C_out, 1ull, 1ull}, new_bias);
                        auto new_add = std::make_shared<Add>(new_conv->output(0), new_bias_const->output(0));
                        auto add_shape = new_add->get_output_shape(0);
                        auto W_out = add_shape[1] * add_shape[3] / weights_shape[1];
                        OutputVector upstream;
                        upstream.push_back(new_add->output(0));
                        InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                        if (fq_after) {
                            upstream.push_back(helper::InsertFQLayer(fq_after, new_add)->output(0));
                        }
                        new_transpose = std::make_shared<ov::op::v1::Transpose>(
                            upstream[0],
                            Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                        auto new_reshape = std::make_shared<Reshape>(
                            new_transpose->output(0),
                            Constant::create(ngraph::element::i64,
                                             Shape{4},
                                             std::initializer_list<decltype(N)>{N, 1ull, W_out, weights_shape[1]})
                                ->output(0),
                            false);
                        parts.push_back(new_reshape->output(0));
                    }
                } else {
                    auto conv_shape = new_conv->get_output_shape(0);
                    auto W_out = conv_shape[1] * conv_shape[3] / weights_shape[1];
                    OutputVector upstream;
                    upstream.push_back(new_conv->output(0));
                    InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                    if (fq_after) {
                        upstream.push_back(helper::InsertFQLayer(fq_after, new_conv)->output(0));
                    }
                    new_transpose = std::make_shared<ov::op::v1::Transpose>(
                        upstream[0],
                        Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                    auto new_reshape = std::make_shared<Reshape>(
                        new_transpose->output(0),
                        Constant::create(ngraph::element::i64,
                                         Shape{4},
                                         std::initializer_list<decltype(N)>{N, 1ull, W_out, weights_shape[1]})
                            ->output(0),
                        false);
                    parts.push_back(new_reshape->output(0));
                }
            }
            auto new_concat = std::make_shared<Concat>(parts, 2);
            if (transpose_after != nullptr) {
                ngraph::replace_node(transpose_after, new_concat);
            } else {
                ngraph::replace_node(reshape_after, new_concat);
            }

            is_graph_modfied = true;

        }
        // ========== 2D stride (S,1) cases ===========
        else if (strides[1] == 1) {  // zero insertion in H direction only

            Output<Node> source = transpose_before->input_value(0);
            std::vector<std::vector<std::vector<size_t>>> kernel_list;  // list of kernel layouts
            std::vector<std::vector<int32_t>> input_list;               // list of input endpoints

            BuildKernelMap1D(H_new,
                             H_pad_outer,
                             H_pad_additional,
                             strides[0],
                             weights_shape[2],
                             true,
                             kernel_list,
                             input_list);

            // Insert convolutions
            OutputVector parts;
            for (size_t n = 0; n < input_list.size(); n++) {
                auto new_reshape = std::make_shared<Reshape>(
                    source,
                    Constant::create(ngraph::element::i64, Shape{2}, {N * H, W * C})->output(0),
                    false);
                size_t H_start = input_list[n][0];
                size_t H_stop = input_list[n][1] + 1;
                auto slice_start = Constant::create(ngraph::element::i64,
                                                    Shape{2},
                                                    std::initializer_list<decltype(H_start)>{H_start, 0ull});
                auto slice_stop = Constant::create(ngraph::element::i64, Shape{2}, {H_stop, W * C});
                auto slice_step = Constant::create(ngraph::element::i64, Shape{2}, {1ull, 1ull});
                auto new_slice =
                    std::make_shared<op::v8::Slice>(new_reshape->output(0), slice_start, slice_stop, slice_step);
                new_reshape = std::make_shared<Reshape>(
                    new_slice->output(0),
                    Constant::create(ngraph::element::i64, Shape{4}, {N, H_stop - H_start, W, C})->output(0),
                    false);
                auto new_transpose = std::make_shared<ov::op::v1::Transpose>(
                    new_reshape->output(0),
                    Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                // Extract partial kernels
                //   The trick here is to artificially increase the output channels to produce more outputs per step
                //
                size_t C_out = kernel_list[n].size() *
                               weights_shape[1];  // increase output channels (for TransConv 0 dim is input)
                size_t C_in = C;                  // preserve input channels
                size_t K_h = kernel_list[n][0]
                                 .size();  // include only kernel elements that would not be multiplied by zero padding
                size_t K_w = weights_shape[3];
                const float* weight_ptr = weights_const->get_data_ptr<float>();
                std::vector<float> new_weights(C_out * C_in * K_h * K_w, 0.0f);
                float* new_weight_ptr = new_weights.data();
                size_t i_step = weights_shape[1] * weights_shape[2] * weights_shape[3];
                size_t j_step = weights_shape[2] * weights_shape[3];
                bool print_stuff = false;
                if (print_stuff) {
                    printf("prev weights\n");
                    for (size_t i = 0; i < weights_shape[0]; i++) {
                        for (size_t j = 0; j < weights_shape[1]; j++) {
                            for (size_t k = 0; k < weights_shape[2]; k++) {
                                for (size_t n = 0; n < K_w; n++) {
                                    if ((j == 2) && (k == 1)) {
                                        printf("%e\n", *(weight_ptr + i * i_step + j * j_step + k * K_w + n));
                                    }
                                    // printf("%d %d %d %d: %f\n", (int)i, (int)j, (int)k, (int)n, *(weight_ptr + i *
                                    // i_step + j * j_step + k * K_w + n));
                                }
                            }
                        }
                    }
                    printf("bias\n");
                    auto bias_const =
                        std::dynamic_pointer_cast<Constant>(add_after->input_value(1).get_node_shared_ptr());
                    const float* bias_ptr = bias_const->get_data_ptr<float>();
                    for (size_t j = 0; j < weights_shape[1]; j++) {
                        printf("%e\n", *(bias_ptr + j));
                    }
                    printf("weights\n");
                }
                for (size_t i = 0; i < C_out; i++) {
                    for (size_t j = 0; j < C_in; j++) {
                        size_t k = i / weights_shape[1];
                        for (size_t m = 0; m < kernel_list[n][k].size(); m++) {
                            size_t i_prev = j;                     // input channel
                            size_t j_prev = i % weights_shape[1];  // output channel
                            size_t k_prev = kernel_list[n][k][m];  // kernel element
                            if (k_prev == PAD_VALUE) {
                                for (size_t n = 0; n < K_w; n++) {
                                    *(new_weight_ptr + i * C_in * K_h * K_w + j * K_h * K_w + m * K_w + n) = 0;
                                    if (print_stuff) {
                                        // printf("%d %d %d: %f\n", (int)i, (int)j, (int)m, *(new_weight_ptr + i * C_in
                                        // * K_h + j * K_h + m));
                                    }
                                }
                            } else {
                                for (size_t n = 0; n < K_w; n++) {
                                    *(new_weight_ptr + i * C_in * K_h * K_w + j * K_h * K_w + m * K_w + n) = *(
                                        weight_ptr + i_prev * i_step + j_prev * j_step + k_prev * K_w + (K_w - n - 1));
                                    if (print_stuff) {
                                        // printf("%d %d %d: %f\n", (int)i, (int)j, (int)m, *(new_weight_ptr + i * C_in
                                        // * K_h + j * K_h + m));
                                    }
                                }
                            }
                        }
                    }
                }
                ov::Strides new_strides = {1, 1};
                ov::CoordinateDiff new_pads_begin = {0, (int64_t)W_pad_outer};
                ov::CoordinateDiff new_pads_end = {0, (int64_t)W_pad_outer};
                ov::Strides new_dilations = {1, 1};
                auto new_weights_const =
                    Constant::create(ngraph::element::f32, Shape{C_out, C_in, K_h, K_w}, new_weights);
                auto new_conv = std::make_shared<Convolution>(new_transpose->output(0),
                                                              new_weights_const->output(0),
                                                              new_strides,
                                                              new_pads_begin,
                                                              new_pads_end,
                                                              new_dilations,
                                                              op::PadType::EXPLICIT);
                if (add_after != nullptr) {  // need to repeat bias vector to match new convolution
                    auto bias_const =
                        std::dynamic_pointer_cast<Constant>(add_after->input_value(1).get_node_shared_ptr());
                    if (bias_const != nullptr) {
                        const float* bias_ptr = bias_const->get_data_ptr<float>();
                        std::vector<float> new_bias(C_out, 0.0f);
                        float* new_bias_ptr = new_bias.data();
                        for (size_t i = 0; i < C_out; i++) {
                            size_t j = i % weights_shape[1];
                            *(new_bias_ptr + i) = *(bias_ptr + j);
                        }
                        auto new_bias_const =
                            Constant::create(ngraph::element::f32, Shape{1ull, C_out, 1ull, 1ull}, new_bias);
                        auto new_add = std::make_shared<Add>(new_conv->output(0), new_bias_const->output(0));
                        auto add_shape = new_add->get_output_shape(0);
                        auto H_out = add_shape[1] * add_shape[2] / weights_shape[1];
                        auto W_out = add_shape[3];
                        OutputVector upstream;
                        upstream.push_back(new_add->output(0));
                        InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                        if (fq_after) {
                            upstream.push_back(helper::InsertFQLayer(fq_after, new_add)->output(0));
                        }
                        new_transpose = std::make_shared<ov::op::v1::Transpose>(
                            upstream[0],
                            Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                        auto output_shape = new_transpose->output(0).get_shape();
                        if (output_shape[3] == weights_shape[1]) {
                            auto new_reshape = std::make_shared<Reshape>(
                                new_transpose->output(0),
                                Constant::create(ngraph::element::i64, Shape{4}, {N, H_out, W_out, weights_shape[1]})
                                    ->output(0),
                                false);
                            upstream[0] = new_reshape->output(0);
                        } else {
                            auto tmp_H = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] /
                                         weights_shape[1];
                            auto tmp_W = weights_shape[1];
                            auto new_reshape = std::make_shared<Reshape>(
                                new_transpose->output(0),
                                Constant::create(ngraph::element::i64, Shape{2}, {tmp_H, tmp_W})->output(0),
                                false);
                            new_transpose = std::make_shared<ov::op::v1::Transpose>(
                                new_reshape->output(0),
                                Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            new_reshape = std::make_shared<Reshape>(
                                new_transpose->output(0),
                                Constant::create(ngraph::element::i64, Shape{4}, {N, H_out, W_out, weights_shape[1]})
                                    ->output(0),
                                false);
                            upstream[0] = new_reshape->output(0);
                        }
                        parts.push_back(upstream[0]);
                    }
                } else {
                    auto conv_shape = new_conv->get_output_shape(0);
                    auto H_out = conv_shape[1] * conv_shape[2] / weights_shape[1];
                    auto W_out = conv_shape[3];
                    OutputVector upstream;
                    upstream.push_back(new_conv->output(0));
                    InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                    if (fq_after) {
                        upstream.push_back(helper::InsertFQLayer(fq_after, new_conv)->output(0));
                    }
                    new_transpose = std::make_shared<ov::op::v1::Transpose>(
                        upstream[0],
                        Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                    auto output_shape = new_transpose->output(0).get_shape();
                    if (output_shape[3] == weights_shape[1]) {
                        auto new_reshape = std::make_shared<Reshape>(
                            new_transpose->output(0),
                            Constant::create(ngraph::element::i64, Shape{4}, {N, H_out, W_out, weights_shape[1]})
                                ->output(0),
                            false);
                        parts.push_back(new_reshape->output(0));
                    } else {
                        printf("Warning 2D unwrapping method is not yet tested.  This network may be incorrect.\n");
                        auto tmp_H = output_shape[0] * output_shape[1] * output_shape[2];
                        auto tmp_W = output_shape[3];
                        auto new_reshape = std::make_shared<Reshape>(
                            new_transpose->output(0),
                            Constant::create(ngraph::element::i64, Shape{2}, {tmp_H, tmp_W})->output(0),
                            false);
                        new_transpose = std::make_shared<ov::op::v1::Transpose>(
                            new_reshape->output(0),
                            Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                        auto num_splits = output_shape[3] / weights_shape[1];
                        auto new_split = std::make_shared<Split>(new_transpose->output(0),
                                                                 Constant::create(element::i64, Shape{}, {0}),
                                                                 num_splits);
                        for (uint32_t i = 0; i < num_splits; i++) {
                            new_transpose = std::make_shared<ov::op::v1::Transpose>(
                                new_split->output(i),
                                Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            new_reshape = std::make_shared<Reshape>(
                                new_transpose->output(0),
                                Constant::create(ngraph::element::i64,
                                                 Shape{4},
                                                 {N, H_out / num_splits, W_out, weights_shape[1]})
                                    ->output(0),
                                false);
                            parts.push_back(new_reshape->output(0));
                        }
                    }
                }
            }
            auto new_concat = std::make_shared<Concat>(parts, 1);
            if (transpose_after != nullptr) {
                ngraph::replace_node(transpose_after, new_concat);
            } else {
                ngraph::replace_node(reshape_after, new_concat);
            }

            is_graph_modfied = true;

        } else {
            continue;
        }
    }
    return is_graph_modfied;
}

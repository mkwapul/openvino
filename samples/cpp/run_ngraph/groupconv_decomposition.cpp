/* ============================================================================
 * INTEL CONFIDENTIAL
 *
 * Copyright 2021 Intel Corporation All Rights Reserved.
 *
 * The source code contained or described herein and all documents related to
 * the source code ("Material") are owned by Intel Corporation or its suppliers
 * or licensors. Title to the Material remains with Intel Corporation or its
 * suppliers and licensors. The Material contains trade secrets and proprietary
 * and confidential information of Intel or its suppliers and licensors. The
 * Material is protected by worldwide copyright and trade secret laws and
 * treaty provisions. No part of the Material may be used, copied, reproduced,
 * modified, published, uploaded, posted, transmitted, distributed, or
 * disclosed in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other intellectual
 * property right is granted to or conferred upon you by disclosure or delivery
 * of the Materials, either expressly, by implication, inducement, estoppel or
 * otherwise. Any license under such intellectual property rights must be
 * express and approved by Intel in writing.
 * ============================================================================
 */

#include "groupconv_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include "decomp_helper.hpp"

using namespace ngraph;
using namespace op;

bool ngraph::pass::GroupConvolutionDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto conv = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node);
        if (nullptr == conv) {
            continue;
        }

        const Output<Node>& input = conv->input_value(0);
        const Output<Node>& weights = conv->input_value(1);
        auto input_shape = input.get_shape();
        auto weights_shape = weights.get_shape();
        auto output_shape = conv->get_output_shape(0);
        auto auto_pad = conv->get_auto_pad();
        auto dilations = conv->get_dilations();
        auto pads_begin = conv->get_pads_begin();
        auto pads_end = conv->get_pads_end();
        auto strides = conv->get_strides();
        auto weights_const =
            std::dynamic_pointer_cast<ngraph::opset1::Constant>(conv->input_value(1).get_node_shared_ptr());
        const float* weight_ptr = weights_const->get_data_ptr<float>();

        // only support 4D input with N=1, 5D filters, 2D stride, 2D dilation, 2D padding
        if (input_shape.size() != 4 || weights_shape.size() != 5 || output_shape.size() != 4 ||
            pads_begin.size() != 2 || pads_end.size() != 2 || dilations.size() != 2 || strides.size() != 2 ||
            input_shape[0] != 1) {
            continue;
        }

        // find Transpose-->Convolution--><Add>-->Transpose pattern else skip
        const Output<Node>& parent = conv->input_value(0);
        auto children = conv->output(0).get_target_inputs();
        if (children.size() != 1) {
            continue;
        }
        auto transpose_before =
            std::dynamic_pointer_cast<ngraph::opset1::Transpose>(parent.get_node()->shared_from_this());
        if (transpose_before == nullptr) {
            continue;
        }
        auto add_after =
            std::dynamic_pointer_cast<ngraph::opset1::Add>(children.begin()->get_node()->shared_from_this());
        auto prelu_after =
            std::dynamic_pointer_cast<ngraph::opset1::PRelu>(children.begin()->get_node()->shared_from_this());
        auto relu_after =
            std::dynamic_pointer_cast<ngraph::opset1::Relu>(children.begin()->get_node()->shared_from_this());
        auto sigmoid_after =
            std::dynamic_pointer_cast<ngraph::opset1::Sigmoid>(children.begin()->get_node()->shared_from_this());
        auto tanh_after =
            std::dynamic_pointer_cast<ngraph::opset1::Tanh>(children.begin()->get_node()->shared_from_this());
        auto transpose_after =
            std::dynamic_pointer_cast<ngraph::opset1::Transpose>(children.begin()->get_node()->shared_from_this());
        if (add_after != nullptr) {
            auto add_children = add_after->output(0).get_target_inputs();
            if (add_children.size() != 1) {
                continue;
            }
            prelu_after =
                std::dynamic_pointer_cast<ngraph::opset1::PRelu>(add_children.begin()->get_node()->shared_from_this());
            relu_after =
                std::dynamic_pointer_cast<ngraph::opset1::Relu>(add_children.begin()->get_node()->shared_from_this());
            sigmoid_after = std::dynamic_pointer_cast<ngraph::opset1::Sigmoid>(
                add_children.begin()->get_node()->shared_from_this());
            tanh_after =
                std::dynamic_pointer_cast<ngraph::opset1::Tanh>(add_children.begin()->get_node()->shared_from_this());
            transpose_after = std::dynamic_pointer_cast<ngraph::opset1::Transpose>(
                add_children.begin()->get_node()->shared_from_this());
        }
        if (transpose_after == nullptr) {
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
                transpose_after = std::dynamic_pointer_cast<ngraph::opset1::Transpose>(
                    act_children.begin()->get_node()->shared_from_this());
            } else {
                continue;
            }
        }
        if (transpose_after == nullptr) {
            continue;
        }

        auto N = input_shape[0];
        auto C = input_shape[1];
        auto H = input_shape[2];
        auto W = input_shape[3];
        auto G = weights_shape[0];
        auto Co = weights_shape[1];
        auto Ci = weights_shape[2];
        auto Kh = weights_shape[3];
        auto Kw = weights_shape[4];
        auto Hnew = (H + pads_begin[0] + pads_end[0] - Kh) / strides[0] + 1;
        auto Wnew = (W + pads_begin[1] + pads_end[1] - Kw) / strides[1] + 1;
        auto H_pad = H;
        auto W_pad = W;
        auto Hnew_pad = Hnew;
        auto Wnew_pad = Wnew;
        OutputVector upstream;
        std::shared_ptr<ov::op::v0::Constant> out_padding_const = nullptr;

        upstream.push_back(transpose_before->input_value(0));
        if (((H * W) % 32) != 0) {  // pad input to avoid 64B unaligned channel splitting
            uint64_t num_padding_elements = 0;
            if ((H == 1) || (W == 1)) {
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                    upstream[0],
                    op::Constant::create(ngraph::element::i64, Shape{2}, {H * W, C})->output(0),
                    false);
                upstream[0] = new_reshape->output(0);
                if (H == 1) {  // 1D case - pad in W
                    W_pad = W + 32 - (W % 32);
                    Wnew_pad = (W_pad + pads_begin[1] + pads_end[1] - Kw) / strides[1] + 1;
                    num_padding_elements = C * (W_pad - W);
                } else if (W == 1) {  // 1D case - pad in H
                    H_pad = H + 32 - (H % 32);
                    Hnew_pad = (H_pad + pads_begin[0] + pads_end[0] - Kh) / strides[0] + 1;
                    num_padding_elements = C * (H_pad - H);
                }
                std::vector<float> padding(num_padding_elements, 0.0f);
                auto padding_const =
                    op::Constant::create(ngraph::element::f32, Shape{num_padding_elements / C, C}, padding);
                upstream.push_back(padding_const->output(0));
                auto new_concat = std::make_shared<ngraph::opset1::Concat>(upstream, 0);
                upstream.pop_back();
                upstream[0] = new_concat->output(0);
            } else {  // 2D case
                W_pad = W + 32 - (W % 32);
                Wnew_pad = (W_pad + pads_begin[1] + pads_end[1] - Kw) / strides[1] + 1;
                num_padding_elements = C * (W_pad - W);
                std::vector<float> padding(num_padding_elements, 0.0f);
                auto padding_const =
                    op::Constant::create(ngraph::element::f32, Shape{1, 1, num_padding_elements / C, C}, padding);
                upstream.push_back(padding_const->output(0));
                auto split = std::make_shared<ngraph::opset1::Split>(
                    upstream[0],
                    ngraph::opset1::Constant::create(element::i64, Shape{}, {1}),
                    H);
                OutputVector parts;
                for (uint32_t h = 0; h < H; h++) {
                    upstream[0] = split->output(h);
                    auto new_concat = std::make_shared<ngraph::opset1::Concat>(upstream, 2);
                    parts.push_back(new_concat->output(0));
                }
                upstream.pop_back();
                auto new_concat = std::make_shared<ngraph::opset1::Concat>(parts, 1);
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                    new_concat->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{2}, {H * W_pad, C})->output(0),
                    false);
                upstream[0] = new_reshape->output(0);
            }
        } else {
            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                upstream[0],
                op::Constant::create(ngraph::element::i64, Shape{2}, {H * W, C})->output(0),
                false);
            upstream[0] = new_reshape->output(0);
        }
        if (((Hnew_pad * Wnew_pad) % 32) != 0) {  // pad output to avoid 64B unaligned channel concatenation
            uint64_t num_padding_elements = 0;
            ov::Shape pad_shape;
            if (Hnew_pad == 1) {  // 1D case - pad in W
                auto Wnew_pad2 = Wnew_pad + 32 - (Wnew_pad % 32);
                num_padding_elements = Co * (Wnew_pad2 - Wnew_pad);
                pad_shape = {N, Hnew_pad, Wnew_pad2 - Wnew_pad, Co};
                Wnew_pad = Wnew_pad2;
            } else if (Wnew_pad == 1) {  // 1D case - pad in H
                auto Hnew_pad2 = Hnew_pad + 32 - (Hnew_pad % 32);
                num_padding_elements = Co * (Hnew_pad2 - Hnew_pad);
                pad_shape = {N, Hnew_pad2 - Hnew_pad, Wnew_pad, Co};
                Hnew_pad = Hnew_pad2;
            } else {
                continue;  // 2D unaligned case not yet implemented
            }
            std::vector<float> padding(num_padding_elements, 0.0f);
            out_padding_const = op::Constant::create(ngraph::element::f32, pad_shape, padding);
        }
        auto new_transpose =
            std::make_shared<op::Transpose>(upstream[0], op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
        auto split =
            std::make_shared<ngraph::opset1::Split>(new_transpose->output(0),
                                                    ngraph::opset1::Constant::create(element::i64, Shape{}, {0}),
                                                    G);

        OutputVector parts;
        for (uint32_t g = 0; g < G; g++) {
            std::vector<float> new_weights(Co * Ci * Kh * Kw, 0.0f);
            float* new_weight_ptr = new_weights.data();
            for (size_t i = 0; i < Co; i++) {
                for (size_t j = 0; j < Ci; j++) {
                    for (size_t k = 0; k < Kh; k++) {
                        for (size_t m = 0; m < Kw; m++) {
                            *(new_weight_ptr + i * Ci * Kh * Kw + j * Kh * Kw + k * Kw + m) =
                                *(weight_ptr + g * Co * Ci * Kh * Kw + i * Ci * Kh * Kw + j * Kh * Kw + k * Kw + m);
                        }
                    }
                }
            }
            auto new_weights_const = op::Constant::create(ngraph::element::f32, Shape{Co, Ci, Kh, Kw}, new_weights);
            new_weights_const->set_friendly_name("SharedWeights");
            upstream[0] = split->output(g);
            if ((H_pad == 1) || (W_pad == 1)) {
                auto new_transpose =
                    std::make_shared<op::Transpose>(upstream[0],
                                                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                upstream[0] = new_transpose->output(0);
            }
            auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                upstream[0],
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, H_pad, W_pad, C / G})->output(0),
                false);
            auto new_transpose =
                std::make_shared<op::Transpose>(new_reshape->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            auto new_conv = std::make_shared<opset1::Convolution>(new_transpose->output(0),
                                                                  new_weights_const->output(0),
                                                                  strides,
                                                                  pads_begin,
                                                                  pads_end,
                                                                  dilations,
                                                                  auto_pad);
            new_conv->set_friendly_name("ReplaceGroupConv");

            if (add_after != nullptr) {
                auto bias_const =
                    std::dynamic_pointer_cast<opset1::Constant>(add_after->input_value(1).get_node_shared_ptr());
                const float* bias_ptr = bias_const->get_data_ptr<float>();
                std::vector<float> new_bias(Co, 0.0f);
                float* new_bias_ptr = new_bias.data();
                for (size_t i = 0; i < Co; i++) {
                    *(new_bias_ptr + i) = *(bias_ptr + g * Co + i);
                }
                auto new_bias_const = op::Constant::create(ngraph::element::f32, Shape{1ull, Co, 1ull, 1ull}, new_bias);
                auto new_add = std::make_shared<opset1::Add>(new_conv->output(0), new_bias_const->output(0));
                upstream[0] = new_add->output(0);
                InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                new_transpose =
                    std::make_shared<op::Transpose>(upstream[0],
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            } else {
                upstream[0] = new_conv->output(0);
                InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                new_transpose =
                    std::make_shared<op::Transpose>(upstream[0],
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            }
            upstream[0] = new_transpose->output(0);
            if (out_padding_const) {
                size_t pad_dim = (Hnew_pad == 1) ? 2 : 1;
                upstream.push_back(out_padding_const->output(0));
                auto new_concat = std::make_shared<ngraph::opset1::Concat>(upstream, pad_dim);
                upstream.pop_back();
                upstream[0] = new_concat->output(0);
            }
            parts.push_back(upstream[0]);
        }
        auto new_concat = std::make_shared<ngraph::opset1::Concat>(parts, 1);
        auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
            new_concat->output(0),
            op::Constant::create(ngraph::element::i64, Shape{2}, {G * Co, Hnew_pad * Wnew_pad})->output(0),
            false);
        new_transpose = std::make_shared<op::Transpose>(new_reshape->output(0),
                                                        op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
        upstream[0] = new_transpose->output(0);
        if ((Hnew_pad > Hnew) || (Wnew_pad > Wnew)) {  // remove padding
            auto slice_start = op::Constant::create(ngraph::element::i64, Shape{2}, {0ull, 0ull});
            auto slice_stop = op::Constant::create(ngraph::element::i64, Shape{2}, {Hnew * Wnew, C});
            auto slice_step = op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, 1ull});
            auto new_slice = std::make_shared<v8::Slice>(upstream[0], slice_start, slice_stop, slice_step);
            upstream[0] = new_slice->output(0);
        }
        new_reshape = std::make_shared<ngraph::opset1::Reshape>(
            upstream[0],
            op::Constant::create(ngraph::element::i64, Shape{4}, {N, Hnew, Wnew, G * Co})->output(0),
            false);

        ngraph::replace_node(transpose_after, new_reshape);

        is_graph_modfied = true;
    }

    return is_graph_modfied;
}

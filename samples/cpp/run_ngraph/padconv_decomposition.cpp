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

#include "padconv_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include "decomp_helper.hpp"

using namespace ngraph;
using namespace op;

bool ngraph::pass::PadConvolutionDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node);
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
        bool nchw_only = true;  // keep graph NCHW and rely on layout transformation to convert to NHWC for GNA

        // only support 4D input with N=1, 5D filters, 2D stride, 2D dilation, 2D padding
        if (input_shape.size() != 4 || weights_shape.size() != 4 || output_shape.size() != 4 ||
            pads_begin.size() != 2 || pads_end.size() != 2 || dilations.size() != 2 || strides.size() != 2 ||
            input_shape[0] != 1) {
            continue;
        }

        // find Convolution--><Add>--><Activation> pattern else skip
        const Output<Node>& parent = conv->input_value(0);
        auto children = conv->output(0).get_target_inputs();
        if (children.size() != 1) {
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
        }

        size_t N = input_shape[0];
        size_t C = input_shape[1];
        size_t H = input_shape[2];
        size_t W = input_shape[3];
        size_t Co = weights_shape[0];
        size_t Kh = weights_shape[2];
        size_t Kw = weights_shape[3];

        if ((Kh > H) && (Kw > W)) {  // oversized in both H and W
            continue;                // don't yet support this case
        } else if (Kh > H) {         // oversized in H only

            size_t Kh_new = H;
            size_t Kw_new = Kw;
            size_t Kh_parts = H + pads_begin[0] + pads_end[0] - Kh + 1;
            size_t Co_new = Co * Kh_parts;

            const float* weight_ptr = weights_const->get_data_ptr<float>();
            std::vector<float> new_weights(Co_new * C * Kh_new * Kw_new, 0.0f);
            float* new_weight_ptr = new_weights.data();
            ov::Shape new_weights_shape;
            ov::CoordinateDiff new_pads_begin;
            ov::CoordinateDiff new_pads_end;
            new_weights_shape.push_back(Co_new);
            new_weights_shape.push_back(C);
            new_weights_shape.push_back(Kh_new);
            new_weights_shape.push_back(Kw_new);
            new_pads_begin.push_back(0);              // remove padding in H
            new_pads_begin.push_back(pads_begin[1]);  // preserve padding in W
            new_pads_end.push_back(0);                // remove padding in H
            new_pads_end.push_back(pads_end[1]);      // preserve padding in W
            for (size_t i = 0; i < Co; i++) {
                for (size_t ih = 0; ih < Kh_parts; ih++) {
                    for (size_t j = 0; j < C; j++) {
                        auto kernel = weight_ptr + i * C * Kh * Kw + j * Kh * Kw;
                        auto new_kernel =
                            new_weight_ptr + (i * Kh_parts + ih) * C * Kh_new * Kw_new + j * Kh_new * Kw_new;
                        for (size_t k = 0; k < Kh_new; k++) {
                            size_t n = k + pads_begin[0] - ih;
                            for (size_t m = 0; m < Kw_new; m++) {
                                new_kernel[k * Kw_new + m] = kernel[n * Kw + m];
                            }
                        }
                    }
                }
            }
            auto new_weights_const = op::Constant::create(ngraph::element::f32, new_weights_shape, new_weights);

            auto new_conv = std::make_shared<opset1::Convolution>(input,
                                                                  new_weights_const->output(0),
                                                                  strides,
                                                                  new_pads_begin,
                                                                  new_pads_end,
                                                                  dilations,
                                                                  PadType::EXPLICIT);

            OutputVector upstream;
            upstream.push_back(new_conv->output(0));
            if (add_after != nullptr) {  // need to repeat bias vector to match new convolution
                auto bias_const =
                    std::dynamic_pointer_cast<opset1::Constant>(add_after->input_value(1).get_node_shared_ptr());
                if (bias_const != nullptr) {
                    const float* bias_ptr = bias_const->get_data_ptr<float>();
                    std::vector<float> new_bias(Co_new, 0.0f);
                    float* new_bias_ptr = new_bias.data();
                    for (size_t i = 0; i < Co_new; i++) {
                        size_t j = i % weights_shape[1];
                        *(new_bias_ptr + i) = *(bias_ptr + j);
                    }
                    auto new_bias_const =
                        op::Constant::create(ngraph::element::f32, Shape{1ull, Co_new, 1ull, 1ull}, new_bias);
                    auto new_add = std::make_shared<opset1::Add>(new_conv->output(0), new_bias_const->output(0));
                    auto add_shape = new_add->get_output_shape(0);
                    upstream[0] = new_add->output(0);
                    InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                }
            } else {
                auto conv_shape = new_conv->get_output_shape(0);
                upstream[0] = new_conv->output(0);
                InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
            }

            if (nchw_only) {
                // unfold output channels into H
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                    upstream[0],
                    op::Constant::create(ngraph::element::i64, Shape{4}, output_shape)->output(0),
                    false);
                if (prelu_after) {
                    ngraph::replace_node(prelu_after, new_reshape);
                } else if (relu_after) {
                    ngraph::replace_node(relu_after, new_reshape);
                } else if (sigmoid_after) {
                    ngraph::replace_node(sigmoid_after, new_reshape);
                } else if (tanh_after) {
                    ngraph::replace_node(tanh_after, new_reshape);
                } else if (add_after) {
                    ngraph::replace_node(add_after, new_reshape);
                } else {
                    ngraph::replace_node(conv, new_reshape);
                }

            } else {
                auto new_transpose =
                    std::make_shared<op::Transpose>(upstream[0],
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                    new_transpose->output(0),
                    op::Constant::create(ngraph::element::i64,
                                         Shape{4},
                                         {output_shape[0], output_shape[2], output_shape[3], output_shape[1]})
                        ->output(0),
                    false);
                new_transpose =
                    std::make_shared<op::Transpose>(new_reshape->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));

                if (prelu_after) {
                    ngraph::replace_node(prelu_after, new_transpose);
                } else if (relu_after) {
                    ngraph::replace_node(relu_after, new_transpose);
                } else if (sigmoid_after) {
                    ngraph::replace_node(sigmoid_after, new_transpose);
                } else if (tanh_after) {
                    ngraph::replace_node(tanh_after, new_transpose);
                } else if (add_after) {
                    ngraph::replace_node(add_after, new_transpose);
                } else {
                    ngraph::replace_node(conv, new_transpose);
                }
            }

            is_graph_modfied = true;

        } else if (Kw > W) {  // oversized in W only

            size_t Kh_new = Kh;
            size_t Kw_new = W;
            size_t Kw_parts = W + pads_begin[1] + pads_end[1] - Kw + 1;
            size_t Co_new = Co * Kw_parts;

            const float* weight_ptr = weights_const->get_data_ptr<float>();
            std::vector<float> new_weights(Co_new * C * Kh_new * Kw_new, 0.0f);
            float* new_weight_ptr = new_weights.data();
            ov::Shape new_weights_shape;
            ov::CoordinateDiff new_pads_begin;
            ov::CoordinateDiff new_pads_end;
            new_weights_shape.push_back(Co_new);
            new_weights_shape.push_back(C);
            new_weights_shape.push_back(Kh_new);
            new_weights_shape.push_back(Kw_new);
            new_pads_begin.push_back(pads_begin[0]);  // preserve padding in H
            new_pads_begin.push_back(0);              // remove padding in W
            new_pads_end.push_back(pads_end[0]);      // preserve padding in H
            new_pads_end.push_back(0);                // remove padding in W
            for (size_t i = 0; i < Co; i++) {
                for (size_t iw = 0; iw < Kw_parts; iw++) {
                    for (size_t j = 0; j < C; j++) {
                        auto kernel = weight_ptr + i * C * Kh * Kw + j * Kh * Kw;
                        auto new_kernel = new_weight_ptr + (iw * Co + i) * C * Kh_new * Kw_new + j * Kh_new * Kw_new;
                        for (size_t k = 0; k < Kh_new; k++) {
                            for (size_t m = 0; m < Kw_new; m++) {
                                size_t n = m + pads_begin[1] - iw;
                                new_kernel[k * Kw_new + m] = kernel[k * Kw + n];
                            }
                        }
                    }
                }
            }
            auto new_weights_const = op::Constant::create(ngraph::element::f32, new_weights_shape, new_weights);

            auto new_conv = std::make_shared<opset1::Convolution>(input,
                                                                  new_weights_const->output(0),
                                                                  strides,
                                                                  new_pads_begin,
                                                                  new_pads_end,
                                                                  dilations,
                                                                  PadType::EXPLICIT);

            OutputVector upstream;
            upstream.push_back(new_conv->output(0));
            if (add_after != nullptr) {  // need to repeat bias vector to match new convolution
                auto bias_const =
                    std::dynamic_pointer_cast<opset1::Constant>(add_after->input_value(1).get_node_shared_ptr());
                if (bias_const != nullptr) {
                    const float* bias_ptr = bias_const->get_data_ptr<float>();
                    std::vector<float> new_bias(Co_new, 0.0f);
                    float* new_bias_ptr = new_bias.data();
                    for (size_t i = 0; i < Co_new; i++) {
                        size_t j = i % weights_shape[1];
                        *(new_bias_ptr + i) = *(bias_ptr + j);
                    }
                    auto new_bias_const =
                        op::Constant::create(ngraph::element::f32, Shape{1ull, Co_new, 1ull, 1ull}, new_bias);
                    auto new_add = std::make_shared<opset1::Add>(new_conv->output(0), new_bias_const->output(0));
                    auto add_shape = new_add->get_output_shape(0);
                    upstream[0] = new_add->output(0);
                    InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
                }
            } else {
                auto conv_shape = new_conv->get_output_shape(0);
                upstream[0] = new_conv->output(0);
                InsertActivation(upstream, prelu_after, relu_after, sigmoid_after, tanh_after);
            }

            if (nchw_only) {
                // unfold output channels into W
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                    new_conv->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{2}, {Kw_parts, output_shape[1] * output_shape[2]})
                        ->output(0),
                    false);
                auto new_transpose =
                    std::make_shared<op::Transpose>(new_reshape->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                    new_transpose->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{4}, output_shape)->output(0),
                    false);
                if (prelu_after) {
                    ngraph::replace_node(prelu_after, new_reshape);
                } else if (relu_after) {
                    ngraph::replace_node(relu_after, new_reshape);
                } else if (sigmoid_after) {
                    ngraph::replace_node(sigmoid_after, new_reshape);
                } else if (tanh_after) {
                    ngraph::replace_node(tanh_after, new_reshape);
                } else if (add_after) {
                    ngraph::replace_node(add_after, new_reshape);
                } else {
                    ngraph::replace_node(conv, new_reshape);
                }

            } else {
                std::vector<float> zeros(output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3], 0.0f);
                auto new_transpose =
                    std::make_shared<op::Transpose>(new_conv->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                    new_transpose->output(0),
                    op::Constant::create(ngraph::element::i64,
                                         Shape{4},
                                         {output_shape[0], output_shape[2], output_shape[3], output_shape[1]})
                        ->output(0),
                    false);
                new_transpose =
                    std::make_shared<op::Transpose>(new_reshape->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                if (prelu_after) {
                    ngraph::replace_node(prelu_after, new_transpose);
                } else if (relu_after) {
                    ngraph::replace_node(relu_after, new_transpose);
                } else if (sigmoid_after) {
                    ngraph::replace_node(sigmoid_after, new_transpose);
                } else if (tanh_after) {
                    ngraph::replace_node(tanh_after, new_transpose);
                } else if (add_after) {
                    ngraph::replace_node(add_after, new_transpose);
                } else {
                    ngraph::replace_node(conv, new_transpose);
                }
            }

            is_graph_modfied = true;

        } else {
            continue;
        }
    }

    return is_graph_modfied;
}

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

#include "l2norm_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;

bool ngraph::pass::L2NormDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto reducel2 = std::dynamic_pointer_cast<op::v4::ReduceL2>(node);
        if (nullptr == reducel2) {
            continue;
        }

        const Output<Node>& parent = reducel2->input_value(0);
        ov::Shape reducel2_in_shape = parent.get_shape();
        ov::Shape reducel2_out_shape = reducel2->get_output_shape(0);
        ov::Output<ov::Node> reduction_axes = reducel2->input_value(1);
        auto axis_const_data =
            std::dynamic_pointer_cast<Constant>(reduction_axes.get_node_shared_ptr())->get_data_ptr();
        int64_t axis_value = *((int64_t*)axis_const_data);

        // find <ReduceL2>--><Clamp>--><Div> pattern
        // or <ReduceL2>--><Div> pattern else skip
        auto children = reducel2->output(0).get_target_inputs();
        if (children.size() != 1) {
            continue;
        }
        auto clamp_after =
            std::dynamic_pointer_cast<ngraph::opset1::Clamp>(children.begin()->get_node()->shared_from_this());
        auto div_after =
            std::dynamic_pointer_cast<ngraph::opset1::Divide>(children.begin()->get_node()->shared_from_this());
        if (div_after != nullptr) {
            const Output<Node>& div_parent = div_after->input_value(0);
            if (div_parent != parent) {
                continue;
            }
        } else if (clamp_after != nullptr) {
            auto children = clamp_after->output(0).get_target_inputs();
            if (children.size() != 1) {
                continue;
            }
            div_after =
                std::dynamic_pointer_cast<ngraph::opset1::Divide>(children.begin()->get_node()->shared_from_this());
            if (div_after != nullptr) {
                const Output<Node>& div_parent = div_after->input_value(0);
                if (div_parent != parent) {
                    continue;
                }
            } else {
                continue;
            }
        }

        size_t N = 1, C = 1, H, W;
        if (reducel2_in_shape.size() == 4) {
            N = reducel2_in_shape[0];
            C = reducel2_in_shape[1];
            H = reducel2_in_shape[2];
            W = reducel2_in_shape[3];
        } else if (reducel2_in_shape.size() == 3) {
            C = reducel2_in_shape[0];
            H = reducel2_in_shape[1];
            W = reducel2_in_shape[2];
        } else if (reducel2_in_shape.size() == 2) {
            H = reducel2_in_shape[0];
            W = reducel2_in_shape[1];
        } else {
            continue;
        }

        if (((reducel2_in_shape.size() == 2) && (axis_value != 1)) ||
            ((reducel2_in_shape.size() == 3) && (axis_value != 2)) ||
            ((reducel2_in_shape.size() == 4) && (axis_value != 3))) {
            continue;  // only support W axis
        }

        if (N != 1) {
            continue;  // Batch case not yet implemented
        } else if (C != 1) {
            continue;
        } else {
            OutputVector upstream;
            upstream.push_back(parent);
            size_t H_new = H;
            size_t W_new = W;

            // Check if average must be split
            size_t num_parts = 1;
            while (W_new / num_parts > 768) {  // 768 is maximum GNA1/2 kernel size
                num_parts *= 2;
            }
            // Abort if W_new is not divisible by power of 2
            if ((W_new / num_parts) * num_parts != W_new) {
                continue;
            }

            // Create MVN averaging weights --------
            std::vector<float> avg_weights(8 * W_new / num_parts, 1.0f);
            std::vector<float> avg_broadcast(8 * W_new * num_parts, 0.0f);
            std::vector<float> minus_half(H_new * W_new, -0.5f);
            for (size_t i = 0; i < W_new * num_parts; i++) {
                avg_broadcast[i * 8] = 1.0f;
            }
            auto avg_weights_const =
                op::Constant::create(ngraph::element::f32, Shape{8, W_new / num_parts, 1, 1}, avg_weights);
            auto avg_broadcast_const =
                op::Constant::create(ngraph::element::f32, Shape{W_new, 8 * num_parts, 1, 1}, avg_broadcast);
            auto minus_half_const = op::Constant::create(ngraph::element::f32, Shape{1, H_new * W_new}, minus_half);

            auto squared = std::make_shared<op::v1::Multiply>(parent, parent);
            squared->set_friendly_name("L2NormSqrDiff");
            auto squared_reshape = std::make_shared<ngraph::opset1::Reshape>(
                squared->output(0),
                op::Constant::create(ngraph::element::i64,
                                     Shape{4},
                                     std::initializer_list<decltype(N)>{N, H_new * num_parts, 1ull, W_new / num_parts})
                    ->output(0),
                false);
            auto transposed_input_3 =
                std::make_shared<op::Transpose>(squared_reshape->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            auto transposed_avg_conv_3 = std::make_shared<opset1::Convolution>(transposed_input_3->output(0),
                                                                               avg_weights_const->output(0),
                                                                               Strides{1, 1},
                                                                               CoordinateDiff{0, 0},
                                                                               CoordinateDiff{0, 0},
                                                                               Strides{1, 1},
                                                                               PadType::VALID);
            transposed_avg_conv_3->set_friendly_name("L2NormAvg3");
            auto avg_conv_3 =
                std::make_shared<op::Transpose>(transposed_avg_conv_3->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            auto reshape_avg_conv_3 = std::make_shared<ngraph::opset1::Reshape>(
                avg_conv_3->output(0),
                op::Constant::create(ngraph::element::i64,
                                     Shape{4},
                                     std::initializer_list<decltype(N)>{N, 1ull, H_new, 8 * num_parts})
                    ->output(0),
                false);
            auto transposed_input_4 =
                std::make_shared<op::Transpose>(reshape_avg_conv_3->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            auto transposed_avg_conv_4 = std::make_shared<opset1::Convolution>(transposed_input_4->output(0),
                                                                               avg_broadcast_const->output(0),
                                                                               Strides{1, 1},
                                                                               CoordinateDiff{0, 0},
                                                                               CoordinateDiff{0, 0},
                                                                               Strides{1, 1},
                                                                               PadType::VALID);
            transposed_avg_conv_4->set_friendly_name("L2NormAvg4");
            auto avg_conv_4 =
                std::make_shared<op::Transpose>(transposed_avg_conv_4->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            auto reshape_avg_conv_4 = std::make_shared<ngraph::opset1::Reshape>(
                avg_conv_4->output(0),
                op::Constant::create(ngraph::element::i64,
                                     Shape{2},
                                     std::initializer_list<decltype(H_new)>{1ull, H_new * W_new})
                    ->output(0),
                false);
            OutputVector avg;
            avg.push_back(reshape_avg_conv_4->output(0));
            auto log_var = std::make_shared<ngraph::opset1::Log>(avg[0]);
            log_var->set_friendly_name("L2NormLogVar");
            auto log_inv_stdev = std::make_shared<op::v1::Multiply>(log_var->output(0), minus_half_const->output(0));
            log_inv_stdev->set_friendly_name("L2NormLogInvStdev");
            auto inv_stdev = std::make_shared<ngraph::opset1::Exp>(log_inv_stdev->output(0));
            inv_stdev->set_friendly_name("L2NormInvStdev");
            auto normalized_output = std::make_shared<op::v1::Multiply>(parent, inv_stdev->output(0));
            normalized_output->set_friendly_name("L2NormOutput");
            auto l2normalize_output =
                (reducel2_in_shape.size() == 3)
                    ? std::make_shared<ngraph::opset1::Reshape>(
                          normalized_output->output(0),
                          op::Constant::create(ngraph::element::i64, Shape{3}, {C, H, W})->output(0),
                          false)
                    : std::make_shared<ngraph::opset1::Reshape>(
                          normalized_output->output(0),
                          op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),
                          false);
            ngraph::replace_node(div_after, l2normalize_output);

            is_graph_modfied = true;
        }
    }
    return is_graph_modfied;
}

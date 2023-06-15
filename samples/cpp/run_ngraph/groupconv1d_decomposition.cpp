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

#include "groupconv1d_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;

NGRAPH_RTTI_DEFINITION(ngraph::pass::GroupConvolution1dDecomposition, "GroupConvolution1dDecomposition");
bool ngraph::pass::GroupConvolution1dDecomposition::run_on_model(const std::shared_ptr<ov::Model>& m) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : m->get_ordered_ops()) {
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
        Output<Node>& parent = conv->input_value(0);
        auto children = conv->output(0).get_target_inputs();
        if (children.size() != 1) {
            continue;
        }
        auto transpose_before =
            std::dynamic_pointer_cast<ngraph::opset1::Transpose>(parent.get_node()->shared_from_this());
        if (transpose_before == nullptr) {
            continue;
        }
        parent = transpose_before->input_value(0);
        auto add_after =
            std::dynamic_pointer_cast<ngraph::opset1::Add>(children.begin()->get_node()->shared_from_this());
        auto transpose_after =
            std::dynamic_pointer_cast<ngraph::opset1::Transpose>(children.begin()->get_node()->shared_from_this());
        if (add_after != nullptr) {
            auto add_children = add_after->output(0).get_target_inputs();
            if (add_children.size() != 1) {
                continue;
            }
            transpose_after = std::dynamic_pointer_cast<ngraph::opset1::Transpose>(
                add_children.begin()->get_node()->shared_from_this());
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

        if ((W != Kw) || (pads_begin[0] != 0) || (pads_begin[1] != 0) || (pads_end[0] != 0) ||
            (pads_end[1] != 0)) {  // if cannot be converted to 1D then abort
            continue;
        }
        auto new_shape = parent.get_shape();
        new_shape[2] = new_shape[1] * new_shape[2];
        new_shape[1] = 1;
        auto new_reshape = std::make_shared<ngraph::opset1::Reshape>(
            parent,
            op::Constant::create(ngraph::element::i64, Shape{4}, new_shape)->output(0),
            false);
        new_reshape->set_friendly_name("ReshapeTo1D");
        auto new_transpose =
            std::make_shared<op::Transpose>(new_reshape->output(0),
                                            op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
        new_transpose->set_friendly_name(transpose_before->get_friendly_name());
        auto new_weights_const = op::Constant::create(ngraph::element::f32, Shape{G, Co, Ci, 1, Kh * Kw}, weight_ptr);
        new_weights_const->set_friendly_name("ReplaceWeights");
        auto new_strides = strides;
        new_strides[1] = Kw * strides[0];
        new_strides[0] = 1;
        auto new_conv = std::make_shared<opset1::GroupConvolution>(new_transpose->output(0),
                                                                   new_weights_const->output(0),
                                                                   new_strides,
                                                                   pads_begin,
                                                                   pads_end,
                                                                   dilations,
                                                                   auto_pad);
        new_conv->set_friendly_name("ReplaceGroupConv");
        if (add_after != nullptr) {
            auto bias_const =
                std::dynamic_pointer_cast<opset1::Constant>(add_after->input_value(1).get_node_shared_ptr());
            const float* bias_ptr = bias_const->get_data_ptr<float>();
            auto new_bias_const = op::Constant::create(ngraph::element::f32, Shape{1ull, Co, 1ull, 1ull}, bias_ptr);
            auto new_add = std::make_shared<opset1::Add>(new_conv->output(0), new_bias_const->output(0));
            new_transpose =
                std::make_shared<op::Transpose>(new_add->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
        } else {
            new_transpose =
                std::make_shared<op::Transpose>(new_conv->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
        }

        ngraph::replace_node(transpose_after, new_transpose);

        is_graph_modfied = true;
    }

    return is_graph_modfied;
}

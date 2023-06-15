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

#include "to_nhwc.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;

NGRAPH_RTTI_DEFINITION(ngraph::pass::ToNHWC, "ToNHWC");
bool ngraph::pass::ToNHWC::run_on_model(const std::shared_ptr<ov::Model>& m) {
    std::vector<std::shared_ptr<ov::Node>> to_add, to_add_new;  // list of nodes to add to path
    std::vector<std::shared_ptr<ov::Node>> path, new_path;
    bool is_graph_modfied = false;
    // Traverse nGraph Function in topological order to find starting points for search path
    for (auto& node : m->get_ordered_ops()) {
        auto parameter = std::dynamic_pointer_cast<Parameter>(node);
        if (nullptr == parameter) {
            continue;
        }
        std::shared_ptr<ov::op::v0::Parameter> new_parameter;
        auto output_shape = node->get_output_shape(0);
        auto N = output_shape[0];
        if (output_shape.size() == 1) {
            new_parameter = std::make_shared<ngraph::opset8::Parameter>(ov::element::f32, ov::Shape{N});
        } else if (output_shape.size() == 2) {
            auto C = output_shape[1];
            new_parameter = std::make_shared<ngraph::opset8::Parameter>(ov::element::f32, ov::Shape{N, C});
        } else if (output_shape.size() == 3) {
            auto C = output_shape[1];
            auto H = output_shape[2];
            new_parameter = std::make_shared<ngraph::opset8::Parameter>(ov::element::f32, ov::Shape{N, H, C});
        } else if (output_shape.size() == 4) {
            auto C = output_shape[1];
            auto H = output_shape[2];
            auto W = output_shape[3];
            new_parameter = std::make_shared<ngraph::opset8::Parameter>(ov::element::f32, ov::Shape{N, H, W, C});
        } else {
            fprintf(stderr, "Support for >4D inputs not yet added!  Aborting to_nhwc transformation...\n");
            exit(-1);
        }
        new_path.push_back(new_parameter);
        path.push_back(node);
    }
    // walk all paths until there are none left, convert all tensors to NHWC
    while (path.size() > 0) {
        for (uint32_t i = 0; i < path.size(); i++) {
            auto children = path[i]->output(0).get_target_inputs();
            for (auto p = children.begin(); p != children.end(); p++) {
                auto input_shape = p->get_shape();
                auto output_index = p->get_source_output().get_index();
                auto input_index = p->get_index();
                auto node = p->get_node()->shared_from_this();
                auto result = std::dynamic_pointer_cast<opset1::Result>(node);
                auto fake_quantize = std::dynamic_pointer_cast<opset1::FakeQuantize>(node);
                auto reshape = std::dynamic_pointer_cast<opset1::Reshape>(node);
                auto convolution = std::dynamic_pointer_cast<opset1::Convolution>(node);
                auto groupconvolution = std::dynamic_pointer_cast<opset1::GroupConvolution>(node);
                auto transconvolution = std::dynamic_pointer_cast<opset1::ConvolutionBackpropData>(node);
                auto matmul = std::dynamic_pointer_cast<opset1::MatMul>(node);
                auto add = std::dynamic_pointer_cast<opset1::Add>(node);
                auto multiply = std::dynamic_pointer_cast<opset1::Multiply>(node);
                auto subtract = std::dynamic_pointer_cast<opset1::Subtract>(node);
                auto transpose = std::dynamic_pointer_cast<opset1::Transpose>(node);
                auto concat = std::dynamic_pointer_cast<opset1::Concat>(node);
                auto split = std::dynamic_pointer_cast<opset1::Split>(node);
                auto variadic_split = std::dynamic_pointer_cast<opset1::VariadicSplit>(node);
                auto exp = std::dynamic_pointer_cast<opset1::Exp>(node);
                auto log = std::dynamic_pointer_cast<opset1::Log>(node);
                auto maxpool = std::dynamic_pointer_cast<opset1::MaxPool>(node);
                auto relu = std::dynamic_pointer_cast<opset1::Relu>(node);
                auto sigmoid = std::dynamic_pointer_cast<opset1::Sigmoid>(node);
                auto tanh = std::dynamic_pointer_cast<opset1::Tanh>(node);

                if (result) {
                    auto new_result = std::make_shared<ngraph::opset8::Result>(new_path[i]->output(output_index));
                    ngraph::replace_node(result, new_result);
                    is_graph_modfied = true;
                } else if (fake_quantize) {
                    fprintf(stderr, "Found FakeQuantize.  Aborting to_nhwc transformation...\n");
                    exit(-1);
                } else if (reshape) {
                    auto output_shape = node->get_output_shape(0);
                    auto output_size = output_shape.size();
                    std::shared_ptr<ov::op::v1::Reshape> new_reshape = nullptr;
                    if (output_size < 3) {
                        new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                            new_path[i]->output(output_index),
                            op::Constant::create(ngraph::element::i64, Shape{output_size}, output_shape)->output(0),
                            false);
                    } else if (output_size == 3) {
                        new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                            new_path[i]->output(output_index),
                            op::Constant::create(ngraph::element::i64,
                                                 Shape{3},
                                                 {output_shape[0], output_shape[2], output_shape[1]})
                                ->output(0),
                            false);
                    } else if (output_size == 4) {
                        new_reshape = std::make_shared<ngraph::opset1::Reshape>(
                            new_path[i]->output(output_index),
                            op::Constant::create(ngraph::element::i64,
                                                 Shape{4},
                                                 {output_shape[0], output_shape[2], output_shape[3], output_shape[1]})
                                ->output(0),
                            false);
                    } else {
                        fprintf(stderr, "Support for >4D inputs not yet added!  Aborting to_nhwc transformation...\n");
                        exit(-1);
                    }
                    to_add.push_back(reshape);
                    to_add_new.push_back(new_reshape);
                } else if (convolution) {
                    auto auto_pad = convolution->get_auto_pad();
                    auto dilations = convolution->get_dilations();
                    auto pads_begin = convolution->get_pads_begin();
                    auto pads_end = convolution->get_pads_end();
                    auto strides = convolution->get_strides();
                    auto weights_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(
                        convolution->input_value(1).get_node_shared_ptr());
                    auto input_dim = input_shape.size();
                    std::shared_ptr<ov::op::v1::Transpose> new_transpose = nullptr;
                    OutputVector upstream, new_upstream;
                    upstream.push_back(convolution->output(0));
                    if ((input_dim != 3) && (input_dim != 4)) {
                        fprintf(stderr,
                                "Only 3D and 4D convolutions are supported!  Aborting to_nhwc transformation...\n");
                        exit(-1);
                    }
                    if (input_dim == 3) {
                        new_transpose = std::make_shared<op::Transpose>(
                            new_path[i]->output(output_index),
                            op::Constant::create(element::Type_t::i64, Shape{3}, {0, 2, 1}));
                    } else {
                        new_transpose = std::make_shared<op::Transpose>(
                            new_path[i]->output(output_index),
                            op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                    }
                    auto new_conv = std::make_shared<opset1::Convolution>(new_transpose->output(0),
                                                                          weights_const->output(0),
                                                                          strides,
                                                                          pads_begin,
                                                                          pads_end,
                                                                          dilations,
                                                                          auto_pad);
                    new_upstream.push_back(new_conv->output(0));
                    auto conv_children = convolution->output(0).get_target_inputs();
                    if (conv_children.size() == 1) {
                        auto bias = std::dynamic_pointer_cast<opset1::Add>(
                            conv_children.begin()->get_node()->shared_from_this());
                        if (bias) {
                            auto bias_const =
                                std::dynamic_pointer_cast<opset1::Constant>(bias->get_input_node_shared_ptr(1));
                            if (bias_const) {
                                auto new_bias = std::make_shared<opset1::Add>(new_upstream[0], bias_const->output(0));
                                upstream[0] = bias->output(0);
                                new_upstream[0] = new_bias->output(0);
                            }
                        }
                        auto bias_children = upstream[0].get_target_inputs();
                        if (bias_children.size() == 1) {
                            auto act_relu =
                                std::dynamic_pointer_cast<opset1::Relu>(upstream[0].get_node()->shared_from_this());
                            auto act_sigmoid =
                                std::dynamic_pointer_cast<opset1::Sigmoid>(upstream[0].get_node()->shared_from_this());
                            auto act_tanh =
                                std::dynamic_pointer_cast<opset1::Tanh>(upstream[0].get_node()->shared_from_this());
                            if (act_relu) {
                                auto new_relu = std::make_shared<ngraph::opset1::Relu>(new_upstream[0]);
                                new_upstream[0] = new_relu->output(0);
                            } else if (act_sigmoid) {
                                auto new_sigmoid = std::make_shared<ngraph::opset1::Sigmoid>(new_upstream[0]);
                                new_upstream[0] = new_sigmoid->output(0);
                            } else if (act_tanh) {
                                auto new_tanh = std::make_shared<ngraph::opset1::Tanh>(new_upstream[0]);
                                new_upstream[0] = new_tanh->output(0);
                            }
                        }
                    }
                    if (input_dim == 3) {
                        new_transpose = std::make_shared<op::Transpose>(
                            upstream[0],
                            op::Constant::create(element::Type_t::i64, Shape{3}, {0, 2, 1}));
                    } else {
                        new_transpose = std::make_shared<op::Transpose>(
                            upstream[0],
                            op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                    }
                    to_add.push_back(convolution);
                    to_add_new.push_back(new_transpose);
                } else if (groupconvolution) {
                    auto auto_pad = groupconvolution->get_auto_pad();
                    auto dilations = groupconvolution->get_dilations();
                    auto pads_begin = groupconvolution->get_pads_begin();
                    auto pads_end = groupconvolution->get_pads_end();
                    auto strides = groupconvolution->get_strides();
                    auto weights_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(
                        groupconvolution->input_value(1).get_node_shared_ptr());
                    std::shared_ptr<ov::op::v1::Transpose> new_transpose = nullptr;
                    if (input_shape.size() == 3) {
                        new_transpose = std::make_shared<op::Transpose>(
                            new_path[i]->output(output_index),
                            op::Constant::create(element::Type_t::i64, Shape{3}, {0, 2, 1}));
                        auto new_conv = std::make_shared<opset1::Convolution>(new_transpose->output(0),
                                                                              weights_const->output(0),
                                                                              strides,
                                                                              pads_begin,
                                                                              pads_end,
                                                                              dilations,
                                                                              auto_pad);
                        new_transpose = std::make_shared<op::Transpose>(
                            new_conv->output(0),
                            op::Constant::create(element::Type_t::i64, Shape{3}, {0, 2, 1}));
                    } else if (input_shape.size() == 4) {
                        new_transpose = std::make_shared<op::Transpose>(
                            new_path[i]->output(output_index),
                            op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                        auto new_conv = std::make_shared<opset1::Convolution>(new_transpose->output(0),
                                                                              weights_const->output(0),
                                                                              strides,
                                                                              pads_begin,
                                                                              pads_end,
                                                                              dilations,
                                                                              auto_pad);
                        new_transpose = std::make_shared<op::Transpose>(
                            new_conv->output(0),
                            op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                    } else {
                        fprintf(
                            stderr,
                            "Only 3D and 4D groupconvolutions are supported!  Aborting to_nhwc transformation...\n");
                        exit(-1);
                    }
                    to_add.push_back(groupconvolution);
                    to_add_new.push_back(new_transpose);
                } else if (transconvolution) {
                    auto auto_pad = transconvolution->get_auto_pad();
                    auto dilations = transconvolution->get_dilations();
                    auto output_padding = transconvolution->get_output_padding();
                    auto pads_begin = transconvolution->get_pads_begin();
                    auto pads_end = transconvolution->get_pads_end();
                    auto strides = transconvolution->get_strides();
                    auto weights_const = std::dynamic_pointer_cast<ngraph::opset1::Constant>(
                        transconvolution->input_value(1).get_node_shared_ptr());
                    std::shared_ptr<ov::op::v1::Transpose> new_transpose = nullptr;
                    if (input_shape.size() == 3) {
                        new_transpose = std::make_shared<op::Transpose>(
                            new_path[i]->output(output_index),
                            op::Constant::create(element::Type_t::i64, Shape{3}, {0, 2, 1}));
                        auto new_conv = std::make_shared<opset1::Convolution>(new_transpose->output(0),
                                                                              weights_const->output(0),
                                                                              strides,
                                                                              pads_begin,
                                                                              pads_end,
                                                                              dilations,
                                                                              auto_pad);
                        new_transpose = std::make_shared<op::Transpose>(
                            new_conv->output(0),
                            op::Constant::create(element::Type_t::i64, Shape{3}, {0, 2, 1}));
                    } else if (input_shape.size() == 4) {
                        new_transpose = std::make_shared<op::Transpose>(
                            new_path[i]->output(output_index),
                            op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                        auto new_conv = std::make_shared<opset1::Convolution>(new_transpose->output(0),
                                                                              weights_const->output(0),
                                                                              strides,
                                                                              pads_begin,
                                                                              pads_end,
                                                                              dilations,
                                                                              auto_pad);
                        new_transpose = std::make_shared<op::Transpose>(
                            new_conv->output(0),
                            op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                    } else {
                        fprintf(
                            stderr,
                            "Only 3D and 4D transconvolutions are supported!  Aborting to_nhwc transformation...\n");
                        exit(-1);
                    }
                    to_add.push_back(transconvolution);
                    to_add_new.push_back(new_transpose);
                } else if (matmul) {
                } else if (add) {
                } else if (multiply) {
                    // check if layer is on the merge list, if so finish processing it, if not add it
                } else if (subtract) {
                } else if (transpose) {
                } else if (concat) {
                } else if (split) {
                } else if (variadic_split) {
                } else if (exp) {
                    auto new_exp = std::make_shared<ngraph::opset1::Exp>(new_path[i]->output(output_index));
                    to_add.push_back(exp);
                    to_add_new.push_back(new_exp);
                } else if (log) {
                    auto new_log = std::make_shared<ngraph::opset1::Log>(new_path[i]->output(output_index));
                    to_add.push_back(log);
                    to_add_new.push_back(new_log);
                } else if (maxpool) {
                    fprintf(stderr, "Stanalone pooling not yet supported!  Aborting to_nhwc transformation...\n");
                    exit(-1);
                } else if (relu) {
                    auto new_relu = std::make_shared<ngraph::opset1::Relu>(new_path[i]->output(output_index));
                    to_add.push_back(relu);
                    to_add_new.push_back(new_relu);
                } else if (sigmoid) {
                    auto new_sigmoid = std::make_shared<ngraph::opset1::Sigmoid>(new_path[i]->output(output_index));
                    to_add.push_back(sigmoid);
                    to_add_new.push_back(new_sigmoid);
                } else if (tanh) {
                    auto new_tanh = std::make_shared<ngraph::opset1::Tanh>(new_path[i]->output(output_index));
                    to_add.push_back(tanh);
                    to_add_new.push_back(new_tanh);
                } else {
                    fprintf(stderr, "Support for operator not yet added!  Aborting to_nhwc transformation...\n");
                    exit(-1);
                }
            }
        }
        path.clear();
        new_path.clear();
        for (uint32_t i = 0; i < to_add.size(); i++) {
            path.push_back(to_add[i]);
            new_path.push_back(to_add_new[i]);
        }
        to_add.clear();
        to_add_new.clear();
    }

    return is_graph_modfied;
}

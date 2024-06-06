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

#include <fstream>
#include <inference_engine.hpp>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/pass/serialize.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"

using namespace InferenceEngine;
using namespace ngraph;
using namespace op;

void setNodeNames(std::shared_ptr<ov::Node> node, char* name) {
    std::string sname(name);
    node->set_friendly_name(sname);
    for (uint32_t i = 0; i < node->outputs().size(); i++) {
        node->get_output_tensor(i).set_names({sname + "_" + std::to_string(i)});
    }
}

void fillRandom(std::vector<float>& data) {
    std::default_random_engine RNG;
    std::normal_distribution<float> gaussian(0.0, 1.0);
    for (uint32_t i = 0; i < data.size(); i++) {
        data[i] = gaussian(RNG);
    }
}

std::shared_ptr<Function> createNgraphFunctionCustomer(std::vector<std::string> ops,
                                                       std::vector<std::vector<size_t>> dims) {
    std::shared_ptr<opset1::Parameter> input_2d = nullptr;
    OutputVector upstream;
    ov::Shape new_shape;
    ngraph::ParameterVector input_list;
    size_t N, H, W, C;
    auto input_dim = dims[0].size();
    if ((ops[0] != "Parameter") || (input_dim < 2) || (input_dim > 4)) {
        fprintf(stderr, "Operator list must start with Parameter of dimension 2-4\n");
        exit(-1);
    }
    if (input_dim == 2) {
        N = dims[0][0];
        H = dims[0][1];
        input_2d = std::make_shared<op::Parameter>(element::Type_t::f32, Shape(std::vector<size_t>{{1, N * H}}));
    } else if (input_dim == 3) {
        N = dims[0][0];
        H = dims[0][1];
        C = dims[0][2];
        input_2d = std::make_shared<op::Parameter>(element::Type_t::f32, Shape(std::vector<size_t>{{1, N * H * C}}));
    } else {
        N = dims[0][0];
        H = dims[0][1];
        W = dims[0][2];
        C = dims[0][3];
        input_2d =
            std::make_shared<op::Parameter>(element::Type_t::f32, Shape(std::vector<size_t>{{1, N * H * W * C}}));
    }
    setNodeNames(input_2d, "input");
    upstream.push_back(input_2d->output(0));
    input_list.push_back(input_2d);

    for (uint32_t i = 0; i < ops.size(); i++) {
        if ((ops[i] == "Convolution") || (ops[i] == "ConvolutionBare")) {
            if (dims[i].size() != 16) {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr,
                        "Should be:  "
                        "(N,H,W,C,Cout,Cin,Kh,Kw,stride0,stride1,pads_begin0,pads_begin1,pads_end0,pads_end1,dilat0,"
                        "dilat1)\n");
                exit(-1);
            }
            size_t N = dims[i][0];
            size_t H = dims[i][1];
            size_t W = dims[i][2];
            size_t C = dims[i][3];
            size_t Cout = dims[i][4];
            size_t Cin = dims[i][5];
            size_t Kh = dims[i][6];
            size_t Kw = dims[i][7];
            auto weight_shape = Shape{Cout, Cin, Kh, Kw};
            auto strides = Strides{dims[i][8], dims[i][9]};
            auto pads_begin = CoordinateDiff{(int64_t)dims[i][10], (int64_t)dims[i][11]};
            auto pads_end = CoordinateDiff{(int64_t)dims[i][12], (int64_t)dims[i][13]};
            auto dilations = Strides{dims[i][14], dims[i][15]};
            auto auto_pad = PadType::EXPLICIT;
            if (ops[i] == "Convolution") {
                if (upstream[0].get_shape() != Shape{N, H, W, C}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, H, W, C})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4D");
                    upstream[0] = input_4d->output(0);
                }
                auto new_transpose =
                    std::make_shared<op::Transpose>(upstream[0],
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                setNodeNames(new_transpose, "Transpose");
                upstream[0] = new_transpose->output(0);
            } else {
                if (upstream[0].get_shape() != Shape{N, C, H, W}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4D");
                    upstream[0] = input_4d->output(0);
                }
            }
            std::vector<float> new_weights(Cout * Cin * Kh * Kw, 0.0f);
            fillRandom(new_weights);
            auto new_weights_const = op::Constant::create(ngraph::element::f32, weight_shape, new_weights);
            setNodeNames(new_weights_const, "Weights");
            auto new_conv = std::make_shared<opset1::Convolution>(upstream[0],
                                                                  new_weights_const->output(0),
                                                                  strides,
                                                                  pads_begin,
                                                                  pads_end,
                                                                  dilations,
                                                                  auto_pad);
            setNodeNames(new_conv, "Convolution");
            upstream[0] = new_conv->output(0);
            if (ops[i] == "Convolution") {
                auto new_transpose =
                    std::make_shared<op::Transpose>(new_conv->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                setNodeNames(new_transpose, "UnTranspose");
                upstream[0] = new_transpose->output(0);
            }
            new_shape = upstream[0].get_shape();

        } else if ((ops[i] == "ConvolutionBackpropData") || (ops[i] == "ConvolutionBackpropDataBare")) {
            if (dims[i].size() != 16) {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr,
                        "Should be:  "
                        "(N,H,W,C,Cout,Cin,Kh,Kw,stride0,stride1,pads_begin0,pads_begin1,pads_end0,pads_end1,dilat0,"
                        "dilat1)\n");
                exit(-1);
            }
            size_t N = dims[i][0];
            size_t H = dims[i][1];
            size_t W = dims[i][2];
            size_t C = dims[i][3];
            size_t Cout = dims[i][4];
            size_t Cin = dims[i][5];
            size_t Kh = dims[i][6];
            size_t Kw = dims[i][7];
            auto weight_shape = Shape{Cin, Cout, Kh, Kw};
            auto strides = Strides{dims[i][8], dims[i][9]};
            auto pads_begin = CoordinateDiff{(int64_t)dims[i][10], (int64_t)dims[i][11]};
            auto pads_end = CoordinateDiff{(int64_t)dims[i][12], (int64_t)dims[i][13]};
            auto dilations = Strides{dims[i][14], dims[i][15]};
            auto auto_pad = PadType::EXPLICIT;
            uint32_t out_pad_h = (H + pads_begin[0] + pads_end[0] - Kh) % strides[0];
            uint32_t out_pad_w = (W + pads_begin[1] + pads_end[1] - Kw) % strides[1];
            auto output_padding = CoordinateDiff{out_pad_h, out_pad_w};
            if (ops[i] == "ConvolutionBackpropData") {
                if (upstream[0].get_shape() != Shape{N, H, W, C}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, H, W, C})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4D");
                    upstream[0] = input_4d->output(0);
                }
                auto new_transpose =
                    std::make_shared<op::Transpose>(upstream[0],
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                setNodeNames(new_transpose, "Transpose");
                upstream[0] = new_transpose->output(0);
            } else {
                if (upstream[0].get_shape() != Shape{N, C, H, W}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4D");
                    upstream[0] = input_4d->output(0);
                }
            }
            std::vector<float> new_weights(Cout * Cin * Kh * Kw, 0.0f);
            fillRandom(new_weights);
            auto new_weights_const = op::Constant::create(ngraph::element::f32, weight_shape, new_weights);
            setNodeNames(new_weights_const, "Weights");
            auto new_conv = std::make_shared<opset1::ConvolutionBackpropData>(upstream[0],
                                                                              new_weights_const->output(0),
                                                                              strides,
                                                                              pads_begin,
                                                                              pads_end,
                                                                              dilations,
                                                                              auto_pad,
                                                                              output_padding);
            setNodeNames(new_conv, "TransposeConvolution");
            upstream[0] = new_conv->output(0);
            if (ops[i] == "ConvolutionBackpropData") {
                auto new_transpose =
                    std::make_shared<op::Transpose>(new_conv->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                setNodeNames(new_transpose, "UnTranspose");
                upstream[0] = new_transpose->output(0);
            }
            new_shape = upstream[0].get_shape();

        } else if ((ops[i] == "GroupConvolution") || (ops[i] == "GroupConvolutionBare")) {
            if (dims[i].size() != 17) {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr,
                        "Should be:  "
                        "(N,H,W,C,G,Cout,Cin,Kh,Kw,stride0,stride1,pads_begin0,pads_begin1,pads_end0,pads_end1,dilat0,"
                        "dilat1)\n");
                exit(-1);
            }
            size_t N = dims[i][0];
            size_t H = dims[i][1];
            size_t W = dims[i][2];
            size_t C = dims[i][3];
            size_t G = dims[i][4];
            size_t Cout = dims[i][5];
            size_t Cin = dims[i][6];
            size_t Kh = dims[i][7];
            size_t Kw = dims[i][8];
            auto weight_shape = Shape{G, Cin, Cout, Kh, Kw};
            auto strides = Strides{dims[i][9], dims[i][10]};
            auto pads_begin = CoordinateDiff{(int64_t)dims[i][11], (int64_t)dims[i][12]};
            auto pads_end = CoordinateDiff{(int64_t)dims[i][13], (int64_t)dims[i][14]};
            auto dilations = Strides{dims[i][15], dims[i][16]};
            auto auto_pad = PadType::EXPLICIT;
            auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                upstream[0],
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, H, W, C})->output(0),
                false);
            setNodeNames(input_4d, "Reshape4D");
            if (ops[i] == "GroupConvolution") {
                auto new_transpose =
                    std::make_shared<op::Transpose>(input_4d->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                setNodeNames(new_transpose, "Transpose");
                upstream[0] = new_transpose->output(0);
            }
            std::vector<float> new_weights(G * Cout * Cin * Kh * Kw, 0.0f);
            fillRandom(new_weights);
            auto new_weights_const = op::Constant::create(ngraph::element::f32, weight_shape, new_weights);
            setNodeNames(new_weights_const, "Weights");
            auto new_conv = std::make_shared<opset1::GroupConvolution>(upstream[0],
                                                                       new_weights_const->output(0),
                                                                       strides,
                                                                       pads_begin,
                                                                       pads_end,
                                                                       dilations,
                                                                       auto_pad);
            setNodeNames(new_conv, "TransposeConvolution");
            upstream[0] = new_conv->output(0);
            if (ops[i] == "GroupConvolution") {
                auto new_transpose =
                    std::make_shared<op::Transpose>(new_conv->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                setNodeNames(new_transpose, "UnTranspose");
                upstream[0] = new_transpose->output(0);
            }
            new_shape = upstream[0].get_shape();

        } else if (ops[i] == "Multiply") {
            if (dims[i].size() == 2) {
                size_t H = dims[i][0];
                size_t W = dims[i][1];
                auto weight_shape = Shape{H, W};
                if (upstream[0].get_shape() != Shape{H, W}) {
                    auto input_2d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{2}, {H, W})->output(0),
                        false);
                    setNodeNames(input_2d, "Reshape2D");
                    upstream[0] = input_2d->output(0);
                }
                std::vector<float> new_weights(H * W, 0.0f);
                fillRandom(new_weights);
                auto new_weights_const = op::Constant::create(ngraph::element::f32, weight_shape, new_weights);
                setNodeNames(new_weights_const, "Weights");
                auto new_multiply = std::make_shared<opset1::Multiply>(upstream[0], new_weights_const->output(0));
                setNodeNames(new_multiply, "Multiply");
                new_shape = new_multiply->get_output_shape(0);
                upstream[0] = new_multiply->output(0);
            } else if (dims[i].size() == 4) {
                size_t N = dims[i][0];
                size_t H = dims[i][1];
                size_t W = dims[i][2];
                size_t C = dims[i][3];
                auto weight_shape = Shape{N, H, W, C};
                if (upstream[0].get_shape() != Shape{N, H, W, C}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, H, W, C})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4D");
                    upstream[0] = input_4d->output(0);
                }
                std::vector<float> new_weights(N * H * W * C, 0.0f);
                fillRandom(new_weights);
                auto new_weights_const = op::Constant::create(ngraph::element::f32, weight_shape, new_weights);
                setNodeNames(new_weights_const, "Weights");
                auto new_multiply = std::make_shared<opset1::Multiply>(upstream[0], new_weights_const->output(0));
                setNodeNames(new_multiply, "Multiply");
                new_shape = new_multiply->get_output_shape(0);
                upstream[0] = new_multiply->output(0);
            } else {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (N,H,W,C)\n");
                fprintf(stderr, "  or        (H,W)\n");
                exit(-1);
            }

        } else if ((ops[i] == "Add") || (ops[i] == "AddBias")) {
            if (dims[i].size() == 2) {
                size_t H = dims[i][0];
                size_t W = dims[i][1];
                auto weight_shape = Shape{H, W};
                if (upstream[0].get_shape() != Shape{H, W}) {
                    auto input_2d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{2}, {H, W})->output(0),
                        false);
                    setNodeNames(input_2d, "Reshape2D");
                    upstream[0] = input_2d->output(0);
                }
                std::vector<float> new_weights(H * W, 0.0f);
                fillRandom(new_weights);
                auto new_weights_const = op::Constant::create(ngraph::element::f32, weight_shape, new_weights);
                setNodeNames(new_weights_const, "Weights");
                auto new_add = std::make_shared<opset1::Add>(upstream[0], new_weights_const->output(0));
                setNodeNames(new_add, "Add");
                new_shape = new_add->get_output_shape(0);
                upstream[0] = new_add->output(0);
            } else if (dims[i].size() == 4) {
                size_t N = dims[i][0];
                size_t H = dims[i][1];
                size_t W = dims[i][2];
                size_t C = dims[i][3];
                if (upstream[0].get_shape() != Shape{N, H, W, C}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, H, W, C})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4D");
                    upstream[0] = input_4d->output(0);
                }
                if (ops[i] == "AddBias") {
                    auto weight_shape = Shape{1, H, 1, 1};  // H plays role of C within NCHW op
                    std::vector<float> new_weights(H, 0.0f);
                    fillRandom(new_weights);
                    auto new_weights_const = op::Constant::create(ngraph::element::f32, weight_shape, new_weights);
                    setNodeNames(new_weights_const, "Bias");
                    auto new_add = std::make_shared<opset1::Add>(upstream[0], new_weights_const->output(0));
                    setNodeNames(new_add, "AddBias");
                    upstream[0] = new_add->output(0);
                } else {
                    auto weight_shape = Shape{N, H, W, C};
                    std::vector<float> new_weights(N * H * W * C, 0.0f);
                    fillRandom(new_weights);
                    auto new_weights_const = op::Constant::create(ngraph::element::f32, weight_shape, new_weights);
                    setNodeNames(new_weights_const, "Weights");
                    auto new_add = std::make_shared<opset1::Add>(upstream[0], new_weights_const->output(0));
                    setNodeNames(new_add, "Add");
                    upstream[0] = new_add->output(0);
                }
                new_shape = upstream[0].get_shape();
            } else {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (N,H,W,C)\n");
                fprintf(stderr, "  or        (H,W)\n");
                exit(-1);
            }

        } else if (ops[i] == "LSTMCell") {
            if (dims[i].size() != 3) {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (N,H,hidden_size)\n");
                exit(-1);
            }
            size_t N = dims[i][0];
            size_t H = dims[i][1];
            size_t hidden_size = dims[i][2];
            if (upstream[0].get_shape() != Shape{N, H}) {
                auto input_2d_new = std::make_shared<ngraph::opset1::Reshape>(
                    upstream[0],
                    op::Constant::create(ngraph::element::i64, Shape{2}, {N, H})->output(0),
                    false);
                setNodeNames(input_2d_new, "NewReshape2D");
                upstream[0] = input_2d_new->output(0);
            }
            std::vector<float> hidden_state(N * hidden_size, 0.0f);
            std::vector<float> cell_state(N * hidden_size, 0.0f);
            std::vector<float> F_weights(4 * hidden_size * H, 0.0f);
            fillRandom(F_weights);
            std::vector<float> R_weights(4 * hidden_size * hidden_size, 0.0f);
            fillRandom(R_weights);
            std::vector<float> bias(4 * hidden_size, 0.0f);
            fillRandom(bias);
            auto hidden_state_const = op::Constant::create(ngraph::element::f32, Shape{N, hidden_size}, hidden_state);
            setNodeNames(hidden_state_const, "HiddenState");
            // auto readvalue_hidden = std::make_shared<opset3::ReadValue>(hidden_state_const->output(0),
            // "hiddenstate"+std::to_string(i)); setNodeNames(readvalue_hidden, "ReadValueHidden");
            auto cell_state_const = op::Constant::create(ngraph::element::f32, Shape{N, hidden_size}, cell_state);
            setNodeNames(cell_state_const, "CellState");
            // auto readvalue_cell = std::make_shared<opset3::ReadValue>(cell_state_const->output(0),
            // "cellstate"+std::to_string(i)); setNodeNames(readvalue_cell, "ReadValueCell");
            auto F_const = op::Constant::create(ngraph::element::f32, Shape{4 * hidden_size, H}, F_weights);
            setNodeNames(F_const, "F_weights");
            auto R_const = op::Constant::create(ngraph::element::f32, Shape{4 * hidden_size, hidden_size}, R_weights);
            setNodeNames(R_const, "R_weights");
            auto bias_const = op::Constant::create(ngraph::element::f32, Shape{4 * hidden_size}, bias);
            setNodeNames(bias_const, "Bias");
            // auto new_lstm = std::make_shared<opset1::LSTMCell>(upstream[0], readvalue_hidden->output(0),
            // readvalue_cell->output(0),
            //     F_const->output(0), R_const->output(0), bias_const->output(0), hidden_size,
            //     ov::op::LSTMWeightsFormat::IFCO);
            auto new_lstm = std::make_shared<opset1::LSTMCell>(upstream[0],
                                                               hidden_state_const->output(0),
                                                               cell_state_const->output(0),
                                                               F_const->output(0),
                                                               R_const->output(0),
                                                               bias_const->output(0),
                                                               hidden_size,
                                                               ov::op::LSTMWeightsFormat::IFCO);
            setNodeNames(new_lstm, "LSTMCell");
            // auto assign_hidden = std::make_shared<opset3::Assign>(new_lstm->output(0),
            // "hiddenstate"+std::to_string(i)); setNodeNames(assign_hidden, "AssignHidden"); auto assign_cell =
            // std::make_shared<opset3::Assign>(new_lstm->output(1), "cellstate"+std::to_string(i));
            // setNodeNames(assign_hidden, "AssignHidden");
            // auto result_hidden = std::make_shared<op::Result>(assign_hidden->output(0));
            // setNodeNames(result_hidden, "ResultHidden");
            // auto result_cell = std::make_shared<op::Result>(assign_cell->output(0));
            // setNodeNames(result_cell, "ResultCell");
            new_shape = new_lstm->get_output_shape(0);
            auto result0 = std::make_shared<op::Result>(new_lstm->output(0));
            auto result1 = std::make_shared<op::Result>(new_lstm->output(1));
            // upstream[0] = assign_hidden->output(0);  // new_lstm->output(0);
            upstream[0] = new_lstm->output(0);

        } else if (ops[i] == "MVN") {
            if (dims[i].size() == 4) {
                size_t N = dims[i][0];
                size_t C = dims[i][1];
                size_t H = dims[i][2];
                size_t W = dims[i][3];
                auto across_channels = false;
                auto normalize_variance = true;
                float epsilon = 1e-5f;
                op::MVNEpsMode eps_mode = op::MVNEpsMode::INSIDE_SQRT;
                if (upstream[0].get_shape() != Shape{N, C, H, W}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4D");
                    upstream[0] = input_4d->output(0);
                }
                auto mvn_1 =
                    std::make_shared<op::v6::MVN>(upstream[0],
                                                  op::Constant::create(ngraph::element::i64, Shape{1}, {-1})->output(0),
                                                  normalize_variance,
                                                  epsilon,
                                                  eps_mode);
                setNodeNames(mvn_1, "Mvn1");
                new_shape = mvn_1->get_output_shape(0);
                upstream[0] = mvn_1->output(0);
            } else if (dims[i].size() == 3) {
                size_t C = dims[i][0];
                size_t H = dims[i][1];
                size_t W = dims[i][2];
                auto across_channels = false;
                auto normalize_variance = true;
                float epsilon = 1e-5f;
                op::MVNEpsMode eps_mode = op::MVNEpsMode::INSIDE_SQRT;
                if (upstream[0].get_shape() != Shape{C, H, W}) {
                    auto input_3d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{3}, {C, H, W})->output(0),
                        false);
                    setNodeNames(input_3d, "Reshape3D");
                    upstream[0] = input_3d->output(0);
                }
                auto mvn_1 =
                    std::make_shared<op::v6::MVN>(upstream[0],
                                                  op::Constant::create(ngraph::element::i64, Shape{1}, {-1})->output(0),
                                                  normalize_variance,
                                                  epsilon,
                                                  eps_mode);
                setNodeNames(mvn_1, "Mvn1");
                new_shape = mvn_1->get_output_shape(0);
                upstream[0] = mvn_1->output(0);
            } else {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (N,C,H,W) or (C,H,W)\n");
                exit(-1);
            }

        } else if (ops[i] == "Softmax") {
            if (dims[i].size() == 5) {
                size_t N = dims[i][0];
                size_t C = dims[i][1];
                size_t H = dims[i][2];
                size_t W = dims[i][3];
                size_t axis = dims[i][4];
                if (upstream[0].get_shape() != Shape{N, C, H, W}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4D");
                    upstream[0] = input_4d->output(0);
                }
                auto new_softmax = std::make_shared<opset1::Softmax>(upstream[0], axis);
                setNodeNames(new_softmax, "Softmax");
                new_shape = new_softmax->get_output_shape(0);
                upstream[0] = new_softmax->output(0);
            } else if (dims[i].size() == 4) {
                size_t C = dims[i][0];
                size_t H = dims[i][1];
                size_t W = dims[i][2];
                size_t axis = dims[i][3];
                if (upstream[0].get_shape() != Shape{C, H, W}) {
                    auto input_3d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{3}, {C, H, W})->output(0),
                        false);
                    setNodeNames(input_3d, "Reshape3D");
                    upstream[0] = input_3d->output(0);
                }
                auto new_softmax = std::make_shared<opset1::Softmax>(upstream[0], axis);
                setNodeNames(new_softmax, "Softmax");
                new_shape = new_softmax->get_output_shape(0);
                upstream[0] = new_softmax->output(0);
            } else {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (N,C,H,W,axis) or (C,H,W,axis)\n");
                exit(-1);
            }
        } else if (ops[i] == "Transpose") {
            if (dims[i].size() == 8) {
                size_t N = dims[i][0];
                size_t H = dims[i][1];
                size_t W = dims[i][2];
                size_t C = dims[i][3];
                size_t I0 = dims[i][4];
                size_t I1 = dims[i][5];
                size_t I2 = dims[i][6];
                size_t I3 = dims[i][7];
                if (upstream[0].get_shape() != Shape{N, H, W, C}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, H, W, C})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4D");
                    upstream[0] = input_4d->output(0);
                }
                auto new_transpose = std::make_shared<op::Transpose>(
                    upstream[0],
                    op::Constant::create(element::Type_t::i64, Shape{4}, {I0, I1, I2, I3}));
                setNodeNames(new_transpose, "Transpose");
                new_shape = new_transpose->get_output_shape(0);
                upstream[0] = new_transpose->output(0);
            } else if (dims[i].size() == 4) {
                size_t H = dims[i][0];
                size_t W = dims[i][1];
                size_t I0 = dims[i][2];
                size_t I1 = dims[i][3];
                if (upstream[0].get_shape() != Shape{H, W}) {
                    auto input_2d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{2}, {H, W})->output(0),
                        false);
                    setNodeNames(input_2d, "Reshape2D");
                    upstream[0] = input_2d->output(0);
                }
                auto new_transpose =
                    std::make_shared<op::Transpose>(upstream[0],
                                                    op::Constant::create(element::Type_t::i64, Shape{2}, {I0, I1}));
                setNodeNames(new_transpose, "Transpose");
                new_shape = new_transpose->get_output_shape(0);
                upstream[0] = new_transpose->output(0);
            } else {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (N,H,W,C,I0,I1,I2,I3)\n");
                fprintf(stderr, "  or        (H,W,I0,I1)\n");
                exit(-1);
            }
        } else if ((ops[i] == "Concat") || (ops[i] == "ConcatZero")) {
            if (dims[i].size() == 9) {
                size_t N1 = dims[i][0];
                size_t H1 = dims[i][1];
                size_t W1 = dims[i][2];
                size_t C1 = dims[i][3];
                size_t N2 = dims[i][4];
                size_t H2 = dims[i][5];
                size_t W2 = dims[i][6];
                size_t C2 = dims[i][7];
                size_t axis = dims[i][8];
                if (upstream[0].get_shape() != Shape{N1, H1, W1, C1}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N1, H1, W1, C1})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4Da");
                    upstream[0] = input_4d->output(0);
                }
                if (ops[i] == "Concat") {
                    auto input2_2d =
                        std::make_shared<op::Parameter>(element::Type_t::f32,
                                                        Shape(std::vector<size_t>{{1, N2 * H2 * W2 * C2}}));
                    setNodeNames(input2_2d, "input2");
                    input_list.push_back(input2_2d);
                    auto input2_4d = std::make_shared<ngraph::opset1::Reshape>(
                        input2_2d->output(0),
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N2, H2, W2, C2})->output(0),
                        false);
                    setNodeNames(input2_4d, "Reshape4Db");
                    upstream.push_back(input2_4d->output(0));
                } else {  // zero padding
                    std::vector<float> zeros(N2 * H2 * W2 * C2, 0.0f);
                    auto zero_const = op::Constant::create(ngraph::element::f32, {N2, H2, W2, C2}, zeros);
                    setNodeNames(zero_const, "ZeroPadding");
                    upstream.push_back(zero_const->output(0));
                }
                auto concat = std::make_shared<ngraph::opset1::Concat>(upstream, axis);
                setNodeNames(concat, "Concat");
                upstream.clear();
                upstream.push_back(concat->output(0));
                new_shape = upstream[0].get_shape();
            } else {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (N1,H1,W1,C1,N2,H2,W2,C2,axis)\n");
                exit(-1);
            }
        } else if ((ops[i] == "MatMul") || (ops[i] == "MatMulConst")) {
            if (dims[i].size() == 6) {
                size_t H1 = dims[i][0];
                size_t W1 = dims[i][1];
                size_t H2 = dims[i][2];
                size_t W2 = dims[i][3];
                bool transpose_a = (dims[i][4] != 0);
                bool transpose_b = (dims[i][5] != 0);
                if (upstream[0].get_shape() != Shape{H1, W1}) {
                    auto input_2d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{2}, {H1, W1})->output(0),
                        false);
                    setNodeNames(input_2d, "Reshape2Da");
                    upstream[0] = input_2d->output(0);
                }
                if (ops[i] == "MatMul") {
                    auto input2_2d =
                        std::make_shared<op::Parameter>(element::Type_t::f32, Shape(std::vector<size_t>{{H2, W2}}));
                    setNodeNames(input2_2d, "input2");
                    input_list.push_back(input2_2d);
                    upstream.push_back(input2_2d->output(0));
                    auto matmul = std::make_shared<op::MatMul>(upstream[0], upstream[1], transpose_a, transpose_b);
                    setNodeNames(matmul, "MatMul");
                    upstream.clear();
                    upstream.push_back(matmul->output(0));
                } else {  // multiply by const
                    if (upstream[0].get_shape() != Shape{H1, W1}) {
                        auto input1_2d = std::make_shared<ngraph::opset1::Reshape>(
                            upstream[0],
                            op::Constant::create(ngraph::element::i64, Shape{2}, {H1, W1})->output(0),
                            false);
                        setNodeNames(input1_2d, "Reshape2Da");
                        upstream[0] = input1_2d->output(0);
                    }
                    std::vector<float> weights(H2 * W2, 0.0f);
                    fillRandom(weights);
                    float* weights_ptr = weights.data();
                    auto weights_const = op::Constant::create(ngraph::element::f32, Shape{H2, W2}, weights);
                    auto matmul = std::make_shared<op::MatMul>(upstream[0], weights_const, transpose_a, transpose_b);
                    setNodeNames(matmul, "MatMul");
                    upstream.clear();
                    upstream.push_back(matmul->output(0));
                }
                new_shape = upstream[0].get_shape();
            } else if (dims[i].size() == 8) {
                size_t H1 = dims[i][0];
                size_t W1 = dims[i][1];
                size_t C1 = dims[i][2];
                size_t H2 = dims[i][3];
                size_t W2 = dims[i][4];
                size_t C2 = dims[i][5];
                bool transpose_a = (dims[i][6] != 0);
                bool transpose_b = (dims[i][7] != 0);
                if (upstream[0].get_shape() != Shape{H1, W1}) {
                    auto input_3d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{3}, {C1, H1, W1})->output(0),
                        false);
                    setNodeNames(input_3d, "Reshape2Da");
                    upstream[0] = input_3d->output(0);
                }
                if (ops[i] == "MatMul") {
                    auto input2_2d = std::make_shared<op::Parameter>(element::Type_t::f32,
                                                                     Shape(std::vector<size_t>{{1ull, C2 * H2 * W2}}));
                    setNodeNames(input2_2d, "input2");
                    input_list.push_back(input2_2d);
                    auto input2_3d = std::make_shared<ngraph::opset1::Reshape>(
                        input2_2d->output(0),
                        op::Constant::create(ngraph::element::i64, Shape{3}, {C2, H2, W2})->output(0),
                        false);
                    setNodeNames(input2_3d, "input2_3D");
                    upstream.push_back(input2_3d->output(0));
                    auto matmul = std::make_shared<op::MatMul>(upstream[0], upstream[1], transpose_a, transpose_b);
                    setNodeNames(matmul, "MatMul");
                    upstream.clear();
                    upstream.push_back(matmul->output(0));
                } else {  // multiply by const
                    if (upstream[0].get_shape() != Shape{C1, H1, W1}) {
                        auto input1_2d = std::make_shared<ngraph::opset1::Reshape>(
                            upstream[0],
                            op::Constant::create(ngraph::element::i64, Shape{3}, {C1, H1, W1})->output(0),
                            false);
                        setNodeNames(input1_2d, "Reshape2Da");
                        upstream[0] = input1_2d->output(0);
                    }
                    std::vector<float> weights(C2 * H2 * W2, 0.0f);
                    fillRandom(weights);
                    float* weights_ptr = weights.data();
                    auto weights_const = op::Constant::create(ngraph::element::f32, Shape{C2, H2, W2}, weights);
                    auto matmul = std::make_shared<op::MatMul>(upstream[0], weights_const, transpose_a, transpose_b);
                    setNodeNames(matmul, "MatMul");
                    upstream.clear();
                    upstream.push_back(matmul->output(0));
                }
                new_shape = upstream[0].get_shape();

            } else {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (H1,W1,H2,W2,0/1,0/1)\n");
                fprintf(stderr, "   or\n");
                fprintf(stderr, "Should be:  (H1,W1,C1,H2,W2,C2,0/1,0/1)\n");
                fprintf(stderr, "   where final 2 params are transpose flags\n");
                exit(-1);
            }
        } else if (ops[i] == "Sigmoid") {
            auto sigmoid = std::make_shared<op::v0::Sigmoid>(upstream[0]);
            setNodeNames(sigmoid, "Sigmoid");
            new_shape = sigmoid->get_output_shape(0);
            upstream[0] = sigmoid->output(0);
        } else if (ops[i] == "ReLU") {
            auto relu = std::make_shared<op::v0::Relu>(upstream[0]);
            setNodeNames(relu, "Relu");
            new_shape = relu->get_output_shape(0);
            upstream[0] = relu->output(0);
        } else if (ops[i] == "Tanh") {
            auto tanh = std::make_shared<op::v0::Tanh>(upstream[0]);
            setNodeNames(tanh, "Tanh");
            new_shape = tanh->get_output_shape(0);
            upstream[0] = tanh->output(0);
        } else if (ops[i] == "Slice") {
            if (dims[i].size() == 12) {
                size_t N = dims[i][0];
                size_t H = dims[i][1];
                size_t W = dims[i][2];
                size_t C = dims[i][3];
                size_t Nb = dims[i][4];
                size_t Hb = dims[i][5];
                size_t Wb = dims[i][6];
                size_t Cb = dims[i][7];
                size_t Ne = dims[i][8];
                size_t He = dims[i][9];
                size_t We = dims[i][10];
                size_t Ce = dims[i][11];
                if (upstream[0].get_shape() != Shape{N, H, W, C}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, H, W, C})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4Da");
                    upstream[0] = input_4d->output(0);
                }
                auto slice_start = op::Constant::create(ngraph::element::i64, Shape{4}, {Nb, Hb, Wb, Cb});
                auto slice_stop = op::Constant::create(ngraph::element::i64, Shape{4}, {Ne, He, We, Ce});
                auto slice_step = op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, 1ull, 1ull, 1ull});
                auto slice = std::make_shared<v8::Slice>(upstream[0], slice_start, slice_stop, slice_step);
                setNodeNames(slice, "Slice");
                new_shape = slice->get_output_shape(0);
                upstream[0] = slice->output(0);
            } else {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (N,H,W,C,Nbegin,Hbegin,Wbegin,Cbegin,Nend,Hend,Wend,Cend)\n");
                exit(-1);
            }
        } else if (ops[i] == "ReduceMean") {
            if (dims[i].size() == 3) {
                size_t H = dims[i][0];
                size_t W = dims[i][1];
                size_t axis0 = dims[i][2];
                if (upstream[0].get_shape() != Shape{H, W}) {
                    auto input_2d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{2}, {H, W})->output(0),
                        false);
                    setNodeNames(input_2d, "Reshape2Da");
                    upstream[0] = input_2d->output(0);
                }
                auto axis = op::Constant::create(ngraph::element::i64, Shape{1}, {axis0});
                auto reducemean = std::make_shared<v1::ReduceMean>(upstream[0], axis, true);
                setNodeNames(reducemean, "ReduceMean");
                new_shape = reducemean->get_output_shape(0);
                upstream[0] = reducemean->output(0);

            } else if (dims[i].size() == 4) {
                size_t C = dims[i][0];
                size_t H = dims[i][1];
                size_t W = dims[i][2];
                size_t axis0 = dims[i][3];
                if (upstream[0].get_shape() != Shape{C, H, W}) {
                    auto input_3d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{3}, {C, H, W})->output(0),
                        false);
                    setNodeNames(input_3d, "Reshape3Da");
                    upstream[0] = input_3d->output(0);
                }
                auto axis = op::Constant::create(ngraph::element::i64, Shape{1}, {axis0});
                auto reducemean = std::make_shared<v1::ReduceMean>(upstream[0], axis, true);
                setNodeNames(reducemean, "ReduceMean");
                new_shape = reducemean->get_output_shape(0);
                upstream[0] = reducemean->output(0);

            } else if (dims[i].size() == 5) {
                size_t N = dims[i][0];
                size_t C = dims[i][1];
                size_t H = dims[i][2];
                size_t W = dims[i][3];
                size_t axis0 = dims[i][4];
                if (upstream[0].get_shape() != Shape{N, C, H, W}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4Da");
                    upstream[0] = input_4d->output(0);
                }
                auto axis = op::Constant::create(ngraph::element::i64, Shape{1}, {axis0});
                auto reducemean = std::make_shared<v1::ReduceMean>(upstream[0], axis, true);
                setNodeNames(reducemean, "ReduceMean");
                new_shape = reducemean->get_output_shape(0);
                upstream[0] = reducemean->output(0);

            } else if (dims[i].size() == 6) {
                size_t N = dims[i][0];
                size_t C = dims[i][1];
                size_t H = dims[i][2];
                size_t W = dims[i][3];
                size_t axis0 = dims[i][4];
                size_t axis1 = dims[i][5];
                if (upstream[0].get_shape() != Shape{N, C, H, W}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, C, H, W})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4Da");
                    upstream[0] = input_4d->output(0);
                }
                auto axis = op::Constant::create(ngraph::element::i64, Shape{2}, {axis0, axis1});
                auto reducemean = std::make_shared<v1::ReduceMean>(upstream[0], axis, true);
                setNodeNames(reducemean, "ReduceMean");
                new_shape = reducemean->get_output_shape(0);
                upstream[0] = reducemean->output(0);

            } else {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (N,H,W,C,axis0,axis1)\n");
                exit(-1);
            }

        } else if (ops[i] == "AvgPool") {
            if (dims[i].size() == 6) {
                size_t N = dims[i][0];
                size_t H = dims[i][1];
                size_t W = dims[i][2];
                size_t C = dims[i][3];
                size_t Ph = dims[i][4];
                size_t Pw = dims[i][5];
                if (upstream[0].get_shape() != Shape{N, H, W, C}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, H, W, C})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4Da");
                    upstream[0] = input_4d->output(0);
                }
                auto avgpool = std::make_shared<v1::AvgPool>(upstream[0],
                                                             Strides{Ph, Pw},
                                                             Shape{0, 0},
                                                             Shape{0, 0},
                                                             Shape{Ph, Pw},
                                                             false,
                                                             ov::op::RoundingType::FLOOR,
                                                             op::PadType::VALID);
                setNodeNames(avgpool, "AvgPool");
                new_shape = avgpool->get_output_shape(0);
                upstream[0] = avgpool->output(0);
            } else {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (N,H,W,C,Ph,Pw)\n");
                exit(-1);
            }

        } else if (ops[i] == "MaxPool") {
            if (dims[i].size() == 6) {
                size_t N = dims[i][0];
                size_t H = dims[i][1];
                size_t W = dims[i][2];
                size_t C = dims[i][3];
                size_t Ph = dims[i][4];
                size_t Pw = dims[i][5];
                if (upstream[0].get_shape() != Shape{N, H, W, C}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {N, H, W, C})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4Da");
                    upstream[0] = input_4d->output(0);
                }
                auto maxpool = std::make_shared<v1::MaxPool>(upstream[0],
                                                             Strides{Ph, Pw},
                                                             Shape{0, 0},
                                                             Shape{0, 0},
                                                             Shape{Ph, Pw},
                                                             ov::op::RoundingType::FLOOR,
                                                             op::PadType::VALID);
                setNodeNames(maxpool, "MaxPool");
                new_shape = maxpool->get_output_shape(0);
                upstream[0] = maxpool->output(0);
            } else {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (N,H,W,C,Ph,Pw)\n");
                exit(-1);
            }
        } else if (ops[i] == "Split") {
            if (dims[i].size() == 4) {
                size_t H = dims[i][0];
                size_t W = dims[i][1];
                size_t axis = dims[i][2];
                size_t num_split = dims[i][3];
                if (upstream[0].get_shape() != Shape{H, W}) {
                    auto input_2d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        op::Constant::create(ngraph::element::i64, Shape{2}, {H, W})->output(0),
                        false);
                    setNodeNames(input_2d, "Reshape2D");
                    upstream[0] = input_2d->output(0);
                }
                const auto axis_node = ngraph::opset1::Constant::create(element::i64, Shape{}, {axis});
                auto split = std::make_shared<ngraph::opset1::Split>(upstream[0], axis_node->output(0), num_split);
                setNodeNames(split, "Split");
                new_shape = split->get_output_shape(0);
                upstream[0] = split->output(0);
            } else {
                fprintf(stderr, "Wrong number of parameters for %s\n", ops[i].c_str());
                fprintf(stderr, "Should be:  (H,W,axis,num_split)\n");
                exit(-1);
            }
        }
    }

    size_t num_elements = 1;
    for (uint32_t i = 0; i < new_shape.size(); i++) {
        num_elements *= new_shape[i];
    }
    auto output_2d = std::make_shared<ngraph::opset1::Reshape>(
        upstream[0],
        op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, num_elements})->output(0),
        false);
    setNodeNames(output_2d, "Reshape2D");
    auto result_full = std::make_shared<op::Result>(output_2d->output(0));
    setNodeNames(result_full, "Result");
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(result_full, input_list, "test_graph");

    return fnPtr;
}

std::string ExtractOpName(char* arg) {
    std::string name(arg);
    size_t pos = name.find_first_of("(");
    if (pos != std::string::npos) {
        name = name.substr(0, pos);
    }
    return name;
}

std::vector<size_t> ExtractDims(char* arg) {
    std::string name(arg);
    std::vector<size_t> dims;
    size_t begin_pos = name.find_first_of("(");
    size_t end_pos = name.find_last_of(")");
    if (begin_pos != std::string::npos) {
        name = name.substr(begin_pos + 1, end_pos - begin_pos - 1);
        while (name.length() > 0) {
            size_t comma_pos = name.find_first_of(",");
            if (comma_pos != std::string::npos) {
                dims.push_back(std::stoi(name.substr(0, comma_pos)));
                name = name.substr(comma_pos + 1, name.length() - comma_pos);
            } else {
                dims.push_back(std::stoi(name));
                name.clear();
            }
        }
    }
    return dims;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage:  make_ngraph model op1(N,H,W,C,param1,param2,...) op2(...) ..." << std::endl;
        return -1;
    }

    std::string xml_name(argv[1]);
    std::string bin_name(argv[1]);
    xml_name += ".xml";
    bin_name += ".bin";
    std::vector<std::string> operators;
    std::vector<std::vector<size_t>> dimensions;
    for (int i = 2; i < argc; i++) {
        operators.push_back(ExtractOpName(argv[i]));
        dimensions.push_back(ExtractDims(argv[i]));
    }

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(xml_name, bin_name, ngraph::pass::Serialize::Version::IR_V11);
    const auto& pass_config = manager.get_pass_config();
    manager.run_passes(createNgraphFunctionCustomer(operators, dimensions));

    return 0;
}

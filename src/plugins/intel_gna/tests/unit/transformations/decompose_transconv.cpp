// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <legacy/ngraph_ops/convolution_ie.hpp>
#include <legacy/ngraph_ops/deconvolution_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_convolutions.hpp>
#include <map>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace testing;

using InputShape = ngraph::PartialShape;
using WeightsShape = ngraph::Shape;

using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using ov::op::v0::Result;
using ov::op::v1::ConvolutionBackpropData;
using ov::op::v1::Transpose;

void setNodeNames(std::shared_ptr<ov::Node> node, std::string name) {
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

std::shared_ptr<ngraph::Function> createNgraphFunctionCustomer(std::vector<std::string> ops,
                                                               std::vector<std::vector<size_t>> dims) {
    std::shared_ptr<Parameter> input_2d = nullptr;
    ov::OutputVector upstream;
    ov::Shape new_shape;
    size_t N, H, W, C;
    auto input_dim = dims[0].size();
    if ((ops[0] != "Parameter") || (input_dim < 2) || (input_dim > 4)) {
        fprintf(stderr, "Operator list must start with Parameter of dimension 2-4\n");
        exit(-1);
    }
    if (input_dim == 2) {
        N = dims[0][0];
        H = dims[0][1];
        input_2d = std::make_shared<Parameter>(ov::element::Type_t::f32, ov::Shape(std::vector<size_t>{{1, N * H}}));
    } else if (input_dim == 3) {
        N = dims[0][0];
        H = dims[0][1];
        C = dims[0][2];
        input_2d =
            std::make_shared<Parameter>(ov::element::Type_t::f32, ov::Shape(std::vector<size_t>{{1, N * H * C}}));
    } else {
        N = dims[0][0];
        H = dims[0][1];
        W = dims[0][2];
        C = dims[0][3];
        input_2d =
            std::make_shared<Parameter>(ov::element::Type_t::f32, ov::Shape(std::vector<size_t>{{1, N * H * W * C}}));
    }
    setNodeNames(input_2d, "input");
    upstream.push_back(input_2d->output(0));

    for (uint32_t i = 0; i < ops.size(); i++) {
        if ((ops[i] == "ConvolutionBackpropData") || (ops[i] == "ConvolutionBackpropDataBare")) {
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
            auto weight_shape = ov::Shape{Cin, Cout, Kh, Kw};
            auto strides = ov::Strides{dims[i][8], dims[i][9]};
            auto pads_begin = ov::CoordinateDiff{(int64_t)dims[i][10], (int64_t)dims[i][11]};
            auto pads_end = ov::CoordinateDiff{(int64_t)dims[i][12], (int64_t)dims[i][13]};
            auto dilations = ov::Strides{dims[i][14], dims[i][15]};
            auto auto_pad = ov::op::PadType::EXPLICIT;
            uint32_t out_pad_h = (H + pads_begin[0] + pads_end[0] - Kh) % strides[0];
            uint32_t out_pad_w = (W + pads_begin[1] + pads_end[1] - Kw) % strides[1];
            auto output_padding = ov::CoordinateDiff{out_pad_h, out_pad_w};
            if (ops[i] == "ConvolutionBackpropData") {
                if (upstream[0].get_shape() != ov::Shape{N, H, W, C}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        Constant::create(ngraph::element::i64, ov::Shape{4}, {N, H, W, C})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4D");
                    upstream[0] = input_4d->output(0);
                }
                auto new_transpose =
                    std::make_shared<Transpose>(upstream[0],
                                                Constant::create(ov::element::Type_t::i64, ov::Shape{4}, {0, 3, 1, 2}));
                setNodeNames(new_transpose, "Transpose");
                upstream[0] = new_transpose->output(0);
            } else {
                if (upstream[0].get_shape() != ov::Shape{N, C, H, W}) {
                    auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
                        upstream[0],
                        Constant::create(ngraph::element::i64, ov::Shape{4}, {N, C, H, W})->output(0),
                        false);
                    setNodeNames(input_4d, "Reshape4D");
                    upstream[0] = input_4d->output(0);
                }
            }
            std::vector<float> new_weights(Cout * Cin * Kh * Kw, 0.0f);
            fillRandom(new_weights);
            auto new_weights_const = Constant::create(ngraph::element::f32, weight_shape, new_weights);
            setNodeNames(new_weights_const, "Weights");
            auto new_conv = std::make_shared<ConvolutionBackpropData>(upstream[0],
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
                    std::make_shared<Transpose>(new_conv->output(0),
                                                Constant::create(ov::element::Type_t::i64, ov::Shape{4}, {0, 2, 3, 1}));
                setNodeNames(new_transpose, "UnTranspose");
                upstream[0] = new_transpose->output(0);
            }
            new_shape = upstream[0].get_shape();
        }
    }

    size_t num_elements = 1;
    for (uint32_t i = 0; i < new_shape.size(); i++) {
        num_elements *= new_shape[i];
    }
    auto output_2d = std::make_shared<ngraph::opset1::Reshape>(
        upstream[0],
        Constant::create(ngraph::element::i64,
                         ov::Shape{2},
                         std::initializer_list<decltype(num_elements)>{1ull, num_elements})
            ->output(0),
        false);
    setNodeNames(output_2d, "Reshape2D");
    auto result_full = std::make_shared<Result>(output_2d->output(0));
    setNodeNames(result_full, "Result");
    std::shared_ptr<ngraph::Function> fnPtr =
        std::make_shared<ngraph::Function>(result_full, ngraph::ParameterVector{input_2d}, "test_graph");

    return fnPtr;
}

class DecomposeTransconvTest : public CommonTestUtils::TestsCommon,
                               public testing::WithParamInterface<std::tuple<InputShape, WeightsShape>> {
public:
    std::shared_ptr<ngraph::Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& weights_shape = std::get<1>(GetParam());

        f = get_initial_function(input_shape, weights_shape);
        f_ref = get_reference_function(input_shape, weights_shape);
    }

private:
    std::shared_ptr<ngraph::Function> get_initial_function(const ngraph::PartialShape& input_shape,
                                                           const ov::Shape& weights_shape) {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(ngraph::element::f32, input_shape);
        auto weights = Constant::create(ngraph::element::f32, weights_shape, {1});
        auto conv = std::make_shared<ngraph::opset1::ConvolutionBackpropData>(input,
                                                                              weights,
                                                                              ov::Strides(spatial_dims, 1),
                                                                              ov::CoordinateDiff(spatial_dims, 0),
                                                                              ov::CoordinateDiff(spatial_dims, 0),
                                                                              ov::Strides(spatial_dims, 1));

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{conv}, ngraph::ParameterVector{input});
    }

    std::shared_ptr<ngraph::Function> get_reference_function(const ngraph::PartialShape& input_shape,
                                                             const ov::Shape& weights_shape) {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(ngraph::element::f32, input_shape);
        auto weights = Constant::create(ngraph::element::f32, weights_shape, {1});
        auto conv = std::make_shared<ngraph::op::DeconvolutionIE>(input,
                                                                  weights,
                                                                  ov::Strides(spatial_dims, 1),
                                                                  ov::Strides(spatial_dims, 1),
                                                                  ov::CoordinateDiff(spatial_dims, 0),
                                                                  ov::CoordinateDiff(spatial_dims, 0),
                                                                  ngraph::element::f32);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{conv}, ngraph::ParameterVector{input});
    }
};

TEST_P(DecomposeTransconvTest, CompareFunctions) {
    const auto orig_shape = f->get_output_partial_shape(0);
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::ConvertConvolutions>();
    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
    ASSERT_TRUE(orig_shape.same_scheme(f->get_output_partial_shape(0)))
        << "Shape " << orig_shape << " is not equal to " << f->get_output_partial_shape(0);
}

INSTANTIATE_TEST_SUITE_P(DecomposeTransconv,
                         DecomposeTransconvTest,
                         testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN},
                                                         WeightsShape{3, 8, 1, 2, 3}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, WeightsShape{3, 8, 1, 2, 3}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64, 64}, WeightsShape{3, 9, 2, 3, 1}),
                                         std::make_tuple(InputShape{3, 3, DYN, 64, 64}, WeightsShape{3, 6, 3, 4, 2}),
                                         std::make_tuple(InputShape{3, 3, 64, DYN, 64}, WeightsShape{3, 5, 3, 4, 3}),
                                         std::make_tuple(InputShape{3, 3, 64, 64, DYN}, WeightsShape{3, 3, 3, 4, 3}),
                                         std::make_tuple(InputShape{1, 3, 64, 64}, WeightsShape{3, 6, 1, 1}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, WeightsShape{3, 7, 1, 1}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64}, WeightsShape{3, 8, 1, 2}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64}, WeightsShape{3, 9, 2, 3}),
                                         std::make_tuple(InputShape{3, 3, DYN, 64}, WeightsShape{3, 6, 3, 4}),
                                         std::make_tuple(InputShape{3, 3, 64, DYN}, WeightsShape{3, 5, 3, 4}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN}, WeightsShape{3, 5, 1}),
                                         std::make_tuple(InputShape{DYN, 3, 10}, WeightsShape{3, 3, 1}),
                                         std::make_tuple(InputShape{2, DYN, 9}, WeightsShape{3, 2, 2}),
                                         std::make_tuple(InputShape{3, 3, DYN}, WeightsShape{3, 1, 3})));

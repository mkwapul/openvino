// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

// TODO: remove unused includes
// TODO: fix "" vs <>
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
#include "transformations/decompose_transconv.hpp"

using namespace testing;

using InputShape = ngraph::PartialShape;
using WeightsShape = ngraph::Shape;

using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using ov::op::v0::Result;
using ov::op::v1::ConvolutionBackpropData;
using ov::op::v1::Transpose;

using DecomposeTransconvParamsType = std::tuple<size_t,  // N
                                                size_t,  // H
                                                size_t,  // W
                                                size_t,  // C
                                                size_t,  // Cout
                                                size_t,  // Cin
                                                size_t,  // Kh
                                                size_t,  // Kw
                                                size_t,  // stride0
                                                size_t,  // stride1
                                                size_t,  // pads_begin0
                                                size_t,  // pads_begin1
                                                size_t,  // pads_end0
                                                size_t,  // pads_end1
                                                size_t,  // dilat0
                                                size_t   // dilat1
                                                >;

struct DecomposeTransconvParam {
    size_t N{};
    size_t H{};
    size_t W{};
    size_t C{};
    size_t Cout{};
    size_t Cin{};
    size_t Kh{};
    size_t Kw{};
    size_t stride0{};
    size_t stride1{};
    size_t pads_begin0{};
    size_t pads_begin1{};
    size_t pads_end0{};
    size_t pads_end1{};
    size_t dilat0{};
    size_t dilat1{};
};

static void setNodeNames(std::shared_ptr<ov::Node> node, std::string name) {
    std::string sname(name);
    node->set_friendly_name(sname);
    for (uint32_t i = 0; i < node->outputs().size(); i++) {
        node->get_output_tensor(i).set_names({sname + "_" + std::to_string(i)});
    }
}

static void fillRandom(std::vector<float>& data) {
    std::default_random_engine RNG;
    std::normal_distribution<float> gaussian(0.0, 1.0);
    for (uint32_t i = 0; i < data.size(); i++) {
        data[i] = gaussian(RNG);
    }
}

class DecomposeTransconvTest : public CommonTestUtils::TestsCommon,
                               public testing::WithParamInterface<DecomposeTransconvParamsType> {
public:
    std::shared_ptr<ov::Model> m, m_ref;

    void SetUp() override;

private:
    static std::shared_ptr<ov::Model> createInitialModel(DecomposeTransconvParam params);

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

void DecomposeTransconvTest::SetUp() {
    DecomposeTransconvParam param;
    std::tie(param.N,
             param.H,
             param.W,
             param.C,
             param.Cout,
             param.Cin,
             param.Kh,
             param.Kw,
             param.stride0,
             param.stride1,
             param.pads_begin0,
             param.pads_begin1,
             param.pads_end0,
             param.pads_end1,
             param.dilat0,
             param.dilat1) = GetParam();

    m = createInitialModel(param);

    // TODO, mkwap: create initial function
    // f_ref = get_reference_function(input_shape, weights_shape);
}

std::shared_ptr<ov::Model> DecomposeTransconvTest::createInitialModel(DecomposeTransconvParam param) {
    ov::OutputVector upstream;
    ov::Shape new_shape;

    std::shared_ptr<Parameter> input_2d =
        std::make_shared<Parameter>(ov::element::Type_t::f32,
                                    ov::Shape(std::vector<size_t>{{1, param.N * param.H * param.W * param.C}}));

    setNodeNames(input_2d, "input");
    upstream.push_back(input_2d->output(0));

    auto weight_shape = ov::Shape{param.Cin, param.Cout, param.Kh, param.Kw};
    auto strides = ov::Strides{param.stride0, param.stride1};
    auto pads_begin =
        ov::CoordinateDiff{static_cast<int64_t>(param.pads_begin0), static_cast<int64_t>(param.pads_begin1)};
    auto pads_end = ov::CoordinateDiff{static_cast<int64_t>(param.pads_end0), static_cast<int64_t>(param.pads_end1)};
    auto dilations = ov::Strides{param.dilat0, param.dilat1};
    auto auto_pad = ov::op::PadType::EXPLICIT;
    uint32_t out_pad_h = (param.H + pads_begin[0] + pads_end[0] - param.Kh) % strides[0];
    uint32_t out_pad_w = (param.W + pads_begin[1] + pads_end[1] - param.Kw) % strides[1];
    auto output_padding = ov::CoordinateDiff{out_pad_h, out_pad_w};

    if (upstream[0].get_shape() != ov::Shape{param.N, param.H, param.W, param.C}) {
        auto input_4d = std::make_shared<ngraph::opset1::Reshape>(
            upstream[0],
            Constant::create(ngraph::element::i64, ov::Shape{4}, {param.N, param.H, param.W, param.C})->output(0),
            false);
        setNodeNames(input_4d, "Reshape4D");
        upstream[0] = input_4d->output(0);
    }
    auto new_transpose =
        std::make_shared<Transpose>(upstream[0],
                                    Constant::create(ov::element::Type_t::i64, ov::Shape{4}, {0, 3, 1, 2}));
    setNodeNames(new_transpose, "Transpose");
    upstream[0] = new_transpose->output(0);

    std::vector<float> new_weights(param.Cout * param.Cin * param.Kh * param.Kw, 0.0f);
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
    new_transpose = std::make_shared<Transpose>(new_conv->output(0),
                                                Constant::create(ov::element::Type_t::i64, ov::Shape{4}, {0, 2, 3, 1}));
    setNodeNames(new_transpose, "UnTranspose");
    upstream[0] = new_transpose->output(0);
    new_shape = upstream[0].get_shape();

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
    std::shared_ptr<ov::Model> model =
        std::make_shared<ngraph::Function>(result_full, ngraph::ParameterVector{input_2d}, "test_graph");

    return model;
}

TEST_P(DecomposeTransconvTest, CompareFunctions) {
    const auto orig_shape = m->get_output_partial_shape(0);
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::TransposeConvolutionDecomposition>();
    manager.run_passes(m);
    ASSERT_NO_THROW(check_rt_info(m));
    auto res = compare_functions(m, m_ref);
    ASSERT_TRUE(res.first) << res.second;
    ASSERT_TRUE(orig_shape.same_scheme(m->get_output_partial_shape(0)))
        << "Shape " << orig_shape << " is not equal to " << m->get_output_partial_shape(0);
}

INSTANTIATE_TEST_SUITE_P(DecomposeTransconv,
                         DecomposeTransconvTest,
                         testing::Values(std::make_tuple(1, 7, 1, 1, 1, 1, 3, 1, 3, 1, 1, 0, 1, 0, 1, 1)));

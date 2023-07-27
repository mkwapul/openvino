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
#include <ngraph/opsets/opset11.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <tuple>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/opsets/opset11.hpp"
#include "transformations/decompose_transconv.hpp"

using namespace testing;

using InputShape = ngraph::PartialShape;
using WeightsShape = ngraph::Shape;

using ov::opset11::Constant;
using ov::opset11::ConvolutionBackpropData;
using ov::opset11::Parameter;
using ov::opset11::Result;
using ov::opset11::Slice;
using ov::opset11::Transpose;

// TODO: split onto strides, shapes etc
struct TransconvInitParam {
    TransconvInitParam(size_t N_,
                       size_t H_,
                       size_t W_,
                       size_t C_,
                       size_t Cout_,
                       size_t Cin_,
                       size_t Kh_,
                       size_t Kw_,
                       size_t stride0_,
                       size_t stride1_,
                       size_t pads_begin0_,
                       size_t pads_begin1_,
                       size_t pads_end0_,
                       size_t pads_end1_,
                       size_t dilat0_,
                       size_t dilat1_)
        : N(N_),
          H(H_),
          W(W_),
          C(C_),
          Cout(Cout_),
          Cin(Cin_),
          Kh(Kh_),
          Kw(Kw_),
          stride0(stride0_),
          stride1(stride1_),
          pads_begin0(pads_begin0_),
          pads_begin1(pads_begin1_),
          pads_end0(pads_end0_),
          pads_end1(pads_end1_),
          dilat0(dilat0_),
          dilat1(dilat1_) {}

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

using SliceLayerParam = std::pair<ngraph::Shape, ngraph::Shape>;  // start, stop
using SliceParam = std::tuple<SliceLayerParam,                    // Slice Layer param
                              ngraph::Shape,                      // Convolution Layer Kernel
                              std::size_t                         // out reshape channel numbers
                              >;

using TransconvRefParam = std::vector<SliceParam>;

using TransconvParam = std::pair<TransconvInitParam, TransconvRefParam>;

using ILS = typename std::initializer_list<std::size_t>;

class DecomposeTransconvTest : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<TransconvParam> {
public:
    std::shared_ptr<ov::Model> m, m_ref;

    void SetUp() override;

private:
    static std::shared_ptr<ov::Model> createInitialModel(const TransconvInitParam& param);
    static std::shared_ptr<ov::Model> createReferenceModel(const TransconvInitParam& param,
                                                           const TransconvRefParam& sliceInputs);
};

void DecomposeTransconvTest::SetUp() {
    TransconvParam param = GetParam();

    m = createInitialModel(param.first);

    m_ref = createReferenceModel(param.first, param.second);

    ov::serialize(m_ref, "DecomposeTransconvTestRerefence.xml", "DecomposeTransconvTestRerefence.bin");
}

std::shared_ptr<ov::Model> DecomposeTransconvTest::createInitialModel(const TransconvInitParam& param) {
    ov::OutputVector upstream;
    ov::Shape new_shape;

    std::shared_ptr<Parameter> input_2d =
        std::make_shared<Parameter>(ov::element::Type_t::f32,
                                    ov::Shape(std::vector<size_t>{{1, param.N * param.H * param.W * param.C}}));

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
        auto input_4d = std::make_shared<ngraph::opset11::Reshape>(
            upstream[0],
            Constant::create(ngraph::element::i64, ov::Shape{4}, {param.N, param.H, param.W, param.C})->output(0),
            false);
        upstream[0] = input_4d->output(0);
    }
    auto new_transpose =
        std::make_shared<Transpose>(upstream[0],
                                    Constant::create(ov::element::Type_t::i64, ov::Shape{4}, {0, 3, 1, 2}));
    upstream[0] = new_transpose->output(0);

    std::vector<float> new_weights(param.Cout * param.Cin * param.Kh * param.Kw, 0.0f);
    auto new_weights_const = Constant::create(ngraph::element::f32, weight_shape, new_weights);
    auto new_conv = std::make_shared<ConvolutionBackpropData>(upstream[0],
                                                              new_weights_const->output(0),
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations,
                                                              auto_pad,
                                                              output_padding);
    upstream[0] = new_conv->output(0);
    new_transpose = std::make_shared<Transpose>(new_conv->output(0),
                                                Constant::create(ov::element::Type_t::i64, ov::Shape{4}, {0, 2, 3, 1}));
    upstream[0] = new_transpose->output(0);
    new_shape = upstream[0].get_shape();

    size_t num_elements = 1;
    for (uint32_t i = 0; i < new_shape.size(); i++) {
        num_elements *= new_shape[i];
    }
    auto output_2d = std::make_shared<ngraph::opset11::Reshape>(
        upstream[0],
        Constant::create(ngraph::element::i64,
                         ov::Shape{2},
                         std::initializer_list<decltype(num_elements)>{1ull, num_elements})
            ->output(0),
        false);
    auto result_full = std::make_shared<Result>(output_2d->output(0));
    std::shared_ptr<ov::Model> model =
        std::make_shared<ngraph::Function>(result_full, ngraph::ParameterVector{input_2d}, "test_graph");

    return model;
}

std::shared_ptr<ov::Model> DecomposeTransconvTest::createReferenceModel(const TransconvInitParam& param,
                                                                        const TransconvRefParam& sliceInputs) {
    ov::Shape new_shape;

    std::shared_ptr<Parameter> input_2d =
        std::make_shared<Parameter>(ov::element::Type_t::f32,
                                    ov::Shape(std::vector<size_t>{{1, param.N * param.H * param.W * param.C}}));

    auto upstrm = input_2d->output(0);

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

    auto input_4d = std::make_shared<ngraph::opset11::Reshape>(
        upstrm,
        Constant::create(ngraph::element::i64, ov::Shape{4}, {param.N, param.H, param.W, param.C})->output(0),
        false);
    upstrm = input_4d->output(0);

    std::size_t hSumOut{};

    ov::OutputVector sliceOutputs;

    for (auto& slice : sliceInputs) {
        SliceLayerParam sliceParam;
        ngraph::Shape convKrnlShape;
        std::size_t sliceOutChannels;
        std::tie(sliceParam, convKrnlShape, sliceOutChannels) = slice;

        auto upstrm_inloop =
            std::make_shared<ngraph::opset11::Reshape>(
                upstrm,
                Constant::create(ngraph::element::i64, ngraph::Shape{2}, {param.N * param.H, param.W * param.C})
                    ->output(0),
                false)
                ->output(0);

        upstrm_inloop =
            std::make_shared<ngraph::opset11::Slice>(
                upstrm_inloop,
                Constant::create(ngraph::element::i64, ngraph::Shape{2}, sliceParam.first)->output(0),   // start
                Constant::create(ngraph::element::i64, ngraph::Shape{2}, sliceParam.second)->output(0),  // stop
                Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 1})->output(0))
                ->output(0);  // step

        upstrm_inloop =
            // TODO, mkwap: use makeReshape
            std::make_shared<ngraph::opset11::Reshape>(
                upstrm_inloop,
                Constant::create(ngraph::element::i64,
                                 ngraph::Shape{4},
                                 ILS{1,
                                     upstrm_inloop.get_shape()[0] /*sliceParam.second[0] - sliceParam.first[0]*/,
                                     1,
                                     upstrm_inloop.get_shape()[1]})
                    ->output(0),
                false)
                ->output(0);

        upstrm_inloop =
            // TODO, mkwap: use makeTranspose
            std::make_shared<Transpose>(upstrm_inloop,
                                        Constant::create(ov::element::Type_t::i64, ov::Shape{4}, {0, 3, 1, 2}))
                ->output(0);

        auto convInShape = upstrm_inloop.get_shape();
        auto convOutDims = ngraph::Shape{
            1,
            convKrnlShape[0],
            (convInShape[2] + 2 * 0 - convKrnlShape[2]) / 1 + 1,
            (convInShape[3] + 2 * 0 - convKrnlShape[3]) / 1 + 1,
        };
        upstrm_inloop = std::make_shared<ngraph::opset11::Convolution>(
            upstrm_inloop,
            Constant::create(
                ngraph::element::f32,
                convKrnlShape,
                // TODO make some multiplication algorithm
                std::vector<float>(convKrnlShape[0] * convKrnlShape[1] * convKrnlShape[2] * convKrnlShape[3], 0.0f))
                ->output(0),  // weights
            ngraph::Strides{1, 1},
            ngraph::CoordinateDiff{0, 0},  // pads_begin
            ngraph::CoordinateDiff{0, 0},  // pads_end
            ngraph::Strides{1, 1},         // dilations
            ov::op::PadType::VALID);

        upstrm_inloop =
            std::make_shared<Transpose>(upstrm_inloop,
                                        Constant::create(ov::element::Type_t::i64, ov::Shape{4}, {0, 2, 3, 1}))
                ->output(0);

        upstrm_inloop = std::make_shared<ngraph::opset11::Reshape>(
                            upstrm_inloop,
                            Constant::create(ngraph::element::i64,
                                             ngraph::Shape{4},
                                             ILS{1,
                                                 (convOutDims[0] * convOutDims[1] * convOutDims[2] * convOutDims[3]) /
                                                     sliceOutChannels,
                                                 1,
                                                 sliceOutChannels})
                                ->output(0),
                            false)
                            ->output(0);

        // TODO mkwap do std::accumulate + std::multiplies
        hSumOut += (convOutDims[0] * convOutDims[1] * convOutDims[2] * convOutDims[3]);

        sliceOutputs.push_back(upstrm_inloop);
    }

    if (sliceOutputs.size() > 0) {
        auto new_concat = std::make_shared<ngraph::opset11::Concat>(sliceOutputs, 1);
        upstrm = new_concat->output(0);
    }

    auto output = std::make_shared<ngraph::opset11::Reshape>(
        upstrm,
        // use summed output from concat instead of hSumOut!!!
        Constant::create(ngraph::element::i64, ov::Shape{2}, ILS{1, hSumOut})->output(0),
        false);

    upstrm = output->output(0);

    auto result_full = std::make_shared<Result>(upstrm);

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

    auto res = compare_functions(m, m_ref);
    ASSERT_TRUE(res.first) << res.second;
    ASSERT_TRUE(orig_shape.same_scheme(m->get_output_partial_shape(0)))
        << "Shape " << orig_shape << " is not equal to " << m->get_output_partial_shape(0);
}

INSTANTIATE_TEST_SUITE_P(
    DecomposeTransconv,
    DecomposeTransconvTest,
    testing::Values(
        // test7:
        std::make_pair(
            TransconvInitParam(1, 7, 1, 1, 1, 1, 3, 1, 3, 1, 1, 0, 1, 0, 1, 1),
            TransconvRefParam{
                {{SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{1, 1}), ngraph::Shape{2, 1, 1, 1}, 1},
                 {SliceLayerParam(ngraph::Shape{1, 0}, ngraph::Shape{6, 1}), ngraph::Shape{3, 1, 1, 1}, 1},
                 {SliceLayerParam(ngraph::Shape{6, 0}, ngraph::Shape{7, 1}), ngraph::Shape{2, 1, 1, 1}, 1}}}),
        // test8:
        std::make_pair(TransconvInitParam(1, 8, 1, 1, 1, 1, 3, 1, 3, 1, 1, 0, 1, 0, 1, 1),
                       TransconvRefParam{{
                           {SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{1, 1}), ngraph::Shape{2, 1, 1, 1}, 1},
                           {SliceLayerParam(ngraph::Shape{1, 0}, ngraph::Shape{8, 1}), ngraph::Shape{3, 1, 1, 1}, 1},
                       }}),
        // test8s:
        std::make_pair(
            TransconvInitParam(1, 8, 1, 1, 1, 1, 3, 1, 2, 1, 1, 0, 1, 0, 1, 1),
            TransconvRefParam{
                {{SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{1, 1}), ngraph::Shape{1, 1, 1, 1}, 1},
                 {SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{7, 1}), ngraph::Shape{2, 1, 2, 1}, 1},
                 {SliceLayerParam(ngraph::Shape{6, 0}, ngraph::Shape{8, 1}), ngraph::Shape{3, 1, 2, 1}, 1}}}),
        // test8-2:
        std::make_pair(TransconvInitParam(1, 8, 1, 1, 2, 1, 3, 1, 3, 1, 1, 0, 1, 0, 1, 1),
                       TransconvRefParam{{
                           {SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{1, 1}), ngraph::Shape{4, 1, 1, 1}, 2},
                           {SliceLayerParam(ngraph::Shape{1, 0}, ngraph::Shape{8, 1}), ngraph::Shape{6, 1, 1, 1}, 2},
                       }}),
        // test8-1-2:
        std::make_pair(TransconvInitParam(1, 8, 1, 2, 1, 2, 3, 1, 3, 1, 1, 0, 1, 0, 1, 1),
                       TransconvRefParam{{
                           {SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{1, 2}), ngraph::Shape{2, 2, 1, 1}, 1},
                           {SliceLayerParam(ngraph::Shape{1, 0}, ngraph::Shape{8, 2}), ngraph::Shape{3, 2, 1, 1}, 1},
                       }}),
        // test8-2-2:
        std::make_pair(TransconvInitParam(1, 8, 1, 2, 2, 2, 3, 1, 3, 1, 1, 0, 1, 0, 1, 1),
                       TransconvRefParam{{
                           {SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{1, 2}), ngraph::Shape{4, 2, 1, 1}, 2},
                           {SliceLayerParam(ngraph::Shape{1, 0}, ngraph::Shape{8, 2}), ngraph::Shape{6, 2, 1, 1}, 2},
                       }}),
        // test8-4-4:
        std::make_pair(TransconvInitParam(1, 8, 1, 4, 4, 4, 3, 1, 3, 1, 1, 0, 1, 0, 1, 1),
                       TransconvRefParam{{
                           {SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{1, 4}), ngraph::Shape{8, 4, 1, 1}, 4},
                           {SliceLayerParam(ngraph::Shape{1, 0}, ngraph::Shape{8, 4}), ngraph::Shape{12, 4, 1, 1}, 4},
                       }}),
        // test8-8-8:
        std::make_pair(TransconvInitParam(1, 8, 1, 8, 8, 8, 3, 1, 3, 1, 1, 0, 1, 0, 1, 1),
                       TransconvRefParam{{
                           {SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{1, 8}), ngraph::Shape{16, 8, 1, 1}, 8},
                           {SliceLayerParam(ngraph::Shape{1, 0}, ngraph::Shape{8, 8}), ngraph::Shape{24, 8, 1, 1}, 8},
                       }}),
        // test16:
        std::make_pair(TransconvInitParam(1, 16, 1, 1, 1, 1, 3, 1, 3, 1, 1, 0, 1, 0, 1, 1),
                       TransconvRefParam{{
                           {SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{1, 1}), ngraph::Shape{2, 1, 1, 1}, 1},
                           {SliceLayerParam(ngraph::Shape{1, 0}, ngraph::Shape{15, 1}), ngraph::Shape{3, 1, 1, 1}, 1},
                           {SliceLayerParam(ngraph::Shape{15, 0}, ngraph::Shape{16, 1}), ngraph::Shape{2, 1, 1, 1}, 1},
                       }}),
        // test16-2:
        std::make_pair(TransconvInitParam(1, 16, 1, 1, 2, 1, 3, 1, 3, 1, 1, 0, 1, 0, 1, 1),
                       TransconvRefParam{{
                           {SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{1, 1}), ngraph::Shape{4, 1, 1, 1}, 2},
                           {SliceLayerParam(ngraph::Shape{1, 0}, ngraph::Shape{15, 1}), ngraph::Shape{6, 1, 1, 1}, 2},
                           {SliceLayerParam(ngraph::Shape{15, 0}, ngraph::Shape{16, 1}), ngraph::Shape{4, 1, 1, 1}, 2},
                       }}),
        // test16-2-2:
        std::make_pair(TransconvInitParam(1, 16, 1, 2, 2, 2, 3, 1, 3, 1, 1, 0, 1, 0, 1, 1),
                       TransconvRefParam{{
                           {SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{1, 2}), ngraph::Shape{4, 2, 1, 1}, 2},
                           {SliceLayerParam(ngraph::Shape{1, 0}, ngraph::Shape{15, 2}), ngraph::Shape{6, 2, 1, 1}, 2},
                           {SliceLayerParam(ngraph::Shape{15, 0}, ngraph::Shape{16, 2}), ngraph::Shape{4, 2, 1, 1}, 2},
                       }}),
        // test19-64-64:
        std::make_pair(
            TransconvInitParam(1, 19, 1, 64, 64, 64, 3, 1, 3, 1, 1, 0, 1, 0, 1, 1),
            TransconvRefParam{{
                {SliceLayerParam(ngraph::Shape{0, 0}, ngraph::Shape{1, 64}), ngraph::Shape{128, 64, 1, 1}, 64},
                {SliceLayerParam(ngraph::Shape{1, 0}, ngraph::Shape{18, 64}), ngraph::Shape{192, 64, 1, 1}, 64},
                {SliceLayerParam(ngraph::Shape{18, 0}, ngraph::Shape{19, 64}), ngraph::Shape{128, 64, 1, 1}, 64},
            }})));

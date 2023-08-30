// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <tuple>
#include <vector>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "openvino/opsets/opset11.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {

using transConvParams = std::tuple<InferenceEngine::SizeVector,  // Kernel size
                                   InferenceEngine::SizeVector,  // Strides
                                   std::vector<ptrdiff_t>,       // Pad begin
                                   std::vector<ptrdiff_t>,       // Pad end
                                   InferenceEngine::SizeVector,  // Dilation
                                   size_t,                       // Num out channels
                                   ngraph::op::PadType           // Padding type
                                   >;

using transConvSubraphParams = std::tuple<transConvParams,                    // Convolution params
                                          InferenceEngine::Precision,         // Net precision
                                          InferenceEngine::SizeVector,        // Input shape
                                          std::map<std::string, std::string>  // Configuration
                                          >;

class DecomposeTransConvTest : public testing::WithParamInterface<transConvSubraphParams>,
                               virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<transConvSubraphParams>& obj);

protected:
    void SetUp() override;
    // TODO: remove
    // InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& inputInfo) const override;
};

std::string DecomposeTransConvTest::getTestCaseName(const testing::TestParamInfo<transConvSubraphParams>& obj) {
    std::ostringstream result;
    transConvParams convParams;
    InferenceEngine::Precision netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::SizeVector inputShape;
    std::map<std::string, std::string> configuration;
    std::tie(convParams, netPrecision, inputShape, configuration) = obj.param;

    InferenceEngine::SizeVector kernelSize, strides, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t numOutChannels;
    ngraph::op::PadType paddingType;
    std::tie(kernelSize, strides, padBegin, padEnd, dilation, numOutChannels, paddingType) = convParams;

    result << "netPRC=" << netPrecision.name() << "_";
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "K" << CommonTestUtils::vec2str(kernelSize) << "_";
    result << "S" << CommonTestUtils::vec2str(strides) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
    result << "O=" << numOutChannels << "_";
    result << "AP=" << paddingType << "_";
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }

    return result.str();
}

using namespace ov::opset11;

/* TODO: remove
InferenceEngine::Blob::Ptr DecomposeTransConvTest::GenerateInput(
    const InferenceEngine::InputInfo& info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 0, 1, 4);
}
*/

void DecomposeTransConvTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_GNA;
    std::map<std::string, std::string> tempConfig;
    transConvParams convParams;
    InferenceEngine::SizeVector inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;

    std::tie(convParams, netPrecision, inputShape, tempConfig) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    InferenceEngine::SizeVector kernelSize, strides, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t numOutChannels;
    ngraph::op::PadType paddingType;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params =
        ngraph::builder::makeParams(ngPrc, {{1, inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3]}});

    // TODO: based on src/plugins/intel_gna/tests/functional/pass_tests/broadcast_const_with_fq.cpp
    auto fakeQuantizeForReshape = ngraph::builder::makeFakeQuantize(params[0],
                                                                    ngPrc,
                                                                    65536,
                                                                    {},
                                                                    {-10.00030517578125},
                                                                    {10.0},
                                                                    {-10.00030517578125},
                                                                    {10.0});

    std::tie(kernelSize, strides, padBegin, padEnd, dilation, numOutChannels, paddingType) = convParams;

    auto reshape_before = std::make_shared<Reshape>(  // params[0],
        fakeQuantizeForReshape->output(0),
        Constant::create(ngraph::element::i64,
                         ov::Shape{4},
                         {inputShape[0], inputShape[2], inputShape[3], inputShape[1]})
            ->output(0),  // HCHW -> NHWC - TODO: do it more gently.
        false);

    auto transpose_before =
        std::make_shared<Transpose>(reshape_before->output(0),
                                    Constant::create(ov::element::Type_t::i64, ov::Shape{4}, {0, 3, 1, 2}))
            ->output(0);

    auto convbpd = ngraph::builder::makeConvolutionBackpropData(transpose_before,
                                                                ngraph::element::f32,
                                                                kernelSize,
                                                                strides,
                                                                padBegin,
                                                                padEnd,
                                                                dilation,
                                                                paddingType,
                                                                numOutChannels);

    auto fakeQuantizeAfterConvBPD = ngraph::builder::makeFakeQuantize(convbpd->output(0),
                                                                      ngPrc,
                                                                      65536,
                                                                      {},
                                                                      {-99.74967956542969},
                                                                      {99.74663543701172},
                                                                      {-99.74967956542969},
                                                                      {99.74663543701172});

    auto transpose_after = std::make_shared<Transpose>(  // convbpd->output(0),
        fakeQuantizeAfterConvBPD->output(0),
        Constant::create(ov::element::Type_t::i64, ov::Shape{4}, {0, 2, 3, 1}));
    //->output(0);

    auto new_shape = transpose_after->output(0).get_shape();
    std::size_t acc = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<std::size_t>());
    auto reshape_after = std::make_shared<Reshape>(transpose_after->output(0),
                                                   Constant::create(ngraph::element::i64,
                                                                    ov::Shape{2},
                                                                    std::vector<std::size_t>{{
                                                                        1,
                                                                        acc,
                                                                    }})
                                                       ->output(0),
                                                   false);

    // auto result = std::make_shared<Result>(reshape_after);

    ngraph::ResultVector result{std::make_shared<Result>(reshape_after)};
    function = std::make_shared<ngraph::Function>(result, params, "convolutionBackpropData");
}

TEST_P(DecomposeTransConvTest, CompareWithRefs) {
    Run();
    // TODO mkwap: what is it? should I use it?
    // CheckPluginRelatedResults(executableNetwork, pluginTypeNode);
};

// TODO, mkwap: pass exec target in more gentle way.
const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}}};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32/*,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32*/};

const ngraph::op::PadType paddingType{ngraph::op::PadType::EXPLICIT};
const size_t numOutChannels{1};

const InferenceEngine::SizeVector inputShape2D{1, 1, 7, 1};
const InferenceEngine::SizeVector kernelSize2D{3, 1};
const InferenceEngine::SizeVector strides2D{3, 1};
const std::vector<ptrdiff_t> padBegin2D{1, 0};
const std::vector<ptrdiff_t> padEnd2D{1, 0};
const InferenceEngine::SizeVector dilation2D{1, 1};
auto convParamsTest7 =
    transConvParams{kernelSize2D, strides2D, padBegin2D, padEnd2D, dilation2D, numOutChannels, paddingType};

const auto paramsTest7 = ::testing::Combine(::testing::Values(convParamsTest7),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(inputShape2D),
                                            ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeTransConvTest7,
                         DecomposeTransConvTest,
                         paramsTest7,
                         DecomposeTransConvTest::getTestCaseName);

auto convParamsTest8 = transConvParams{{3, 1}, {3, 1}, {1, 0}, {1, 0}, {1, 1}, {1}, paddingType};

const auto paramsTest8 = ::testing::Combine(::testing::Values(convParamsTest8),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::SizeVector{1, 1, 8, 1}),
                                            ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeTransConvTest8,
                         DecomposeTransConvTest,
                         paramsTest8,
                         DecomposeTransConvTest::getTestCaseName);

auto convParamsTest8s = transConvParams{{3, 1}, {2, 1}, {1, 0}, {1, 0}, {1, 1}, {1}, paddingType};

const auto paramsTest8s = ::testing::Combine(::testing::Values(convParamsTest8s),
                                             ::testing::ValuesIn(netPrecisions),
                                             ::testing::Values(InferenceEngine::SizeVector{1, 1, 8, 1}),
                                             ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeTransConvTest8s,
                         DecomposeTransConvTest,
                         paramsTest8s,
                         DecomposeTransConvTest::getTestCaseName);

auto convParamsTest8_2 = transConvParams{{3, 1}, {3, 1}, {1, 0}, {1, 0}, {1, 1}, {2}, paddingType};

const auto paramsTest8_2 = ::testing::Combine(::testing::Values(convParamsTest8_2),
                                              ::testing::ValuesIn(netPrecisions),
                                              ::testing::Values(InferenceEngine::SizeVector{1, 1, 8, 1}),
                                              ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeTransConvTest8_2,
                         DecomposeTransConvTest,
                         paramsTest8_2,
                         DecomposeTransConvTest::getTestCaseName);

auto convParamsTest8_1_2 = transConvParams{{3, 1}, {3, 1}, {1, 0}, {1, 0}, {1, 1}, {1}, paddingType};

const auto paramsTest8_1_2 = ::testing::Combine(::testing::Values(convParamsTest8_1_2),
                                                ::testing::ValuesIn(netPrecisions),
                                                ::testing::Values(InferenceEngine::SizeVector{1, 2, 8, 1}),
                                                ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeTransConvTest8_1_2,
                         DecomposeTransConvTest,
                         paramsTest8_1_2,
                         DecomposeTransConvTest::getTestCaseName);

auto convParamsTest8_2_2 = transConvParams{{3, 1}, {3, 1}, {1, 0}, {1, 0}, {1, 1}, {2}, paddingType};

const auto paramsTest8_2_2 = ::testing::Combine(::testing::Values(convParamsTest8_2_2),
                                                ::testing::ValuesIn(netPrecisions),
                                                ::testing::Values(InferenceEngine::SizeVector{1, 2, 8, 1}),
                                                ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeTransConvTest8_2_2,
                         DecomposeTransConvTest,
                         paramsTest8_2_2,
                         DecomposeTransConvTest::getTestCaseName);

auto convParamsTest8_4_4 = transConvParams{{3, 1}, {3, 1}, {1, 0}, {1, 0}, {1, 1}, {4}, paddingType};

const auto paramsTest8_4_4 = ::testing::Combine(::testing::Values(convParamsTest8_4_4),
                                                ::testing::ValuesIn(netPrecisions),
                                                ::testing::Values(InferenceEngine::SizeVector{1, 4, 8, 1}),
                                                ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeTransConvTest8_4_4,
                         DecomposeTransConvTest,
                         paramsTest8_4_4,
                         DecomposeTransConvTest::getTestCaseName);

auto convParamsTest8_8_8 = transConvParams{{3, 1}, {3, 1}, {1, 0}, {1, 0}, {1, 1}, {8}, paddingType};

const auto paramsTest8_8_8 = ::testing::Combine(::testing::Values(convParamsTest8_8_8),
                                                ::testing::ValuesIn(netPrecisions),
                                                ::testing::Values(InferenceEngine::SizeVector{1, 8, 8, 1}),
                                                ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeTransConvTest8_8_8,
                         DecomposeTransConvTest,
                         paramsTest8_8_8,
                         DecomposeTransConvTest::getTestCaseName);

auto convParamsTest16 = transConvParams{{3, 1}, {3, 1}, {1, 0}, {1, 0}, {1, 1}, {1}, paddingType};

const auto paramsTest16 = ::testing::Combine(::testing::Values(convParamsTest16),
                                             ::testing::ValuesIn(netPrecisions),
                                             ::testing::Values(InferenceEngine::SizeVector{1, 1, 16, 1}),
                                             ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeTransConvTest16,
                         DecomposeTransConvTest,
                         paramsTest16,
                         DecomposeTransConvTest::getTestCaseName);

auto convParamsTest16_2 = transConvParams{{3, 1}, {3, 1}, {1, 0}, {1, 0}, {1, 1}, {2}, paddingType};

const auto paramsTest16_2 = ::testing::Combine(::testing::Values(convParamsTest16_2),
                                               ::testing::ValuesIn(netPrecisions),
                                               ::testing::Values(InferenceEngine::SizeVector{1, 1, 16, 1}),
                                               ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeTransConvTest16_2,
                         DecomposeTransConvTest,
                         paramsTest16_2,
                         DecomposeTransConvTest::getTestCaseName);

auto convParamsTest16_2_2 = transConvParams{{3, 1}, {3, 1}, {1, 0}, {1, 0}, {1, 1}, {2}, paddingType};

const auto paramsTest16_2_2 = ::testing::Combine(::testing::Values(convParamsTest16_2_2),
                                                 ::testing::ValuesIn(netPrecisions),
                                                 ::testing::Values(InferenceEngine::SizeVector{1, 2, 16, 1}),
                                                 ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeTransConvTest16_2_2,
                         DecomposeTransConvTest,
                         paramsTest16_2_2,
                         DecomposeTransConvTest::getTestCaseName);

auto convParamsTest19_64_64 = transConvParams{{3, 1}, {3, 1}, {1, 0}, {1, 0}, {1, 1}, {64}, paddingType};

const auto paramsTest19_64_64 = ::testing::Combine(::testing::Values(convParamsTest19_64_64),
                                                   ::testing::ValuesIn(netPrecisions),
                                                   ::testing::Values(InferenceEngine::SizeVector{1, 64, 19, 1}),
                                                   ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_DecomposeTransConvTest19_64_64,
                         DecomposeTransConvTest,
                         paramsTest19_64_64,
                         DecomposeTransConvTest::getTestCaseName);

}  // namespace SubgraphTestsDefinitions

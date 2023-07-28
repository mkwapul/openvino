// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <tuple>
#include <vector>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {

using convBackPropParams = std::tuple<InferenceEngine::SizeVector,  // Kernel size
                                      InferenceEngine::SizeVector,  // Strides
                                      std::vector<ptrdiff_t>,       // Pad begin
                                      std::vector<ptrdiff_t>,       // Pad end
                                      InferenceEngine::SizeVector,  // Dilation
                                      size_t,                       // Num out channels
                                      ngraph::op::PadType           // Padding type
                                      >;

using convBackPropSubraphParams = std::tuple<convBackPropParams,                 // Convolution params
                                             InferenceEngine::Precision,         // Net precision
                                             InferenceEngine::SizeVector,        // Input shape
                                             std::map<std::string, std::string>  // Configuration
                                             >;

class ConvolutionBackpropSubgraphTest : public testing::WithParamInterface<convBackPropSubraphParams>,
                                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convBackPropSubraphParams>& obj);

protected:
    void SetUp() override;
    // TODO: remove
    // std::string pluginTypeNode;
};

std::string ConvolutionBackpropSubgraphTest::getTestCaseName(
    const testing::TestParamInfo<convBackPropSubraphParams>& obj) {
    std::ostringstream result;
    convBackPropParams convParams;
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

void ConvolutionBackpropSubgraphTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_GNA;
    std::map<std::string, std::string> tempConfig;
    convBackPropParams convParams;
    InferenceEngine::SizeVector inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;

    std::tie(convParams, netPrecision, inputShape, tempConfig) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    InferenceEngine::SizeVector kernelSize, strides, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t numOutChannels;
    ngraph::op::PadType paddingType;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    configuration.insert(tempConfig.begin(), tempConfig.end());

    std::tie(kernelSize, strides, padBegin, padEnd, dilation, numOutChannels, paddingType) = convParams;

    auto convbpd = ngraph::builder::makeConvolutionBackpropData(paramOuts[0],
                                                                ngraph::element::f32,
                                                                kernelSize,
                                                                strides,
                                                                padBegin,
                                                                padEnd,
                                                                dilation,
                                                                paddingType,
                                                                numOutChannels);

    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(convbpd)};
    function = std::make_shared<ngraph::Function>(results, params, "convolutionBackpropData");
}

TEST_P(ConvolutionBackpropSubgraphTest, CompareWithRefs) {
    Run();
    // TODO mkwap: what is it? should I use it?
    // CheckPluginRelatedResults(executableNetwork, pluginTypeNode);
};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
                                                                 {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32/*,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32*/};

const ngraph::op::PadType paddingType{ngraph::op::PadType::EXPLICIT};
const size_t numOutChannels{32};

const InferenceEngine::SizeVector inputShape2D{1, 64, 16, 16};
const InferenceEngine::SizeVector kernelSize2D{3, 3};
const InferenceEngine::SizeVector strides2D{2, 2};
const std::vector<ptrdiff_t> padBegin2D{1, 1};
const std::vector<ptrdiff_t> padEnd2D{1, 1};
const InferenceEngine::SizeVector dilation2D{1, 1};
auto convParams2D =
    convBackPropParams{kernelSize2D, strides2D, padBegin2D, padEnd2D, dilation2D, numOutChannels, paddingType};

const auto params2D = ::testing::Combine(::testing::Values(convParams2D),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(inputShape2D),
                                         ::testing::ValuesIn(configs));

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D,
                         ConvolutionBackpropSubgraphTest,
                         params2D,
                         ConvolutionBackpropSubgraphTest::getTestCaseName);

}  // namespace SubgraphTestsDefinitions

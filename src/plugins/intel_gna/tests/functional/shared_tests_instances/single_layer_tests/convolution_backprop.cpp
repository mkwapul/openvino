// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/convolution_backprop.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
                                                                 {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16,
                                                               InferenceEngine::Precision::I32};

const std::vector<size_t> numOutChannels = {16, 32};
const std::vector<size_t> numGroups = {2, 8, 16};
const std::vector<std::vector<size_t>> emptyOutputShape = {{}};
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

// ============= 1D GroupConvolution =============
const std::vector<std::vector<size_t>> inputShapes1D = {{1, 16, 32}};

const std::vector<std::vector<size_t>> kernels1D = {{1}, {3}};
const std::vector<std::vector<size_t>> strides1D = {{1}};
const std::vector<std::vector<ptrdiff_t>> padBegins1D = {{0}};
const std::vector<std::vector<ptrdiff_t>> padEnds1D = {{0}};
const std::vector<std::vector<size_t>> dilations1D = {{1}};

const auto groupConvBackpropData1DParams_ExplicitPadding =
    ::testing::Combine(::testing::ValuesIn(kernels1D),
                       ::testing::ValuesIn(strides1D),
                       ::testing::ValuesIn(padBegins1D),
                       ::testing::ValuesIn(padEnds1D),
                       ::testing::ValuesIn(dilations1D),
                       ::testing::ValuesIn(numOutChannels),
                       ::testing::Values(ngraph::op::PadType::EXPLICIT),
                       ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData1D_ExplicitPadding,
                         ConvolutionBackpropLayerTest,
                         ::testing::Combine(groupConvBackpropData1DParams_ExplicitPadding,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes1D),
                                            ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                         ConvolutionBackpropLayerTest::getTestCaseName);
/*
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropLayerTest,
                         ConvolutionBackpropLayerTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(params),
                                            ::testing::ValuesIn(inputChannels),
                                            ::testing::ValuesIn(outputChannels)),
                         ConcatConvTest::getTestCaseName);
*/
}  // namespace

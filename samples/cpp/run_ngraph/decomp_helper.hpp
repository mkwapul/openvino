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

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "ngraph/opsets/opset3.hpp"

using namespace ngraph;
using namespace op;

#define ADL_GNA_MAX_CHANNELS 96

size_t GetChannels(Output<Node>& parent);
std::shared_ptr<ov::Node> AdlChannelPadTensor(Output<Node> parent);
std::shared_ptr<ov::Node> AdlChannelPadKernel(Output<Node>& weights_const_output, size_t C);
std::shared_ptr<ov::Node> NchwToNhwc(Output<Node> parent);
std::shared_ptr<ov::Node> AdlInsertConvolutionAddRelu(Output<Node> parent,
                                                      std::shared_ptr<opset1::Convolution> conv,
                                                      std::shared_ptr<opset1::Add> add);
std::vector<std::shared_ptr<ov::Node>> AdlInsertSplitConvolutionAddRelu(Output<Node> parent,
                                                                        std::shared_ptr<opset1::Convolution> conv,
                                                                        std::shared_ptr<opset1::Add> add);
std::shared_ptr<ov::Node> AdlInsertConvolutionAddReluHpadCsplit(Output<Node> parent,
                                                                std::shared_ptr<opset1::Convolution> conv,
                                                                std::shared_ptr<opset1::Add> add);
std::shared_ptr<ov::Node> AdlInsertSplitConvolutionAddReluHpadCsplit(std::vector<std::shared_ptr<ov::Node>> parent,
                                                                     std::shared_ptr<opset1::Convolution> conv,
                                                                     std::shared_ptr<opset1::Add> add);
std::shared_ptr<ov::Node> AdlBigTranspose2d(Output<Node> parent);
void InsertActivation(OutputVector& upstream,
                      std::shared_ptr<ov::op::v0::PRelu> prelu,
                      std::shared_ptr<ov::op::v0::Relu> relu,
                      std::shared_ptr<ov::op::v0::Sigmoid> sigmoid,
                      std::shared_ptr<ov::op::v0::Tanh> tanh);

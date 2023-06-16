// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

#define OV_PP_TOSTRING_(...)  #__VA_ARGS__
#define OV_PP_TOSTRING(...)   OV_PP_TOSTRING_(__VA_ARGS__)
#define MATCHER_SCOPE(region) const std::string matcher_name(OV_PP_TOSTRING(region))

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Convert a depthwise separable convolution (represented by a GroupConvolution) to a set of ScaleShift layers
 * (MatMul + Add) Additionally supported are bias and fake quantize layers.
 */
class ConvertDWSCToScaleShifts : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertDWSCToScaleShifts", "0");
    ConvertDWSCToScaleShifts();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov

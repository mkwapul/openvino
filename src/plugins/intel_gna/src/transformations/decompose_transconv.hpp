// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeConvolutionDecomposition transformation breaks down 2d conv into set of 1d conv.
 */
class TransposeConvolutionDecomposition : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("TransposeConvolutionDecomposition", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov

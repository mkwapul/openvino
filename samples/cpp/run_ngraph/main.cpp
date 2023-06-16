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
#include <string>
#include <unordered_map>
#include <vector>

#include "concat_decomposition.hpp"
#include "convert_dwsc_to_scaleshifts.hpp"
#include "custom_to_mvn.hpp"
#include "groupconv1d_decomposition.hpp"
#include "groupconv_decomposition.hpp"
#include "ie_core.hpp"
#include "l2norm_decomposition.hpp"
#include "matmul_decomposition.hpp"
#include "mvn_decomposition.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/pass/serialize.hpp"
#include "openvino/openvino.hpp"
#include "padconv_decomposition.hpp"
#include "softmax_decomposition.hpp"
#include "to_mvn.hpp"
#include "to_nhwc.hpp"
#include "transconv_decomposition.hpp"
#include "transconv_nchw_decomposition.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "transpose_decomposition.hpp"

std::list<std::string> passes({"mvn",
                               "softmax",
                               "bigtranspose",
                               "bigmatmul",
                               "groupconv",
                               "groupconv1d",
                               "transposeconv",
                               "transposeconv_nchw",
                               "dwsc",
                               "lstmcell",
                               "to_mvn",
                               "to_mvn2",
                               "to_nhwc",
                               "l2norm",
                               "concat",
                               "padconv"});

using namespace InferenceEngine;
using namespace ngraph;
using namespace op;

void RegisterPass(ngraph::pass::Manager& manager, std::string name) {
    if (name == "mvn") {
        manager.register_pass<ngraph::pass::MvnDecomposition>();
    } else if (name == "softmax") {
        manager.register_pass<ngraph::pass::SoftmaxDecomposition>();
    } else if (name == "bigtranspose") {
        manager.register_pass<ngraph::pass::TransposeDecomposition>();
    } else if (name == "bigmatmul") {
        manager.register_pass<ngraph::pass::MatMulDecomposition>();
    } else if (name == "groupconv") {
        manager.register_pass<ngraph::pass::GroupConvolutionDecomposition>();
    } else if (name == "groupconv1d") {
        manager.register_pass<ngraph::pass::GroupConvolution1dDecomposition>();
    } else if (name == "transposeconv") {
        manager.register_pass<ngraph::pass::TransposeConvolutionDecomposition>();
    } else if (name == "transposeconv_nchw") {
        manager.register_pass<ngraph::pass::TransposeConvolutionNchwDecomposition>();
    } else if (name == "dwsc") {
        manager.register_pass<ov::intel_gna::pass::ConvertDWSCToScaleShifts>();
    } else if (name == "lstmcell") {
        manager.register_pass<ov::pass::LSTMCellDecomposition>();
    } else if (name == "to_mvn") {
        manager.register_pass<ngraph::pass::ToMvn>();
    } else if (name == "to_mvn2") {
        manager.register_pass<ngraph::pass::CustomToMvn>();
    } else if (name == "to_nhwc") {
        manager.register_pass<ngraph::pass::ToNHWC>();
    } else if (name == "l2norm") {
        manager.register_pass<ngraph::pass::L2NormDecomposition>();
    } else if (name == "concat") {
        manager.register_pass<ngraph::pass::ConcatDecomposition>();
    } else if (name == "padconv") {
        manager.register_pass<ngraph::pass::PadConvolutionDecomposition>();
    }
}

int main(int argc, char* argv[]) {
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    // --------------------------- 1. Load inference engine -------------------------------------
    std::cout << "Loading Inference Engine" << std::endl;

    if (argc < 2) {
        fprintf(stderr, "Missing required arguments\n");
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "\t%s xml_file_name pass1 <pass2 ...>\n", argv[0]);
        fprintf(stderr, "Supported passes:\n");
        for (std::string pass : passes) {
            fprintf(stderr, "\t%s\n", pass.c_str());
        }
        return (-1);
    }

    {
        std::ifstream f(argv[1]);
        if (!f.good()) {
            fprintf(stderr, "File \"%s\" not found!\n", argv[1]);
            exit(-1);
        }
    }

    //--------------------------- 2. Create network using ngraph function -----------------------------------
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    std::string filename = argv[1];
    size_t dot = filename.find_last_of(".");
    std::string name = filename.substr(0, dot);
    ngraph::pass::Manager manager;

    for (int32_t i = 2; i < argc; i++) {
        std::string arg(argv[i]);
        if (std::find(passes.begin(), passes.end(), arg) != passes.end()) {
            RegisterPass(manager, arg);
        } else {
            fprintf(stderr, "Unsupported pass:  %s\n", arg.c_str());
            return (-1);
        }
    }

    manager.register_pass<ngraph::pass::Serialize>(name + "_factorized.xml",
                                                   name + "_factorized.bin",
                                                   ngraph::pass::Serialize::Version::IR_V11);
    const auto& pass_config = manager.get_pass_config();
    manager.run_passes(model);
    return 0;
}

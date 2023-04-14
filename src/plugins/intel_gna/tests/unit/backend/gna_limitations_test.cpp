// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend/gna_limitations.hpp"
#include "common/gna_target.hpp"

#include <gtest/gtest.h>

#include <utility>

using namespace ov::intel_gna::limitations;
using ov::intel_gna::common::kGnaTarget3_0;
using ov::intel_gna::common::kGnaTarget3_5;
using ov::intel_gna::common::kGnaTarget3_6;

struct GNACnn2DValidatorTestParam {
    std::string target;
    std::string whatInvalid;
    std::vector<uint32_t> invalid;
};

const std::vector<uint32_t> kInvalidH_30 = {0, 400};
const std::vector<uint32_t> kInvalidH_35 = {0, 65536};
const std::vector<uint32_t> kInvalidH_36 = {0, 65536};

const std::vector<uint32_t> kInvalidW_30 = {0, 400};
const std::vector<uint32_t> kInvalidW_35 = {0, 65536};
const std::vector<uint32_t> kInvalidW_36 = {0, 65536};

const std::vector<uint32_t> kInvalidC_30 = {0, 1, 400};
const std::vector<uint32_t> kInvalidC_35 = {0, 2049};
const std::vector<uint32_t> kInvalidC_36 = {7, 1023};

const std::vector<uint32_t> kInvalidkH_30 = {0, 8, 400};
const std::vector<uint32_t> kInvalidkH_35 = {0, 257, 2049};
const std::vector<uint32_t> kInvalidkH_36 = {0, 256};

const std::vector<uint32_t> kInvalidkW_30 = {0, 8, 400};
const std::vector<uint32_t> kInvalidkW_35 = {0, 257, 2049};
const std::vector<uint32_t> kInvalidkW_36 = {0, 256};

const std::vector<uint32_t> kInvalidkN_30 = {0, 1, 400};
const std::vector<uint32_t> kInvalidkN_35 = {0, 2049};
const std::vector<uint32_t> kInvalidkN_36 = {0, 2};

const std::vector<uint32_t> kInvalidsH_30 = {0, 400};
const std::vector<uint32_t> kInvalidsH_35 = {0, 2049};
const std::vector<uint32_t> kInvalidsH_36 = {0, 256};

const std::vector<uint32_t> kInvalidsW_30 = {0, 400};
const std::vector<uint32_t> kInvalidsW_35 = {0, 2049};
const std::vector<uint32_t> kInvalidsW_36 = {0, 256};

const std::vector<uint32_t> kInvaliddH_30 = {0, 2, 400};
const std::vector<uint32_t> kInvaliddH_35 = {0, 2, 2049};
const std::vector<uint32_t> kInvaliddH_36 = {0, 2};

const std::vector<uint32_t> kInvaliddW_30 = {0, 2, 400};
const std::vector<uint32_t> kInvaliddW_35 = {0, 2, 2049};
const std::vector<uint32_t> kInvaliddW_36 = {0, 2};

const GNACnn2DValidatorTestParam target_30 {
    kGnaTarget3_0,
    "inH",
    kInvalidH_30,
};

const GNACnn2DValidatorTestParam target_35 {
    kGnaTarget3_5,
    "inH",
    kInvalidH_35,
};

const GNACnn2DValidatorTestParam target_36{
    kGnaTarget3_6,
    "inH",
    kInvalidH_36,
};

const GNACnn2DValidatorTestParam target_30_inW{
    kGnaTarget3_0,
    "inW",
    kInvalidW_30,
};

const GNACnn2DValidatorTestParam target_35_inW{
    kGnaTarget3_5,
    "inW",
    kInvalidW_35,
};

const GNACnn2DValidatorTestParam target_36_inW{
    kGnaTarget3_6,
    "inW",
    kInvalidW_36,
};

const GNACnn2DValidatorTestParam target_30_inC{
    kGnaTarget3_0,
    "inC",
    kInvalidC_30,
};

const GNACnn2DValidatorTestParam target_35_inC{
    kGnaTarget3_5,
    "inC",
    kInvalidC_35,
};

const GNACnn2DValidatorTestParam target_36_inC{
    kGnaTarget3_6,
    "inC",
    kInvalidC_36,
};

const GNACnn2DValidatorTestParam target_30_kH{
    kGnaTarget3_0,
    "kH",
    kInvalidkH_30,
};

const GNACnn2DValidatorTestParam target_35_kH{
    kGnaTarget3_5,
    "kH",
    kInvalidkH_35,
};

const GNACnn2DValidatorTestParam target_36_kH{
    kGnaTarget3_6,
    "kH",
    kInvalidkH_36,
};

const GNACnn2DValidatorTestParam target_30_kW{
    kGnaTarget3_0,
    "kW",
    kInvalidkW_30,
};

const GNACnn2DValidatorTestParam target_35_kW{
    kGnaTarget3_5,
    "kW",
    kInvalidkW_35,
};

const GNACnn2DValidatorTestParam target_36_kW{
    kGnaTarget3_6,
    "kW",
    kInvalidkW_36,
};

const GNACnn2DValidatorTestParam target_30_kN{
    kGnaTarget3_0,
    "inC",
    kInvalidkN_30,
};

const GNACnn2DValidatorTestParam target_35_kN{
    kGnaTarget3_5,
    "inC",
    kInvalidkN_35,
};

const GNACnn2DValidatorTestParam target_36_kN{
    kGnaTarget3_6,
    "inC",
    kInvalidkN_36,
};

const GNACnn2DValidatorTestParam target_30_sH{
    kGnaTarget3_0,
    "sH",
    kInvalidsH_30,
};

const GNACnn2DValidatorTestParam target_35_sH{
    kGnaTarget3_5,
    "sH",
    kInvalidsH_35,
};

const GNACnn2DValidatorTestParam target_36_sH{
    kGnaTarget3_6,
    "sH",
    kInvalidsH_36,
};

const GNACnn2DValidatorTestParam target_30_sW{
    kGnaTarget3_0,
    "sW",
    kInvalidsW_30,
};

const GNACnn2DValidatorTestParam target_35_sW{
    kGnaTarget3_5,
    "sW",
    kInvalidsW_35,
};

const GNACnn2DValidatorTestParam target_36_sW{
    kGnaTarget3_6,
    "sW",
    kInvalidsW_36
};

const GNACnn2DValidatorTestParam target_30_dH{
    kGnaTarget3_0,
    "dH",
    kInvaliddH_30,
};

const GNACnn2DValidatorTestParam target_35_dH{
    kGnaTarget3_5,
    "dH",
    kInvaliddH_35,
};

const GNACnn2DValidatorTestParam target_36_dH{
    kGnaTarget3_6,
    "dH",
    kInvaliddH_36,
};

const GNACnn2DValidatorTestParam target_30_dW{
    kGnaTarget3_0,
    "dW",
    kInvaliddW_30,
};

const GNACnn2DValidatorTestParam target_35_dW{
    kGnaTarget3_5,
    "dW",
    kInvaliddW_35,
};

const GNACnn2DValidatorTestParam target_36_dW{
    kGnaTarget3_6,
    "dW",
    kInvaliddW_36,
};

const std::vector<uint32_t> kInvalidpw_30 = {0, 2, 10};
const GNACnn2DValidatorTestParam target_30_pwH{
    kGnaTarget3_0,
    "windowH",
    kInvalidpw_30,
};
const GNACnn2DValidatorTestParam target_30_pwW{
    kGnaTarget3_0,
    "windowW",
    kInvalidpw_30,
};

const std::vector<uint32_t> kInvalidps_30 = {0, 4, 10};
const GNACnn2DValidatorTestParam target_30_psH{
    kGnaTarget3_0,
    "strideH",
    kInvalidps_30,
};
const GNACnn2DValidatorTestParam target_30_psW{
    kGnaTarget3_0,
    "strideW",
    kInvalidps_30,
};

const std::vector<uint32_t> kInvalidPoolingRange35 = {0, 256};
const GNACnn2DValidatorTestParam target_35_pwH{
    kGnaTarget3_5,
    "windowH",
    kInvalidPoolingRange35,
};
const GNACnn2DValidatorTestParam target_35_pwW{
    kGnaTarget3_5,
    "windowW",
    kInvalidPoolingRange35,
};
const GNACnn2DValidatorTestParam target_35_psH{
    kGnaTarget3_5,
    "strideH",
    kInvalidPoolingRange35,
};
const GNACnn2DValidatorTestParam target_35_psW{
    kGnaTarget3_5,
    "strideW",
    kInvalidPoolingRange35,
};

const std::vector<uint32_t> kInvalidPoolingRange36 = {0, 256};
const GNACnn2DValidatorTestParam target_36_pwH{
    kGnaTarget3_6,
    "windowH",
    kInvalidPoolingRange36,
};
const GNACnn2DValidatorTestParam target_36_pwW{
    kGnaTarget3_6,
    "windowW",
    kInvalidPoolingRange36,
};
const GNACnn2DValidatorTestParam target_36_psH{
    kGnaTarget3_6,
    "strideH",
    kInvalidPoolingRange36,
};
const GNACnn2DValidatorTestParam target_36_psW{
    kGnaTarget3_6,
    "strideW",
    kInvalidPoolingRange36,
};

struct ValidateCnn2DParams {
    std::map<std::string, uint32_t> parameters;
    OvGnaType precision;
    static const bool exceptionMode = false;

    static ValidateCnn2DParams GetValid() {
        ValidateCnn2DParams v;
        v.parameters["inH"] = 16;
        v.parameters["inW"] = 16;
        v.parameters["inC"] = 16;
        v.parameters["kH"] = 2;
        v.parameters["kW"] = 2;
        v.parameters["kN"] = 8;
        v.parameters["sH"] = 1;
        v.parameters["sW"] = 1;
        v.parameters["dH"] = 1;
        v.parameters["dW"] = 1;
        v.precision = OvGnaTypeInt16;
        return v;
    }

    static ValidateCnn2DParams GetValidPooling() {
        ValidateCnn2DParams v;
        v.parameters["windowH"] = 3;
        v.parameters["windowW"] = 3;
        v.parameters["strideH"] = 3;
        v.parameters["strideW"] = 3;
        return v;
    }

    static ValidateCnn2DParams GetValidDwsc() {
        ValidateCnn2DParams v;
        v.parameters["inH"] = 16;
        v.parameters["inW"] = 16;
        v.parameters["inC"] = 16;
        v.parameters["kH"] = 2;
        v.parameters["kW"] = 2;
        v.parameters["kN"] = 8;
        v.parameters["sH"] = 1;
        v.parameters["sW"] = 1;
        v.parameters["dH"] = 1;
        v.parameters["dW"] = 1;
        return v;
    }

    bool ValidateCnn2D(const cnn2d::AbstractValidator& validator) {
        return validator.ValidateCnn2D({},
                                       parameters["inH"],
                                       parameters["inW"],
                                       parameters["inC"],
                                       parameters["kH"],
                                       parameters["kW"],
                                       parameters["kN"],
                                       parameters["sH"],
                                       parameters["sW"],
                                       parameters["dH"],
                                       parameters["dW"],
                                       precision,
                                       exceptionMode);
    }

    bool ValidatePooling2D(const cnn2d::AbstractValidator& validator) {
        return validator.ValidatePooling2D({},
                                           parameters["windowH"],
                                           parameters["windowW"],
                                           parameters["strideH"],
                                           parameters["strideW"],
                                           exceptionMode);
    }

    bool ValidateDwsc(const cnn2d::AbstractValidator& validator) {
        return validator.ValidateDwsc({},
                                      parameters["inH"],
                                      parameters["inW"],
                                      parameters["inC"],
                                      parameters["kH"],
                                      parameters["kW"],
                                      parameters["kN"],
                                      parameters["sH"],
                                      parameters["sW"],
                                      parameters["dH"],
                                      parameters["dW"],
                                      exceptionMode);
    }

    void set(const std::string& what, const uint32_t value) {
        if (what == "precision") {
            precision = static_cast<OvGnaType>(value);
        } else {
            parameters[what] = value;
        }
    }
};

class GNACnn2DValidatorTest : public ::testing::TestWithParam<GNACnn2DValidatorTestParam> {
protected:
    void SetUp() override  {
        validator = cnn2d::AbstractValidator::Create(ov::intel_gna::common::StringToDevice(GetParam().target));
        ASSERT_TRUE(validator != nullptr);
    }
    std::unique_ptr<cnn2d::AbstractValidator> validator;
};

class GNACnn2DValidatorTestPadding : public GNACnn2DValidatorTest {
protected:
    bool isPaddingSupported() {
        static const std::set<std::string> supported{kGnaTarget3_5, kGnaTarget3_6};
        return supported.count(GetParam().target);
    }
};

class GNACnn2DValidatorTestPooling2D : public GNACnn2DValidatorTest {};

class GNADwscValidatorTest : public GNACnn2DValidatorTest {};

namespace {
TEST_P(GNACnn2DValidatorTestPadding, testPaddingSupported) {
    ASSERT_TRUE(validator->ValidateInputPadding("", 1, 1, 1, 1, 2, 2, false) == isPaddingSupported());
}

TEST_P(GNACnn2DValidatorTest, testValidateCnn2DInvalid) {
    auto valid = ValidateCnn2DParams::GetValid();
    for (const auto invalid : GetParam().invalid) {
        valid.set(GetParam().whatInvalid, invalid);
        ASSERT_FALSE(valid.ValidateCnn2D(*validator));
    }
}

TEST_P(GNACnn2DValidatorTestPooling2D, testValidateCnn2DInvalid) {
    auto valid = ValidateCnn2DParams::GetValidPooling();
    for (const auto invalid : GetParam().invalid) {
        valid.set(GetParam().whatInvalid, invalid);
        ASSERT_FALSE(valid.ValidatePooling2D(*validator));
    }
}

TEST_P(GNADwscValidatorTest, testValidateDwscInvalid) {
    auto valid = ValidateCnn2DParams::GetValidDwsc();
    for (const auto invalid : GetParam().invalid) {
        valid.set(GetParam().whatInvalid, invalid);
        ASSERT_FALSE(valid.ValidateDwsc(*validator));
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_GNACnn2DValidatorTestPadding,
                         GNACnn2DValidatorTestPadding,
                         testing::Values(target_30, target_35, target_36));

INSTANTIATE_TEST_SUITE_P(smoke_GNACnn2DValidatorTest,
                         GNACnn2DValidatorTest,
                         testing::Values(target_30,
                                         target_35,
                                         target_30_inW,
                                         target_35_inW,
                                         target_30_inC,
                                         target_35_inC,
                                         target_30_kH,
                                         target_35_kH,
                                         target_30_kW,
                                         target_35_kW,
                                         target_30_kN,
                                         target_35_kN,
                                         target_30_sH,
                                         target_30_sW,
                                         target_30_dH,
                                         target_30_dW,
                                         target_35_sH,
                                         target_35_sW,
                                         target_35_dH,
                                         target_35_dW)
);

INSTANTIATE_TEST_SUITE_P(smoke_GNACnn2DValidatorTestPooling2D,
                         GNACnn2DValidatorTestPooling2D,
                         testing::Values(target_30_pwH,
                                         target_30_pwW,
                                         target_30_psH,
                                         target_30_psW,
                                         target_35_pwH,
                                         target_35_pwW,
                                         target_35_psH,
                                         target_35_psW,
                                         target_36_pwH,
                                         target_36_pwW,
                                         target_36_psH,
                                         target_36_psW));

INSTANTIATE_TEST_SUITE_P(smoke_GNADwscValidatorTest,
                         GNADwscValidatorTest,
                         testing::Values(target_36,
                                         target_36_inW,
                                         target_36_inC,
                                         target_36_kH,
                                         target_36_kW,
                                         target_36_kN,
                                         target_36_sH,
                                         target_36_sW,
                                         target_36_dH,
                                         target_36_dW));

}  // namespace

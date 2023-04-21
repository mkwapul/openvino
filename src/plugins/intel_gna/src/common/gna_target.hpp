// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "gna2-common-api.h"
#include "openvino/runtime/intel_gna/properties.hpp"

namespace ov {
namespace intel_gna {
namespace common {

enum DeviceVersion {
    DeviceVersionNotSet = -1,
    DeviceVersionSoftwareEmulation = 0,
    DeviceVersionEmbedded1_0 = 0x10e,
    DeviceVersion2_0 = 0x20,
    DeviceVersion3_0 = 0x30,
    DeviceVersionEmbedded3_1 = 0x31e,
    DeviceVersion3_5 = 0x35,
    DeviceVersionEmbedded3_5 = 0x35e,
    DeviceVersionEmbedded3_6 = 0x36e,
    DeviceVersionEmbedded4_0 = 0x40e,
    DeviceVersionEmbedded4_0_8CE = 0x40E8,
    DeviceVersionDefault = DeviceVersionEmbedded4_0_8CE
};

class Target {
    DeviceVersion get_default_target() const;

public:
    DeviceVersion detected_device_version = DeviceVersionSoftwareEmulation;
    DeviceVersion user_set_execution_target = DeviceVersionDefault;
    DeviceVersion user_set_compile_target = DeviceVersionDefault;

    DeviceVersion get_effective_execution_target() const;
    DeviceVersion get_effective_compile_target(const bool device_helper) const;
};

static constexpr const char* kGnaTargetUnspecified = "";
static constexpr const char* kGnaTarget1_0 = "GNA_TARGET_1_0";
static constexpr const char* kGnaTarget2_0 = "GNA_TARGET_2_0";
static constexpr const char* kGnaTarget3_0 = "GNA_TARGET_3_0";
static constexpr const char* kGnaTarget3_1 = "GNA_TARGET_3_1";
static constexpr const char* kGnaTarget3_5 = "GNA_TARGET_3_5";
static constexpr const char* kGnaTarget3_5_e = "GNA_TARGET_3_5_E";
static constexpr const char* kGnaTarget3_6 = "GNA_TARGET_3_6";
static constexpr const char* kGnaTarget4_0 = "GNA_TARGET_4_0";

DeviceVersion HwGenerationToDevice(const HWGeneration& target);
HWGeneration DeviceToHwGeneration(const DeviceVersion& target);
DeviceVersion StringToDevice(const std::string& target);
bool IsEmbeddedDevice(const DeviceVersion& target);
DeviceVersion get_tlv_target_from_compile_target(const DeviceVersion& compileTarget);

}  // namespace common
}  // namespace intel_gna
}  // namespace ov

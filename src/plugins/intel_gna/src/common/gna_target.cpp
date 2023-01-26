// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_target.hpp"
#include "log/debug.hpp"

#include <map>
#include <set>

namespace ov {
namespace intel_gna {
namespace common {

static const std::map<std::string, DeviceVersion> StringDeviceMap{{kGnaTarget1_0, DeviceVersionEmbedded1_0},
                                                                  {kGnaTarget2_0, DeviceVersion2_0},
                                                                  {kGnaTarget3_0, DeviceVersion3_0},
                                                                  {kGnaTarget3_1, DeviceVersionEmbedded3_1},
                                                                  {kGnaTarget3_5, DeviceVersion3_5},
                                                                  {kGnaTarget3_6, DeviceVersionEmbedded3_6},
                                                                  {kGnaTarget4_0, DeviceVersionEmbedded4_0},
                                                                  {kGnaTargetUnspecified, DeviceVersionNotSet}};

static const std::map<HWGeneration, DeviceVersion> HWGenerationDeviceMap{
    {HWGeneration::GNA_2_0, DeviceVersion2_0},
    {HWGeneration::GNA_3_0, DeviceVersion3_0},
    {HWGeneration::GNA_3_5, DeviceVersion3_5},
    {HWGeneration::GNA_3_6, DeviceVersionEmbedded3_6},
    {HWGeneration::GNA_4_0, DeviceVersionEmbedded4_0},
    {HWGeneration::UNDEFINED, DeviceVersionNotSet}};

static const std::vector<DeviceVersion> EmbeddedDevices{DeviceVersionEmbedded1_0,
                                                        DeviceVersionEmbedded3_1,
                                                        DeviceVersionEmbedded3_5,
                                                        DeviceVersionEmbedded3_6,
                                                        DeviceVersionEmbedded4_0};

// TODO make sure the message is printed correctly
DeviceVersion HwGenerationToDevice(const HWGeneration& target) {
    const auto f = HWGenerationDeviceMap.find(target);
    if (f != HWGenerationDeviceMap.end()) {
        return f->second;
    }
    THROW_GNA_EXCEPTION << "Unsupported GNA compile/exec target value: " << target;
}
// TODO make sure the message is printed correctly
HWGeneration DeviceToHwGeneration(const DeviceVersion& target) {
    for (auto it = HWGenerationDeviceMap.begin(); it != HWGenerationDeviceMap.end(); ++it) {
        if (it->second == target) {
            return it->first;
        }
    }
    THROW_GNA_EXCEPTION << "Unsupported GNA target value: " << target;
}
// TODO make sure the message is printed correctly
DeviceVersion StringToDevice(const std::string& target) {
    const auto f = StringDeviceMap.find(target);
    if (f != StringDeviceMap.end()) {
        return f->second;
    }
    THROW_GNA_EXCEPTION << "Unsupported GNA target value: " << target;
}

bool IsEmbeddedDevice(const DeviceVersion& target) {
    return std::find(EmbeddedDevices.begin(), EmbeddedDevices.end(), target) != EmbeddedDevices.end();
}

DeviceVersion get_tlv_target_from_compile_target(const DeviceVersion& compile_target) {
    // TODO verify if this is correct, but 1.0 is probably treated as a wrong device
    if (compile_target == DeviceVersionEmbedded1_0) {
        THROW_GNA_EXCEPTION << "Unsupported compile target for TLV export: " << compile_target << "\n";
    }
    return compile_target;
}

DeviceVersion Target::get_default_target() const {
    if (detected_device_version == DeviceVersionSoftwareEmulation) {
        return DeviceVersionDefault;
    }
    return detected_device_version;
}

DeviceVersion Target::get_effective_execution_target() const {
    if (user_set_execution_target == DeviceVersionNotSet) {
        return get_default_target();
    }
    return user_set_execution_target;
}

DeviceVersion Target::get_effective_compile_target(const bool device_helper) const {
    if (user_set_compile_target != DeviceVersionNotSet) {
        return user_set_compile_target;
    } else if (device_helper) {
        return get_effective_execution_target();
    }
    return DeviceVersionDefault;
}

}  // namespace common
}  // namespace intel_gna
}  // namespace ov

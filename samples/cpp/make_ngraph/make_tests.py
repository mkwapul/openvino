#*****************************************************************************
# 
# INTEL CONFIDENTIAL
# Copyright 2018-2023 Intel Corporation
# 
# The source code contained or described herein and all documents related 
# to the source code ("Material") are owned by Intel Corporation or its suppliers 
# or licensors. Title to the Material remains with Intel Corporation or its suppliers 
# and licensors. The Material contains trade secrets and proprietary 
# and confidential information of Intel or its suppliers and licensors.
# The Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified, 
# published, uploaded, posted, transmitted, distributed, or disclosed in any way 
# without Intel's prior express written permission.
# 
# No license under any patent, copyright, trade secret or other intellectual 
# property right is granted to or conferred upon you by disclosure or delivery 
# of the Materials, either expressly, by implication, inducement, estoppel 
# or otherwise. Any license under such intellectual property rights must 
# be express and approved by Intel in writing.
#*****************************************************************************
# make_tests.py : create tests using make_ngraph and generate_ark
#

run_ld_test = True  # study speed versus number of layer descriptors

# test to see impact of layer descriptor loading
if (run_ld_test):
    num_tests = 7
    test_size = 64
    H = 1
    W = 64
    for i in range(num_tests):
        print(".\\Debug\\make_ngraph.exe test"+str(i+1)+" \'Parameter("+str(H)+","+str(W)+")\'", end=" ")
        for j in range(int(test_size)):
            print("\'Multiply("+str(H)+","+str(W)+")\'", end=" ")
        test_size = test_size / 2
        W = 2 * W
        print("")

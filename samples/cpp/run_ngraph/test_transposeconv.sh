#!/bin/bash

#OV_ROOT=${USER}/git/openvino
#export PATH=$OV_ROOT/bin/intel64/Debug/:$PATH

function generate_arks() {
    echo -e "\n==== [`date`] ${FUNCNAME[0]} START"
    python generate_ark.py test7_input.ark 7
    python generate_ark.py test8_input.ark 8
    python generate_ark.py test16_input.ark 16
    python generate_ark.py test32_input.ark 32
    python generate_ark.py test64_input.ark 64
    python generate_ark.py convtrans_input.ark 1216
    echo "==== [`date`] ${FUNCNAME[0]} END"
}

function make_models() {
    echo -e "\n==== [`date`] ${FUNCNAME[0]} START"
    make_ngraph test7 "Parameter(1,7,1,1)" "ConvolutionBackpropData(1,7,1,1,1,1,3,1,3,1,1,0,1,0,1,1)"
    make_ngraph test8 "Parameter(1,8,1,1)" "ConvolutionBackpropData(1,8,1,1,1,1,3,1,3,1,1,0,1,0,1,1)"
    make_ngraph test8s "Parameter(1,8,1,1)" "ConvolutionBackpropData(1,8,1,1,1,1,3,1,2,1,1,0,1,0,1,1)"
    make_ngraph test8-2 "Parameter(1,8,1,1)" "ConvolutionBackpropData(1,8,1,1,2,1,3,1,3,1,1,0,1,0,1,1)"
    make_ngraph test8-1-2 "Parameter(1,8,1,2)" "ConvolutionBackpropData(1,8,1,2,1,2,3,1,3,1,1,0,1,0,1,1)"
    make_ngraph test8-2-2 "Parameter(1,8,1,2)" "ConvolutionBackpropData(1,8,1,2,2,2,3,1,3,1,1,0,1,0,1,1)"
    #make_ngraph test8-2-2-bias "Parameter(1,8,1,2)" "Transpose(1,8,1,2,0,3,1,2)" "ConvolutionBackpropDataBare(1,2,8,1,2,2,3,1,3,1,1,0,1,0,1,1)" "AddBias(1,2,23,1)" "Sigmoid()" "Transpose(1,2,23,1,0,2,3,1)"
    make_ngraph test8-4-4 "Parameter(1,8,1,4)" "ConvolutionBackpropData(1,8,1,4,4,4,3,1,3,1,1,0,1,0,1,1)"
    make_ngraph test8-8-8 "Parameter(1,8,1,8)" "ConvolutionBackpropData(1,8,1,8,8,8,3,1,3,1,1,0,1,0,1,1)"
    make_ngraph test16 "Parameter(1,16,1,1)" "ConvolutionBackpropData(1,16,1,1,1,1,3,1,3,1,1,0,1,0,1,1)"
    make_ngraph test16-2 "Parameter(1,16,1,1)" "ConvolutionBackpropData(1,16,1,1,2,1,3,1,3,1,1,0,1,0,1,1)"
    make_ngraph test16-2-2 "Parameter(1,16,1,2)" "ConvolutionBackpropData(1,16,1,2,2,2,3,1,3,1,1,0,1,0,1,1)"
    make_ngraph test19-64-64 "Parameter(1,19,1,64)" "ConvolutionBackpropData(1,19,1,64,64,64,3,1,3,1,1,0,1,0,1,1)"
    echo "==== [`date`] ${FUNCNAME[0]} END"
}

function run_ngraph_tool() {
    echo -e "\n==== [`date`] ${FUNCNAME[0]} START"
    run_ngraph ./test7.xml transposeconv
    run_ngraph ./test8.xml transposeconv
    run_ngraph ./test8s.xml transposeconv
    run_ngraph ./test8-2.xml transposeconv
    run_ngraph ./test8-1-2.xml transposeconv
    run_ngraph ./test8-2-2.xml transposeconv
    #run_ngraph ./test8-2-2-bias.xml transposeconv
    run_ngraph ./test8-4-4.xml transposeconv
    run_ngraph ./test8-8-8.xml transposeconv
    run_ngraph ./test16.xml transposeconv
    run_ngraph ./test16-2.xml transposeconv
    run_ngraph ./test16-2-2.xml transposeconv
    run_ngraph ./test19-64-64.xml transposeconv
    #run_ngraph ./convtrans.xml transposeconv
    echo "==== [`date`] ${FUNCNAME[0]} END"
}

function make_ref_arks_CPU() {
    echo -e "\n==== [`date`] ${FUNCNAME[0]} START"
    speech_sample -m ./test7.xml -i ./test7_input.ark -d CPU -o test7_mkldnn.ark
    speech_sample -m ./test8.xml -i ./test8_input.ark -d CPU -o test8_mkldnn.ark
    speech_sample -m ./test8s.xml -i ./test8_input.ark -d CPU -o test8s_mkldnn.ark
    speech_sample -m ./test8-2.xml -i ./test8_input.ark -d CPU -o test8-2_mkldnn.ark
    speech_sample -m ./test8-1-2.xml -i ./test16_input.ark -d CPU -o test8-1-2_mkldnn.ark
    speech_sample -m ./test8-2-2.xml -i ./test16_input.ark -d CPU -o test8-2-2_mkldnn.ark
    #speech_sample -m ./test8-2-2-bias.xml -i ./test16_input.ark -d CPU -o test8-2-2-bias_mkldnn.ark
    speech_sample -m ./test8-4-4.xml -i ./test32_input.ark -d CPU -o test8-4-4_mkldnn.ark
    speech_sample -m ./test8-8-8.xml -i ./test64_input.ark -d CPU -o test8-8-8_mkldnn.ark
    speech_sample -m ./test16.xml -i ./test16_input.ark -d CPU -o test16_mkldnn.ark
    speech_sample -m ./test16-2.xml -i ./test16_input.ark -d CPU -o test16-2_mkldnn.ark
    speech_sample -m ./test16-2-2.xml -i ./test32_input.ark -d CPU -o test16-2-2_mkldnn.ark
    speech_sample -m ./test19-64-64.xml -i ./convtrans_input.ark -d CPU -o test19-64-64_mkldnn.ark
    #speech_sample -m ./convtrans.xml -i ./convtrans_input.ark -d CPU -o convtrans_mkldnn.ark
    echo "==== [`date`] ${FUNCNAME[0]} END"
}

function make_ref_arks_GNA() {
    echo -e "\n==== [`date`] ${FUNCNAME[0]} START"
    speech_sample -m ./test7_factorized.xml -i ./test7_input.ark -d CPU -r test7_mkldnn.ark -o test7_factorized_mkldnn.ark
    speech_sample -m ./test8_factorized.xml -i ./test8_input.ark -d CPU -r test8_mkldnn.ark -o test8_factorized_mkldnn.ark
    speech_sample -m ./test8s_factorized.xml -i ./test8_input.ark -d CPU -r test8s_mkldnn.ark -o test8s_factorized_mkldnn.ark
    speech_sample -m ./test8-2_factorized.xml -i ./test8_input.ark -d CPU -r test8-2_mkldnn.ark -o test8-2_factorized_mkldnn.ark
    speech_sample -m ./test8-1-2_factorized.xml -i ./test16_input.ark -d CPU -r test8-1-2_mkldnn.ark -o test8-1-2_factorized_mkldnn.ark
    speech_sample -m ./test8-2-2_factorized.xml -i ./test16_input.ark -d CPU -r test8-2-2_mkldnn.ark -o test8-2-2_factorized_mkldnn.ark
    #speech_sample -m ./test8-2-2-bias_factorized.xml -i ./test16_input.ark -d CPU -r test8-2-2-bias_mkldnn.ark -o test8-2-2-bias_factorized_mkldnn.ark
    speech_sample -m ./test8-4-4_factorized.xml -i ./test32_input.ark -d CPU -r test8-4-4_mkldnn.ark -o test8-4-4_factorized_mkldnn.ark
    speech_sample -m ./test8-8-8_factorized.xml -i ./test64_input.ark -d CPU -r test8-8-8_mkldnn.ark -o test8-8-8_factorized_mkldnn.ark
    speech_sample -m ./test16_factorized.xml -i ./test16_input.ark -d CPU -r test16_mkldnn.ark -o test16_factorized_mkldnn.ark
    speech_sample -m ./test16-2_factorized.xml -i ./test16_input.ark -d CPU -r test16-2_mkldnn.ark -o test16-2_factorized_mkldnn.ark
    speech_sample -m ./test16-2-2_factorized.xml -i ./test32_input.ark -d CPU -r test16-2-2_mkldnn.ark -o test16-2-2_factorized_mkldnn.ark
    speech_sample -m ./test19-64-64_factorized.xml -i ./convtrans_input.ark -d CPU -r test19-64-64_mkldnn.ark -o test19-64-64_factorized_mkldnn.ark
    #speech_sample -m ./convtrans_factorized.xml -i ./convtrans_input.ark -d CPU -r convtrans_mkldnn.ark -o convtrans_factorized_mkldnn.ark
    echo "==== [`date`] ${FUNCNAME[0]} END"
}

function speech_sample_GNA() {
    echo -e "\n==== [`date`] ${FUNCNAME[0]} START"
    speech_sample -m ./test7_factorized.xml -i ./test7_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test7_factorized_mkldnn.ark
    speech_sample -m ./test8_factorized.xml -i ./test8_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test8_factorized_mkldnn.ark
    speech_sample -m ./test8s_factorized.xml -i ./test8_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test8s_factorized_mkldnn.ark
    speech_sample -m ./test8-2_factorized.xml -i ./test8_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test8-2_factorized_mkldnn.ark
    speech_sample -m ./test8-1-2_factorized.xml -i ./test16_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test8-1-2_factorized_mkldnn.ark
    speech_sample -m ./test8-2-2_factorized.xml -i ./test16_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test8-2-2_factorized_mkldnn.ark
    #speech_sample -m ./test8-2-2-bias_factorized.xml -i ./test16_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test8-2-2-bias_factorized_mkldnn.ark
    speech_sample -m ./test8-4-4_factorized.xml -i ./test32_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test8-4-4_factorized_mkldnn.ark
    speech_sample -m ./test8-8-8_factorized.xml -i ./test64_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test8-8-8_factorized_mkldnn.ark
    speech_sample -m ./test16_factorized.xml -i ./test16_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test16_factorized_mkldnn.ark
    speech_sample -m ./test16-2_factorized.xml -i ./test16_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test16-2_factorized_mkldnn.ark
    speech_sample -m ./test16-2-2_factorized.xml -i ./test32_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test16-2-2_factorized_mkldnn.ark
    speech_sample -m ./test19-64-64_factorized.xml -i ./convtrans_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r test19-64-64_factorized_mkldnn.ark
    #speech_sample -m ./convtrans_factorized.xml -i ./convtrans_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r convtrans_factorized_mkldnn.ark
    echo "==== [`date`] ${FUNCNAME[0]} END"
}

#generate_arks
#make_models
#run_ngraph_tool
#make_ref_arks_CPU
#make_ref_arks_GNA
speech_sample_GNA

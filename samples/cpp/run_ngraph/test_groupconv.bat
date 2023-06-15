.\Debug\make_ngraph.exe test8 Parameter(1,8,1,8) GroupConvolution(1,8,1,8,2,4,1,3,1,2,1,1,0,1,0,1,1)
.\Debug\make_ngraph.exe test8d Parameter(1,1,8,8) GroupConvolution(1,1,8,8,8,1,1,1,3,1,2,0,0,0,0,1,1)
.\Debug\make_ngraph.exe test8d-bias Parameter(1,1,8,8) Transpose(1,1,8,8,0,3,1,2) GroupConvolutionBare(1,1,8,8,8,1,1,1,3,1,2,0,0,0,0,1,1) AddBias(1,8,1,3) Sigmoid() Transpose(1,8,1,3,0,2,3,1)
.\Debug\make_ngraph.exe test8-16 Parameter(1,8,1,16) GroupConvolution(1,8,1,16,4,4,1,3,1,2,1,1,0,1,0,1,1)
.\Debug\make_ngraph.exe test16 Parameter(1,16,1,8) GroupConvolution(1,16,1,8,2,4,1,3,1,2,1,1,0,1,0,1,1)
.\Debug\make_ngraph.exe test16-16 Parameter(1,16,1,16) GroupConvolution(1,16,1,16,8,2,1,3,1,2,1,1,0,1,0,1,1)
.\Debug\make_ngraph.exe test19-64-64 Parameter(1,1,79,64) GroupConvolution(1,1,79,64,64,1,1,1,6,1,4,0,0,0,0,1,1)

.\Debug\run_ngraph.exe .\test8.xml groupconv
.\Debug\run_ngraph.exe .\test8d.xml groupconv
.\Debug\run_ngraph.exe .\test8d-bias.xml groupconv
.\Debug\run_ngraph.exe .\test8-16.xml groupconv
.\Debug\run_ngraph.exe .\test16.xml groupconv
.\Debug\run_ngraph.exe .\test16-16.xml groupconv
.\Debug\run_ngraph.exe .\test19-64-64.xml groupconv
.\Debug\run_ngraph.exe .\depthconv.xml groupconv
copy test8d.xml test8d_dwsc.xml
copy test8d.bin test8d_dwsc.bin
copy test8d-bias.xml test8d-bias_dwsc.xml
copy test8d-bias.bin test8d-bias_dwsc.bin
copy depthconv.xml depthconv_dwsc.xml
copy depthconv.bin depthconv_dwsc.bin
copy test19-64-64.xml test19-64-64_dwsc.xml 
copy test19-64-64.bin test19-64-64_dwsc.bin 
.\Debug\run_ngraph.exe .\test8d_dwsc.xml dwsc
.\Debug\run_ngraph.exe .\test8d-bias_dwsc.xml dwsc
.\Debug\run_ngraph.exe .\test19-64-64_dwsc.xml dwsc
.\Debug\run_ngraph.exe .\depthconv_dwsc.xml dwsc

.\Debug\speech_sample.exe -m .\test8.xml -i .\test64_input.ark -d CPU -o test8_mkldnn.ark
.\Debug\speech_sample.exe -m .\test8d.xml -i .\test64_input.ark -d CPU -o test8d_mkldnn.ark
.\Debug\speech_sample.exe -m .\test8d-bias.xml -i .\test64_input.ark -d CPU -o test8d-bias_mkldnn.ark
.\Debug\speech_sample.exe -m .\test8-16.xml -i .\test128_input.ark -d CPU -o test8-16_mkldnn.ark
.\Debug\speech_sample.exe -m .\test16.xml -i .\test128_input.ark -d CPU -o test16_mkldnn.ark
.\Debug\speech_sample.exe -m .\test16-16.xml -i .\test256_input.ark -d CPU -o test16-16_mkldnn.ark
.\Debug\speech_sample.exe -m .\test19-64-64.xml -i .\depthconv_input.ark -d CPU -o test19-64-64_mkldnn.ark
.\Debug\speech_sample.exe -m .\depthconv.xml -i .\depthconv_input.ark -d CPU -o depthconv_mkldnn.ark
.\Debug\speech_sample.exe -m .\test8d_dwsc.xml -i .\test64_input.ark -d CPU -o test8d_dwsc_mkldnn.ark
.\Debug\speech_sample.exe -m .\test8d-bias_dwsc.xml -i .\test64_input.ark -d CPU -o test8d-bias_dwsc_mkldnn.ark
.\Debug\speech_sample.exe -m .\test19-64-64_dwsc.xml -i .\depthconv_input.ark -d CPU -o test19-64-64_dwsc_mkldnn.ark
.\Debug\speech_sample.exe -m .\depthconv_dwsc.xml -i .\depthconv_input.ark -d CPU -o depthconv_dwsc_mkldnn.ark

.\Debug\speech_sample.exe -m .\test8_factorized.xml -i .\test64_input.ark -d CPU -r test8_mkldnn.ark -o test8_factorized_mkldnn.ark 
.\Debug\speech_sample.exe -m .\test8d_factorized.xml -i .\test64_input.ark -d CPU -r test8d_mkldnn.ark -o test8d_factorized_mkldnn.ark 
.\Debug\speech_sample.exe -m .\test8d-bias_factorized.xml -i .\test64_input.ark -d CPU -r test8d-bias_mkldnn.ark -o test8d-bias_factorized_mkldnn.ark 
.\Debug\speech_sample.exe -m .\test8-16_factorized.xml -i .\test128_input.ark -d CPU -r test8-16_mkldnn.ark -o test8-16_factorized_mkldnn.ark 
.\Debug\speech_sample.exe -m .\test16_factorized.xml -i .\test128_input.ark -d CPU -r test16_mkldnn.ark -o test16_factorized_mkldnn.ark 
.\Debug\speech_sample.exe -m .\test16-16_factorized.xml -i .\test256_input.ark -d CPU -r test16-16_mkldnn.ark -o test16-16_factorized_mkldnn.ark 
.\Debug\speech_sample.exe -m .\test19-64-64_factorized.xml -i .\depthconv_input.ark -d CPU -r test19-64-64_mkldnn.ark -o test19-64-64_factorized_mkldnn.ark 
.\Debug\speech_sample.exe -m .\depthconv_factorized.xml -i .\depthconv_input.ark -d CPU -r depthconv_mkldnn.ark -o depthconv_factorized_mkldnn.ark 
.\Debug\speech_sample.exe -m .\test8d_dwsc_factorized.xml -i .\test64_input.ark -d CPU -r test8d_dwsc_mkldnn.ark -o test8d_dwsc_factorized_mkldnn.ark 
.\Debug\speech_sample.exe -m .\test8d-bias_dwsc_factorized.xml -i .\test64_input.ark -d CPU -r test8d-bias_dwsc_mkldnn.ark -o test8d-bias_dwsc_factorized_mkldnn.ark 
.\Debug\speech_sample.exe -m .\test19-64-64_dwsc_factorized.xml -i .\depthconv_input.ark -d CPU -r test19-64-64_dwsc_mkldnn.ark -o test19-64-64_dwsc_factorized_mkldnn.ark 
.\Debug\speech_sample.exe -m .\depthconv_dwsc_factorized.xml -i .\depthconv_input.ark -d CPU -r depthconv_dwsc_mkldnn.ark -o depthconv_dwsc_factorized_mkldnn.ark 


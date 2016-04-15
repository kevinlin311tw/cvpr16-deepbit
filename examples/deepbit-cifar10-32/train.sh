# training stage 1: minimal quantization loss + balance binary codes 
../../build/tools/caffe train -solver  solver_stage1.prototxt -weights ../../models/vggnet/VGG_ILSVRC_16_layers.caffemodel -gpu 0 2>&1 | tee log_deepbit32_stage1.txt
sleep 1
# training stage 2: minimal quantization loss + balance binary codes + rotation invariance
../../build/tools/caffe train -solver  solver_stage2.prototxt -weights DeepBit32_stage1_iter_50000.caffemodel -gpu 0 2>&1 | tee log_deepbit32_stage2.txt
sleep 1
# final stage: remove redundant parameter and output final deepbit model
../../build/tools/caffe train -solver  solver_final.prototxt -weights DeepBit32_stage2_iter_5000.caffemodel -gpu 0 2>&1 | tee log_deepbit32_fianl.txt

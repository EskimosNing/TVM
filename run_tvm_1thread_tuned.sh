export TVM_NUM_THREADS=1

log_path=./logs/tvm-gpu-1thread-titanx-tuned
mkdir -p $log_path
#python3 tvm_untuned2.py -m ./models/resnet50.onnx -d gpu  | tee $log_path/resnet50


python3 tvm_tuned2.py -m ./models/resnet50.onnx -t gpu  | tee $log_path/resnet50
# for i in `cat list`./run
# do
   
#    #python3 tvm_tuned.py -m ./models/$i.onnx -t x86  | tee $log_path/$i
#    #python3 tvm_tuned.py -m ./models/$i.onnx -t x86  | tee $log_path/$i
#     python3 tvm_untuned.py -m ./models/$i.onnx -d gpu  | tee $log_path/$i
# done

unset TVM_NUM_THREADS

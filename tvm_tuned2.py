import sys
import os
import argparse

from numpy.lib.stride_tricks import as_strided
from tvm.contrib import utils
from tvm import relay
import tvm
import timeit
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import numpy as np
import tvm.contrib.graph_runtime as runtime
from tvm.relay.op.tensor import mod
import time
from tvm.contrib import graph_executor
from scipy.special import softmax
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target', type=str, default='gpu')
parser.add_argument('-m', '--model', type=str, default='./models/resnet50_v2.onnx')
parser.add_argument('-l', '--log', type=str, default='./tvm-log/gpu/onnx/1batch/gpu_onnx_1batch_resnet50.log')
#parser.add_argument('-l', '--log', type=str, default=None)

args = parser.parse_args()

def get_model(model_path):
    import onnx

    ox = onnx.load(model_path)
    name = ox.graph.input[0].name
    input_shape = [i.dim_value for i in ox.graph.input[0].type.tensor_type.shape.dim]
    shape_dict = {name: input_shape}

    return ox, shape_dict

def get_target():
    # if args.target == 'x86' or args.target == 'cpu':
    #     target = tvm.target.create('llvm -mcpu=core-avx2')
    if args.target == 'gpu':
        target = tvm.target.cuda(model="titanx")

    return target

def get_logfile():
    if args.log:
        return args.log
    model_name = args.model.split('/')[-1][:-5]
    log_filepath = './tvm-log/' + args.target + '/onnx/1batch'
    if not os.path.exists(log_filepath):
        os.makedirs(log_filepath)
    log_file = log_filepath + '/' + '_'.join([args.target, 'onnx', '1batch', model_name]) + '.log'

    return log_file

def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               ):

    print("funtion tune_tasks. log filename =" , log_filename)
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
    #os.mknod(tmp_log_file) 
    print("the length of tasks:"+ str(len(tasks)))
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # op_name = tsk.workload[0]
        # if op_name == 'conv2d':
        #     func_create = 'topi_x86_conv2d_NCHWc'
        # elif op_name == 'depthwise_conv2d_nchw':
        #     func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        # else:
        #     raise ValueError("Tuning {} is not supported on x86".format(op_name))

        # task = autotvm.task.create(func_create, args=tsk.args,
        #                           target=tsk.target, template_key='direct')
        # task.workload = tsk.workload
        # tsk = task


        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        n_trial_temp = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=n_trial_temp,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial_temp, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file),],)

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)




def tuning(tuning_option,
           model_path=None,
           dtype='float32',
           input_name='data',
           device_key=None,
           use_android=False):
    print('Extract tasks...')

    ox, shape_dict = get_model(model_path)
    #mod, params, input_shape, out_shape = get_network(network, batch_size=1)
    mod, params = relay.frontend.from_onnx(ox, shape_dict) #1
    input_shape = shape_dict[input_name]
    target = get_target()
    # tasks = autotvm.task.extract_from_program(mod['main'], target=target,
    #                                          params = params,
    #                                          ops=(relay.op.nn.conv2d,))
    tasks = autotvm.task.extract_from_program(mod['main'], target=target,
                                             params = params,
                                             ops=(relay.op.get("nn.conv2d"),))
    
    log_file = tuning_option['log_filename']

    # run tuning tasks
    if os.path.exists(log_file):
        print(log_file + " exists, skipping...")
    else:
        print(log_file + " doesn't exist")
        print('Tuning...')
        print(time.strftime('[tune start localtime] %Y-%m-%d %H:%M:%S', time.localtime()))
        tune_tasks(tasks, **tuning_option)
    # compile kernels with history best records
    func = mod['main']
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3): #2
            graph, lib, params = relay.build(func, target, params=params)

    return graph, lib, params


#dev = tvm.device(str(target), 0)
#module = graph_executor.GraphModule(lib["default"](dev))
def tuning_model(model_path):
    dtype='float32'

    print("model_path:"+ model_path)
    ox, shape_dict = get_model(model_path)
    input_name = list(shape_dict.keys())[0]
    device_key = None
    if args.target == 'gpu':
        device_key = 'titanx'
    use_android = False

    log_file = get_logfile()
    print("log_file:"+log_file)
    other_option = {
        'model_path': model_path,
        'dtype': dtype,
        'input_name': input_name,
        'device_key': device_key,
        'use_android': use_android
    }

    if args.target == 'gpu':
        measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                # runner=autotvm.RPCRunner(
                    
                #     device_key,
                #     '127.0.0.1', 9190,
                #     number=20, repeat=3, timeout=4, min_repeat_ms=150)
                runner=autotvm.LocalRunner(
                    number=20, repeat=3, timeout=4, min_repeat_ms=150
                )
        )
    n_trial = 200
    #n_trial = 1500

    tuning_option = {
        'log_filename': log_file,
        'tuner': 'xgb',
        'n_trial': n_trial,
        'early_stopping': 80, #300
        'measure_option': measure_option
    }

    graph, lib, params = tuning(tuning_option, **other_option)
    return graph, lib, params

def speed(graph, lib, params, shape_dict, dtype='float32'):
    if args.target == 'gpu':
        #ctx = tvm.gpu(1)
        ctx=tvm.device("cuda", 0)
    input_name = list(shape_dict.keys())[0]
    input_shape = list(shape_dict.values())[0]
    #
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    #
    module = runtime.create(graph, lib, ctx)
    #module = runtime.GraphModule(lib["default"](ctx))
    module = graph_executor.GraphModule(lib["default"](ctx))
    module.set_input(input_name, data_tvm)
    module.set_input(**params)
    
    ftimer = module.module.time_evaluator('run', ctx, number=1, repeat=15)
    prof_res = np.array(ftimer().results)

    return prof_res

from PIL import Image
def evelution(graph, lib, params, shape_dict, dtype='float32'):
    #预处理
    start = time.time()
#long running
#do something other
    pretime=np.zeros(1010) 
    
    for i in range(1010):
        start = time.time()
        img_path = os.path.expanduser("/workspace/dlcompiler-comparison/TVM/data/imagenet_cat.png")
        # Resize it to 224x224
        resized_image = Image.open(img_path).resize((224, 224))
        img_data = np.asarray(resized_image).astype("float32")
        # Our input image is in HWC layout while ONNX expects CHW input, so convert the array
        img_data = np.transpose(img_data, (2, 0, 1))

        # Normalize according to the ImageNet input specification
        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

        # Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
        img_data = np.expand_dims(norm_img_data, axis=0)
        end = time.time()
        pretime[i]=(end-start)*1000

    preave=0.
    for i in range(1010):
        if i <= 9:
            continue
        else:
            preave=preave+pretime[i]
    preave=preave/1000
    print("preave: %s" % (preave))
    img_path = os.path.expanduser("/workspace/dlcompiler-comparison/TVM/data/imagenet_cat.png")
        # Resize it to 224x224
    resized_image = Image.open(img_path).resize((224, 224))
    img_data = np.asarray(resized_image).astype("float32")
        # Our input image is in HWC layout while ONNX expects CHW input, so convert the array
    img_data = np.transpose(img_data, (2, 0, 1))

        # Normalize according to the ImageNet input specification
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

        # Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
    img_data = np.expand_dims(norm_img_data, axis=0)
    if args.target == 'gpu':
        #ctx = tvm.gpu(1)
        ctx=tvm.device("cuda", 1)
    input_name = list(shape_dict.keys())[0]
    input_shape = list(shape_dict.values())[0]
    
    #
    #data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    #
    module = runtime.create(graph, lib, ctx)
    #module = runtime.GraphModule(lib["default"](ctx))
    #module = graph_executor.GraphModule(lib["default"](ctx))
    
    module.set_input(input_name, img_data)
    module.set_input(**params)
    module.run()
    
    output_shape = (1, 1000)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
    
    timing_number = 1000
    timing_repeat = 10
    unoptimized = (
        np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
        * 1000
        / timing_number
    )
    print(1)
    unoptimized = {
        "mean": np.mean(unoptimized),
        "median": np.median(unoptimized),
        "std": np.std(unoptimized),
    }

    print(unoptimized)
    #ftimer = module.module.time_evaluator('run', ctx, number=1, repeat=15)
    #prof_res = np.array(ftimer().results)
    #后处理输出
    # Download a list of labels
    posttime=np.zeros(1010)
    for i in range(1010):
        start = time.time()
        labels_path = os.path.expanduser("/workspace/dlcompiler-comparison/TVM/data/synset.txt")
        with open(labels_path, "r") as f:
            labels = [l.rstrip() for l in f]

        # Open the output and read the output tensor
        scores = softmax(tvm_output)
        scores = np.squeeze(scores)
        ranks = np.argsort(scores)[::-1]
        end = time.time()
        posttime[i]=(end-start)*1000
    postave=0.    
    for i in range(1010):
        if i <=9:
            continue
        else:
            postave=postave+posttime[i]
    postave=postave/1000
    print("postave: %s" % (postave))
    
            
    labels_path = os.path.expanduser("/workspace/dlcompiler-comparison/TVM/data/synset.txt")
    with open(labels_path, "r") as f:
        labels = [l.rstrip() for l in f]

    # Open the output and read the output tensor
    scores = softmax(tvm_output)
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]
    for rank in ranks[0:5]:
        print("class='%s' with probability=%f" % (labels[rank], scores[rank]))    
    #timing_number = 10
    #timing_repeat = 10
    # timing_number = 10000
    # timing_repeat = 10
    # optimized = (
    #     np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    #     * 1000
    #     / timing_number
    # )
    # optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}


    # print("optimized: %s" % (optimized))
    # print("unoptimized: %s" % (unoptimized))

def moduleRun():
    pass
if __name__ == '__main__':
    print(time.strftime('[start localtime] %Y-%m-%d %H:%M:%S', time.localtime()))
    graph, lib, params = tuning_model(args.model)
    print(time.strftime('[tune end localtime] %Y-%m-%d %H:%M:%S', time.localtime()))
    _, shape_dict = get_model(args.model)
    #prof_res = speed(graph, lib, params, shape_dict)
    evelution(graph, lib, params, shape_dict)
    print(time.strftime('[localtime] %Y-%m-%d %H:%M:%S', time.localtime()))
    model_name = args.model.split('/')[-1][:-5]
    print(model_name)
    print(list(shape_dict.values())[0])
    #for i in range(5, 15):
    #    print('-- {}, iteration time(s) is {:.4f}'.format(i, prof_res[i]))

    #print('@@ {}, average time(s) is {:.4f}'.format(model_name, np.mean(prof_res[5:])))
    print('FINISH')

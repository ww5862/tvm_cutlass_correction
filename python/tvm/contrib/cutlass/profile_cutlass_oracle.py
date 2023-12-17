#compile
import os
import subprocess
import multiprocessing

#runtime
import ctypes

#tvm
import tvm
import numpy as np
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor

#Making graph
import matplotlib.pyplot as plt

#get cutlass tailor data and get or set cutlass, cublas, Oracle Data
import pandas as pd
import json

#elapsed tunning time
import time as timSec

class Data:
    def __init__(self, data, dim):
        self.data = data
        self.dim = dim
        
        self.len = len(self.data)
        self.gpuMalloc = None
    
    def _loadMalloc(self):
        real_path = os.path.dirname(__file__)
        path = f"{real_path}/Malloc/gpuMalloc.so"
        c_module = ctypes.cdll.LoadLibrary(path)
        
        return c_module
    
    def mallocGPU(self):
        c_module = self._loadMalloc()
        malloc_gpu = c_module.MallocFloat
        
        (batch, row, column) = self.dim
        
        data = (ctypes.c_float * self.len)(*self.data)
        data_len = (ctypes.c_int)(self.len)
        
        self.gpuMalloc = malloc_gpu(data, (ctypes.c_int)(batch), (ctypes.c_int)(row), (ctypes.c_int)(column), data_len)
        
        # c_module.verify((ctypes.c_void_p)(self.gpuMalloc), data_len)
        
    def resetGPU(self):
        c_module = self._loadMalloc()
        reset = c_module.resetGPU()
    
    def memSetZeroGPU(self):
        c_module = self._loadMalloc()
        memSetGPU = c_module.memsetZero
        
        memSetGPU((ctypes.c_void_p)(self.gpuMalloc), (ctypes.c_int)(self.len))
    
    def freeGPU(self):
        c_module = self._loadMalloc()
        
        free_gpu = c_module.freeMalloc
        free_gpu((ctypes.c_void_p)(self.gpuMalloc))
        
    def verify(self):
        c_module = self._loadMalloc()
        
        verify = c_module.verify
        
        verify((ctypes.c_void_p)(self.gpuMalloc), (ctypes.c_int)(self.len))

class Cublas:
    def __init__(self, filename=None, object_filename = None, op=None):
        self.filename = filename
        self.object_filename = object_filename
        
        self.target_compile = None
        self.compile_option = None
        
        self.dim = None
        self.time = None
        
        for key, value in op.items():
            if "compile" in key:
                self.target_compile = value
            elif "option" in key:
                self.compile_option = value
    def get_object_rlt(self):
        return self.object_filename_rlt
    
    def elasped_time(self):
        return self.time
    
    def get_dim(self):
        return self.dim
    
    def _loadBackend(self):
        path = self.object_filename_rlt
        c_module = ctypes.cdll.LoadLibrary(path)
        
        return c_module
    
    def compile(self, batch=24, m=512, n=512, k=64):
        dim_opt = f"-DBatch={batch} -DM={m} -DN={n} -DK={k}"
        
        self.dim = (batch, m, n, k)
        
        self.object_filename_rlt = f"./tmp_cublas/{self.object_filename}_{batch}_{m}_{n}_{k}"
        
        if os.path.exists(self.object_filename_rlt):
            return None
        
        compile = f"{self.target_compile} {self.filename} {self.compile_option} {dim_opt} -o {self.object_filename_rlt}"
        os.system(compile)
        
    def eval(self, tensorA:Data, tensorB:Data, tensorC:Data, warmup=20, repeat=100):
        c_module = self._loadBackend()
        batch_cublas = c_module.batch_cublas
        
        (batch, m, n, k) = self.dim
        
        assert tensorA.dim[0] == tensorB.dim[0] and tensorA.dim[2] == tensorB.dim[1]
        assert tensorC.dim[0] == tensorA.dim[0] and tensorC.dim[1] == tensorA.dim[1] and tensorC.dim[2] == tensorB.dim[2]
        assert batch == tensorA.dim[0] and m == tensorA.dim[1] and n == tensorB.dim[2] and k == tensorA.dim[2]
        
        batch_cublas.restype = ctypes.c_float
        time = batch_cublas((ctypes.c_void_p)(tensorA.gpuMalloc), (ctypes.c_void_p)(tensorB.gpuMalloc), (ctypes.c_void_p)(tensorC.gpuMalloc), (ctypes.c_int)(warmup), (ctypes.c_int)(repeat))
        self.time = time

class TVMTunning:
    def __init__(self, dim):
        self.dim = dim
    def eval(self, sm=75, trial=900, tunning=True):
        (batch, m, n, k) = self.dim
        
        input = relay.var("input", dtype="float32", shape=(batch, m, k))
        weight = relay.var("weight", dtype="float32", shape=(batch, n, k))
        
        mod = relay.nn.batch_matmul(input, weight, out_dtype="float32", transpose_a=False, transpose_b=True)
        mod = tvm.IRModule.from_expr(mod)
        mod = relay.transform.InferType()(mod)
        
        input_arr = np.random.uniform(1, -1, size=(batch, m, k)).astype("float32")
        weight_arr = np.random.uniform(1, -1, size=(batch, n, k)).astype("float32")
        
        params = {"weight": weight_arr}
        cuda = tvm.target.Target("cuda")
        log_file = f"./tvm_log/{batch}_{m}_{n}_{k}_gemm.json"
        
        elapsed_time = None
        
        if tunning is True:
            tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, cuda)
            
            measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)
            tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=trial,
                runner=measure_ctx.runner,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            )
            
            start_time = timSec.time()
            tuner.tune(tune_option)
            end_time = timSec.time()
            
            elapsed_time = end_time - start_time
            
        
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=cuda, params=params)
                
        dev = tvm.device(str(cuda), 0)
        module = graph_executor.GraphModule(lib["default"](dev))
        
        module.set_input("input", input_arr)
        tmp = module.benchmark(dev, number=1, repeat=20)
        rlt = module.benchmark(dev, number=1, repeat=100)
        
        return rlt.mean * 1000, elapsed_time

class Cutlass:
    def __init__(self, filename=None, object_filename=None, op=None):
        self.filename = filename
        self.object_filename = object_filename
        self.target_compile = None
        self.compile_option = None
        
        self.option_dim = None
        
        
        self.object_filename_rlt = None
        self.split_k = 1
        self.time = None
        
        for key, value in op.items():
            if "compile" in key:
                self.target_compile = value
            else:
                self.compile_option = value
    
    def _loadBackend(self):
        path = self.object_filename_rlt
        
        if os.path.exists(self.object_filename_rlt):
            c_module = ctypes.cdll.LoadLibrary(path)
        else:
            return None
        
        return c_module
        
    def compile(self, opt=[]):
        block_shape = opt[0]
        warp_shape = opt[1]
        kstage = opt[2]
        alginment = opt[3]
        
        self.option_dim = opt
        
        dim_opt_block = f"-DMM={block_shape[0]} -DMN={block_shape[1]} -DMk={block_shape[2]} "
        dim_opt_warp = f"-DWM={warp_shape[0]} -DWN={warp_shape[1]} -DWK={warp_shape[2]} "
        dim_opt = dim_opt_block + dim_opt_warp + f"-DKStage={kstage[0]} -DAl={alginment[0]} "
        
        real_path = os.path.dirname(__file__)
        self.object_filename_rlt = f"{real_path}/tmp_cutlass/{self.object_filename}_{block_shape[0]}_{block_shape[1]}_{block_shape[2]}_{warp_shape[0]}_{warp_shape[1]}_{warp_shape[2]}_{kstage[0]}_{alginment[0]}"
        if os.path.exists(self.object_filename_rlt):
            return None
        
        print(f"compile: {self.object_filename_rlt}")
        compile = f"{self.target_compile} {self.filename} {self.compile_option} {dim_opt} -o {self.object_filename_rlt}"
        os.system(compile)
    
    def compile_all(self, opt):
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.map(self.compile, opt)
        
    def eval(self, tensorA:Data, tensorB:Data, tensorC:Data, split_k = 1, warmup=20, repeat=100):
        c_module = self._loadBackend()
        
        if c_module is None:
            return -1
        
        cutlass_batch = c_module.cutlass_batch
        
        cutlass_batch.restype = ctypes.c_float
        time = cutlass_batch((ctypes.c_void_p)(tensorA.gpuMalloc), (ctypes.c_void_p)(tensorB.gpuMalloc), (ctypes.c_void_p)(tensorC.gpuMalloc), (ctypes.c_int)(split_k), (ctypes.c_int)(warmup), (ctypes.c_int)(repeat))
        return time

class Tailor:
    def __init__(self, fileName):
        self.fileName = fileName
        self.tailorLog = {}
        
    def loadCSV(self):
        df = pd.read_csv(self.fileName)
        
        for index, row in df.iloc[:-2].iterrows():
            m = int(row["m"])
            n = int(row["n"])
            k = int(row["k"])
            
            key = f"{[m, n, k]}"
            
            tileShape = row["tile"]
            
            tileShape = tileShape.split("_")
            cutlass_execute_file = [tileShape[0], tileShape[1]]
            
            split_k = int(tileShape[2])
            
            execute = [cutlass_execute_file, split_k]
            
            if key in self.tailorLog:
                self.tailorLog[key].append(execute)
            else:
                self.tailorLog.update({key: []})

class CompareCutlass:
    def __init__(self, dim=(), printing=True):
        self.printing = printing
        
        self.dim = dim
        self.default = ((128, 128, 8), (32, 64, 8), (2,), (1,))
        (self.batch, self.m, self.n, self.k) = dim
        
        self.opt_rlt = None
    
    def _iswarp(self, shared_m, shared_n, warp_m, warp_n):
        if((shared_m / warp_m) * (shared_n / warp_n)) > 26:
            return False
        
        if (shared_m % warp_m) != 0 or (shared_n % warp_n) != 0:
            return False
        
        tmp_warp = (shared_m / warp_m) * (shared_n / warp_n) * 32
        
        # if tmp_warp < shared_m or tmp_warp < shared_n:
        #     return False
        # if tmp_warp % shared_m != 0 and shared_m % tmp_warp:
        #     return False
        # if tmp_warp % shared_n != 0 and shared_n % tmp_warp:
        #     return False
        
        warpNumThreadsM = 8
        
        if warp_m <= warp_n:
            warpNumThreadsM = 4
        
        warpNumThreadsN = 32 / warpNumThreadsM
        
        ThreadTileM = warp_m / warpNumThreadsM
        ThreadTileN = warp_n / warpNumThreadsN
        
        if warp_m % warpNumThreadsM != 0 or warp_n % warpNumThreadsN != 0:
            return False
        
        laneM = min(4, ThreadTileM)
        laneN = min(4, ThreadTileN)
        
        paddingM = 8
        paddingN = 8
        
        if paddingM % laneM != 0 or paddingN % laneN:
            return False
        
        if warp_m / warpNumThreadsM < 0 or warp_n / warpNumThreadsN < 0:
            return False
        
        if warp_m % warpNumThreadsM != 0 or warp_n % warpNumThreadsN != 0:
            return False
        
        iterationA = warp_m / warpNumThreadsM
        iterationB = warp_n / warpNumThreadsN
    
        if iterationA % laneM != 0 or iterationB % laneN != 0:
            return False
        
        return True
    
    def _create_cutlass_opt(self, m_shape=[], n_shape=[], k_shape=[], kstage_shape=[], alignment=1):
        compile_option = []
        
        for stage in range(kstage_shape[0], kstage_shape[1] + kstage_shape[2], kstage_shape[2]):
            for shared_m in range(m_shape[0], m_shape[1] + m_shape[2], m_shape[2]):
                for shared_n in range(n_shape[0], n_shape[1] + n_shape[2], n_shape[2]):
                    for shared_k in range(k_shape[0], k_shape[1] + k_shape[2], k_shape[2]):
                        for warp_m in range(1, shared_m + 1):
                            for warp_n in range(1, shared_n + 1):
                                if self._iswarp(shared_m, shared_n, warp_m, warp_n) is False:
                                    continue
                                
                                op = [[shared_m, shared_n, shared_k], [warp_m , warp_n, shared_k], [stage], [alignment]]
                                compile_option.append(op)
        return compile_option
    
    def tunning_cutlass(self, tunning_parameter=None, tunning_split_k=None):
        real_path = os.path.dirname(__file__)
        cutlass_file = f"{real_path}/src_cutlass/cutlass_batch_gemm.cu"
        cutlass_object_file = "cutlass_batch_gemm"
        cutlass_op = {
                "compile" : "nvcc",
                "option" : "-O3 -arch=sm_86 --std=c++17 -I /home/local_guest/tvm/3rdparty/cutlass/include -I /home/local_guest/tvm/3rdparty/cutlass/cutlass/examples -I /home/local_guest/tvm/3rdparty/cutlass/tools/util/include --shared -Xcompiler -fPIC"
            }
        
        m_shape = [32, 256, 32]
        n_shape = [32, 256, 32]
        k_shape = [8, 8, 8]
        kstage = [2, 2, 1]
        
        self.opt_rlt = self._create_cutlass_opt(m_shape, n_shape, k_shape, kstage)
        
        #cutlass compile
        run_cutlass = Cutlass(cutlass_file, cutlass_object_file, cutlass_op)
        run_cutlass.compile_all(opt=self.opt_rlt)
        
        if (tunning_parameter is not None) and (tunning_split_k is not None):
            run_cutlass.compile(tunning_parameter)
            run_cutlass.split_k = tunning_split_k
            
            return run_cutlass, None
        
        cutlass_array = [Cutlass(cutlass_file, cutlass_object_file, cutlass_op) for _ in range(len(self.opt_rlt))]

        tensorA = Data([i for i in range(self.batch * self.m * self.k)], (self.batch, self.m, self.k))
        tensorB = Data([1 for _ in range(self.batch * self.k * self.n)], (self.batch, self.k, self.n))
        tensorC = Data([0 for _ in range(self.batch * self.m * self.n)], (self.batch, self.m, self.n))
        
        tensorA.resetGPU()
        tensorA.mallocGPU()
        tensorB.mallocGPU()
        tensorC.mallocGPU()
        
        fastest_time = 10e9
        fastest_cutlass = None
        
        elasped_time = None
        
        start_time = timSec.time()
        
        for split in range(1, 16):
            for i, value in enumerate(cutlass_array):
                value.compile(self.opt_rlt[i])
                tunning_time = value.eval(tensorA, tensorB, tensorC, split_k = split, warmup=20, repeat=10)
                tensorC.memSetZeroGPU()
                if tunning_time < 0:
                    continue
                if fastest_time > tunning_time:
                    fastest_time = tunning_time
                    fastest_cutlass = value
                    fastest_cutlass.split_k = split
        
        end_time = timSec.time()
                    
        tensorA.freeGPU()
        tensorB.freeGPU()
        tensorC.freeGPU()
        
        tensorA.resetGPU()
        
        elasped_time = end_time - start_time
        return fastest_cutlass, elasped_time
    
    def tunning_tailor(self):
        tailor_log = Tailor("231115_sample_3090.csv")
        tailor_log.loadCSV()
        
        log = tailor_log.tailorLog
        log = log[str([self.m, self.n, self.k])]
        
        best_cutlass = None
        
        opt_rlt = []
        for i, value in enumerate(log):
            split_k = value[1]
            
            if split_k == 0:
                split_k = 1
            
            threadblock = value[0][0]
            warpblock = value[0][1]
            
            threadblock = threadblock.split("x")
            warpblock = warpblock.split("x")
            
            tmp_rlt = [[[int(threadblock[0]), int(threadblock[1]), 8], [int(warpblock[0]), int(warpblock[1]), 8], [2], [1]], split_k]
            opt_rlt.append(tmp_rlt)
        
        cutlass_file = "./src_cutlass/cutlass_batch_gemm.cu"
        cutlass_object_file = "cutlass_batch_gemm"
        cutlass_op = {
                "compile" : "nvcc",
                "option" : "-O3 -arch=sm_86 --std=c++17 -I ./cutlass/include -I ./cutlass/examples -I ./cutlass/tools/util/include --shared -Xcompiler -fPIC"
            }
        
        cutlass_array = [Cutlass(cutlass_file, cutlass_object_file, cutlass_op) for _ in range(len(opt_rlt))]
        
        tensorA = Data([i for i in range(self.batch * self.m * self.k)], (self.batch, self.m, self.k))
        tensorB = Data([1 for _ in range(self.batch * self.k * self.n)], (self.batch, self.k, self.n))
        tensorC = Data([0 for _ in range(self.batch * self.m * self.n)], (self.batch, self.m, self.n))
        
        tensorA.resetGPU()
        tensorA.mallocGPU()
        tensorB.mallocGPU()
        tensorC.mallocGPU()
        
        fastest_cutlass = None
        fastest_time = 10e9
        
        for i, value in enumerate(cutlass_array):
            value.compile(opt_rlt[i][0])
            time = value.eval(tensorA, tensorB, tensorC, opt_rlt[i][1], warmup=20, repeat=10)
            
            if time < 0:
                continue
            
            if time < fastest_time:
                fastest_time = time
                fastest_cutlass = value
                fastest_cutlass.split_k = opt_rlt[i][1]
        
        tensorA.freeGPU()
        tensorB.freeGPU()
        tensorC.freeGPU()
        tensorA.resetGPU()
        
        return fastest_cutlass
    
    def benchmark_TVM(self, trial=100, tunning=True):
        tvm_tunnning = TVMTunning(self.dim)
        time, tunning_time = tvm_tunnning.eval(trial=trial, tunning=tunning)
        
        return time, tunning_time

    def benchmark_cutlass(self, best_cutlass:Cutlass, repeat=100):
        tensorA = Data([i for i in range(self.batch * self.m * self.k)], (self.batch, self.m, self.k))
        tensorB = Data([1 for _ in range(self.batch * self.k * self.n)], (self.batch, self.k, self.n))
        tensorC = Data([0 for _ in range(self.batch * self.m * self.n)], (self.batch, self.m, self.n))
        
        tensorA.resetGPU()
        tensorA.mallocGPU()
        tensorB.mallocGPU()
        tensorC.mallocGPU()
        
        time = best_cutlass.eval(tensorA, tensorB, tensorC, split_k=best_cutlass.split_k, warmup=20, repeat=100)
        
        tensorA.freeGPU()
        tensorB.freeGPU()
        tensorC.freeGPU()
        
        tensorA.resetGPU()
        
        return time
    
    def benchmark_cuBLAS(self, repeat=100):
        # print(f"Matrix Dimension: {self.batch}, {self.m}, {self.n}, {self.k}")
        
        cublas_file = "./src_cublas/batch_gemm.cu"
        cublas_object_file = "batch_gemm"
        cublas_op = {
                "compile" : "nvcc",
                "option" : "-O3 -arch=sm_86 --std=c++17 -lcublas --shared -Xcompiler -fPIC",
            }
                
        #cublas
        run_cublas = Cublas(cublas_file, cublas_object_file, cublas_op)
        run_cublas.compile(batch=self.batch, m=self.m, n=self.n, k=self.k)
        
        #make data
        tensorA = Data([i for i in range(self.batch * self.m * self.k)], (self.batch, self.m, self.k))
        tensorB = Data([1 for _ in range(self.batch * self.n * self.k)], (self.batch, self.k, self.n))
        tensorC = Data([0 for _ in range(self.batch * self.m * self.n)], (self.batch, self.m, self.n))
        
        tensorA.resetGPU()
        
        #copy data host to device
        tensorA.mallocGPU()
        tensorB.mallocGPU()
        tensorC.mallocGPU()
        
        run_cublas.eval(tensorA=tensorA, tensorB=tensorB, tensorC=tensorC, warmup=20, repeat=100)
        time = run_cublas.elasped_time()
        
        tensorA.freeGPU()
        tensorB.freeGPU()
        tensorC.freeGPU()
        
        tensorA.resetGPU()
        
        return time
    
class Visualize:
    def __init__(self, name):
        self.name = name
    
    def plotTvmAndCutlass(self, time, time2, dim_option, test_k):
        f = plt.figure(figsize=(15, 10))
        x = [str(i) for i in test_k]
        
        for i in range(1, len(dim_option)+1):
            ax = plt.subplot(2, 2, i)
            x = [str(i) for i in test_k]
            
            ax.bar([i*2 for i in range(len(test_k))], time[i-1], label=f"cutlass n={dim_option[i-1][0]}, m={dim_option[i-1][1]}")
            ax.bar([i*2 + 0.83 for i in range(len(test_k))], time2[i-1], label=f"TVM n={dim_option[i-1][0]}, m={dim_option[i-1][1]}")
            
            ax.set_xticks([i * 2 + 0.5 for i in range(len(test_k))], x)
            plt.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.11))
            # plt.xlabel("K")
            # plt.ylabel("Speedup")
    
        plt_str = f"batch_gemm_{self.name}_TVM_Auto_tuning_cutlass.png"
        plt.savefig(plt_str)
        plt.clf()

class LogFile:
    def __init__(self, fileName):
        self.fileName = fileName
        self.logData = {}
        
    def writeLog(self, dim, parameter, split_k, cublas_time, cutlass_time, ansor_time, oracle_tunning=0, ansor_tunning=0,tailor_time=None):
        tmp_parameter = []
        tmp_parameter.append(parameter)
        tmp_parameter.append(split_k)
        
        if tailor_time is None:
            json_attr = {"dim": dim, "tunning": tmp_parameter, "cublas": cublas_time, "cutlass": cutlass_time,
                         "ansor": ansor_time, "cutlass_tunning_time": oracle_tunning, "ansor_tunning_time": ansor_tunning}
        else:
            json_attr = {"dim": dim, "tunning": tmp_parameter, "cublas": cublas_time, "cutlass": cutlass_time, "ansor": ansor_time, "tailor": tailor_time}
        
        with open(self.fileName, 'a') as f:
            json.dump(json_attr, f)
            f.write('\n')
    
    def loadLog(self):    
        with open(self.fileName, 'r') as f:
            for line in f:
                json_data = json.loads(line)                
                self.logData.update({f"{json_data['dim']}": {'option_dim': json_data['tunning'][0],
                                                             'split_k': json_data['tunning'][1], 'cublas': json_data['cublas'],
                                                             'cutlass': json_data['cutlass'], 'ansor': json_data['ansor']}})
                
                if ('cutlass_tunning_time' in json_data) and ('ansor_tunning_time' in json_data):
                    self.logData[f"{json_data['dim']}"].update({'cutlass_tunning_time': json_data['cutlass_tunning_time'], 'ansor_tunning_time': json_data['ansor_tunning_time']}) 
    
    def loadTime(self, dim):
        dim = f"{[dim[0], dim[1], dim[2], dim[3]]}"
        
        if dim not in self.logData:
            return None, None, None
        
        log = self.logData[dim]
        cublas_time = log['cublas']
        cutlass_time = log['cutlass']
        ansor_time = log['ansor']
        
        return cublas_time, cutlass_time, ansor_time
    
    def loadTunningTime(self, dim):
        dim = f"{[dim[0], dim[1], dim[2], dim[3]]}"
        
        if dim not in self.logData:
            return None, None
        
        log = self.logData[dim]
        
        if ('cutlass_tunning_time' not in log) and ('ansor_tunning_time' not in log):
            return None, None
        
        oracle_tunning_time = log['cutlass_tunning_time']
        ansor_tunning_time = log['ansor_tunning_time']
        
        return oracle_tunning_time, ansor_tunning_time
    
    def loadParameter(self, dim):
        dim = f"{[dim[0], dim[1], dim[2], dim[3]]}"
        
        if dim not in self.logData:
            return None, None
        
        log = self.logData[dim]
        parameter = log['option_dim']
        split_k = log['split_k']
        
        return parameter, split_k
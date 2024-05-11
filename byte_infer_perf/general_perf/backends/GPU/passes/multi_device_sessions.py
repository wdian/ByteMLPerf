
import time
import threading
import numpy as np
import onnxruntime as ort
TypeMap = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(uint8)" : np.uint8,
    "tensor(int8)": np.int8,
    "tensor(int32)": np.int32,
    "tensor(int64)" : np.int64,
    "tensor(uint64)" : np.uint64,
}
class MultiDeviceInferenceSession():
    def __init__(self, model_path, providers, device_ids=None):
        self.lock = threading.Lock()
        self.model_path = model_path
        self.providers = providers
        self.session_dict = {}
        self.device_ids = device_ids
        self.task_dict = {}
        self.use_maca = False
        self.initSessions()

    def initSessions(self):
        if("MACAExecutionProvider" not in self.providers):
            self.device_ids = [0]
            self.session_dict[0] = ort.InferenceSession(self.model_path, providers = self.providers, provider_options = [])
            self.task_dict[0] = 0
        else:
            self.use_maca = True
            if((self.device_ids is None) or len(self.device_ids) == 0):
                self.device_ids = [i for i in range(ort.getDeviceCount())]
            for device_id in self.device_ids:
                provider_options = [{"device_id":device_id}]         
                self.session_dict[device_id] = ort.InferenceSession(self.model_path, providers = self.providers, provider_options = provider_options)
                self.task_dict[device_id] = 0
    
    def gainDeviceNum(self):
        return len(self.device_ids)
    
    def getIobinding(self, sess, input_dict, input_loc_str, output_loc_str, device_id):
        input_nodes = sess.get_inputs()
        input_type_map = {}
        for i_n in input_nodes:
            input_type_map[i_n.name]=i_n.type
        
        output_names = self.gainNodeNames(sess.get_outputs())
        io_binding = sess.io_binding()
        for key in input_dict.keys():
            input_numpy = input_dict[key]
            if isinstance(input_numpy, np.ndarray) == False:
                input_numpy = np.array(input_numpy, dtype=TypeMap[input_type_map[key]])
            io_binding.bind_ortvalue_input(key, 
                ort.OrtValue.ortvalue_from_numpy(input_numpy, input_loc_str, device_id=device_id))
        for o_n in output_names:
            io_binding.bind_output(o_n, output_loc_str,device_id=device_id)
        return io_binding
    def gainNodeNames(self,nodes):
        names = []
        for n in nodes:
            names.append(n.name)
        return names  
    def gainFPS(self, inputs_dict, thread_num = 16, infer_num = 16, warm_num=16):
        if self.use_maca:
            input_memory_info = "maca" ## maca, maca_pinned, cpu
            output_memory_info = "maca" ## maca, maca_pinned, cpu
        else:
            input_memory_info = "cpu" ## maca, maca_pinned, cpu
            output_memory_info = "cpu" ## maca, maca_pinned, cpu
        infer_num = infer_num * self.gainDeviceNum()
        task_list = [i for i in range(infer_num)]
        threadLock = threading.Lock()
        def infer_task(sess, io_binding):
            stop = False
            while True:
                threadLock.acquire()
                if len(task_list) == 0:
                    stop = True
                else:
                    task_list.pop()
                threadLock.release()
                if stop:
                    break
                else:
                    sess.run_with_iobinding(io_binding)
        ## warming up
        #for _ in range(warm_num):
            # Run inference.
            #self.run(None, inputs_dict)
        sess_iobindind_dict = {}
        for d in self.device_ids:
            sess = self.getSesssion(d)
            sess_iobindind_dict[sess] = self.getIobinding(sess, inputs_dict,input_memory_info,output_memory_info, d) 
            for i in range(warm_num):
                sess.run_with_iobinding(sess_iobindind_dict[sess])
        
        thread_num = thread_num * self.gainDeviceNum()
        threads = []
        for i in range(thread_num):
            execute_id = self.device_ids[i % len(self.device_ids)] 
            sess = self.getSesssion(execute_id)
            threads.append(threading.Thread(target=infer_task, args=(sess, sess_iobindind_dict[sess])))
        start = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        time_cost = time.time() - start
        latency = time_cost/(infer_num)
        fps = 1./latency 
        # self.initSessions()
        return fps
    
    def benchmark(self, inputs_dict, batch_size, infer_num = 16, warm_num=16):
        if self.use_maca:
            input_memory_info = "maca" ## maca, maca_pinned, cpu
            output_memory_info = "maca" ## maca, maca_pinned, cpu
        else:
            input_memory_info = "cpu" ## maca, maca_pinned, cpu
            output_memory_info = "cpu" ## maca, maca_pinned, cpu
        infer_num = infer_num * self.gainDeviceNum()

        sess_iobindind_dict = {}
        for d in self.device_ids:
            sess = self.getSesssion(d)
            sess_iobindind_dict[sess] = self.getIobinding(sess, inputs_dict,input_memory_info,output_memory_info, d) 
            for i in range(warm_num):
                sess.run_with_iobinding(sess_iobindind_dict[sess])
        

        times_range = []
        for d in self.device_ids:
            sess = self.getSesssion(d)
            sess_iobindind_dict[sess] = self.getIobinding(sess, inputs_dict,input_memory_info,output_memory_info, d) 
            for i in range(infer_num):
                start_time = time.time()
                sess.run_with_iobinding(sess_iobindind_dict[sess])
                end_time = time.time()
                times_range.append(end_time - start_time)

        times_range.sort()
        tail_latency = round(times_range[int(len(times_range) * 0.99)]/ batch_size * 1000, 3)
        avg_latency = round(sum(times_range) / len(times_range) / batch_size * 1000, 3)
        
        return avg_latency, tail_latency
        
    def gainFPSCPUIO(self, inputs_dict, thread_num = 16, infer_num = 16, warm_num=8):
        infer_num = infer_num * self.gainDeviceNum()
        task_list = [i for i in range(infer_num)]
        threadLock = threading.Lock()
        def infer_task(sess,ort_inputs, device_id):
            stop = False
            while True:
                threadLock.acquire()
                if len(task_list) == 0:
                    stop = True
                else:
                    task_list.pop()
                threadLock.release()
                if stop:
                    break
                else:
                    sess.run(None, ort_inputs, device_id)
        ## warming up
        for _ in range(warm_num):
            # Run inference.
            self.run(None, inputs_dict)
        
        thread_num = thread_num * self.gainDeviceNum()
        threads = []
        for i in range(thread_num):
            threads.append(threading.Thread(target=infer_task, args=(self, inputs_dict, i)))
        start = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        time_cost = time.time() - start
        latency = time_cost/(infer_num)
        fps = 1./latency 
        return fps
        
    def gainBestDeviceId(self):
        self.lock.acquire()
        device_id = 0
        task_num = 1e16
        for key in self.task_dict.keys():
            if(self.task_dict[key] < task_num):
                device_id = key
                task_num = self.task_dict[key] 
        self.task_dict[device_id] += 1
        self.lock.release()
        return device_id
    def reduceTask(self, device_id):
        self.lock.acquire()
        self.task_dict[device_id] -= 1
        self.lock.release()
    def addTask(self, device_id):
        self.lock.acquire()
        self.task_dict[device_id] += 1
        self.lock.release()
    def getSesssion(self, device_id):
        return self.session_dict[device_id]
    def run(self, output_names, input_dict,device_id=None):
        if(device_id is None):
            execute_id = self.gainBestDeviceId()
            sess = self.getSesssion(execute_id)
            output = sess.run(output_names,input_dict)
            self.reduceTask(execute_id)
        else:
            execute_id = self.device_ids[device_id % len(self.device_ids)] 
            sess = self.getSesssion(execute_id)
            output = sess.run(output_names,input_dict)
        return output
    # def run(self, output_names, input_dict, device_id):
    #     self.addTask(device_id)
    #     sess = self.getSesssion(device_id)
    #     output = sess.run(output_names,input_dict)
    #     self.reduceTask(device_id)
    #     return output
    def get_inputs(self):
        return self.getSesssion(self.device_ids[0]).get_inputs()
        

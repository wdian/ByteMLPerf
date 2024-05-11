import os
import time
import logging
from typing import Any, Dict
import numpy as np
import onnxruntime as ort
from general_perf.datasets.data_loader import Dataset
from general_perf.backends import runtime_backend
from general_perf.backends.GPU.passes.multi_device_sessions import MultiDeviceInferenceSession

log = logging.getLogger("RuntimeBackendGPU")


INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": np.int64,
    "INT32": np.int32,
    "INT64": np.int64,
}

class RuntimeBackendGPU(runtime_backend.RuntimeBackend):
    def __init__(self, engine=None, configs=None,  batch_size=1, precision="fp32"):
        super(RuntimeBackendGPU, self).__init__()
        self.hardware_type = "GPU"
        self.need_reload = False

        self._engine = engine
        self.configs = configs
        self.batch_size = batch_size
        self.precision = precision

        self.runner_name = "OnnxRT"
        self.compiled_dir = (os.path.split(os.path.abspath(__file__))[0] + "/compiled_models")
    
    @ property
    def engine(self, engine):
        return self._engine


    def version(self) -> str:
        return "1.0.0"
    
    def load(self, batch_size): 
        if self.batch_size != batch_size:
            self.batch_size = batch_size

        self.input_type = self.configs['input_type']
        self.framework = self.configs['framework']
        self.model_name = self.configs['model']

        # create engine
        sess_option = ort.SessionOptions()
        sess_option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        provider_list = ["MACAExecutionProvider"]
        # provider_list = ["CPUExecutionProvider"]

        model_file = self.model_info['model_path']
        run_sesion = MultiDeviceInferenceSession(model_file, providers=provider_list)
        self._engine = run_sesion
        return self._engine

    def _tuning_input(self, feeds: Dict[str, Any]):
        i = 0
        for k,v in feeds.items():
            if not isinstance(v, list): continue
            feeds[k] = np.array(v, dtype= INPUT_TYPE[self.input_type[i]])
            i += 1
        return feeds

    def get_loaded_batch_size(self):
        return self.batch_size

    def predict(self, feeds, test_benchmark=False):
        self._tuning_input(feeds)
        results = self._engine.run(None, feeds)
        if str(self.model_name).startswith("bert-tf"):
            results = np.split(results[0], 2, axis=-1)

        return results
    

    def benchmark(self, dataloader: Dataset, thread_num=8):
        iterations = self.workload['iterations']
        input_feeds = dataloader.get_fake_samples(self.batch_size, shape=self.configs["input_shape"], input_type=self.input_type)
        input_feeds = self._tuning_input(input_feeds)
        avg_latency, tail_latency  = self._engine.benchmark(input_feeds, 
                                                            batch_size=self.batch_size, 
                                                            infer_num=iterations)      #  ms/sample
        qps = round(1000 / avg_latency, 2)

        qps_info = {
            "BS": self.batch_size,
            "QPS": qps,
            "AVG_Latency": avg_latency,
            "P99_Latency": tail_latency,
            }
        return qps_info
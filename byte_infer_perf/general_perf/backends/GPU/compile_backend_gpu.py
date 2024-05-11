

import os
import json
import onnx
import subprocess
import logging
import importlib
from typing import Any, Dict

import numpy as np
import yaml
from general_perf.tools import  torch_to_onnx
from general_perf.backends import compile_backend

log = logging.getLogger("CompileBackendGPU")




class CompileBackendGPU(compile_backend.CompileBackend):
    def __init__(self):
        super(CompileBackendGPU, self).__init__()
        self.hardware_type = "GPU"
        self.need_reload = False
        self.model_runtimes = []
        
        self.configs = None
        self.precision = 'fp32'

        self.packrunner = False 
        self.current_dir = os.path.split(os.path.abspath(__file__))[0]
        self._resnet = False

    def version(self) -> str:
        return "1.0.0"

    def get_interact_profile(self, config):
        model_profile = []
        file_path = os.path.join("general_perf/backends/GPU/", "interact_infos", config["model_info"]["model"] + '.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                model_profile = json.load(f)
        else:
            log.info(
                'File path: {} does not exist, please check'.format(file_path))

        return model_profile
    
    def pre_optimize(self, configs: Dict[str, Any]):
        model_name = configs["model_info"]["model"]
        model_path = configs["model_info"]["model_path"]
        model_format = configs["model_info"]["model_format"]
        # suffix = configs["model_info"]["model"].split("-")[1]

        if model_format != "onnx":
            model_dir, base_name = os.path.split(model_path)
            # model_name = "".join(base_name.strip("/").split(".")[:-1])
            optimized_model = os.path.join(model_dir, model_name.strip("/") + ".onnx")
            layout = configs["model_info"].get("layout", None)
            convert_param = {}
            
            if model_format in {"saved_model", "pb"} and layout == "NCHW":
                convert_param["inputs_as_nchw"] = configs["model_info"]["inputs"] 

            if not os.path.exists(optimized_model) or self._resnet:
                log.info("===== Convert model to onnx format".format(optimized_model))
                self.ConvertOnnxModel(configs, model_path, optimized_model, model_format=model_format, specify_params=convert_param)
            else:
                log.info("{} file exists, skip ONNX conversion".format(optimized_model))

            configs["model_info"]["model_path"] = optimized_model    
            configs["model_info"]["framework"] = "Onnx"
            configs["model_info"]["framework_version"] = "1.2.0"
            configs["model_info"]["model_format"] = "onnx"
        else:
            model_dir, _ = os.path.split(model_path)
            optimized_model = os.path.join(model_dir, model_name.strip("/") + ".onnx")
            extra_params = {}
            extra_params["input_shape"] = "/".join([str(v).replace(" ","") for _,v in 
                                                    configs["model_info"]["input_shape"].items()])
            self.optimizeModel(model_path, optimized_model, specify_params=extra_params)
            configs["model_info"]["model_path"] = optimized_model    
            configs["model_info"]["framework_version"] = "1.2.0"

        self.workload = configs['workload']
        self.model_info = configs['model_info']
        self.precision = configs["model_info"]["model_precision"]

        return configs


    def get_best_batch_size(self):
         return self.workload.get("batch_sizes", 1)


    def compile(self,configs: Dict[str, Any], dataloader=None):
        suffix = configs["model_info"]["model_format"]

        if self.precision.lower() == "fp16":
            model_dir, base_name = os.path.split(configs["model_info"]["model_path"])
            half_model = os.path.join(model_dir, base_name.strip("/").strip(".onnx") + f"_half.{suffix}")
            if not os.path.exists(half_model) or self._resnet:
                self.castHalfModel(configs["model_info"]["model_path"], half_model)
            configs['model_info']["model_path"] = half_model

        elif self.precision.lower() == "int8":
            cfg_path = os.path.join(self.current_dir, "cfg", configs["model_info"]["model"] + ".yaml")
            int8_model = self.quantizeteModel(cfg_path)
            configs['model_info']["model_path"] = int8_model


        result = {
            "model": 
                configs['model_info']['model'],
            "framework": 
                configs['model_info']['framework'],
            "compile_precision": 
                configs['model_info']['model_precision'],
            "input_type": 
                configs['model_info']['input_type'].split(","),
            "input_shape": 
                configs['model_info']['input_shape'],
            "max_batch_size": 
                configs['model_info']['max_batch_size'],
            "compile_status": "success",
            "sg_percent": 100,
            "segments": [
                {
                    "sg_idx": 0,
                    "is_fallback": False,
                    "input_tensor_map":
                        configs['model_info']['input_shape'],
                    "output_tensor_map":
                        configs['model_info']['outputs'],
                    "compiled_model": [
                        {
                            "compiled_bs": 1,
                            "compiled_obj": configs['model_info']['model_path'],
                        },
                    ],
                },
            ]
        }
        self.configs = result
        return result

    def tuning(self, configs: Dict[str, Any]):
        return
    
    @staticmethod
    def optimizeModel(model_path, half_model, specify_params=None, simplify_level=1):
        cmd = "python -m maca_converter --model_type onnx --model_path {} --output {} --simplify {}"\
              .format(model_path, half_model, simplify_level) 

        cmd_list = [s.strip() for s in cmd.split(" ")]
        if specify_params is not None:
            for k, v in specify_params.items():
                if f"--{k}" in cmd_list: 
                    flag_idx = cmd_list.index(f"--{k}")
                    print(f"Replace --{k}: {cmd_list[flag_idx + 1]} ===> {v}")
                    cmd_list[flag_idx + 1] = str(v)
                    cmd = " ".join(cmd_list)
                else:
                    cmd = cmd + f" --{k} {v}"
        log.info(f"=====Optimize Onnx Model")
        log.debug(f"=====Optimize Command line: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        if not os.path.exists(half_model) or not half_model.endswith(".onnx"):
            raise RuntimeWarning(f"Optimize onnx model failed,\nCommand line:{cmd}")


    @staticmethod
    def quantizeteModel(yaml_file: str, mode: str = "auto", run_cmd=False):
        with open(yaml_file, "r") as f:
            content = yaml.load(f, Loader=yaml.Loader)
            export_file = content["export_model"]
        
        if not os.path.exists(export_file):
            if run_cmd:
                quantize_cmd = "python -m maca_quantizer -c {} -m {}".format(yaml_file, mode)
                subprocess.run(quantize_cmd, shell=True, check=True)
            else:
                mxq = importlib.import_module(name='.maca_quantize_runner', package='maca_quantizer')
                obj = mxq.MacaQuantizeRunner(yaml_file, mode=mode)
                obj.run()
        return export_file


    @staticmethod
    def castHalfModel(model_path, half_model, specify_params=None, simplify_level=0, run_cmd=False):
        if run_cmd:
            cmd = "python -m maca_converter --model_type onnx --model_path {} --output {} --fp32_to_fp16 1 --simplify {}"\
                .format(model_path, half_model, simplify_level) 

            cmd_list = [s.strip() for s in cmd.split(" ")]
            if specify_params is not None:
                for k, v in specify_params.items():
                    if f"--{k}" in cmd_list: 
                        flag_idx = cmd_list.index(f"--{k}")
                        print(f"Replace --{k}: {cmd_list[flag_idx + 1]} ===> {v}")
                        cmd_list[flag_idx + 1] = str(v)
                        cmd = " ".join(cmd_list)
                    else:
                        cmd = cmd + f" --{k} {v}"
            log.info(f"===== Convert to half model")
            log.debug(f"Convert to half model cmd: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
        else:
            from maca_converter.float16 import convert_float_to_float16
            from maca_converter import operation 
            onnx_model = onnx.load(model_path)
            log.info('===== Begin model convert fp32-->fp16:')
            onnx_model_ = convert_float_to_float16(onnx_model, keep_io_types=True)
            delete = operation.eliminate_redundant_reshape(onnx_model_)
            while delete == True:
                delete = operation.eliminate_redundant_reshape(onnx_model_)   

            operation.eliminate_unused_input_initializer(onnx_model_)
            operation.eliminate_unused_constant_node(onnx_model_)
            operation.remove_unused_initializer(onnx_model_)
            onnx.save(onnx_model_, half_model)
            
        if not os.path.exists(half_model) or not half_model.endswith(".onnx"):
            raise RuntimeWarning(f"Convert fp16 onnx model failed,\nCommand line:{cmd}")
        

    def ConvertOnnxModel(self, configs, model_path, onnx_model, model_format, specify_params=None, simplify_level=1):
        if model_format in {"saved_model"}:
            model_type = "tf-sm"
            cmd = "python -m maca_converter  --model_type {}  --model_path  {}  --output {}  --simplify {}"\
                .format(model_type, model_path, onnx_model, simplify_level) 
            cmd_list = [s.strip() for s in cmd.split(" ")]
            if specify_params is not None:
                for k, v in specify_params.items():
                    if f"--{k}" in cmd_list: 
                        flag_idx = cmd_list.index(f"--{k}")
                        print(f"Replace --{k}: {cmd_list[flag_idx + 1]} ===> {v}")
                        cmd_list[flag_idx + 1] = str(v)
                        cmd = " ".join(cmd_list)
                    else:
                        cmd = cmd + f" --{k} {v}"
            log.debug(f"Convert Command line: {cmd}")
            subprocess.run(cmd, shell=True, check=True)

        elif model_format=="pt":
            torch_to_onnx.torch_to_onnx(model_path, onnx_model)

            extra_params = {}
            if configs["model_info"]["model"].startswith("swin-large"):
                # TODO: BLACK rasie error 
                extra_params["input_shape"] = "/".join([str(v).replace(" ","") for _,v in configs["model_info"]["input_shape"].items()])

            self.optimizeModel(onnx_model, onnx_model, specify_params=extra_params)

        if not os.path.exists(onnx_model) or not onnx_model.endswith(".onnx"):
            raise RuntimeWarning(f"Convert onnx model failed,\nCommand line:{cmd}")
    


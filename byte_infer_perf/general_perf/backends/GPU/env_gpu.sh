MACA_ROOT_PATH=${1}

if [ ! -n "$MACA_ROOT_PATH" ]; then
    # echo "MACA_ROOT_PATH IS  NULL"
    MACA_ROOT_PATH=/opt/maca
fi
echo "MACA_PATH: ${MACA_ROOT_PATH}"

export MACA_PATH=${MACA_ROOT_PATH}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge

export DEVINFO_ROOT=${MACA_PATH}
export CUDA_PATH=${MACA_PATH}/tools/wcuda
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin/
export LD_LIBRARY_PATH=${MACA_PATH}/lib/:${MACA_PATH}/mxgpu_llvm/lib/:${LD_LIBRARY_PATH}
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}

export ISU_FASTMODEL=1  # must be set, otherwise may induce precision error
export USE_TDUMP=OFF    # optional, use to control whether generating debug file
export TMEM_LOG=OFF     # optional, use to control whether generating debug file
export DEBUG_ITRACE=0   # optional, use to control whether generating debug file

# export MACA_QUANTIZER_USING_MXGPU=0   # 0:Disable  1:Enable(default)
# export MACA_LAUNCH_BLOCKING=1
# export MXLOG_LEVEL=debug
# export MCDNN_LOG_ENABLE=1
# export MCDNN_LOG_LEVEL=5


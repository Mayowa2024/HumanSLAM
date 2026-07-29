# HumanSLAM GPU OCR environment

HumanSLAM uses PyTorch/TensorRT and PaddleOCR in one Python process. PyTorch
2.8.0+cu129 and PaddlePaddle 3.2.0 pin different patch releases of the NVIDIA
CUDA Python runtime packages even though the newer CUDA 12.9 libraries work
with both frameworks.

For this machine, keep PyTorch's CUDA 12.9 dependencies and install Paddle
without allowing pip to downgrade them:

```bash
python3 -m pip uninstall -y paddlepaddle
python3 -m pip install --user \
  paddlepaddle-gpu==3.2.0 \
  --no-deps \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python3 -m pip install --user nvidia-nvtx-cu11==11.8.86
```

The legacy NVTX package supplies `libnvToolsExt.so.1`, which Paddle's binary
requires. It does not replace the CUDA 12.9 compute libraries used by PyTorch.

Verify both frameworks in the same process:

```bash
python3 - <<'PY'
import torch
import paddle

print(torch.__version__, torch.cuda.is_available())
print(paddle.__version__, paddle.device.is_compiled_with_cuda())
print(torch.cuda.get_device_name(0))
print(paddle.device.get_device())
PY
```

Pip may report Paddle metadata conflicts because it asks for exact CUDA 12.6
patch versions. The runtime combination above was tested on this machine with
an NVIDIA 575.57.08 driver: PyTorch, PaddlePaddle, Places365 TensorRT, YOLO
TensorRT, and PaddleOCR all executed successfully on the RTX 4070.

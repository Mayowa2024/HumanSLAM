# Model weights

Runtime model binaries are intentionally not stored in the Git repository.
TensorRT engines are hardware/runtime-specific, and the current scene engine
also exceeds GitHub's normal per-file limit.

Place these files in this directory before running HumanSLAM:

| File | SHA-256 of the currently tested local model |
|---|---|
| `resnet50_places365_embed.engine` | `49a1a7ff933df4327e22f62e7a954af1910cca6a24c16fa9f806b1601d0c478b` |
| `resnet50_places365_embed.onnx` | `4b580071b0a318c305b57258889050d3016269bd5339edd461482078c652efb6` |
| `yolo26n-seg.engine` | `96d46d253d7cb060164076f7ccf8db4ee452566e9a37eb3888260922daba8495` |

The ONNX file is an export/rebuild artefact and is not loaded during normal
runtime. Use `tools/build_places365_embedding_engine.py` to build a compatible
scene TensorRT engine on the target machine.

Model licences and redistribution terms remain those of their respective model
authors. Do not publish third-party weights without checking those terms.

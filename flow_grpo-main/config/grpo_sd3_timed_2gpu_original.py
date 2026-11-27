"""Keep original general_ocr_sd3_1gpu hyperparameters on two GPUs.

World size changes: 128 images / 16 prompt groups per epoch instead of
64 images / 8 groups. This is not a globally batch-equivalent reproduction.
"""
import importlib.util
import os
from pathlib import Path


def get_config():
    path = Path(__file__).with_name("grpo.py")
    spec = importlib.util.spec_from_file_location("sd3_original_hyperparameters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.general_ocr_sd3_1gpu()
    # Recompute activations in backward without changing batch or objective.
    config.activation_checkpointing = True
    # Model location and output identification.
    config.pretrained.model = os.environ.get("SD3_MODEL_PATH", config.pretrained.model)
    config.run_name = "sd3_ocr_timed_2gpu_original"
    config.save_dir = "logs/ocr/sd3_timed_2gpu_original"
    return config

"""Single-GPU SD3 OCR timing smoke-run configuration."""
import importlib.util
from pathlib import Path


def get_config():
    path = Path(__file__).with_name("grpo_sd3_timed_2gpu.py")
    spec = importlib.util.spec_from_file_location("sd3_timing_two_gpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config()
    config.run_name = "sd3_ocr_timed_1gpu"
    # A group must fit within world_size * sampling batch.
    config.sample.train_batch_size = 2
    config.train.batch_size = 2
    config.sample.num_image_per_prompt = 2
    config.sample.test_batch_size = 1
    config.resolution = 384
    # Original reference-forward code uses transformer.module, which is absent
    # without DDP. Disable KL for this diagnostic configuration.
    config.train.beta = 0.0
    config.save_dir = "logs/ocr/sd3_timed_1gpu"
    assert config.sample.train_batch_size % config.sample.num_image_per_prompt == 0
    return config

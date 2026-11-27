"""Small-batch SD3 OCR timing configuration for two 24GB GPUs."""
import importlib.util
import os
from pathlib import Path


def get_config():
    path = Path(__file__).with_name("grpo.py")
    spec = importlib.util.spec_from_file_location("sd3_timing_base_grpo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.general_ocr_sd3_1gpu()
    config.run_name = "sd3_ocr_timed_2gpu"
    config.pretrained.model = os.environ.get("SD3_MODEL_PATH", config.pretrained.model)
    config.use_lora = True
    config.mixed_precision = "fp16"
    config.resolution = 512
    # Sampler requires (world_size * per_rank_batch) % group_size == 0.
    config.sample.train_batch_size = 1
    config.train.batch_size = 1
    config.sample.num_image_per_prompt = 2
    # Eight prompt groups per epoch, with smaller groups than the original.
    config.sample.num_batches_per_epoch = 8
    config.train.gradient_accumulation_steps = 4
    config.sample.test_batch_size = 1
    config.save_dir = "logs/ocr/sd3_timed_2gpu"
    assert 2 * config.sample.train_batch_size % config.sample.num_image_per_prompt == 0
    return config

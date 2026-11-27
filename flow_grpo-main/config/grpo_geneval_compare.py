"""Low-GPU FlowGRPO configs for comparison with iGRPO on GenEval.

Both variants use a global group of eight images and collect eight images per
outer iteration. Existing FlowGRPO configs are left untouched.
"""

import imp
import os

grpo = imp.load_source("flowgrpo_original_config", os.path.join(os.path.dirname(__file__), "grpo.py"))


def _comparison(gpu_count: int):
    if gpu_count not in (1, 2):
        raise ValueError("comparison configs support one or two GPUs")
    config = grpo.geneval_sd3()
    per_gpu_batch = 8 // gpu_count
    config.run_name = f"compare_geneval_flowgrpo_{gpu_count}gpu"
    config.sample.train_batch_size = per_gpu_batch
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 4 if gpu_count == 1 else 2
    config.train.batch_size = per_gpu_batch
    config.train.gradient_accumulation_steps = 1
    config.num_epochs = 2500
    config.eval_freq = 50
    config.save_freq = 50
    config.save_dir = os.path.join("logs", "compare_geneval", f"flowgrpo_{gpu_count}gpu")
    return config


def geneval_1gpu():
    return _comparison(1)


def geneval_2gpu():
    return _comparison(2)


def get_config(name):
    variants = {
        "geneval_1gpu": geneval_1gpu,
        "geneval_2gpu": geneval_2gpu,
    }
    if name not in variants:
        raise ValueError(f"unknown FlowGRPO GenEval comparison config: {name}")
    return variants[name]()

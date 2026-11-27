"""Standalone iGRPO experiment configs; existing FlowGRPO configs are untouched."""

import imp
import os

base = imp.load_source("igrpo_base", os.path.join(os.path.dirname(__file__), "base.py"))


def _common(dataset_name: str, reward_name: str):
    config = base.get_config()
    config.method = "igrpo"
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset", dataset_name)
    config.resolution = 512
    config.use_lora = True
    config.mixed_precision = "fp16"

    # Paper values (Sec. 4): 24 samples/prompt for SD3, AdamW 1e-5,
    # weight decay 1e-4, <=2500 iterations. Per-GPU batch was 3 on 32 H100s.
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 1
    config.sample.eval_num_steps = 40
    config.sample.eval_guidance_scale = 4.5
    config.train.batch_size = 3
    config.train.gradient_accumulation_steps = 1
    config.train.learning_rate = 1e-5
    config.train.adam_weight_decay = 1e-4
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-4
    config.train.beta = 0.0
    config.train.ema = False
    config.num_epochs = 2500

    # Not disclosed in the paper. These are explicit reproduction defaults,
    # not claimed paper hyperparameters.
    config.igrpo = {
        "exploration": 0.2,
        "min_t": 0.05,
        "max_t": 0.95,
        "min_delta_t": 0.025,
        "max_delta_t": 0.10,
        "time_sampling": "uniform",
        "reference": "base",
        "log_prob_reduction": "mean",
    }
    config.reward_fn = {reward_name: 1.0}
    config.prompt_fn = "geneval" if dataset_name == "geneval" else "general_ocr"
    config.save_dir = os.path.join("logs", "igrpo", dataset_name)
    config.run_name = f"igrpo_sd35m_{dataset_name}"
    return config


def geneval_reward():
    """Paper configuration (2): SD3.5-M optimized with GenEval reward."""
    return _common("geneval", "geneval")


def clipscore_reward():
    """Paper configuration (1): Pick-a-Pic prompts optimized with CLIPScore."""
    return _common("pickscore", "clipscore")


def _geneval_comparison(gpu_count: int):
    """Match the low-GPU FlowGRPO comparison's global group/sample count."""
    if gpu_count not in (1, 2):
        raise ValueError("comparison configs support one or two GPUs")
    config = _common("geneval", "geneval")
    per_gpu_batch = 8 // gpu_count
    config.run_name = f"compare_geneval_igrpo_{gpu_count}gpu"
    config.sample.train_batch_size = per_gpu_batch
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 4 if gpu_count == 1 else 2
    config.train.batch_size = per_gpu_batch
    config.train.gradient_accumulation_steps = 1
    config.eval_freq = 50
    config.save_freq = 50
    config.save_dir = os.path.join("logs", "compare_geneval", f"igrpo_{gpu_count}gpu")
    return config


def geneval_1gpu():
    return _geneval_comparison(1)


def geneval_2gpu():
    return _geneval_comparison(2)


def get_config(name):
    variants = {
        "geneval_reward": geneval_reward,
        "clipscore_reward": clipscore_reward,
        "geneval_1gpu": geneval_1gpu,
        "geneval_2gpu": geneval_2gpu,
    }
    if name not in variants:
        raise ValueError(f"unknown iGRPO config: {name}")
    return variants[name]()

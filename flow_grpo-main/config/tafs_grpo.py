"""TAFS-GRPO configs isolated from FlowGRPO and iGRPO."""

import imp
import os

base = imp.load_source("tafs_base", os.path.join(os.path.dirname(__file__), "base.py"))


def sd35m_geneval_available_rewards():
    """Runnable subset using reward implementations already in this repo."""
    config = base.get_config()
    config.method = "tafs_grpo"
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset", "geneval")
    config.resolution = 512
    config.use_lora = True
    config.mixed_precision = "fp16"
    config.num_epochs = 1500
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 48
    config.sample.num_batches_per_epoch = 2
    config.sample.test_batch_size = 3
    config.train.batch_size = 3
    # The trainer collates both 24-trajectory distributed micro-batches before
    # one update, yielding the paper's effective group/update size of 48.
    config.train.gradient_accumulation_steps = 1
    config.train.learning_rate = 3e-4
    config.train.adam_weight_decay = 1e-4
    config.train.clip_range = 1e-4
    config.train.beta = 0.04
    config.train.num_inner_epochs = 1
    config.train.ema = False
    config.tafs = {
        "num_predictions": 4,
        "sigmas": (1.0, 0.75, 0.5, 0.25),
        "velocity_convention": "diffusers",
        "reference": "base",
    }
    # The paper equally combines GenEval, PickScore, CLIPScore and HPS-v2.1.
    # HPS-v2.1 is not implemented in this repository, so this runnable config
    # explicitly uses the other three rather than silently claiming exactness.
    config.reward_fn = {"geneval": 1 / 3, "pickscore": 1 / 3, "clipscore": 1 / 3}
    config.prompt_fn = "geneval"
    config.save_freq = 50
    config.eval_freq = 50
    config.save_dir = os.path.join("logs", "tafs_grpo", "sd35m_geneval")
    config.run_name = "tafs_grpo_sd35m_geneval"
    return config


def get_config(name):
    variants = {"sd35m_geneval_available_rewards": sd35m_geneval_available_rewards}
    if name not in variants:
        raise ValueError(f"unknown TAFS-GRPO config: {name}")
    return variants[name]()

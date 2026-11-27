"""Isolated SD3.5-M TAFS-GRPO trainer for arXiv:2602.01591v2."""

import contextlib
import hashlib
import json
import os
import random
import sys

from absl import app, flags
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableDiffusion3Pipeline
from ml_collections import config_flags
from peft import LoraConfig, PeftModel, get_peft_model
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import flow_grpo.rewards
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from tafs_grpo.core import (
    anneal_transition,
    clean_prediction,
    gaussian_kl_same_variance,
    stage_group_advantages,
    stage_grpo_loss,
)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/tafs_grpo.py", "TAFS-GRPO config")


def load_prompts(root):
    path = os.path.join(root, "train_metadata.jsonl")
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return rows


def prompt_group_id(prompt):
    return int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "big") % (2**63 - 1)


def encode(prompts, pipeline, device):
    with torch.no_grad():
        positive, pooled = encode_prompt(
            [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3],
            [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3],
            prompts,
            128,
        )
        negative, negative_pooled = encode_prompt(
            [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3],
            [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3],
            [""] * len(prompts),
            128,
        )
    return tuple(value.to(device) for value in (positive, pooled, negative, negative_pooled))


def velocity(transformer, state, sigma, positive, pooled, negative, negative_pooled, guidance):
    timestep = torch.as_tensor(sigma, device=state.device, dtype=torch.float32)
    if timestep.ndim == 0:
        timestep = timestep.expand(state.shape[0])
    timestep = timestep * 1000.0
    if guidance > 1:
        output = transformer(
            hidden_states=torch.cat([state, state]).to(positive.dtype),
            timestep=torch.cat([timestep, timestep]),
            encoder_hidden_states=torch.cat([negative, positive]),
            pooled_projections=torch.cat([negative_pooled, pooled]),
            return_dict=False,
        )[0]
        unconditional, conditional = output.chunk(2)
        return unconditional + guidance * (conditional - unconditional)
    return transformer(
        hidden_states=state.to(positive.dtype), timestep=timestep,
        encoder_hidden_states=positive, pooled_projections=pooled, return_dict=False,
    )[0]


@torch.no_grad()
def decode(pipeline, latent):
    latent = latent / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
    image = pipeline.vae.decode(latent.to(pipeline.vae.dtype), return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type="pt")


def adapter_disabled(model):
    return model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()


def save_model(transformer, accelerator, root, step):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        target = os.path.join(root, "checkpoints", f"checkpoint-{step}", "lora")
        os.makedirs(target, exist_ok=True)
        accelerator.unwrap_model(transformer).save_pretrained(target)


def main(_):
    config = FLAGS.config
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        project_config=ProjectConfiguration(project_dir=config.save_dir, total_limit=config.num_checkpoint_limit),
    )
    set_seed(config.seed, device_specific=True)
    device = accelerator.device
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False).to(device, dtype=torch.float32)
    for text_encoder in (pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3):
        text_encoder.requires_grad_(False).to(device)
    pipeline.transformer.requires_grad_(not config.use_lora).to(device)
    if config.use_lora:
        targets = [
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
        ]
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, config.train.lora_path, is_trainable=True
            )
        else:
            pipeline.transformer = get_peft_model(
                pipeline.transformer,
                LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=targets),
            )
    transformer = pipeline.transformer
    parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        eps=config.train.adam_epsilon, weight_decay=config.train.adam_weight_decay,
    )
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    rewards_fn = flow_grpo.rewards.multi_score(device, config.reward_fn)
    rows = load_prompts(config.dataset)
    rng = random.Random(config.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    sigmas = tuple(float(value) for value in config.tafs.sigmas)
    if len(sigmas) != config.tafs.num_predictions or sigmas[0] != 1.0:
        raise ValueError("TAFS expects num_predictions sigmas beginning at 1.0")
    if not all(left > right > 0 for left, right in zip(sigmas, sigmas[1:])):
        raise ValueError("TAFS training sigmas must be strictly decreasing and positive")
    local_trajectories = config.sample.train_batch_size * config.sample.num_batches_per_epoch
    global_trajectories = local_trajectories * accelerator.num_processes
    if global_trajectories != config.sample.num_image_per_prompt:
        raise ValueError(
            f"expected global group {config.sample.num_image_per_prompt}, got {global_trajectories}; "
            "paper config requires 8 GPUs"
        )
    reward_stages = len(sigmas) - 1
    latent_side = config.resolution // pipeline.vae_scale_factor
    channels = accelerator.unwrap_model(transformer).config.in_channels
    os.makedirs(config.save_dir, exist_ok=True)

    for step in range(config.num_epochs):
        row = rows[order[step % len(order)]]
        prompt_text = row["prompt"]
        prompts = [prompt_text] * local_trajectories
        metadata = [row] * local_trajectories
        positive, pooled, negative, negative_pooled = encode(prompts, pipeline, device)
        guidance = config.sample.guidance_scale
        state = torch.randn(
            local_trajectories, channels, latent_side, latent_side,
            device=device, dtype=torch.float32,
        )
        source_states, source_sigmas, actions, old_log_probs, stage_rewards = [], [], [], [], []
        transformer.eval()
        with torch.no_grad(), accelerator.autocast():
            first_output = velocity(
                transformer, state, sigmas[0], positive, pooled, negative, negative_pooled, guidance
            )
            clean = clean_prediction(state, first_output, sigmas[0], config.tafs.velocity_convention)
            source_state, source_sigma = state, sigmas[0]
            for target_sigma in sigmas[1:]:
                transition = anneal_transition(clean, target_sigma)
                next_output = velocity(
                    transformer, transition.state, target_sigma,
                    positive, pooled, negative, negative_pooled, guidance,
                )
                next_clean = clean_prediction(
                    transition.state, next_output, target_sigma, config.tafs.velocity_convention
                )
                images = decode(pipeline, next_clean)
                reward_dict, _ = rewards_fn(images, prompts, metadata, only_strict=True)
                stage_rewards.append(torch.as_tensor(reward_dict["avg"], device=device).float())
                source_states.append(source_state.detach())
                source_sigmas.append(torch.full((local_trajectories,), source_sigma, device=device))
                actions.append(transition.state.detach())
                old_log_probs.append(transition.log_prob.detach())
                source_state, source_sigma, clean = transition.state, target_sigma, next_clean

        rewards = torch.stack(stage_rewards, dim=1)
        group_ids = torch.full(
            (local_trajectories,), prompt_group_id(prompt_text), device=device, dtype=torch.long
        )
        gathered_rewards = accelerator.gather(rewards)
        gathered_groups = accelerator.gather(group_ids)
        gathered_advantages = stage_group_advantages(gathered_rewards, gathered_groups)
        start = accelerator.process_index * local_trajectories
        advantages = gathered_advantages[start:start + local_trajectories].to(device)
        source_states = torch.stack(source_states, dim=1)
        source_sigmas = torch.stack(source_sigmas, dim=1)
        actions = torch.stack(actions, dim=1)
        old_log_probs = torch.stack(old_log_probs, dim=1)

        transformer.train()
        with accelerator.accumulate(transformer), accelerator.autocast():
            new_log_probs, new_means, reference_means = [], [], []
            for stage, target_sigma in enumerate(sigmas[1:]):
                output = velocity(
                    transformer, source_states[:, stage], source_sigmas[:, stage],
                    positive, pooled, negative, negative_pooled, guidance,
                )
                predicted_clean = clean_prediction(
                    source_states[:, stage], output, source_sigmas[:, stage],
                    config.tafs.velocity_convention,
                )
                rescored = anneal_transition(predicted_clean, target_sigma, state=actions[:, stage])
                new_log_probs.append(rescored.log_prob)
                new_means.append(rescored.mean)
                if config.train.beta > 0:
                    unwrapped = accelerator.unwrap_model(transformer)
                    with torch.no_grad(), adapter_disabled(unwrapped):
                        reference_output = velocity(
                            transformer, source_states[:, stage], source_sigmas[:, stage],
                            positive, pooled, negative, negative_pooled, guidance,
                        )
                        reference_clean = clean_prediction(
                            source_states[:, stage], reference_output, source_sigmas[:, stage],
                            config.tafs.velocity_convention,
                        )
                        reference_means.append((1 - target_sigma) * reference_clean)
            new_log_probs = torch.stack(new_log_probs, dim=1)
            policy_loss, metrics = stage_grpo_loss(
                new_log_probs, old_log_probs,
                advantages.clamp(-config.train.adv_clip_max, config.train.adv_clip_max),
                config.train.clip_range,
            )
            loss = policy_loss
            if config.train.beta > 0:
                kl_terms = []
                for stage, target_sigma in enumerate(sigmas[1:]):
                    kl_terms.append(gaussian_kl_same_variance(
                        new_means[stage], reference_means[stage],
                        torch.tensor(target_sigma, device=device),
                    ))
                kl = torch.stack(kl_terms, dim=1).mean()
                loss = loss + config.train.beta * kl
                metrics["reference_kl"] = kl.detach()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(parameters, config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        mean_reward = accelerator.reduce(rewards.mean(), reduction="mean")
        if accelerator.is_main_process and (step % 10 == 0 or step + 1 == config.num_epochs):
            detail = " ".join(f"{key}={value.item():.6f}" for key, value in metrics.items())
            print(f"step={step + 1} reward={mean_reward.item():.6f} loss={loss.item():.6f} {detail}")
        if config.save_freq > 0 and (step + 1) % config.save_freq == 0:
            save_model(transformer, accelerator, config.save_dir, step + 1)

    save_model(transformer, accelerator, config.save_dir, config.num_epochs)


if __name__ == "__main__":
    app.run(main)

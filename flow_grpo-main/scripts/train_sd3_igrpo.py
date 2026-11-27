"""Independent SD3.5-M iGRPO trainer (Ren et al., ICML 2026).

Unlike ``train_sd3.py``, collection is a one-step Gaussian action followed by
a no-grad collapse to x0 and immediate reward. Evaluation still uses the
standard multi-step pipeline. See ``igrpo/README.md`` for disclosed/assumed
hyperparameters and known paper ambiguities.
"""

import contextlib
import hashlib
import json
import os
import sys
from collections import defaultdict

# Make the repository-local ``igrpo`` package importable for direct launches.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from absl import app, flags
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableDiffusion3Pipeline
from ml_collections import config_flags
from peft import LoraConfig, PeftModel, get_peft_model
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

import flow_grpo.rewards
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from igrpo.core import (
    collapse_to_x0,
    gaussian_kl_same_variance,
    group_advantages,
    instant_transition,
    ppo_grpo_loss,
)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/igrpo.py", "iGRPO configuration")


class PromptDataset(Dataset):
    def __init__(self, root, split):
        jsonl = os.path.join(root, f"{split}_metadata.jsonl")
        text = os.path.join(root, f"{split}.txt")
        if os.path.exists(jsonl):
            with open(jsonl, encoding="utf-8") as handle:
                self.rows = [json.loads(line) for line in handle]
        elif os.path.exists(text):
            with open(text, encoding="utf-8") as handle:
                self.rows = [{"prompt": line.strip()} for line in handle if line.strip()]
        else:
            raise FileNotFoundError(f"no {split}.txt or {split}_metadata.jsonl in {root}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return row["prompt"], row


class DistributedGroupSampler(Sampler):
    """Distribute complete prompt groups across ranks without local duplication."""
    def __init__(self, dataset, per_rank_batch, group_size, replicas, rank, seed):
        total = per_rank_batch * replicas
        if total % group_size:
            raise ValueError(
                f"world_size*train_batch_size ({total}) must be divisible by group_size ({group_size})"
            )
        self.dataset = dataset
        self.per_rank_batch = per_rank_batch
        self.group_size = group_size
        self.replicas = replicas
        self.rank = rank
        self.seed = seed

    def __iter__(self):
        epoch = 0
        while True:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            count = self.per_rank_batch * self.replicas // self.group_size
            chosen = torch.randperm(len(self.dataset), generator=generator)[:count].tolist()
            all_indices = [index for index in chosen for _ in range(self.group_size)]
            start = self.rank * self.per_rank_batch
            yield all_indices[start : start + self.per_rank_batch]
            epoch += 1


def collate(examples):
    prompts, metadata = zip(*examples)
    return list(prompts), list(metadata)


def encode(prompts, pipeline, device):
    with torch.no_grad():
        prompt, pooled = encode_prompt(
            [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3],
            [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3],
            prompts,
            128,
        )
    return prompt.to(device), pooled.to(device)


def model_velocity(transformer, latents, normalized_t, prompt, pooled):
    # SD3's transformer consumes the scheduler convention (1000 -> 0), while
    # equations in the paper use normalized rectified-flow time (1 -> 0).
    timestep = normalized_t.float() * 1000.0
    return transformer(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=prompt,
        pooled_projections=pooled,
        return_dict=False,
    )[0]


def decode_latents(pipeline, latents):
    value = latents / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
    value = pipeline.vae.decode(value.to(pipeline.vae.dtype), return_dict=False)[0]
    return pipeline.image_processor.postprocess(value, output_type="pt")


def prompt_group_id(prompt):
    return int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)


def adapter_disabled(model):
    return model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()


def save_lora(transformer, accelerator, path, step):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        target = os.path.join(path, f"checkpoint-{step}")
        os.makedirs(target, exist_ok=True)
        accelerator.unwrap_model(transformer).save_pretrained(target)


def main(_):
    config = FLAGS.config
    project = ProjectConfiguration(project_dir=config.save_dir, total_limit=config.num_checkpoint_limit)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        project_config=project,
    )
    set_seed(config.seed, device_specific=True)
    device = accelerator.device

    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False).to(device, dtype=torch.float32)
    for encoder in (pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3):
        encoder.requires_grad_(False).to(device)
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
        parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        eps=config.train.adam_epsilon,
        weight_decay=config.train.adam_weight_decay,
    )
    dataset = PromptDataset(config.dataset, "train")
    sampler = DistributedGroupSampler(
        dataset, config.sample.train_batch_size, config.sample.num_image_per_prompt,
        accelerator.num_processes, accelerator.process_index, config.seed,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=0)
    # The sampler is already rank-aware; preparing the DataLoader would shard
    # it a second time. Only models/optimizer go through Accelerate here.
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    reward_fn = flow_grpo.rewards.multi_score(device, config.reward_fn)
    autocast = accelerator.autocast
    data_iterator = iter(loader)

    if accelerator.is_main_process:
        os.makedirs(config.save_dir, exist_ok=True)
        print("iGRPO reproduction settings:", config.igrpo)

    for step in range(config.num_epochs):
        try:
            prompts, metadata = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            prompts, metadata = next(data_iterator)
        prompt_embeds, pooled = encode(prompts, pipeline, device)
        batch = len(prompts)
        group_ids = torch.tensor([prompt_group_id(prompt) for prompt in prompts], device=device)

        channels = transformer.module.config.in_channels if hasattr(transformer, "module") else transformer.config.in_channels
        latent_side = config.resolution // pipeline.vae_scale_factor
        x_t = torch.randn(batch, channels, latent_side, latent_side, device=device, dtype=torch.float32)
        settings = config.igrpo
        t = torch.empty(batch, device=device).uniform_(settings.min_t, settings.max_t)
        max_dt = torch.minimum(torch.full_like(t, settings.max_delta_t), t)
        min_dt = torch.minimum(torch.full_like(t, settings.min_delta_t), max_dt)
        delta_t = min_dt + torch.rand_like(t) * (max_dt - min_dt)

        transformer.eval()
        with torch.no_grad(), autocast():
            velocity = model_velocity(transformer, x_t, t, prompt_embeds, pooled)
            transition = instant_transition(x_t, velocity, t, delta_t, settings.exploration)
            t_after = t - delta_t
            collapse_velocity = model_velocity(
                transformer, transition.action, t_after, prompt_embeds, pooled
            )
            x0 = collapse_to_x0(transition.action, collapse_velocity, t_after)
            images = decode_latents(pipeline, x0)
        rewards, reward_metadata = reward_fn(images, prompts, metadata, only_strict=True)
        reward = torch.as_tensor(rewards["avg"], device=device, dtype=torch.float32)

        # Groups may span ranks in other launch layouts, so normalize globally.
        all_reward = accelerator.gather(reward)
        all_group = accelerator.gather(group_ids)
        all_advantage = group_advantages(all_reward, all_group)
        rank_start = accelerator.process_index * reward.numel()
        advantage = all_advantage[rank_start : rank_start + reward.numel()].to(device)

        collected = {
            "x_t": x_t.detach(), "action": transition.action.detach(), "t": t.detach(),
            "delta_t": delta_t.detach(), "old_log_prob": transition.log_prob.detach(),
            "prompt": prompt_embeds.detach(), "pooled": pooled.detach(), "advantage": advantage.detach(),
        }

        transformer.train()
        metrics = defaultdict(list)
        for _inner in range(config.train.num_inner_epochs):
            with accelerator.accumulate(transformer), autocast():
                new_velocity = model_velocity(
                    transformer, collected["x_t"], collected["t"], collected["prompt"], collected["pooled"]
                )
                rescored = instant_transition(
                    collected["x_t"], new_velocity, collected["t"], collected["delta_t"],
                    settings.exploration, action=collected["action"],
                )
                policy_loss, policy_metrics = ppo_grpo_loss(
                    rescored.log_prob, collected["old_log_prob"],
                    collected["advantage"].clamp(-config.train.adv_clip_max, config.train.adv_clip_max),
                    config.train.clip_range,
                )
                loss = policy_loss
                if config.train.beta > 0:
                    unwrapped = accelerator.unwrap_model(transformer)
                    with torch.no_grad(), adapter_disabled(unwrapped):
                        reference_velocity = model_velocity(
                            transformer, collected["x_t"], collected["t"], collected["prompt"], collected["pooled"]
                        )
                        reference = instant_transition(
                            collected["x_t"], reference_velocity, collected["t"], collected["delta_t"],
                            settings.exploration, action=collected["action"],
                        )
                    kl = gaussian_kl_same_variance(rescored.mean, reference.mean, rescored.std).mean()
                    loss = loss + config.train.beta * kl
                    metrics["kl"].append(kl.detach())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        reduced_reward = accelerator.reduce(reward.mean(), reduction="mean")
        if accelerator.is_main_process and (step % 10 == 0 or step + 1 == config.num_epochs):
            print(
                f"step={step + 1} reward={reduced_reward.item():.6f} "
                f"loss={loss.detach().item():.6f} ratio={policy_metrics['ratio_mean'].item():.6f}"
            )
        if config.save_freq > 0 and (step + 1) % config.save_freq == 0:
            save_lora(transformer, accelerator, config.save_dir, step + 1)

    save_lora(transformer, accelerator, config.save_dir, config.num_epochs)


if __name__ == "__main__":
    app.run(main)

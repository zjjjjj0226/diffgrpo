"""Probe whether iGRPO post-action collapses provide meaningful CLIP rewards.

The probe first follows one deterministic SD3 ODE trajectory, so every x_t is
a reachable intermediate state. For each selected time it samples four iGRPO
actions, collapses each action to x0, decodes it, and measures CLIPScore.
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

# Support direct execution via ``python scripts/probe_igrpo_instant_reward.py``
# even when this checkout has not been installed with ``pip install -e .``.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffusers import StableDiffusion3Pipeline
from PIL import Image, ImageDraw, ImageFont
import torch

from flow_grpo.clip_scorer import ClipScorer
from igrpo.core import collapse_to_x0, instant_transition


@dataclass
class ProbeRecord:
    candidate: int
    requested_t: float
    actual_t: float
    delta_t: float
    t_after: float
    clipscore: float
    image: str


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--output-dir", default="logs/igrpo/instant_reward_probe")
    parser.add_argument("--times", type=float, nargs="+", default=[0.8, 0.6, 0.4, 0.2])
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, default=40)
    parser.add_argument("--delta-t", type=float, default=None,
                        help="Single-step size; default is 1/ode_steps.")
    parser.add_argument("--exploration", type=float, default=0.2,
                        help="Paper coefficient a; the paper does not disclose its value.")
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14",
                        help="Hugging Face ID or a fully downloaded local CLIP directory.")
    parser.add_argument("--clip-local-files-only", action="store_true",
                        help="Never contact the Hub when loading CLIP.")
    parser.add_argument("--require-clipscore", action="store_true",
                        help="Abort instead of saving unscored images if CLIP cannot load.")
    return parser.parse_args()


def expand_batch(value, count):
    if value is None:
        return None
    return value.expand(count, *value.shape[1:]).contiguous()


def predict_velocity(transformer, latent, timestep, prompt, pooled, neg_prompt, neg_pooled, guidance):
    batch = latent.shape[0]
    timestep = torch.as_tensor(timestep, device=latent.device, dtype=torch.float32)
    if timestep.ndim == 0:
        timestep = timestep.expand(batch)
    if guidance > 1.0:
        model_latent = torch.cat([latent, latent], dim=0).to(prompt.dtype)
        model_t = torch.cat([timestep, timestep], dim=0)
        model_prompt = torch.cat([neg_prompt, prompt], dim=0)
        model_pooled = torch.cat([neg_pooled, pooled], dim=0)
        output = transformer(
            hidden_states=model_latent,
            timestep=model_t,
            encoder_hidden_states=model_prompt,
            pooled_projections=model_pooled,
            return_dict=False,
        )[0]
        unconditional, conditional = output.chunk(2)
        return unconditional + guidance * (conditional - unconditional)
    return transformer(
        hidden_states=latent.to(prompt.dtype),
        timestep=timestep,
        encoder_hidden_states=prompt,
        pooled_projections=pooled,
        return_dict=False,
    )[0]


@torch.no_grad()
def collect_reachable_states(pipeline, prompt, args, generator):
    """Collect nearest requested t values along a deterministic Euler ODE path."""
    prompt_embed, negative_embed, pooled, negative_pooled = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt="",
        negative_prompt_2=None,
        negative_prompt_3=None,
        do_classifier_free_guidance=args.guidance_scale > 1.0,
        device=args.device,
        num_images_per_prompt=1,
        max_sequence_length=128,
    )
    pipeline.scheduler.set_timesteps(args.ode_steps, device=args.device)
    latent = pipeline.prepare_latents(
        1,
        pipeline.transformer.config.in_channels,
        args.resolution,
        args.resolution,
        prompt_embed.dtype,
        args.device,
        generator,
        None,
    ).float()
    sigmas = pipeline.scheduler.sigmas[:-1].detach().float().cpu()
    chosen = {
        requested: int(torch.argmin(torch.abs(sigmas - requested)).item())
        for requested in args.times
    }
    states = {}
    for index, timestep in enumerate(pipeline.scheduler.timesteps):
        for requested, target_index in chosen.items():
            if index == target_index:
                states[requested] = {
                    "latent": latent.detach().clone(),
                    "actual_t": float(sigmas[index]),
                    "model_timestep": float(timestep),
                }
        velocity = predict_velocity(
            pipeline.transformer, latent, timestep, prompt_embed, pooled,
            negative_embed, negative_pooled, args.guidance_scale,
        )
        latent = pipeline.scheduler.step(velocity, timestep, latent, return_dict=False)[0].float()
    return states, (prompt_embed, negative_embed, pooled, negative_pooled)


@torch.no_grad()
def decode(pipeline, latents):
    latents = latents / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
    images = pipeline.vae.decode(latents.to(pipeline.vae.dtype), return_dict=False)[0]
    return pipeline.image_processor.postprocess(images, output_type="pt")


def tensor_to_pil(image):
    array = (image.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round()
    return Image.fromarray(array.astype("uint8"))


def make_grid(images, records, times, candidates, resolution, output):
    header = 44
    label_width = 88
    grid = Image.new("RGB", (label_width + len(times) * resolution, header + candidates * resolution), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    for column, requested in enumerate(times):
        matching = next(record for record in records if record.requested_t == requested)
        title = f"requested t={requested:.3f} | actual={matching.actual_t:.3f}"
        draw.text((label_width + column * resolution + 8, 14), title, fill="black", font=font)
    lookup = {(record.candidate, record.requested_t): record for record in records}
    for row in range(candidates):
        draw.text((8, header + row * resolution + 12), f"candidate {row + 1}", fill="black", font=font)
        for column, requested in enumerate(times):
            image = images[(row, requested)]
            x = label_width + column * resolution
            y = header + row * resolution
            grid.paste(image, (x, y))
            score = lookup[(row, requested)].clipscore
            score_text = "unavailable" if math.isnan(score) else f"{score:.4f}"
            draw.rectangle((x + 5, y + resolution - 29, x + 152, y + resolution - 5), fill=(255, 255, 255))
            draw.text((x + 10, y + resolution - 24), f"CLIPScore {score_text}", fill="black", font=font)
    grid.save(output, quality=95)


def main():
    args = parse_args()
    # diffusers 0.33.x accesses ``device.type`` in prepare_latents, so keep a
    # torch.device throughout instead of forwarding the CLI string.
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if any(not 0.0 < value < 1.0 for value in args.times):
        raise ValueError("all --times must be in (0, 1)")
    if args.candidates < 1:
        raise ValueError("--candidates must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    pipeline.set_progress_bar_config(disable=False)
    pipeline.transformer.eval()
    pipeline.vae.eval()
    states, embeddings = collect_reachable_states(pipeline, args.prompt, args, generator)
    prompt, negative, pooled, negative_pooled = embeddings
    # CLIP and SD3 text encoders are not needed simultaneously after prompt
    # encoding. Offloading avoids an unnecessary multi-GB memory peak.
    for encoder in (pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3):
        encoder.to("cpu")
    if args.device.type == "cuda":
        torch.cuda.empty_cache()
    clip_error = None
    try:
        scorer = ClipScorer(
            args.device,
            model_name_or_path=args.clip_model,
            local_files_only=args.clip_local_files_only,
        )
    except (OSError, ConnectionError) as error:
        if args.require_clipscore:
            raise
        scorer = None
        clip_error = f"{type(error).__name__}: {error}"
        print("WARNING: CLIP could not be loaded; Collapse images will still be generated.")
        print("WARNING:", clip_error)
    delta_default = 1.0 / args.ode_steps
    records = []
    pil_images = {}

    with torch.no_grad():
        for requested in args.times:
            state = states[requested]
            actual_t = state["actual_t"]
            delta_t = min(args.delta_t or delta_default, actual_t - 1e-5)
            x_t = expand_batch(state["latent"], args.candidates)
            prompt_batch = expand_batch(prompt, args.candidates)
            pooled_batch = expand_batch(pooled, args.candidates)
            neg_batch = expand_batch(negative, args.candidates)
            neg_pooled_batch = expand_batch(negative_pooled, args.candidates)
            t = torch.full((args.candidates,), actual_t, device=args.device)
            dt = torch.full_like(t, delta_t)
            velocity = predict_velocity(
                pipeline.transformer, x_t, state["model_timestep"], prompt_batch, pooled_batch,
                neg_batch, neg_pooled_batch, args.guidance_scale,
            )
            actions = instant_transition(
                x_t, velocity, t, dt, args.exploration, generator=generator,
            )
            t_after = t - dt
            collapse_velocity = predict_velocity(
                pipeline.transformer, actions.action, t_after * 1000.0,
                prompt_batch, pooled_batch, neg_batch, neg_pooled_batch, args.guidance_scale,
            )
            x0 = collapse_to_x0(actions.action, collapse_velocity, t_after)
            image_tensors = decode(pipeline, x0)
            if scorer is None:
                scores = torch.full((args.candidates,), float("nan"))
            else:
                scores = scorer(image_tensors, [args.prompt] * args.candidates).float().cpu()
            for candidate in range(args.candidates):
                filename = f"candidate_{candidate + 1:02d}_t_{requested:.3f}_actual_{actual_t:.3f}.png"
                path = os.path.join(args.output_dir, filename)
                image = tensor_to_pil(image_tensors[candidate])
                image.save(path)
                pil_images[(candidate, requested)] = image
                records.append(ProbeRecord(
                    candidate=candidate,
                    requested_t=requested,
                    actual_t=actual_t,
                    delta_t=delta_t,
                    t_after=actual_t - delta_t,
                    clipscore=float(scores[candidate]),
                    image=filename,
                ))

    grid_path = os.path.join(args.output_dir, "instant_reward_grid.png")
    make_grid(pil_images, records, args.times, args.candidates, args.resolution, grid_path)
    with open(os.path.join(args.output_dir, "scores.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ProbeRecord.__annotations__.keys())
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    summary = {
        "prompt": args.prompt,
        "model": args.model,
        "seed": args.seed,
        "ode_steps": args.ode_steps,
        "guidance_scale": args.guidance_scale,
        "exploration": args.exploration,
        "clip_model": args.clip_model,
        "clip_error": clip_error,
        "records": [asdict(record) for record in records],
    }
    with open(os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Saved grid: {grid_path}")
    for requested in args.times:
        values = [record.clipscore for record in records if record.requested_t == requested]
        finite = [value for value in values if math.isfinite(value)]
        if finite:
            print(f"t={requested:.3f}: mean={sum(finite) / len(finite):.4f}, min={min(finite):.4f}, max={max(finite):.4f}")
        else:
            print(f"t={requested:.3f}: CLIPScore unavailable")


if __name__ == "__main__":
    main()

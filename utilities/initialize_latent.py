import glob
import os
import torch
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth_img2img import preprocess_video
from einops import rearrange
import torch.nn.functional as F


def load_source_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f"noisy_latents_{t}.pt")
    assert os.path.exists(latents_t_path), f"Missing latents at t {t} path {latents_t_path}"
    latents = torch.load(latents_t_path).float()
    return latents


def initialize_noisy_latent(model, video_for_latents):
    config = model.config
    device = config["device"]
    noisy_latent = prepare_latents(model, preprocess_video(video_for_latents))
    noisy_latent = noisy_latent[:, :, : config["max_frames"]]

    if config["random_init"]:
        noisy_latent = torch.randn(
            noisy_latent.shape,
            generator=torch.Generator(device=device).manual_seed(config["seed"]),
            device=device,
            dtype=noisy_latent.dtype,
        )

    elif config["high_freq_replacement_init"]:
        if config["downsample_factor"] > 1:
            noise = torch.randn(
                noisy_latent.shape,
                generator=torch.Generator(device=device).manual_seed(config["seed"]),
                device=device,
                dtype=noisy_latent.dtype,
            )
            noisy_latent = high_freqs_replacement_downsampling(noisy_latent, noise, config)

    return noisy_latent


@torch.no_grad()
def prepare_latents(model, video, batch_size=1):
    generator = None
    latents_path = model.config["latents_path"]
    device = model.device
    dtype = torch.float32
    video = video.to(device=device, dtype=dtype)

    # change from (b, c, f, h, w) -> (b * f, c, w, h)
    bsz, channel, frames, width, height = video.shape
    video = video.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

    if video.shape[1] == 4:
        init_latents = video
    else:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        elif isinstance(generator, list):
            init_latents = [
                model.vae.encode(video[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = [
                model.vae.encode(video[i : i + 1]).latent_dist.sample(generator) for i in range(video.shape[0])
            ]
            init_latents = torch.cat(init_latents, dim=0)

        init_latents = model.vae.config.scaling_factor * init_latents

    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `video` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        init_latents = torch.cat([init_latents], dim=0)

    noisest = max(
        [
            int(x.split("_")[-1].split(".")[0])
            for x in glob.glob(os.path.join(latents_path, f"noisy_latents_*.pt"))
        ]
    )
    latents_path = os.path.join(latents_path, f"noisy_latents_{noisest}.pt")
    noisy_latent = torch.load(latents_path).to(device)

    bsz, channel, frames, width, height = noisy_latent.shape
    noisy_latent = noisy_latent.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

    alpha_prod_T = model.scheduler.alphas_cumprod[noisest]
    mu_T, sigma_T = alpha_prod_T**0.5, (1 - alpha_prod_T) ** 0.5
    eps = (noisy_latent - mu_T * init_latents) / sigma_T

    # get latents
    init_latents = model.scheduler.add_noise(init_latents, eps, model.scheduler.timesteps[0])

    latents = init_latents

    latents = latents[None, :].reshape((bsz, frames, latents.shape[1]) + latents.shape[2:]).permute(0, 2, 1, 3, 4)

    return latents



def high_freqs_replacement_downsampling(noisy_latent, noise, config):
    new_h, new_w = (
        noisy_latent.shape[-2] // config["downsample_factor"],
        noisy_latent.shape[-1] // config["downsample_factor"],
    )
    noise = rearrange(noise, "b c f h w -> (b f) c h w")
    noise_down = F.interpolate(noise, size=(new_h, new_w), mode="bilinear", align_corners=True, antialias=True)
    noise_up = F.interpolate(
        noise_down, size=(noise.shape[-2], noise.shape[-1]), mode="bilinear", align_corners=True, antialias=True
    )
    high_freqs = noise - noise_up
    noisy_latent = rearrange(noisy_latent, "b c f h w -> (b f) c h w")
    noisy_latent_down = F.interpolate(
        noisy_latent, size=(new_h, new_w), mode="bilinear", align_corners=True, antialias=True
    )
    low_freqs = F.interpolate(
        noisy_latent_down,
        size=(noisy_latent.shape[-2], noisy_latent.shape[-1]),
        mode="bilinear",
        align_corners=True,
        antialias=True,
    )
    noisy_latent = low_freqs + high_freqs
    noisy_latent = rearrange(noisy_latent, "(b f) c h w -> b c f h w", f=config["max_frames"])
    return noisy_latent


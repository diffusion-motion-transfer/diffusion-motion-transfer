import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import DDIMScheduler, VideoToVideoSDPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth_img2img import preprocess_video
from omegaconf import OmegaConf
from torchvision.io import read_video
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from transformers import logging
from utilities.utils import save_video

# suppress partial model loading warning
logging.set_verbosity_error()

class Preprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = "cuda"
        self.config = config
        self.use_depth = False

        print("Loading video model")
        self.model_key = "cerspense/zeroscope_v2_576w"
        self.resolution = (576, 320)

        self.pipeline = VideoToVideoSDPipeline.from_pretrained(self.model_key, torch_dtype=torch.float16)
        self.pipeline = self.pipeline.to("cuda")
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_model_cpu_offload()

        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        print("video model loaded")

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(
            negative_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([text_embeddings, uncond_embeddings])
        return text_embeddings

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path, save_latents=True):
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]] if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t**0.5
                mu_prev = alpha_prod_t_prev**0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                if save_latents:
                    torch.save(latent, os.path.join(save_path, f"noisy_latents_{t}.pt"))
        torch.save(latent, os.path.join(save_path, f"noisy_latents_{t}.pt"))
        return latent

    @torch.no_grad()
    def ddim_sample(self, x, cond):
        timesteps = self.scheduler.timesteps
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond.repeat(x.shape[0], 1, 1)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t**0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev**0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (x - sigma * eps) / mu
                x = mu_prev * pred_x0 + sigma_prev * eps

        return x

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        image = self.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        video = video.float()
        return video

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path, save_path, inversion_prompt="", negative_prompt=""):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, negative_prompt)[[0]]
        if data_path.endswith(".mp4"):
            video = read_video(data_path, pts_unit="sec")[0].permute(0, 3, 1, 2).cuda() / 255
            video = [ToPILImage()(video[i]).resize(self.resolution) for i in range(video.shape[0])]
        else:
            images = list(Path(data_path).glob("*.png")) + list(Path(data_path).glob("*.jpg"))
            images = sorted(images, key=lambda x: int(x.stem))
            video = [Image.open(img).resize(self.resolution) for img in images]

        video = video[: self.config["max_number_of_frames"]]
        save_video([np.array(img) for img in video], str(Path(save_path) / f"original.mp4"))

        video = preprocess_video(video)
        bsz, channel, frames, width, height = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
        latents = self.vae.config.scaling_factor * self.vae.encode(video.cuda().half()).latent_dist.sample()
        latents = latents[None, :].reshape((1, frames, latents.shape[1]) + latents.shape[2:]).permute(0, 2, 1, 3, 4)

        inverted_x = self.ddim_inversion(cond, latents, save_path, save_latents=True)

        if self.config["save_ddim_reconstruction"]:
            latent_reconstruction = self.ddim_sample(inverted_x, cond)
            decoded_latent = self.decode_latents(latent_reconstruction.half())
            video = tensor2vid(decoded_latent)
            save_video(video, str(Path(save_path) / f"result.mp4"))


def run(opt):
    save_path = opt.save_dir
    os.makedirs(save_path, exist_ok=True)

    model = Preprocess(opt)
    model.extract_latents(
        data_path=opt.video_path,
        num_steps=config["n_timesteps"],
        save_path=save_path,
        inversion_prompt=opt.prompt,
        negative_prompt=opt.negative_prompt,
    )


if __name__ == "__main__":
    # # ==============this code added==================================================================:
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace("132.76.81.120", port=12345, stdoutToServer=True, stderrToServer=True)
    # # ================================================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/preprocess_config.yaml")
    opt = parser.parse_args()
    config = OmegaConf.load(opt.config_path)

    run(config)

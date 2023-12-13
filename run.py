import argparse
import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import TextToVideoSDPipeline, DDIMScheduler
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid
from einops import rearrange
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torchvision.transforms import ToTensor
from tqdm import tqdm
from transformers import logging

from utilities.guidance_utils import (
    get_timesteps,
    register_time,
    register_guidance,
    calculate_losses,
    register_batch,
)
from utilities.initialize_latent import (
    initialize_noisy_latent,
    load_source_latents_t,
)
from utilities.utils import (
    isinstance_str,
    clean_memory, save_video,
)

# suppress partial model loading warning
logging.set_verbosity_error()


class MyIdentityUnsqueeze(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x.unsqueeze(0)


class MyIdentity(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.train_mode = True

    def forward(self, x, *args, **kwargs):
        if self.train_mode:
            return x
        else:
            return self.module(x, *args, **kwargs)


class Guidance(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]

        model_key = "cerspense/zeroscope_v2_576w"
        self.resolution = (576, 320)

        print("Loading video model")

        self.video_pipe = TextToVideoSDPipeline.from_pretrained(
            model_key,
            torch_dtype=torch.float16,
        )
        self.video_pipe = self.video_pipe.to("cuda")
        self.video_pipe.scheduler = DDIMScheduler.from_config(self.video_pipe.scheduler.config)
        self.video_pipe.scheduler.set_timesteps(config["n_timesteps"], device="cuda")

        self.video_pipe.scheduler.timesteps, self.guidance_schedule = get_timesteps(
            self.video_pipe.scheduler,
            config["n_timesteps"],
            config["max_guidance_timestep"],
            config["min_guidance_timestep"],
        )
        self.video_pipe.enable_vae_slicing()
        self.vae = self.video_pipe.vae
        self.tokenizer = self.video_pipe.tokenizer
        self.text_encoder = self.video_pipe.text_encoder
        self.unet = self.video_pipe.unet
        self.scheduler = copy.deepcopy(self.video_pipe.scheduler)
        print("video model loaded")

        # load images and latents
        (
            self.video,
            self.noisy_latent,
        ) = self.get_data()


        # Optimization params:
        self.optim_lr = config["optim_lr"]
        self.optimization_steps = config["optimization_steps"]
        self.scale_range = np.linspace(config["scale_range"][0], config["scale_range"][1], len(self.guidance_schedule))

        with torch.no_grad():
            self.text_embeds = self.video_pipe._encode_prompt(
                config["source_prompt"],
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=config["negative_prompt"],
            ).flip(
                0
            )  # prompt first, negative prompt second

            self.guidance_embeds = self.video_pipe._encode_prompt(
                config["target_prompt"],
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=config["negative_prompt"],
            ).flip(
                0
            )  # prompt first, negative prompt second

        self.current_latents = {
            t.item(): load_source_latents_t(t, self.config["latents_path"])[:, :, : config["max_frames"]]
            for t in self.scheduler.timesteps
        }


    # Copied from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.decode_latents
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
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def get_data(self):
        # load video
        data_path = self.config["data_path"]
        images = list(Path(data_path).glob("*.png")) + list(Path(data_path).glob("*.jpg"))
        images = sorted(images, key=lambda x: int(x.stem))
        video = torch.stack([ToTensor()(Image.open(img).resize(self.resolution)) for img in images]).cuda()
        video = video[: self.config["max_frames"]]

        # prepare noisy latent
        if self.config["seed"] is None:
            seed = torch.randint(0, 1000000, (1,)).item()
            self.config["seed"] = seed
            with open(Path(self.config["output_path"], "seed.txt"), "w") as f:
                f.write(str(seed))

        video_for_latents = [np.array(Image.open(img).resize(self.resolution)) for img in images]
        noisy_latent = initialize_noisy_latent(self, video_for_latents)

        return (
            video,
            noisy_latent,
        )

    @staticmethod
    def my_pass(*args, **kwargs):
        if len(args) == 0:
            return kwargs["hidden_states"]
        else:
            return args[0]

    def change_mode(self, train=True):
        if len(self.config["up_res_dict"]) != 0:
            index = max(self.config["up_res_dict"].keys())
            for i, block in enumerate(self.unet.up_blocks):
                if i > index:
                    if train:
                        self.unet.up_blocks[i].original_forward = self.unet.up_blocks[i].forward
                        self.unet.up_blocks[i].forward = self.my_pass
                    else:
                        self.unet.up_blocks[i].forward = self.unet.up_blocks[i].original_forward

        if self.unet.conv_norm_out:
            if train:
                self.unet.conv_norm_out.original_forward = self.unet.conv_norm_out.forward
                self.unet.conv_norm_out.forward = self.my_pass
            else:
                self.unet.conv_norm_out.forward = self.unet.conv_norm_out.original_forward

            if train:
                self.unet.conv_act.original_forward = self.unet.conv_act.forward
                self.unet.conv_act.forward = self.my_pass
            else:
                self.unet.conv_act.forward = self.unet.conv_act.original_forward

        if train:
            self.unet.conv_out.original_forward = self.unet.conv_out.forward
            self.unet.conv_out.forward = self.my_pass
        else:
            self.unet.conv_out.forward = self.unet.conv_out.original_forward

    def guidance_step(self, x, i, t):
        register_batch(self, 1)
        module_names = ["ModuleWithConvGuidance", "ModuleWithGuidance"]
        scaler = GradScaler()
        self.change_mode(train=True)
        optimized_x = x.clone().detach().requires_grad_(True)

        if self.config["with_lr_decay"]:
            lr = self.scale_range[i]
        else:
            lr = self.optim_lr

        optimizer = torch.optim.Adam([optimized_x], lr=lr)

        latents = self.current_latents[t.item()]

        with torch.no_grad():
            # latent features
            orig_features_pos = {}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                self.unet(latents, t, encoder_hidden_states=self.text_embeds[[0]], return_dict=False)[0]
            for _, module in self.unet.named_modules():
                if isinstance_str(module, module_names):
                    orig_features_pos[module.block_name] = module.saved_features[0]

        for _ in tqdm(range(self.optimization_steps)):
            optimizer.zero_grad()

            # target features
            target_features_pos = {}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                self.unet(optimized_x, t, encoder_hidden_states=self.guidance_embeds[[0]].detach(), return_dict=False)[
                    0
                ]
            for _, module in self.unet.named_modules():
                if isinstance_str(module, module_names):
                    target_features_pos[module.block_name] = module.saved_features[0]

            losses_log = {}
            total_loss = 0

            for (orig_name, orig_features), (target_name, target_features) in zip(
                orig_features_pos.items(), target_features_pos.items()
            ):
                assert orig_name == target_name

                losses = calculate_losses(orig_features.detach(), target_features, self.config)
                for key, value in losses.items():
                    losses_log[f"Loss/{orig_name}/{key}/time_step{t.item()}"] = value.item()
                total_loss += losses["total_loss"]

            losses_log[f"Loss/total_loss/time_step{t.item()}"] = total_loss.item()
            scaler.scale(total_loss).backward()

            scaler.step(optimizer)
            scaler.update()
            del losses_log, total_loss, losses, target_features_pos
            for _, module in self.unet.named_modules():
                if isinstance_str(module, module_names):
                    module.saved_features = None

        return optimized_x


    @torch.no_grad()
    def denoise_step(self, x, t):
        if t in self.guidance_schedule:
            self.change_mode(train=False)

        register_batch(self, 2)
        latent_model_input = torch.cat([x, x], dim=0)
        text_embed_input = self.guidance_embeds

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input, return_dict=False)[0]

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

        bsz, channel, frames, width, height = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

        denoised_latent = self.scheduler.step(noise_pred, t, x)["prev_sample"]
        denoised_latent = denoised_latent[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

        return denoised_latent

    @torch.no_grad()
    def undo_step(self, sample, timestep):
        n = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        for i in range(n):
            beta = self.scheduler.betas[timestep + i]
            noise = torch.randn_like(sample)
            sample = (1 - beta) ** 0.5 * sample + beta**0.5 * noise

        return sample


    def run(self):
        clean_memory()
        x = self.noisy_latent
        register_guidance(self)

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
            register_time(self, t.item())

            if t in self.guidance_schedule:
                x = self.guidance_step(x, i, t)
            x = self.denoise_step(x, t)
            if self.config["restart_sampling"] and t != self.scheduler.timesteps[-1]:
                x = self.undo_step(x, self.scheduler.timesteps[i + 1])
                x = self.denoise_step(x, t)

            clean_memory()
            clean_memory()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            decoded_frames = self.decode_latents(x)
        decoded_frames = np.array(tensor2vid(decoded_frames))  # f h w c numpy [0,255]
        # save frames
        Path(self.config["output_path"], "result_frames").mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(decoded_frames):
            Image.fromarray(frame).save(Path(self.config["output_path"], "result_frames", f"{i:04d}.png"))

        save_video(decoded_frames, os.path.join(self.config["output_path"], "result.mp4"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/guidance_config.yaml")
    opt = parser.parse_args()
    config = OmegaConf.load(opt.config_path)
    Path(config["output_path"]).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, Path(config["output_path"]) / "config.yaml")

    guidance = Guidance(config)
    guidance.run()

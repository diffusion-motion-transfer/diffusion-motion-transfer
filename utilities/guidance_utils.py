from math import sqrt
from utilities.utils import isinstance_str
import torch
import torch.nn.functional as F
from einops import rearrange


@torch.autocast(device_type="cuda", dtype=torch.float32)
def calculate_losses(orig_features, target_features, config):
    orig = orig_features
    target = target_features

    orig = orig.detach()

    total_loss = 0
    losses = {}
    if config["features_loss_weight"] > 0:
        if config["global_averaging"]:
            orig = orig.mean(dim=(2, 3), keepdim=True)
            target = target.mean(dim=(2, 3), keepdim=True)

        features_loss = compute_feature_loss(orig, target)
        total_loss += config["features_loss_weight"] * features_loss
        losses["features_mse_loss"] = features_loss

    if config["features_diff_loss_weight"] > 0:
        features_diff_loss = 0
        orig = orig.mean(dim=(2, 3), keepdim=True)  # t d 1 1
        target = target.mean(dim=(2, 3), keepdim=True)

        for i in range(len(orig)):
            orig_anchor = orig[i]
            target_anchor = target[i]
            orig_diffs = orig - orig_anchor  # t d 1 1
            target_diffs = target - target_anchor  # t d 1 1
            features_diff_loss += 1 - F.cosine_similarity(target_diffs, orig_diffs.detach(), dim=1).mean()
        features_diff_loss /= len(orig)

        total_loss += config["features_diff_loss_weight"] * features_diff_loss
        losses["features_diff_loss"] = features_diff_loss

    losses["total_loss"] = total_loss
    return losses


def compute_feature_loss(orig, target):
    features_loss = 0
    for i, (orig_frame, target_frame) in enumerate(zip(orig, target)):
        features_loss += 1 - F.cosine_similarity(target_frame, orig_frame.detach(), dim=0).mean()
    features_loss /= len(orig)
    return features_loss


def get_timesteps(scheduler, num_inference_steps, max_guidance_timestep, min_guidance_timestep):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * max_guidance_timestep), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    t_end = int(num_inference_steps * min_guidance_timestep)
    timesteps = scheduler.timesteps[t_start * scheduler.order :]
    if t_end > 0:
        guidance_schedule = scheduler.timesteps[t_start * scheduler.order : -t_end * scheduler.order]
    else:
        guidance_schedule = scheduler.timesteps[t_start * scheduler.order :]
    return timesteps, guidance_schedule


def register_time(model, t):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "t", t)


def register_batch(model, b):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "b", b)


def register_guidance(model):
    guidance_schedule = model.guidance_schedule
    num_frames = model.video.shape[0]
    h = model.video.shape[2]
    w = model.video.shape[3]

    class ModuleWithConvGuidance(torch.nn.Module):
        def __init__(self, module, guidance_schedule, num_frames, h, w, block_name, config, module_type):
            super().__init__()
            self.module = module
            self.guidance_schedule = guidance_schedule
            self.num_frames = num_frames
            assert module_type in [
                "spatial_convolution",
            ]
            self.module_type = module_type
            if self.module_type == "spatial_convolution":
                self.starting_shape = "(b t) d h w"
            self.h = h
            self.w = w
            self.block_name = block_name
            self.config = config
            self.saved_features = None

        def forward(self, input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.module.norm1(hidden_states)
            hidden_states = self.module.nonlinearity(hidden_states)

            if self.module.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.module.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.module.downsample is not None:
                input_tensor = self.module.downsample(input_tensor)
                hidden_states = self.module.downsample(hidden_states)

            hidden_states = self.module.conv1(hidden_states)

            if temb is not None:
                temb = self.module.time_emb_proj(self.module.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.module.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.module.norm2(hidden_states)

            if temb is not None and self.module.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.module.nonlinearity(hidden_states)

            hidden_states = self.module.dropout(hidden_states)
            hidden_states = self.module.conv2(hidden_states)

            if self.config["guidance_before_res"] and (self.t in self.guidance_schedule):
                self.saved_features = rearrange(
                    hidden_states, f"{self.starting_shape} -> b t d h w", t=self.num_frames
                )

            if self.module.conv_shortcut is not None:
                input_tensor = self.module.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.module.output_scale_factor

            if not self.config["guidance_before_res"] and (self.t in self.guidance_schedule):
                self.saved_features = rearrange(
                    output_tensor, f"{self.starting_shape} -> b t d h w", t=self.num_frames
                )

            return output_tensor

    class ModuleWithGuidance(torch.nn.Module):
        def __init__(self, module, guidance_schedule, num_frames, h, w, block_name, config, module_type):
            super().__init__()
            self.module = module
            self.guidance_schedule = guidance_schedule
            self.num_frames = num_frames
            assert module_type in [
                "temporal_attention",
                "spatial_attention",
                "temporal_convolution",
                "upsampler",
            ]
            self.module_type = module_type
            if self.module_type == "temporal_attention":
                self.starting_shape = "(b h w) t d"
            elif self.module_type == "spatial_attention":
                self.starting_shape = "(b t) (h w) d"
            elif self.module_type == "temporal_convolution":
                self.starting_shape = "(b t) d h w"
            elif self.module_type == "upsampler":
                self.starting_shape = "(b t) d h w"
            self.h = h
            self.w = w
            self.block_name = block_name
            self.config = config

        def forward(self, x, *args, **kwargs):
            if not isinstance(args, tuple):
                args = (args,)
            out = self.module(x, *args, **kwargs)
            t = self.num_frames
            if self.module_type == "temporal_attention":
                size = out.shape[0] // self.b
            elif self.module_type == "spatial_attention":
                size = out.shape[1]
            elif self.module_type == "temporal_convolution":
                size = out.shape[2] * out.shape[3]
            elif self.module_type == "upsampler":
                size = out.shape[2] * out.shape[3]

            if self.t in self.guidance_schedule:
                h, w = int(sqrt(size * self.h / self.w)), int(sqrt(size * self.h / self.w) * self.w / self.h)
                self.saved_features = rearrange(
                    out, f"{self.starting_shape} -> b t d h w", t=self.num_frames, h=h, w=w
                )

            return out

    up_res_dict = model.config["up_res_dict"]
    for res in up_res_dict:
        module = model.unet.up_blocks[res]
        samplers = module.upsamplers
        if model.config["use_upsampler_features"]:
            if samplers is not None:
                for i in range(len(samplers)):
                    submodule = samplers[i]
                    samplers[i] = ModuleWithGuidance(
                        submodule,
                        guidance_schedule,
                        num_frames,
                        h,
                        w,
                        block_name=f"decoder_res{res}_upsampler",
                        config=model.config,
                        module_type="upsampler",
                    )
        for block in up_res_dict[res]:
            block_name = f"decoder_res{res}_block{block}"
            if model.config["use_conv_features"]:
                block_name_conv = f"{block_name}_spatial_convolution"
                submodule = module.resnets[block]
                module.resnets[block] = ModuleWithConvGuidance(
                    submodule,
                    guidance_schedule,
                    num_frames,
                    h,
                    w,
                    block_name=block_name_conv,
                    config=model.config,
                    module_type="spatial_convolution",
                )

            if model.config["use_temp_conv_features"]:
                block_name_conv = f"{block_name}_temporal_convolution"
                submodule = module.temp_convs[block]
                module.temp_convs[block] = ModuleWithGuidance(
                    submodule,
                    guidance_schedule,
                    num_frames,
                    h,
                    w,
                    block_name=block_name_conv,
                    config=model.config,
                    module_type="temporal_convolution",
                )

            if res == 0:  # UpBlock3D does not have attention
                if model.config["use_spatial_attention_features"]:
                    block_name_spatial = f"{block_name}_spatial_attn1"
                    submodule = module.attentions[block].transformer_blocks[0]
                    assert isinstance_str(submodule, "BasicTransformerBlock")
                    submodule.attn1 = ModuleWithGuidance(
                        submodule.attn1,
                        guidance_schedule,
                        num_frames,
                        h,
                        w,
                        block_name=block_name_spatial,
                        config=model.config,
                        module_type="spatial_attention",
                    )
                if model.config["use_temporal_attention_features"]:
                    submodule = module.temp_attentions[block].transformer_blocks[0]
                    assert isinstance_str(submodule, "BasicTransformerBlock")
                    block_name_temp = f"{block_name}_temporal_attn1"
                    submodule.attn1 = ModuleWithGuidance(
                        submodule.attn1,
                        guidance_schedule,
                        num_frames,
                        h=h,
                        w=w,
                        block_name=block_name_temp,
                        config=model.config,
                        module_type="temporal_attention",
                    )
                    block_name_temp = f"{block_name}_temporal_attn2"
                    submodule.attn2 = ModuleWithGuidance(
                        submodule.attn2,
                        guidance_schedule,
                        num_frames,
                        h=h,
                        w=w,
                        block_name=block_name_temp,
                        config=model.config,
                        module_type="temporal_attention",
                    )

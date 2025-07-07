# Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer
<a href="https://diffusion-motion-transfer.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/abs/2311.17009"><img src="https://img.shields.io/badge/arXiv-2311.17009-b31b1b.svg"></a>

This is the official implementation of the paper:

[**Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer**](https://diffusion-motion-transfer.github.io/)
<br/>

[Danah Yatim*](https://www.linkedin.com/in/danah-yatim-4b15231b5/),
[Rafail Fridman*](https://www.linkedin.com/in/rafail-fridman/),
[Omer Bar-Tal](https://omerbt.github.io/),
[Yoni Kasten](https://ykasten.github.io/),
[Tali Dekel](https://www.weizmann.ac.il/math/dekel/)
<br/>
(*equal contribution)

https://github.com/diffusion-motion-transfer/diffusion-motion-transfer/assets/22198039/4fe912d4-0975-4580-af7f-19fd73b0cbfe



Introducing a zero-shot method for transferring motion across objects and scenes. without any training or finetuning.

>We present a new method for text-driven motion transfer -- synthesizing a video that complies with an input text prompt describing the target objects and scene while maintaining an input video's motion and scene layout. Prior methods are confined to transferring motion across two subjects within the same or closely related object categories and are applicable for limited domains (e.g., humans). 
In this work, we consider a significantly more challenging setting in which the target and source objects differ drastically in shape and fine-grained motion characteristics (e.g., translating a jumping dog into a dolphin).  To this end, we leverage a pre-trained and fixed text-to-video diffusion model, which provides us with generative and motion priors. The pillar of our method is a new space-time feature loss derived directly from the model. This loss guides the generation process to preserve the overall motion of the input video while complying with the target object in terms of shape and fine-grained motion traits. 

For more, visit the [project webpage](https://diffusion-motion-transfer.github.io/).

# Installation
Clone the repo and create a new environment:
```
git clone https://github.com/diffusion-motion-transfer/diffusion-motion-transfer.git
cd diffusion-motion-transfer
conda create --name dmt python=3.9
conda activate dmt
```
Install our environment requirements:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

# Motion Transfer
* Our method is designed for transferring motion across objects and scenes
* Our method is based on [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w) text-to-video model. Therefore, we can edit videos of **24 frames**.
* in some cases the combination of target object and input video motion is out of distribution for the T2V model, which can lead to visual artifacts in the generated video. It may be necessary to sample several seeds.
* Method was tested to run on a single NVIDIA A40 48GB, and takes ~32GB of video memory. It takes approximately 7 minutes on a single NVIDIA A40 48GB.

# Preprocess
To preprocess a video, update configuration file `configs/preprocess_config.yaml':

Arguments to update:
* ```video_path``` - the input video frames should be located in this path
* ```save_dir``` - the latents will be saved in this path
* ```prompt``` - empty string or a string describing the video content

Optional arguments to update:
* ```--save_ddim_reconstruction``` if True, the reconstructed video will be saved in ```--save_dir```

After updating config file, run the following command:
```
python preprocess_video_ddim.py --config_path configs/preprocess_config.yaml
```
Once the preprocessing is done, the latents will be saved in the ```save_dir``` path. 

# Editing
To edit the video, update configuration file `configs/guidance_config.yaml`
Arguments to update:
* ```data_path``` - the input video frames should be located in this path
* ```output_path``` - the edited video will be saved in this path
* ```latents_path``` - the latents of the input video should be located in this path
* ```source_prompt``` - prompt used for inversion
* ```target_prompt``` - prompt used for editing
    
Optional arguments to update:
* ```negative_prompt``` - prompt used for unconditional classifier free guidance
*  ```seed``` - By default it is randomly chosen, to specify seed change thise value.
*  ```optimization_step``` - number of optimization steps for each denoising step
* ```optim_lr``` - learning rate
* ```with_lr_decay```  - if True, overrides `optim_lr`, and the learning rate will decay during the optimization process in the range of `scale_range`

After updating the config file, run the following command:
```
python run.py --config_path configs/guidance_config.yaml
```

Once the method is done, the video will be saved to the ```output_path``` under `result.mp4`.


# Tips
* To get better samples from the T2V model, we used the prefix text ```"Amazing quality, masterpiece, "``` for inversion and edits.
* If the video contains more complex motion/small objects, try increasing number of optimization steps - ```optimization_step: 30```.
* For large deviation in structure between the source and target objects, try using a lower lr - ```scale_range:[0.005, 0.002]```,
*  or adding the source object to the negative prompt text.

# Measuring motion fidelity
We also provide the code for calculating the motion fidelity metric introduced in the paper (Section 5.1).
To calculate the motion fidelity metric, first follow the instructions [here](https://github.com/facebookresearch/co-tracker) to install Co-Tracker and download their checkpoint.
Then, run the following command:
```
python motion_fidelity_score.py --config_path configs/motion_fidelity_config.yaml
```



# Citation
```
@article{yatim2023spacetime,
        title = {Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer},
        author = {Yatim, Danah and Fridman, Rafail and Bar-Tal, Omer and Kasten, Yoni and Dekel, Tali},
        journal={arXiv preprint arxiv:2311.17009},
        year={2023}
        }
```

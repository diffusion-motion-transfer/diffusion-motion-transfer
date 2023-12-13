import gc
import random

import numpy as np
import torch
from typing import Union, List
from torchvision.io import write_video

video_codec = "libx264"
video_options = {
    "crf": "17",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
    "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
}

def save_video(video, path):
    write_video(
        path,
        video,
        fps=10,
        video_codec=video_codec,
        options=video_options,
    )

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def isinstance_str(x: object, cls_name: Union[str, List[str]]):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """
    if type(cls_name) == str:
        for _cls in x.__class__.__mro__:
            if _cls.__name__ == cls_name:
                return True
    else:
        for _cls in x.__class__.__mro__:
            if _cls.__name__ in cls_name:
                return True
    return False
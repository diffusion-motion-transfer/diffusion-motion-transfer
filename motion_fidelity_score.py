import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path
from einops import rearrange
from omegaconf import OmegaConf


def get_similarity_matrix(tracklets1, tracklets2):
    displacements1 = tracklets1[:, 1:] - tracklets1[:, :-1]
    displacements1 = displacements1 / displacements1.norm(dim=-1, keepdim=True)

    displacements2 = tracklets2[:, 1:] - tracklets2[:, :-1]
    displacements2 = displacements2 / displacements2.norm(dim=-1, keepdim=True)

    similarity_matrix = torch.einsum("ntc, mtc -> nmt", displacements1, displacements2).mean(dim=-1)
    return similarity_matrix


def get_score(similarity_matrix):
    similarity_matrix_eye = similarity_matrix - torch.eye(similarity_matrix.shape[0]).to(similarity_matrix.device)
    # for each row find the most similar element
    max_similarity, _ = similarity_matrix_eye.max(dim=1)
    average_score = max_similarity.mean()
    return {
        "average_score": average_score.item(),
    }


def get_tracklets(model, video_path, mask=None):
    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().cuda()
    pred_tracks_small, pred_visibility_small = model(video, grid_size=55, segm_mask=mask)
    pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c ")
    return pred_tracks_small


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/motion_fidelity_score_config.yaml")
    opt = parser.parse_args()
    config = OmegaConf.load(opt.config_path)

    model = CoTrackerPredictor(checkpoint=config.cotracker_model_path)
    model = model.cuda()

    video_path = config.edit_video_path
    original_video_path = config.original_video_path

    if config.use_mask:  # calculate trajectories only on the foreground of the video
        segm_mask = np.array(
            Image.open(config.mask_path)
        )
        segm_mask = torch.tensor(segm_mask).float() / 255
        # get bounding box mask from segmentation mask - rectangular mask that covers the segmentation mask
        box_mask = torch.zeros_like(segm_mask)
        minx = segm_mask.nonzero()[:, 0].min()
        maxx = segm_mask.nonzero()[:, 0].max()
        miny = segm_mask.nonzero()[:, 1].min()
        maxy = segm_mask.nonzero()[:, 1].max()
        box_mask[minx:maxx, miny:maxy] = 1
        box_mask = box_mask[None, None]
    else:
        box_mask = None

    edit_tracklets = get_tracklets(model, video_path, mask=box_mask)
    original_tracklets = get_tracklets(model, original_video_path, mask=box_mask)
    similarity_matrix = get_similarity_matrix(edit_tracklets, original_tracklets)
    similarity_scores_dict = get_score(similarity_matrix)

    save_dict = {
        "similarity_matrix": similarity_matrix.cpu(),
        "similarity_scores": similarity_scores_dict,
    }
    torch.save(save_dict, Path(config.output_path) / "metrics.pt")
    print("Motion similarity score: ", similarity_scores_dict["average_score"])

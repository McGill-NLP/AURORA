from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from PIL import Image, ImageDraw, ImageFont
import textwrap
import json
import os
from dataset_loading import get_dataset
from edit_dataset import EditITMDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
from collections import defaultdict

sys.path.append("./stable_diffusion")
from stable_diffusion.ldm.util import instantiate_from_config

# Assuming CFGDenoiser and other dependencies are correctly set up,
# no changes needed there for image aspect ratio handling.


def calculate_clip_similarity(generated_images, original_image, clip_model, preprocess, device):
    original_image_processed = preprocess(original_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        original_features = clip_model.encode_image(original_image_processed)

    similarities = []
    for img in generated_images:
        img_processed = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            generated_features = clip_model.encode_image(img_processed)
        similarity = torch.nn.functional.cosine_similarity(generated_features, original_features, dim=-1)
        similarities.append(similarity.item())

        #concat both img and original_image for visualization
        # original_image_np = np.array(original_image)
        # img_np = np.array(img)
        # both = np.concatenate((original_image_np, img_np), axis=1)
        # both = Image.fromarray(both)
        # if not os.path.exists('eval_output/edit_itm/flickr_edit_clip_sim/'):
        #     os.makedirs('eval_output/edit_itm/flickr_edit_clip_sim/')
        # random_id = random.randint(0, 100000)
        # both.save(f'eval_output/edit_itm/flickr_edit_clip_sim/{similarity.item()}_{random_id}.png')

    # average_similarity = sum(similarities) / len(similarities)
    # dist = 1 - average_similarity
    dists = [1 - sim for sim in similarities]
    return dists


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale, conditional_only=False):
        # cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_z = z.repeat(3, 1, 1, 1)
        # cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_sigma = sigma.repeat(3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        if conditional_only:
            return out_cond
        else:
            return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)

def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def calculate_accuracy(losses):
    correct_count = 0
    for loss in losses:
        if loss[0] < min(loss[1:]):
            correct_count += 1
    return correct_count, len(losses)  # Return counts for aggregation


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="aurora-mixratio-15-15-1-1-42k-steps.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--task", default='flickr_edit', type=str)
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--samples", default=4, type=int)
    parser.add_argument("--size", default=512, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument('--targets', type=str, nargs='*', help="which target groups for mmbias",default='')
    parser.add_argument("--device", default=0, type=int, help="GPU device index")
    parser.add_argument("--log_imgs", action="store_true")
    parser.add_argument("--conditional_only", action="store_true")
    parser.add_argument("--metric", default="latent", type=str)
    parser.add_argument("--split", default='test', type=str)
    parser.add_argument("--skip", default=1, type=int)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval()
    model.to(dtype=torch.float)
    model = model.to(device)
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = EditITMDataset(split=args.split, task=args.task, min_resize_res=args.size, max_resize_res=args.size, crop_res=args.size)
    dataloader= DataLoader(dataset,batch_size=args.batchsize,num_workers=1,worker_init_fn=None,shuffle=False, persistent_workers=True)

    if os.path.exists(f'itm_evaluation/{args.split}/{args.task}/{args.ckpt.replace("/", "_")}_results.json'):
        with open(f'itm_evaluation/{args.split}/{args.task}/{args.ckpt.replace("/", "_")}_results.json', 'r') as f:
            results = json.load(f)
            results = defaultdict(dict, results)
    else:
        results = defaultdict(dict)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if len(batch['input'][0].shape) < 3:
            continue
        for j, prompt in enumerate(batch['texts']):
            # check if we already have results for this image
            img_id = batch['path'][0] + f'_{i}'
            # if img_id in results and ('pos' in results[img_id] and 'neg' in results[img_id]):
            #     continue

            with torch.no_grad(), autocast("cuda"), model.ema_scope():
                prompt = prompt[0]
                cond = {}
                cond["c_crossattn"] = [model.get_learned_conditioning([prompt])]
                input_image = batch['input'][0].to(device)
                cond["c_concat"] = [model.encode_first_stage(input_image.unsqueeze(0)).mode()]
                scaled_input = model.scale_factor * input_image

                uncond = {}
                uncond["c_crossattn"] = [null_token]
                uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

                sigmas = model_wrap.get_sigmas(args.steps)
                # move everything to the device
                cond = {k: [v.to(device) for v in vs] for k, vs in cond.items()}
                uncond = {k: [v.to(device) for v in vs] for k, vs in uncond.items()}

                cond["c_concat"][0] = cond["c_concat"][0].repeat(args.samples, 1, 1, 1)
                cond["c_crossattn"][0] = cond["c_crossattn"][0].repeat(args.samples, 1, 1)
                uncond["c_concat"][0] = uncond["c_concat"][0].repeat(args.samples, 1, 1, 1)
                uncond["c_crossattn"][0] = uncond["c_crossattn"][0].repeat(args.samples, 1, 1)

                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": args.cfg_text,
                    "image_cfg_scale": args.cfg_image,
                    "conditional_only": args.conditional_only,
                }
                # torch.manual_seed(i)
                torch.manual_seed(42)
                z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args, disable=True)
                x = model.decode_first_stage(z)
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                
            ######## LOG IMAGES ########
                input_image_pil = ((input_image + 1) * 0.5).clamp(0, 1)
                input_image_pil = input_image_pil.permute(1, 2, 0)  # Change from CxHxW to HxWxC for PIL
                input_image_pil = (input_image_pil * 255).type(torch.uint8).cpu().numpy()

                for k in range(2):
                    x_ = 255.0 * rearrange(x[k], "c h w -> h w c")
                    edited_image = x_.type(torch.uint8).cpu().numpy()
                    both = np.concatenate((input_image_pil, edited_image), axis=1)
                    both = Image.fromarray(both)
                    out_base = f'itm_evaluation/{args.split}/{args.task}/{args.ckpt.replace("/", "_")}'
                    if not os.path.exists(out_base):
                        os.makedirs(out_base)
                    prompt_str = prompt.replace(' ', '_')[0:100]
                    both.save(f'{out_base}/{i}_{"correct" if j == 0 else "incorrect"}_sample{k}_{prompt}.png')
                
            ######## CLIP ########

                edited_images = []
                for k in range(args.samples):
                    x_ = 255.0 * rearrange(x[k], "c h w -> h w c")
                    edited_image = Image.fromarray(x_.type(torch.uint8).cpu().numpy())
                    edited_images.append(edited_image)
                input_image_pil = ((input_image + 1) * 0.5).clamp(0, 1)
                input_image_pil = input_image_pil.permute(1, 2, 0)  # Change from CxHxW to HxWxC for PIL
                input_image_pil = (input_image_pil * 255).type(torch.uint8).cpu().numpy()
                input_image_pil = Image.fromarray(input_image_pil)
                dists_clip = calculate_clip_similarity(edited_images, input_image_pil, clip_model, preprocess, device)

            ######## LATENT ########
                z = z.flatten(1)
                original_latent = cond["c_concat"][0].flatten(1)
                dists_latent = torch.norm(z - original_latent, dim=1, p=2).cpu().numpy().tolist()
                cos_sim = torch.nn.functional.cosine_similarity(z, original_latent, dim=1).cpu().numpy().tolist()
                cos_dists_latent = [1 - sim for sim in cos_sim]
            ######## SAVE RESULTS ########
                img_id = batch['path'][0] + f'_{i}'
                results[img_id]['pos' if j == 0 else 'neg'] = {
                    "prompt" : prompt,
                    "clip": dists_clip,
                    "latent_l2": dists_latent,
                    "latent_cosine": cos_dists_latent
                }
                with open(f'itm_evaluation/{args.split}/{args.task}/{args.ckpt.replace("/", "_")}_results.json', 'w') as f:
                    json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

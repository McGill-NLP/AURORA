from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
import os
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class FinetuneDataset(Dataset):
    def __init__(
        self,
        path: str = '',
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.5,
        msr_vtt_cc_full: bool = False,
        mix: list[str] = ['magicbrush', 'something', 'hq'],
        mix_factors: list[float] = [40, 1, 1],
        copy_prob: float = 0.0,
        kubric_100k: bool = False,
    ):
        self.split = split
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.mix_factors = mix_factors
        self.msr_vtt_cc_full = msr_vtt_cc_full
        self.copy_prob = copy_prob

        self.data = []
        for dataset in mix:
            if dataset != 'hq':
                for _ in range(mix_factors[mix.index(dataset)]):
                    if kubric_100k and dataset == 'kubric':
                        self.data.extend(json.load(open(f'data/{dataset}/train_100k.json', 'r')))
                        print("LODADED KUBRIC 100K")
                    else:
                        self.data.extend(json.load(open(f'data/{dataset}/train.json', 'r')))
                    # if dataset == 'msr-vtt-cc':
                    #     self.data.extend(json.load(open(f'data/{dataset}/train_gpt.json', 'r')))

        if split == 'val':
            self.data = self.data[:2]
 
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        # if i < len(self.data):
        ex = self.data[i]
        img_path0 = ex['input']
        img_path1 = ex['output']
        prompt = ex['instruction']
        dataset = img_path0.split('/')[1]
        if dataset == 'kubric':
            subtask = img_path0.split('/')[2]
        else:
            subtask = '___'
        
        if type(prompt) == list:
            prompt = prompt[0]
        spatial = 'left' in prompt.lower() or 'right' in prompt.lower()
        image_1 = Image.open(img_path1).convert('RGB') if i < len(self.data) else img_path1

        if subtask not in ['closer', 'counting', 'further_location', 'rotate']:
            if self.copy_prob > 0 and torch.rand(1) < self.copy_prob:
                image_0 = Image.open(img_path1).convert('RGB') if i < len(self.data) else img_path1
            else:
                image_0 = Image.open(img_path0).convert('RGB') if i < len(self.data) else img_path0
        else:
            image_0 = Image.open(img_path0).convert('RGB') if i < len(self.data) else img_path0

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip_prob = 0.0 if spatial else self.flip_prob
        flip = torchvision.transforms.RandomHorizontalFlip(float(flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


class MagicEditDataset(Dataset):
    def __init__(
        self,
        path: str = '../../change_descriptions/something-something',
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        debug: bool = False,
    ):
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        print("Dataset params")
        print(self.min_resize_res, self.max_resize_res, self.crop_res, self.flip_prob)

        #clean json (if first and last are not both present, remove)
        split = "train" if split == "train" else "dev"
        self.dataset = load_dataset("osunlp/MagicBrush")[split]

        # if split == 'dev':
        #     eval_data = json.load(open('eval_data/video_edit.json', 'r'))
        #     dummy_image = Image.new('RGB', (1, 1), (0, 0, 0))
        #     eval_data = {
        #         'source_img': [Image.open(x['img0']) for x in eval_data],
        #         'target_img': [Image.open(x['img1']) for x in eval_data],
        #         'instruction': [x['edit'] if type(x['edit']) == str else x['edit'][0] for x in eval_data],
        #         'img_id': ['' for _ in eval_data],
        #         'turn_index': np.array([1 for _ in eval_data], dtype=np.int32),
        #         'mask_img': [dummy_image for _ in eval_data]  # Replace each entry with the dummy image
        #     }
        #     eval_dataset = HuggingFaceDataset.from_dict(eval_data)
        #     self.dataset = concatenate_datasets([self.dataset, eval_dataset])

        if debug:
            self.dataset = self.dataset.shuffle(seed=42).select(range(50))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> dict[str, Any]:
        
        prompt = self.dataset[i]['instruction']
        if type(prompt) == list:
            prompt = prompt[0]
        image_0 = self.dataset[i]['source_img']
        image_1 = self.dataset[i]['target_img']
        if image_0.mode == 'RGBA':
            image_0 = image_0.convert('RGB')
        if image_1.mode == 'RGBA':
            image_1 = image_1.convert('RGB')
        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


class FrameEditDataset(Dataset):
    def __init__(
        self,
        path: str = '../../change_descriptions/something-something',
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        task: str = 'flickr30k_text',
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        debug: bool = False,
    ):
        self.split = split
        self.task = task
        if split == "train":
            path = os.path.join(path, 'train.json')
            self.json = json.load(open(path, 'r'))
            np.random.shuffle(self.json)
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        #clean json (if first and last are not both present, remove)
        if split == 'train':
            new_json = []
            for i in range(len(self.json)):
                video_id = self.json[i]['id']
                img_path0 = f'../../change_descriptions/something-something/frames/{video_id}/first.jpg'
                img_path1 = f'../../change_descriptions/something-something/frames/{video_id}/last.jpg'
                if os.path.exists(img_path0) and os.path.exists(img_path1):
                    new_json.append(self.json[i])
            self.json = new_json
        if debug:
            self.json = self.json[:50]

    def __len__(self) -> int:
        return len(self.json)

    def __getitem__(self, i: int) -> dict[str, Any]:
        if self.split == 'train':
            video_id = self.json[i]['id']
            img_path0 = f'../../change_descriptions/something-something/frames/{video_id}/first.jpg'
            img_path1 = f'../../change_descriptions/something-something/frames/{video_id}/last.jpg'
            prompt = self.json[i]['label']
        
        image_0 = Image.open(img_path0).convert('RGB')
        image_1 = Image.open(img_path1).convert('RGB')

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        # image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        # image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_0 = image_0.resize((self.crop_res, self.crop_res))
        image_1 = image_1.resize((self.crop_res, self.crop_res))

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)
    
        # if i ever wanna reverse time
        # if torch.rand(1) > 0.5:
        #     image_0, image_1 = image_1, image_0
        #     prompt = caption0
        if self.split == 'train':
            return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))
        else:
            return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=texts))

class EditITMDataset(Dataset):
    def __init__(
        self,
        path: str = '../../change_descriptions/something-something',
        split: str = "test",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        task: str = 'flickr30k_text',
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        debug: bool = False,
    ):
        self.split = split
        self.task = task
        # if task == 'flickr_edit':
        #     path = 'data/flickr_edit/valid.json' if split == 'val' else 'data/flickr_edit/test.json'
        #     self.json = json.load(open(path, 'r'))
        #     #clean json, if "pos" key is empty string, remove
        #     self.json = [x for x in self.json if x['pos'] != '']
        if task == 'whatsup':
            path = 'data/whatsup/itm_test.json' if split == 'test' else 'data/whatsup/itm_valid.json'
            self.json = json.load(open(path, 'r'))
        elif task == 'svo':
            path = 'data/svo/itm_test.json' if split == 'test' else 'data/svo/itm_valid.json'
            self.json = json.load(open(path, 'r'))
        else:
            path = f'data/{task}/valid.json'
            self.json = json.load(open(path, 'r'))
            self.json = [x for x in self.json if x.get('pos', '') != '']
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        if debug:
            self.json = self.json[:50]

    def __len__(self) -> int:
        return len(self.json)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ex = self.json[i]
        pos = ex.get('pos', '')
        if pos == '':
            pos = ex['prompt']
        neg = ex.get('neg', '')
        if neg == '':
            neg = ex['prompt']
        img_path0 = ex['input']
        texts = [pos, neg]
        # if self.task == 'whatsup' or self.task == 'svo':
        #     img_path0 = f"data/{self.task}/images/{ex['image']}" if self.task == 'flickr_edit' else ex['image']
        #     texts = [ex['pos'], ex['neg']]
        # else:
        #     img_path0 = ex['input']
        #     texts = ex['pos'], ex['prompt']
        # subtasks = ex['type'] if self.task == 'flickr_edit' else ''
        try:
            image_0 = Image.open(img_path0).convert('RGB')
            reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
            image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
            image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        except:
            image_0 = 0

        return dict(input=image_0, texts=texts, path=img_path0)

class OldFrameEditDataset(Dataset):
    def __init__(
        self,
        path: str = '../../change_descriptions/something-something',
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        task: str = 'flickr30k_text',
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        debug: bool = False,
    ):
        if split == "train":
            path = os.path.join(path, 'train.json')
        elif split == "val":
            path = os.path.join(path, 'validation.json')
        self.json = json.load(open(path, 'r'))
        np.random.shuffle(self.json)
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        #clean json (if first and last are not both present, remove)
        new_json = []
        for i in range(len(self.json)):
            video_id = self.json[i]['id']
            img_path0 = f'../../change_descriptions/something-something/frames/{video_id}/first.jpg'
            img_path1 = f'../../change_descriptions/something-something/frames/{video_id}/last.jpg'
            if os.path.exists(img_path0) and os.path.exists(img_path1):
                new_json.append(self.json[i])
        self.json = new_json
        if debug:
            self.json = self.json[:50]

    def __len__(self) -> int:
        return len(self.json)

    def __getitem__(self, i: int) -> dict[str, Any]:
        video_id = self.json[i]['id']
        img_path0 = f'../../change_descriptions/something-something/frames/{video_id}/first.jpg'
        img_path1 = f'../../change_descriptions/something-something/frames/{video_id}/last.jpg'
        prompt = self.json[i]['label']
        image_0 = Image.open(img_path0)
        image_1 = Image.open(img_path1)

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        # if i ever wanna reverse time
        # if torch.rand(1) > 0.5:
        #     image_0, image_1 = image_1, image_0
        #     prompt = caption0
            
        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


class EditDataset(Dataset):
    def __init__(
        self,
        path: str = 'data/clip-filtered-dataset',
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        self.split = split
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        self.genhowto = open('data/genhowto/genhowto_train_clip0.7_filtered.txt', 'r').readlines()
        # self.genhowto = open('data/genhowto/genhowto_train.txt', 'r').readlines()
        self.genhowto = [x.strip() for x in self.genhowto]

        new_genhowto = []
        for i in range(len(self.genhowto)):
            img_path, prompt, prompt2 = self.genhowto[i].split(':')
            new_genhowto.append((img_path, prompt, 'action'))
            new_genhowto.append((img_path, prompt2, 'state'))
        self.genhowto = new_genhowto

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]
        # shuffle seeds and genhowto
        # np.random.seed(42)
        # np.random.shuffle(self.seeds)
        # np.random.shuffle(self.genhowto)

    def __len__(self) -> int:
        return len(self.seeds) + len(self.genhowto)

    def __getitem__(self, i: int) -> dict[str, Any]:
        if i < len(self.seeds):
            name, seeds = self.seeds[i]
            propt_dir = Path(self.path, name)
            seed = seeds[torch.randint(0, len(seeds), ()).item()]
            with open(propt_dir.joinpath("prompt.json")) as fp:
                prompt = json.load(fp)["edit"]
            image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))
            image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))
        else:
            ex = self.genhowto[i - len(self.seeds)]
            # img_path, prompt, prompt2 = ex.split(':')
            # img_path = img_path.replace('changeit_detected_without_test', 'changeit_detected')
            # img_path = 'data/genhowto/' + img_path
            # full_img = Image.open(img_path).convert('RGB')
            # image_0 = full_img.crop((0, 0, full_img.width // 3, full_img.height))
            # image_1 = full_img.crop((full_img.width * 2 // 3, 0, full_img.width, full_img.height))
            # image_2 = full_img.crop((full_img.width // 3, 0, full_img.width * 2 // 3, full_img.height))
            # if torch.rand(1) > 0.5:
            #     image_1 = image_2
            #     prompt = prompt2
            img_path, prompt, type = ex
            img_path = img_path.replace('changeit_detected_without_test', 'changeit_detected')
            img_path = 'data/genhowto/' + img_path
            full_img = Image.open(img_path).convert('RGB')
            image_0 = full_img.crop((0, 0, full_img.width // 3, full_img.height))
            if type == 'action':
                image_1 = full_img.crop((full_img.width // 3, 0, full_img.width * 2 // 3, full_img.height))
            else:
                image_1 = full_img.crop((full_img.width * 2 // 3, 0, full_img.width, full_img.height))
            

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


class GenHowToDataset(Dataset):
    def __init__(
        self,
        path: str = 'data/clip-filtered-dataset',
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        self.split = split
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        self.genhowto = open('data/genhowto/genhowto_train.txt', 'r').readlines()
        self.genhowto = [x.strip() for x in self.genhowto]

        new_genhowto = []
        for i in range(len(self.genhowto)):
            img_path, prompt, prompt2 = self.genhowto[i].split(':')
            new_genhowto.append((img_path, prompt, 'action'))
            new_genhowto.append((img_path, prompt2, 'state'))
        self.genhowto = new_genhowto
        np.random.shuffle(self.genhowto)

        # with open(Path(self.path, "seeds.json")) as f:
        #     self.seeds = json.load(f)

        # split_0, split_1 = {
        #     "train": (0.0, splits[0]),
        #     "val": (splits[0], splits[0] + splits[1]),
        #     "test": (splits[0] + splits[1], 1.0),
        # }[split]

        # idx_0 = math.floor(split_0 * len(self.seeds))
        # idx_1 = math.floor(split_1 * len(self.seeds))
        # self.seeds = self.seeds[idx_0:idx_1]
        # shuffle seeds and genhowto
        # np.random.seed(42)
        # np.random.shuffle(self.seeds)
        # np.random.shuffle(self.genhowto)

    def __len__(self) -> int:
        return len(self.genhowto)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ex = self.genhowto[i]
        img_path, prompt, type = ex
        img_path = img_path.replace('changeit_detected_without_test', 'changeit_detected')
        img_path = 'data/genhowto/' + img_path
        full_img = Image.open(img_path).convert('RGB')
        image_0 = full_img.crop((0, 0, full_img.width // 3, full_img.height))
        if type == 'action':
            image_1 = full_img.crop((full_img.width // 3, 0, full_img.width * 2 // 3, full_img.height))
        else:
            image_1 = full_img.crop((full_img.width * 2 // 3, 0, full_img.width, full_img.height))
        

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


class EditDatasetEval(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 256,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.res = res

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]
        propt_dir = Path(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        with open(propt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)
            edit = prompt["edit"]
            input_prompt = prompt["input"]
            output_prompt = prompt["output"]

        image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))

        reize_res = torch.randint(self.res, self.res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")

        return dict(image_0=image_0, input_prompt=input_prompt, edit=edit, output_prompt=output_prompt)

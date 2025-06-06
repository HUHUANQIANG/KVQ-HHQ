import decord
import tqdm
import glob
import os.path as osp
import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import os
import cv2
import difflib
from functools import lru_cache

import random
import copy

#import skvideo.io

random.seed(42)

def get_resize_input(video,
    size_h=448,
    size_w=448,
    arp=False,
    **kwargs,):

    h,w = video.shape[-2:]

    ratio = max(size_h/ h, size_w/ w)
    try:
        if arp:
            h_new = max(int(h*ratio), size_h)
            w_new = max(int(w*ratio), size_w)
            resize_opt = torchvision.transforms.Resize((h_new, w_new), interpolation=3)
            rand_pos_h = int(h_new/2 - size_h/2)
            rand_pos_w = int(w_new/2 - size_w/2)
            #rand_pos_h = torch.randint(int(h_new - size_h)+1,(1,1))[0,0]
            #rand_pos_w = torch.randint(int(w_new - size_w)+1,(1,1))[0,0]
            video = resize_opt(video)
            video = video[:, :, rand_pos_h: rand_pos_h + size_h, rand_pos_w: rand_pos_w + size_w]
        else:
            resize_opt = torchvision.transforms.Resize((size_h, size_w), interpolation=3)
            video = resize_opt(video)
    except:
        a=1
    return video


def get_single_sample(
    video,
    sample_type="resize",
    **kwargs,
):
    #video = get_spatial_fragments(video, **kwargs)
    video = get_resize_input(video, **kwargs)
    return video


def get_spatial_and_temporal_samples(
    video_path,
    is_train=False,
    augment=False,
):
    video = {}
    imgs = cv2.imread(video_path)
    b, g, r = cv2.split(imgs)
    imgs = cv2.merge([r,g,b])
    imgs = torch.tensor(imgs)
    H, W,_ = imgs.shape
    imgs = imgs.unsqueeze(0).expand(16,H, W,3).permute(3, 0, 1, 2)

    imgs = get_single_sample(imgs)
    return imgs

    
import numpy as np
import random
    
    

class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        ## opt is a dictionary that includes options for video sampling
        
        super().__init__()
        
        
        self.video_infos = []
        self.ann_file = opt["anno_file"]
        self.data_prefix = opt["data_prefix"]
        
        self.data_prefix_filenames = os.listdir(self.data_prefix)
        self.opt = opt
        self.sample_types = opt["sample_types"]
        self.data_backend = opt.get("data_backend", "disk")
        self.augment = opt.get("augment", False)
        if self.data_backend == "petrel":
            from petrel_client import client
            self.client = client.Client(enable_mc=True)
        
        self.phase = opt["phase"]
        self.crop = opt.get("random_crop", False)
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        
        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            try:
                with open(self.ann_file, "r") as fin:
                    lines = fin.readlines()
                    for line in lines:
                        filename, scores_str = line.strip().split(" ", 1)
                        scores = scores_str.split(", ")
                        label = []
                        for score in scores:
                            # Extract the position and value from each score
                            position, value = score.strip("()").split(":")
                            value = float(value)
                            label.append(value)
                        self.video_infos.append(dict(filename=filename, label=label))


            except:
                #### No Label Testing
                video_filenames = sorted(glob.glob(self.data_prefix+"/*.mp4"))
                print(video_filenames)
                for filename in video_filenames:
                    self.video_infos.append(dict(filename=filename, label=-1))


    def refresh_hypers(self):
        if not hasattr(self, "initial_sample_types"):
            self.initial_sample_types = copy.deepcopy(self.sample_types)
        
        types = self.sample_types
        
        if "fragments_up" in types:
            ubh, ubw = self.initial_sample_types["fragments_up"]["fragments_h"] + 1, self.initial_sample_types["fragments_up"]["fragments_w"] + 1
            lbh, lbw = self.initial_sample_types["fragments"]["fragments_h"] + 1, self.initial_sample_types["fragments"]["fragments_w"] + 1
            dh, dw = types["fragments_up"]["fragments_h"], types["fragments_up"]["fragments_w"]

            types["fragments_up"]["fragments_h"] = random.randrange(max(lbh, dh-1), min(ubh, dh+2))
            types["fragments_up"]["fragments_w"] = random.randrange(max(lbw, dw-1), min(ubw, dw+2))
            
        if "resize_up" in types:
        
            types["resize_up"]["size_h"] = types["fragments_up"]["fragments_h"] * types["fragments_up"]["fsize_h"]
            types["resize_up"]["size_w"] = types["fragments_up"]["fragments_w"] * types["fragments_up"]["fsize_w"]
        
        self.sample_types.update(types)

        #print("Refreshed sample hyper-paremeters:", self.sample_types)

        
    def __getitem__(self, index):
        video_info = self.video_infos[index]
        filename = self.data_prefix + video_info["filename"]
        label = video_info["label"]

        ## Read Original Frames
        ## Process Frames
        data = {}
        data["fragments"] = get_spatial_and_temporal_samples(filename,self.phase == "train", self.augment and (self.phase == "train"),)
        imgs = cv2.imread(filename)
        b, g, r = cv2.split(imgs)
        imgs = cv2.merge([r,g,b])
        imgs = torch.tensor(imgs)
        H, W,_ = imgs.shape
        imgs = imgs.unsqueeze(0).expand(16, H, W,3).permute(3, 0, 1, 2)
        data["origin"] = imgs
        
        for k, v in data.items():
            data[k] = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
        
        data["gt_label"] = label
        data["name"] = osp.basename(video_info["filename"])
        
        return data
    
    def __len__(self):
        return len(self.video_infos)

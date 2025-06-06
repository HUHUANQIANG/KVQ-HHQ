import torch
import cv2
import random
import os.path as osp
import os
import matplotlib.pyplot as plt
import vqa.models as models
import local_perception_data as datasets
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from collections import OrderedDict
import argparse
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

from time import time
from tqdm import tqdm
import pickle
import math

import wandb
import yaml

from functools import reduce
from thop import profile
import copy


def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        #print(np.std(gt), np.mean(gt))
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments"]

def profile_inference(inf_set, model, device):
    video = {}
    data = inf_set[0]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device).unsqueeze(0)
    with torch.no_grad():
        flops, params = profile(model, (video, ))
    print(f"The FLOps of the Variant is {flops/1e9:.1f}G, with Params {params/1e6:.2f}M.")

def inference_set(inf_loader, model, device, best_, wandb_flag=False, ddp_flag=False, save_model=False, suffix='s', save_name="divide"):

    results = []
    best_s, best_p, best_k, best_r= best_
    s_list = []
    p_list = []
 
    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video, video_up = {}, {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                ## Reshape into clips

        with torch.no_grad():
            _, feat, _, result["pr_labels"], weight = model(video)
            result["pr_labels"] = result["pr_labels"].mean(2)
            result["pr_labels"] = F.avg_pool2d(result["pr_labels"], kernel_size = (2, 2), stride = (2, 2)).reshape(49)
            result["pr_labels"], feat = result["pr_labels"].cpu().numpy(), feat.cpu().numpy()
        

        result["gt_label"] = torch.tensor(data["gt_label"])
        del video, video_up
        # result['frame_inds'] = data['frame_inds']
        # del data
        results.append(result)
        gt = [item for item in result["gt_label"]]
        pr = [item for item in result["pr_labels"]]
        pr = rescale(pr, gt)
        s_sample = spearmanr(gt, pr)[0]
        p_sample = pearsonr(gt, pr)[0]
        s_list.append(s_sample)
        p_list.append(p_sample)

        
    ## generate the demo video for video quality localization
    gt_labels = [item for r in results for item in r["gt_label"]]
    pr_labels = [item for r in results for item in r["pr_labels"]]
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())
    print(s)
    print(p)
    print(np.mean(s_list))
    print(np.mean(p_list))

    del results, result #, video, video_up
    torch.cuda.empty_cache()

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )

    return best_s, best_p, best_k, best_r, s, p, k, r

    # torch.save(results, f'{args.save_dir}/results_{dataset.lower()}_s{32}*{32}_ens{args.famount}.pkl')


def init_seeds(seed=0):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="./configs/local_perception.yaml", help="the config file"
    )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ## adaptively choose the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ## defining model and loading checkpoint
    bests_ = []
    model = getattr(models, config["model"]["type"])(**config["model"]["args"]).to(device)
    path = config["test_load_path"]
    state_dict = torch.load(path, map_location=device)["state_dict"]
    from collections import OrderedDict
    i_state_dict = OrderedDict()
    for key in state_dict.keys():
        if "module" in key:
            i_state_dict[key[7:]] = state_dict[key]
        else:
            i_state_dict[key] = state_dict[key]
    model.load_state_dict(i_state_dict, strict=True)
    model.to(device)

    val_datasets = {}
    for key in config["data"]:
        if key.startswith("val"):
            val_datasets[key] = getattr(datasets, 
                                        config["data"][key]["type"])(config["data"][key]["args"])
    val_loaders = {}
    for key, val_dataset in val_datasets.items():
        val_loaders[key] = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, num_workers=config["num_workers"], pin_memory=True,
            )

    bests = {}
    for key in val_loaders:
        model.eval()
        bests[key] = -1,-1,-1,1000
        profile_inference(val_dataset, model, device)
        bests[key] = inference_set(
            val_loaders[key],
            model,
            device, bests[key][0:4], wandb_flag=config["wandb"]["flag"], ddp_flag=config['ddp'], save_model=config["save_model"], save_name=config["name"],
            suffix=key+"_s",
        )



if __name__ == "__main__":
    main()
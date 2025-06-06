import torch
import cv2
import random
import os.path as osp
import os
import matplotlib.pyplot as plt
import vqa.models as models
import vqa.datasets as datasets
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

import time
from tqdm import tqdm
import pickle
import math

import wandb
import yaml

from functools import reduce
from thop import profile
import copy
import os
torch.autograd.set_detect_anomaly(True) 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致

def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split(",")
            filename, _, _, label = line_split
            label = float(label)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, label=label))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
    )


def rank_loss(y_pred, y):
    # sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    # y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    # sigma, m = torch.std_mean(y, unbiased=False)
    # y = (y - m) / (sigma + 1e-8)
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def rescaled_l2_loss(y_pred, y):
    y_pred_rs = (y_pred - y_pred.mean()) / y_pred.std()
    y_rs = (y - y.mean()) / (y.std() + eps)
    return torch.nn.functional.mse_loss(y_pred_rs, y_rs)

def rplcc_loss(y_pred, y, eps=1e-8):
    ## Literally (1 - PLCC) / 2
    cov = torch.sum((y_pred - y_pred.mean())*(y - y.mean()))
    std = (torch.std(y_pred) + eps) * (torch.std(y) + eps)
    return (1 - cov / std) / 2

def self_similarity_loss(f, f_hat, f_hat_detach=False):
    if f_hat_detach:
        f_hat = f_hat.detach()
    return 1 - torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()

def contrastive_similarity_loss(f, f_hat, f_hat_detach=False, eps=1e-8):
    if f_hat_detach:
        f_hat = f_hat.detach()
    intra_similarity = torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()
    cross_similarity = torch.nn.functional.cosine_similarity(f, f_hat, dim=0).mean()
    return (1 - intra_similarity) / (1 - cross_similarity + eps)

def lpc(y, y_patch):
    B = y.shape[0]
    y = y.view(B, -1)
    y_patch = y_patch.view(B, -1)
    mean_y, std_y = y.mean(1), y.std(1)
    mean_y_patch, std_y_patch = y_patch.mean(1), y_patch.std(1)
    y = (y - mean_y.unsqueeze(1)) / (std_y.unsqueeze(1) + 1e-8)
    y_patch = (y_patch - mean_y_patch.unsqueeze(1)) / (std_y_patch.unsqueeze(1)+ 1e-8)
    
    lpc_loss = 1 - torch.nn.functional.cosine_similarity(y, y_patch).mean()

    return lpc_loss



def Smooth_L1_Loss(y_pred, y):
    loss = torch.nn.SmoothL1Loss(beta=0.01)
    return loss(y_pred, y/100)

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments"]


#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --config configs/fast-vqa.yaml > log/fast_vqa.log
class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"

def finetune_epoch(ft_loader, model, model_ema, optimizer, scheduler, device, epoch=-1, wandb_flag=False, ddp_flag=False, 
                   need_upsampled=True, need_feat=True, need_fused=False, need_separate_sup=False):
    model.train()
    iter_loss = AverageMeter('Iter loss')
    iter_p_loss = AverageMeter('Iter p_loss')
    iter_r_loss = AverageMeter('Iter r_loss')
    wandb_sum_p_loss = 0.0
    wandb_sum_rank_loss = 0.0
    wandb_sum_lpc_loss = 0.0
    wandb_sum_total_loss = 0.0
    wandb_count = 0
    wandb_log_count = 16
    lpc_flag = True

    for i, data in enumerate(tqdm(ft_loader, desc=f"Training in epoch {epoch}")):
        optimizer.zero_grad()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
        
        
        y = data["gt_label"].float().detach().to(device).unsqueeze(-1)
        if lpc_flag:
            B, C, T, H, W = video["fragments"].shape
            grid = 14
            time_grid = 16
            patch_video = video["fragments"].view(B,C,time_grid, T//time_grid, grid, H//grid, grid, W//grid).permute(0,2,4,6,1,3,5,7).reshape(time_grid*grid*grid*B,C,T//time_grid,H//grid, W//grid)
            scores, feat, act_mag, scores_local, weight, scores_local_patch  = model(video, epoch= epoch, inference=False,reduce_scores=True, patch_flag=True, patch_video = patch_video)
            scores_local_patch = scores_local_patch.reshape(B, time_grid,grid,grid, T//2//time_grid, scores_local.shape[-2]//grid, scores_local.shape[-1]//grid).permute(0,1,4,2,5,3,6).reshape(B,T//2,scores_local.shape[-2],scores_local.shape[-1])
        else:
            scores, feat, act_mag, scores_local, weight, scores_local_patch  = model(video, epoch= epoch, inference=False,reduce_scores=True)

        y_pred = scores
        y_pred = y_pred.mean((-3, -2, -1))
        
        lpc_weight = 0.02

        if ddp_flag==True:
            with torch.no_grad():
                all_y_pred = [torch.zeros_like(y_pred) for _ in range(torch.cuda.device_count())]
                torch.distributed.all_gather(all_y_pred, y_pred)
                all_y = [torch.zeros_like(y) for _ in range(torch.cuda.device_count())]
                torch.distributed.all_gather(all_y, y)

                if lpc_flag:
                    all_scores = [torch.zeros_like(scores_local) for _ in range(torch.cuda.device_count())]
                    torch.distributed.all_gather(all_scores, scores_localn)
                    all_scores_patch = [torch.zeros_like(scores_local_patch) for _ in range(torch.cuda.device_count())]
                    torch.distributed.all_gather(all_scores_patch, scores_local_patch)

            all_y_pred[int(os.environ["LOCAL_RANK"])] = y_pred
            y_pred = torch.cat(all_y_pred)
            y = torch.cat(all_y)
            p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)

            if lpc_flag:
                all_scores[int(os.environ["LOCAL_RANK"])] = scores_local
                scores_local = torch.cat(all_scores)
                scores_local_patch = torch.cat(all_scores_patch)
                lpc_loss  = lpc(scores_local, scores_local_patch)
                loss = p_loss + 0.5 * r_loss + lpc_weight*lpc_loss
            else:
                loss = p_loss + 0.5 * r_loss
        else:
            p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)

            if lpc_flag:
                lpc_loss  = lpc(scores_local, scores_local_patch)
                loss = p_loss + 0.5 * r_loss  + + lpc_weight*lpc_loss
            else:
                loss = p_loss + 0.5 * r_loss


        wandb_sum_p_loss = wandb_sum_p_loss + p_loss.item()*y.shape[0]
        wandb_sum_rank_loss = wandb_sum_rank_loss + r_loss.item()*y.shape[0]
        if lpc_flag:
            wandb_sum_lpc_loss = wandb_sum_lpc_loss + lpc_loss.item()*y.shape[0]
        wandb_sum_total_loss = wandb_sum_total_loss + loss.item()*y.shape[0]
        wandb_count = wandb_count + y.shape[0]
        if wandb_flag and (wandb_count >= wandb_log_count or i==len(ft_loader)-1):
            if ddp_flag==False or (ddp_flag and int(os.environ["LOCAL_RANK"])==0): 
                wandb.log(
                    {
                        "train/plcc_loss": wandb_sum_p_loss/wandb_count,
                        "train/rank_loss": wandb_sum_rank_loss/wandb_count,
                        "train/lpc_loss": wandb_sum_lpc_loss/wandb_count,
                        "train/total_loss": wandb_sum_total_loss/wandb_count,
                    }
                )
                wandb_sum_p_loss = 0.0
                wandb_sum_rank_loss = 0.0
                wandb_sum_lpc_loss = 0.0
                wandb_sum_total_loss = 0.0
                wandb_count = 0
        iter_p_loss.update(p_loss.item(), y.shape[0])
        iter_r_loss.update(r_loss.item(), y.shape[0])
        iter_loss.update(loss.item())

        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        optimizer.step()
        scheduler.step()

        
        if model_ema is not None:
            model_params = dict(model.named_parameters())
            model_ema_params = dict(model_ema.named_parameters())
            for k in model_params.keys():
                model_ema_params[k].data.mul_(0.999).add_(
                    model_params[k].data, alpha=1 - 0.999
                )
    model.eval()


    return iter_loss.avg, iter_p_loss.avg, iter_r_loss.avg

    
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

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video, video_up = {}, {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                ## Reshape into clips
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
            if key + "_up" in data:
                video_up[key] = data[key+"_up"].to(device)
                ## Reshape into clips
                b, c, t, h, w = video_up[key].shape
                video_up[key] = video_up[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
            #.unsqueeze(0)
        with torch.no_grad():
            result["pr_labels"], feat, _, _, _ = model(video)
            result["pr_labels"], feat = result["pr_labels"].cpu().numpy(), feat.cpu().numpy()
        result["gt_label"] = data["gt_label"].item()
        del video, video_up
        # result['frame_inds'] = data['frame_inds']
        # del data
        results.append(result)
        
    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"][:]) for r in results]

    if ddp_flag:
        gt_labels = torch.tensor(gt_labels).to(device)
        pr_labels = torch.tensor(pr_labels).to(device)
        with torch.no_grad():
            all_gt_labels = [torch.zeros_like(gt_labels) for _ in range(torch.cuda.device_count())]
            torch.distributed.all_gather(all_gt_labels, gt_labels)
            all_pr_labels = [torch.zeros_like(pr_labels) for _ in range(torch.cuda.device_count())]
            torch.distributed.all_gather(all_pr_labels, pr_labels)
        gt_labels = torch.cat(all_gt_labels)
        pr_labels = torch.cat(all_pr_labels)
        gt_labels = gt_labels.tolist()
        pr_labels = pr_labels.tolist()
    pr_labels = rescale(pr_labels, gt_labels)
    print(len(pr_labels))


    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    if wandb_flag:
        if ddp_flag==False or (ddp_flag and int(os.environ["LOCAL_RANK"])==0): 
            wandb.log({f"val_{suffix}/SRCC-{suffix}": s, f"val_{suffix}/PLCC-{suffix}": p, f"val_{suffix}/KRCC-{suffix}": k, f"val_{suffix}/RMSE-{suffix}": r})
    del results, result #, video, video_up
    torch.cuda.empty_cache()

    if s + p > best_s + best_p and save_model:
        state_dict = model.state_dict()
        os.makedirs('results/'+save_name+'/weights', exist_ok=True)
        torch.save(
            {
                "state_dict": state_dict,
                "validation_results": best_,
            },
            f"results/{save_name}/weights/{save_name}_{suffix}_dev_v0.0.pth",
        )

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )

    if wandb_flag:
        if ddp_flag==False or (ddp_flag and int(os.environ["LOCAL_RANK"])==0): 
            wandb.log(
                {
                    f"val_{suffix}/best_SRCC-{suffix}": best_s,
                    f"val_{suffix}/best_PLCC-{suffix}": best_p,
                    f"val_{suffix}/best_KRCC-{suffix}": best_k,
                    f"val_{suffix}/best_RMSE-{suffix}": best_r,
                }
            )

    if ddp_flag==False or (ddp_flag and int(os.environ["LOCAL_RANK"])==0): 
        print(
            f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
        )

    return best_s, best_p, best_k, best_r, s, p, k, r

    # torch.save(results, f'{args.save_dir}/results_{dataset.lower()}_s{32}*{32}_ens{args.famount}.pkl')


def draw_train(loss, loss_p, loss_r, save_path, ema=False, linear=False):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(loss, "ro-", label="Train loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 3, 2)
    plt.plot(loss_p, "ro-", label="Train loss_p")
    plt.xlabel("epoch")
    plt.ylabel("loss_p")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(loss_r, "ro-", label="Train loss_r")
    plt.xlabel("epoch")
    plt.ylabel("loss_r")
    plt.legend()

    filename = 'results/'+ save_path +'/draws'
    os.makedirs(filename, exist_ok=True)
    filename += '/train'
    if ema:
        filename += '_ema'
    if linear:
        filename += '_linear'    
    filename += '.jpg'

    plt.savefig(filename)
    plt.close()

def draw_valid(s, p, k, r, save_path, ema=False, linear=False):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.plot(s, "ro-", label="s")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("s")

    plt.subplot(2, 2, 2)
    plt.plot(p, "ro-", label="p")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("p")

    plt.subplot(2, 2, 3)
    plt.plot(k, "ro-", label="k")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("k")

    plt.subplot(2, 2, 4)
    plt.plot(r, "ro-", label="r")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("r")

    filename = 'results/'+ save_path +'/draws'
    os.makedirs(filename, exist_ok=True)
    filename += '/valid'
    if ema:
        filename += '_ema'
    if linear:
        filename += '_linear'    
    filename += '.jpg'
    plt.savefig(filename)
    plt.close()


def init_seeds(seed=0):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="./configs/kvq.yaml", help="the config file"
    )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ## adaptively choose the device
    if config["ddp"] == True:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"


    if config['ddp']==False or (config['ddp'] and local_rank==0):
        print(config)

    a = torch.tensor([0]).to(device)
    ## defining model and loading checkpoint
    model = getattr(models, config["model"]["type"])(**config["model"]["args"]).to(device)
    from collections import OrderedDict
    i_state_dict = OrderedDict()


    
    if config.get("split_seed", -1) > 0:
        num_splits = 10
    else:
        num_splits = 1
        
    for split in range(num_splits):
        
        val_datasets = {}
        for key in config["data"]:
            if key.startswith("val"):
                val_datasets[key] = getattr(datasets, 
                                            config["data"][key]["type"])(config["data"][key]["args"])


        val_loaders = {}
        valid_sampler = {}
        val_s={}
        val_p={}
        val_k={}
        val_r={}
        linear_val_s={}
        linear_val_p={}
        linear_val_k={}
        linear_val_r={}
        val_s_ema={}
        val_p_ema={}
        val_k_ema={}
        val_r_ema={}
        linear_val_s_ema={}
        linear_val_p_ema={}
        linear_val_k_ema={}
        linear_val_r_ema={}
        for key, val_dataset in val_datasets.items():
            val_s[key]=[]
            val_p[key]=[]
            val_k[key]=[]
            val_r[key]=[]
            linear_val_s[key]=[]
            linear_val_p[key]=[]
            linear_val_k[key]=[]
            linear_val_r[key]=[]
            val_s_ema[key]=[]
            val_p_ema[key]=[]
            val_k_ema[key]=[]
            val_r_ema[key]=[]
            linear_val_s_ema[key]=[]
            linear_val_p_ema[key]=[]
            linear_val_k_ema[key]=[]
            linear_val_r_ema[key]=[]
            if config['ddp']:
                valid_sampler[key] = torch.utils.data.distributed.DistributedSampler(val_datasets[key],shuffle=False,)
                val_loaders[key] = torch.utils.data.DataLoader(
                    val_dataset, batch_size=1, num_workers=config["num_workers"], pin_memory=True,sampler=valid_sampler[key],
                )
            else:
                val_loaders[key] = torch.utils.data.DataLoader(
                    val_dataset, batch_size=1, num_workers=config["num_workers"], pin_memory=True,
                )


        train_datasets = {}
        for key in config["data"]:
            if key.startswith("train"):
                train_dataset = getattr(datasets, config["data"][key]["type"])(config["data"][key]["args"])
                train_datasets[key] = train_dataset
        
        train_loaders = {}
        train_sampler = {}
        for key, train_dataset in train_datasets.items():
            if config['ddp']:
                init_seeds(42 + local_rank)
                train_sampler[key] = torch.utils.data.distributed.DistributedSampler(train_datasets[key],shuffle=True,)
                train_loaders[key] = torch.utils.data.DataLoader(
                    train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], sampler=train_sampler[key],
                )
            else:
                train_loaders[key] = torch.utils.data.DataLoader(
                    train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=True,
                )

        if config["wandb"]["flag"]:
            if config['ddp']==False or (config['ddp'] and local_rank==0):
                run = wandb.init(
                    project=config["wandb"]["project_name"],
                    name=config["name"]+f'_{split}' if num_splits > 1 else config["name"],
                    reinit=True,
                    config=config,
                )
        
        if "load_path" in config:
            state_dict = torch.load(config["load_path"], map_location=device)

            if "state_dict" in state_dict:
                ### migrate training weights from mmaction
                state_dict = state_dict["state_dict"]
                from collections import OrderedDict

                i_state_dict = OrderedDict()
                for key in state_dict.keys():
                    if "head" in key:
                        continue
                    if "cls" in key:
                        tkey = key.replace("cls", "vqa")
                    elif "backbone" in key:
                        i_state_dict[key] = state_dict[key]
                        i_state_dict["fragments_"+key] = state_dict[key]
                        if "attn" in key:
                            i_state_dict["fragments_"+key.replace(".attn.",".attn.cross_windowattn.")] = state_dict[key]
                    else:
                        i_state_dict[key] = state_dict[key]
            t_state_dict = model.state_dict()
            for key, value in t_state_dict.items():
                if key in i_state_dict and i_state_dict[key].shape != value.shape:
                    print(key)
                    i_state_dict.pop(key)

            print(model.load_state_dict(i_state_dict, strict=False))
        # if "load_path" in config:
        #     state_dict = torch.load(config["test_load_path"], map_location=device)["state_dict"]
        #     i_state_dict = OrderedDict()
        #     for state_dict_key in state_dict.keys():
        #         if "module" in state_dict_key:
        #             i_state_dict[state_dict_key[7:]] = state_dict[state_dict_key]
        #         else:
        #             i_state_dict[state_dict_key] = state_dict[state_dict_key]
        #     model.load_state_dict(i_state_dict, strict=False)
        if "test_load_path" in config:
            state_dict = torch.load(config["test_load_path"], map_location=device)["state_dict"]
            i_state_dict = OrderedDict()
            for state_dict_key in state_dict.keys():
                if "module" in state_dict_key:
                    i_state_dict[state_dict_key[7:]] = state_dict[state_dict_key]
                else:
                    i_state_dict[state_dict_key] = state_dict[state_dict_key]
            print(model.load_state_dict(i_state_dict, strict=False))


        if config['ddp']==False or (config['ddp'] and local_rank==0):  
            print(model)

        if config["ema"]:
            from copy import deepcopy
            model_ema = deepcopy(model)
        else:
            model_ema = None

        profile_inference(val_dataset, model, device)

        # finetune the model
        param_groups=[]
        for key, value in dict(model.named_parameters()).items():
            if "backbone" in key:
                if "cross_windowattn" in key:
                    param_groups += [{"params": value, "lr": config["optimizer"]["lr"] * config["optimizer"]["backbone_lr_mult"]}]
                else:
                    param_groups += [{"params": value, "lr": config["optimizer"]["lr"] * config["optimizer"]["backbone_lr_mult"]}]
            else:
                param_groups += [{"params": value, "lr": config["optimizer"]["lr"]}]

        optimizer = torch.optim.AdamW(lr=config["optimizer"]["lr"], params=param_groups,
                                    weight_decay=config["optimizer"]["wd"],
                                    )
        warmup_iter = 0
        for train_loader in train_loaders.values():
            warmup_iter += int(config["warmup_epochs"] * len(train_loader))
        max_iter = int((config["num_epochs"] + config["l_num_epochs"]) * len(train_loader))
        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda,
        )

        bests = {}
        bests_n = {}
        for key in val_loaders:
            bests[key] = -1,-1,-1,1000
            bests_n[key] = -1,-1,-1,1000


        if config['ddp'] == True:
        # DistributedDataParallel
            model.to(device)
            model= torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

            model_ema.to(device)
            model_ema= torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_ema)
            model_ema = DDP(model_ema, device_ids=[local_rank], output_device=local_rank)

        Finetune_loss = []
        Finetune_p_loss = []
        Finetune_r_loss = []


        for epoch in range(config["num_epochs"]):
            if config['ddp']==False or (config['ddp'] and local_rank==0):
                print(f"Finetune Epoch {epoch}:")

            if config['ddp']:
                for key, train_dataset in train_datasets.items():
                    train_sampler[key].set_epoch(epoch)
                for key, val_dataset in val_datasets.items():
                    valid_sampler[key].set_epoch(epoch) 
            for key, train_loader in train_loaders.items():
                loss,p_loss,r_loss = finetune_epoch(
                    train_loader, model, model_ema, optimizer, scheduler, device,
                    epoch,config["wandb"]["flag"], config['ddp'],
                    config.get("need_upsampled", False), config.get("need_feat", False), config.get("need_fused", False), 
                )
                Finetune_loss.append(loss)
                Finetune_p_loss.append(p_loss)
                Finetune_r_loss.append(r_loss)
            for key in val_loaders:
                bests[key] = inference_set(
                    val_loaders[key],
                    model_ema if model_ema is not None else model,
                    device, bests[key][0:4], wandb_flag=config["wandb"]["flag"], ddp_flag=config['ddp'], save_model=config["save_model"], save_name=config["name"],
                    suffix=key+"_s",
                )
                val_s[key].append(bests[key][4])
                val_p[key].append(bests[key][5])
                val_k[key].append(bests[key][6])
                val_r[key].append(bests[key][7])
                    
        if config["num_epochs"] > 0:
            draw_train(Finetune_loss, Finetune_p_loss,Finetune_r_loss, save_path=config["name"], ema=False, linear=False)
            for key in val_loaders:
                if config['ddp']==False or (config['ddp'] and local_rank==0):
                    draw_valid(val_s[key], val_p[key], val_k[key], val_r[key], save_path=config["name"], ema=False, linear=False)
                    print(
                        f"""For the finetuning process on {key} with {len(val_loaders[key])} videos,
                        the best validation accuracy of the model-s is as follows:
                        SROCC: {bests[key][0]:.4f}
                        PLCC:  {bests[key][1]:.4f}
                        KROCC: {bests[key][2]:.4f}
                        RMSE:  {bests[key][3]:.4f}."""
                    )
                    draw_valid(val_s_ema[key], val_p_ema[key], val_k_ema[key], val_r_ema[key], save_path=config["name"], ema=True, linear=False)
                    print(
                        f"""For the finetuning process on {key} with {len(val_loaders[key])} videos,
                        the best validation accuracy of the model-n is as follows:
                        SROCC: {bests_n[key][0]:.4f}
                        PLCC:  {bests_n[key][1]:.4f}
                        KROCC: {bests_n[key][2]:.4f}
                        RMSE:  {bests_n[key][3]:.4f}."""
                    )

        
        if config["wandb"]["flag"]:
            if config['ddp']==False or (config['ddp'] and local_rank==0):
                run.finish()



if __name__ == "__main__":
    main()

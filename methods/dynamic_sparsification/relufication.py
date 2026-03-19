#!/usr/bin/env python3
import os
import re
import ast
import math
import random
import logging
import shutil
import copy
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Type

import numpy as np
import torch
import torchvision
from torch import nn, autocast
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from omegaconf import OmegaConf
import PIL.Image as PImage
import PIL.ImageDraw as PImageDraw
from PIL import Image
from tqdm import tqdm

import torch_fidelity

from architectures.moe.moe_layers import ExecuteAllExperts, CustomKernelExperts
from architectures.moe.moefication import add_routers, MoeficationMoE, replace_with_moes
from architectures.moe.dsti import dsti_mlp_filter_condition, replace_with_relu, find_gelu_activations
from architectures.pretrained import get_var_d16

from common import get_default_args, INIT_NAME_MAP, LOSS_NAME_MAP
from eval import autoregressive_infer_cfg_with_expert_plot
from train import TrainingContext, setup_accelerator, setup_data, setup_optimization, setup_files_and_logging, setup_state, make_vae
from utils import (load_model, save_state, remove_hooks, save_final,
                   Mixup, get_lrs, get_module_name,
                   add_save_inputs_hook, add_save_output_norm_hook)
from utils_var import arg_util
from utils_var.misc import create_npz_from_sample_folder
from trainer import VARTrainer
import dist
from collections import OrderedDict
import matplotlib.pyplot as plt



class RouterTrainingContext(TrainingContext):
    moe_modules: Dict[str, nn.Module] = None
    captured_layer_name_map: Dict[str, str] = None
    saved_inputs: Dict = None
    saved_output_norms: Dict = None
    hook_handles: List = None
    router_criterion_type: Type = None
    router_criterion: Callable = None
    initial_model = None



def setup_model(args, tc):
    assert args.model_class == 'dsti_router'

    model, tc.model_vae = get_var_d16()
    model.set_dense_module()
    
    init_path = Path(args.path_file_ft)
    final_state = torch.load(init_path, map_location=args.device)
    state_dict = final_state['model_state']
    model_arg = final_state['args'].model_args
    activations_to_sparsify = find_gelu_activations(model, **model_arg)
    model = replace_with_relu(model, activations_to_sparsify)
    model = model.to(args.device)
    new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(new_state_dict, strict=False)
    tc.model = tc.accelerator.prepare(model)
    tc.model.eval()

    for relu_index in args.relu_index_switch:
        model.set_scale_switch(relu_index)

        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')

        tc.model.rng.manual_seed(seed)
        rng = tc.model.rng

        sample_folder  = f'relu_index_{relu_index}'  # Where to save the 50,000 PNGs
        os.makedirs(sample_folder, exist_ok=True) 

        num_classes        = 1000               
        samples_per_class  = 10                 
        all_labels = np.repeat(np.arange(num_classes), samples_per_class)

        B = args.batch_size_eff           
        cfg, top_p, top_k = 1.5, 0.96, 900
        more_smooth       = False

        sparsity_list_mean = []
        sparsity_list_std = []
        sparsity_list_avg_layer = []
        for start in range(0, len(all_labels), B):
            end          = min(start + B, len(all_labels))
            batch_labels = torch.as_tensor(all_labels[start:end], device="cuda")
            B_cur        = len(batch_labels)              # last batch may be smaller

            with torch.no_grad(), torch.autocast("cuda", enabled=True, dtype=torch.float16):
                recon_B3HW, final_sparsity_mean, final_sparsity_std, avg_layer_sparsity = tc.model.autoregressive_infer_cfg(
                    B=B_cur,
                    label_B=batch_labels,
                    g_seed=None,
                    cfg=cfg,
                    top_k=top_k,
                    top_p=top_p,
                    more_smooth=more_smooth,
                    rng=rng,
                )

            sparsity_list_mean.append(torch.tensor(final_sparsity_mean))
            sparsity_list_std.append(torch.tensor(final_sparsity_std))
            sparsity_list_avg_layer.append(torch.tensor(avg_layer_sparsity))

            # -----------------------------------------------------------------------
            # Save individual PNGs and collect first-of-class examples
            # -----------------------------------------------------------------------
            np_images = (recon_B3HW.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            for i, img_array in enumerate(np_images):
                cls         = batch_labels[i].item()
                global_idx  = start + i

                # 1) Save individual PNG
                Image.fromarray(img_array).save(
                    os.path.join(sample_folder, f"class_{cls:04d}_{global_idx:05d}.png")
                )
        sparsity_list_mean = torch.stack(sparsity_list_mean)
        sparsity_list_std = torch.stack(sparsity_list_std)
        sparsity_list_avg_layer = torch.stack(sparsity_list_avg_layer)
        # ---------------------------------------------------------------------------
        # FID / IS metrics
        # ---------------------------------------------------------------------------
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=sample_folder,
            fid_statistics_file="adm_in256_stats.npz",
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )

        logging.info("*" * 100)
        logging.info(f'relu index switch {relu_index}')
        logging.info(f'Hoyer Sparsity mean: {sparsity_list_mean.mean(dim=0)}')
        logging.info(f'Hoyer Sparsity std: {sparsity_list_std.mean(dim=0)}')
        logging.info(f'Hoyer Sparsity avg layer: {sparsity_list_avg_layer.mean(dim=0)}')
        logging.info(f"Final FID)")
        logging.info(metrics_dict)
        logging.info("*" * 100)
        shutil.rmtree(sample_folder)
  
def train(args):
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    args: arg_util.Args = arg_util.init_dist_and_get_args(args)
    logging.info('Configured logging')
    tc = TrainingContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_model(args, tc)


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()






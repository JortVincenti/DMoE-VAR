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
import types      
import numpy as np
import torch
import torchvision
from torch import nn, autocast
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from architectures.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from omegaconf import OmegaConf
import PIL.Image as PImage
import PIL.ImageDraw as PImageDraw
from PIL import Image
from tqdm import tqdm
from fvcore.nn import flop_count_table
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
                   add_save_inputs_hook, add_save_output_norm_hook, count_scaled_dot_product_attention_ops, count_embedding_ops)
from utils_var import arg_util
from utils_var.misc import create_npz_from_sample_folder
from trainer import VARTrainer
import dist
from collections import OrderedDict
import matplotlib.pyplot as plt
import copy
import logging
import random, re, time
from pathlib import Path
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from fvcore.nn.jit_handles import elementwise_flop_counter
import fvcore.nn.jit_handles as _jit
from fvcore.nn.jit_handles import linear_flop_jit, addmm_flop_jit
import warnings
from omegaconf import OmegaConf
from utils_var import arg_util                       # your helper
from train      import (TrainingContext, setup_accelerator,
                        setup_files_and_logging, setup_model)
from eval       import autoregressive_infer_cfg_with_expert_plot
from architectures.moe.moefication import MoeficationMoE
from architectures.moe.moe_layers import MoELayer, ExecuteAllExperts, CustomKernelExperts
from architectures.pretrained        import get_var_d16
from fvcore.nn.jit_handles import get_shape
from collections import Counter
import logging
import time
from collections import defaultdict
from typing import List, Tuple, Dict
from fvcore.nn.jit_handles import (
    elementwise_flop_counter,
    linear_flop_jit,
    addmm_flop_jit,
    matmul_flop_jit,
    bmm_flop_jit,
    conv_flop_jit,
    einsum_flop_jit,
)
import torch
from accelerate import Accelerator
from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table
from sklearn.metrics import roc_auc_score
from torch.nn import MultiheadAttention, LayerNorm
from torchvision.models.vision_transformer import MLP
from utils import flop_count, get_module_by_name, remove_hooks, find_module_names, add_save_activations_hook
from contextlib import contextmanager
import torch.nn as nn


def _zero_ops(*_a, **_kw):                         # → Counter()   (0 FLOPs)
    return Counter()


OP_HANDLERS = {
    # ------------------------------------------------------------------
    #  built‑in fvcore high‑fidelity counters
    # ------------------------------------------------------------------
    "aten::linear":  linear_flop_jit,           # covers nn.Linear scripted
    "aten::addmm":   addmm_flop_jit,            # older Linear path
    "aten::matmul":  matmul_flop_jit,
    "aten::bmm":     bmm_flop_jit,
    "aten::_convolution": conv_flop_jit,        # conv2d/3d
    "aten::einsum":  einsum_flop_jit,
    # -------------------- Arithmetic elementwise (1 FLOP / element) --------------------
    'aten::add':  elementwise_flop_counter(0, 1),
    'aten::add_': elementwise_flop_counter(0, 1),
    'aten::radd': elementwise_flop_counter(0, 1),
    'aten::sub':  elementwise_flop_counter(0, 1),
    'aten::sub_': elementwise_flop_counter(0, 1),
    'aten::rsub': elementwise_flop_counter(0, 1),
    'aten::mul':  elementwise_flop_counter(0, 1),
    'aten::mul_': elementwise_flop_counter(0, 1),
    'aten::rmul': elementwise_flop_counter(0, 1),
    'aten::div':  elementwise_flop_counter(0, 1),
    'aten::div_': elementwise_flop_counter(0, 1),
    'aten::rdiv': elementwise_flop_counter(0, 1),
    'aten::exp':  elementwise_flop_counter(0, 1),
    'aten::abs':  elementwise_flop_counter(0, 1),
    'aten::ne':   elementwise_flop_counter(0, 1),
    'aten::lt':   elementwise_flop_counter(0, 1),

    # -------------------- Nonlinear activations (heavier than 1) -----------------------
    # Heuristics: sigmoid/tanh ≈ 4 scalar ops, SiLU ≈ 4, GELU ≈ 6
    'aten::sigmoid': elementwise_flop_counter(0, 4),
    'aten::tanh':    elementwise_flop_counter(0, 4),
    'aten::silu_':   elementwise_flop_counter(0, 4),
    'aten::gelu':    elementwise_flop_counter(0, 6),

    # -------------------- Softmax family (reduction + multiple ew ops) -----------------
    # softmax: subtract max + exp + sum + div ≈ 1 reduction + 3 elementwise
    'aten::softmax':     elementwise_flop_counter(1, 3),
    # log_softmax: softmax + log ≈ 1 reduction + 4 elementwise
    'aten::log_softmax': elementwise_flop_counter(1, 4),

    # -------------------- Reductions / scans -------------------------------------------
    'aten::sum':   elementwise_flop_counter(1, 0),  # N-1 adds
    'aten::mean':  elementwise_flop_counter(1, 1),  # sum + final divide
    'aten::cumsum':elementwise_flop_counter(1, 0),  # prefix-scan
    'aten::argmax':elementwise_flop_counter(1, 0),  # compare chain
    # (optional) keep topk as zero because we can’t model O(N log k) here reliably
    'aten::topk':  elementwise_flop_counter(0, 0),

    # -------------------- Indexing / data movement (counted as 0 FLOPs) ----------------
    'aten::one_hot':   elementwise_flop_counter(0, 0),
    'aten::flatten':   elementwise_flop_counter(0, 0),
    'aten::unflatten': elementwise_flop_counter(0, 0),
    'aten::scatter':   elementwise_flop_counter(0, 0),
    'aten::scatter_':  elementwise_flop_counter(0, 0),
    'aten::gather':    elementwise_flop_counter(0, 0),
    'aten::gather_':   elementwise_flop_counter(0, 0),
    'aten::fill_':     elementwise_flop_counter(0, 0),  # write only
    'aten::dropout_':  elementwise_flop_counter(0, 0),  # mask no arithmetic counted

    # -------------------- Pooling (comparisons = reduction) ----------------------------
    'aten::adaptive_max_pool2d': elementwise_flop_counter(1, 0),
    # -------------------- Custom you already have --------------------------------------
    'aten::scaled_dot_product_attention': count_scaled_dot_product_attention_ops,
    'aten::embedding':                    count_embedding_ops,

    # -------------------- Zero-ops  -------
    "aten::amin":    _zero_ops,
    "aten::cumsum_": _zero_ops,
    "aten::masked_fill_": _zero_ops,
    "aten::silu":    _zero_ops, 
    "aten::le":      _zero_ops,
    "aten::linalg_vector_norm": _zero_ops, 
    "aten::multinomial": _zero_ops,
    "aten::sort":        _zero_ops,
    "aten::clamp_min":   _zero_ops,
    "aten::expand_as":   _zero_ops,
    "aten::clamp_max":   _zero_ops,
    "aten::transpose_":  _zero_ops,
    "aten::upsample_bicubic2d": _zero_ops, 
    "aten::repeat":      _zero_ops,

    # -------------------- Done seperately  -------
    "profiler::_record_function_enter_new": _zero_ops,
    "profiler::_record_function_exit":      _zero_ops,
    "prim::PythonOp.MoeFirstLayerImplementation":        _zero_ops,
    "prim::PythonOp.MoeSecondLayerAtomicImplementation": _zero_ops,
    "prim::PythonOp.MoeSecondLayerMergingImplementation":_zero_ops,
    'prim::CallFunction': _zero_ops,
}

class ARInferenceWrapper(torch.nn.Module):
    def __init__(self, tc,
                 cfg=4.0, top_k=900, top_p=0.95, rng=0,
                 tau=1.0, fwd_mode="dynk_max", taus=None, expert_index_switch=0, more_smooth=False, lvl_pos=None):
        super().__init__()
        self.tc, self.cfg, self.top_k, self.top_p = tc, cfg, top_k, top_p
        self.rng, self.tau, self.fwd_mode = rng, tau, fwd_mode
        self.model = tc.model                  
        self.taus = taus
        self.router_stats = None
        self.expert_index_switch = expert_index_switch
        self.more_smooth = more_smooth
        self.lvl_pos = lvl_pos

    def forward(self, label_B, si, num_scales, cond_BD, next_token_map, f_hat, pn, cur_L):

        if torch.is_tensor(si):
            si = int(si.item())  
            num_scales = int(num_scales.item())
            pn = pn.item()

        B = label_B.size(0)

        ratio = si / self.model.num_stages_minus_1
        cond_BD_or_gss = self.model.shared_ada_lin(cond_BD)

        x = next_token_map
        total_flops = 0
        for index, b in enumerate(self.model.blocks):                   
            if self.taus:
                b.ffn.tau = self.taus[si]
            else:
                b.ffn.tau = self.tau
            x, block_flops = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None, current_scale=si)
            if torch.is_tensor(block_flops):
                total_flops += block_flops.sum().item()
            else:
                total_flops += block_flops


        logits_BlV = self.model.get_logits(x, cond_BD, current_scale=si)
        
        t = self.cfg * ratio
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

        idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=self.rng, top_k=self.top_k, top_p=self.top_p, num_samples=1)[:, :, 0]
       
        if not self.more_smooth:  # default case
            h_BChw = self.model.vae_quant_proxy[0].embedding(idx_Bl)
        else:
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
            h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio),
                                            tau=gum_t, hard=False, dim=-1, rng=self.rng) @ \
                    self.model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

        seq_len = h_BChw.shape[1]
        side    = int(math.sqrt(seq_len))
        h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.model.Cvae, side, side)

        f_hat, next_token_map = self.model.vae_quant_proxy[0].get_next_autoregressive_input(si, num_scales, f_hat, h_BChw)

        if si != self.model.num_stages_minus_1:
            next_token_map = next_token_map.view(B, self.model.Cvae, -1).transpose(1, 2)
            next_token_map = self.model.word_embed(next_token_map) + self.lvl_pos[:, cur_L:cur_L + self.model.patch_nums[si + 1] ** 2]     
            next_token_map = next_token_map.repeat(2, 1, 1)

        self.router_stats = total_flops
        return f_hat, next_token_map


def setup_model(args, tc):
    assert args.model_class == 'dsti_router'

    # Base class
    model, tc.model_vae = get_var_d16()
    tc.initial_model = copy.deepcopy(model)
    model.set_dense_module(scale_switch=args.expert_index_switch)


    init_path = Path(args.path_file_ft)
    final_state = torch.load(init_path, map_location=args.device)
    state_dict = final_state['model_state']
    model_arg = final_state['args'].model_args

    
    activations_to_sparsify = find_gelu_activations(model, **model_arg)
    model = replace_with_relu(model, activations_to_sparsify)

    model = model.to(args.device)
    new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())

    model.load_state_dict(new_state_dict, strict=False)


    final_path = Path(args.path_file_moe)
    final_state = torch.load(final_path, map_location=args.device)
    state_dict = final_state['model_state']
    model_arg = final_state['args'].model_args
    model, _ = replace_with_moes(model, **model_arg, module_filter_contition=dsti_mlp_filter_condition)
    model = model.to(args.device)
    model.load_state_dict(state_dict, strict=False)
    tc.moe_modules = add_routers(model, args.model_args)

    if args.use_router:
        final_router_path = Path(args.final_path_save + ".pth")
        final_state = torch.load(final_router_path, map_location=args.device)
        state_dict = final_state['model_state']
        model_arg = final_state['args'].model_args
        model.load_state_dict(state_dict, strict=False)

    tc.model = tc.accelerator.prepare(model)


def count_flops_for_all_taus(args, tc):

    setup_model(args, tc)                 
    device = args.device
    results = []
    tau = args.dsti_tau_to_eval[0]
    tc.model.set_scale_switch(args.expert_index_switch)
    tc.model.eval()
    
    for tau in args.dsti_tau_to_eval:
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

        forward_mode = 'dynk_max'
        num_classes        = 1
        samples_per_class  = 1
        all_labels = np.repeat(np.arange(num_classes), samples_per_class)

        # ---------------------------------------------------------------------------
        # Hyper-parameters
        # ---------------------------------------------------------------------------
        B = 1
        cfg, top_p, top_k = 1.5, 0.96, 900
        average_flops = []
        total_flops = 0
        total_moe_flops = 0
        total_trans_flops = 0
        for start in range(0, len(all_labels), B):
            end          = min(start + B, len(all_labels))
            batch_labels = torch.as_tensor(all_labels[start:end], device="cuda")
            B_cur        = len(batch_labels)

            with torch.no_grad(), torch.autocast("cuda", enabled=True, dtype=torch.float16):
                label_B = batch_labels
                # Prepare conditioning tokens and positional embeddings.
                cond_BD = sos = tc.model.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=tc.model.num_classes)), dim=0))
                lvl_pos = tc.model.lvl_embed(tc.model.lvl_1L) + tc.model.pos_1LC
                next_token_map = sos.unsqueeze(1).expand(2 * B, tc.model.first_l, -1) \
                                + tc.model.pos_start.expand(2 * B, tc.model.first_l, -1) \
                                + lvl_pos[:, :tc.model.first_l]

                # Initialize latent representation.
                f_hat = sos.new_zeros(B, tc.model.Cvae, tc.model.patch_nums[-1], tc.model.patch_nums[-1])

                for b in tc.model.blocks:
                    b.dense_blocks.attn.kv_caching(True)
                    b.attn.kv_caching(True)
                    b.ffn.forward_mode = forward_mode
                    b.ffn.experts.forward_mode = 'triton_atomic'

                # Get all MoE modules within the blocks.
                moe_modules = [m for b in tc.model.blocks for m in b.modules() if hasattr(m, 'gate') and hasattr(m, 'router')]
                original_gates = {m: m.gate for m in moe_modules}
                num_scales = len(tc.model.patch_nums)
                cur_L = 0
                
                wrapper = ARInferenceWrapper(
                    tc,
                    cfg       = cfg,
                    top_k     = top_k,
                    top_p     = top_p,
                    rng       = tc.model.rng,
                    tau       = (1.0 if args.dsti_tau_as_list else tau),
                    fwd_mode  = forward_mode,
                    taus       = (tau if args.dsti_tau_as_list else False),
                    expert_index_switch = args.expert_index_switch,
                    lvl_pos = lvl_pos
                ).to(device).eval()

                for si, pn in enumerate(tc.model.patch_nums):
                    logging.info(f'si: {si}, pn: {pn}')
                    cur_L += pn * pn

                    flops = (FlopCountAnalysis(wrapper, (batch_labels, si, num_scales, cond_BD, next_token_map, f_hat, pn, cur_L))
                            .set_op_handle(**OP_HANDLERS))
                    total_batch = flops.total()

                    old_token_map = next_token_map.clone()
                    old_f_hat = f_hat.clone()

                    f_hat, next_token_map = wrapper.forward(batch_labels, si, num_scales, cond_BD, next_token_map, f_hat, pn, cur_L)
                    router_stats = wrapper.router_stats
                    total_batch = total_batch

                    if router_stats > 0:
                        dummy_lbl = torch.tensor([0], device="cuda")       # any valid class id
                        one_token_one_expert_costs = benchmark_expert_costs_from_wrapper(
                            wrapper, batch_labels, si, num_scales, cond_BD, old_token_map, old_f_hat, pn, cur_L, OP_HANDLERS
                        )
                        average_flops.append((total_batch, router_stats * one_token_one_expert_costs['blocks.0.ffn']))
                    else:
                        average_flops.append((total_batch, 0))
                    
                    logging.info(f'total_batch: {total_batch}')
                    if router_stats > 0:
                        logging.info(f'router_stats: {router_stats}, moe_flops: { one_token_one_expert_costs["blocks.0.ffn"]}')
                    else:
                        logging.info(f'router_stats: {router_stats}, moe_flops: {0}')
                    logging.info('-'*100)
            scale=0
            for trans_flops, moe_flops in average_flops:
                flops_scale = (trans_flops+moe_flops)
                total_flops += flops_scale
                total_moe_flops += moe_flops
                total_trans_flops += trans_flops
                logging.info(
                    f"scale: {scale} | total FLOPs = {(flops_scale)/1e9:,.3f} G | Transformer FLOPs: {(trans_flops)/1e9:,.3f} G | " + 
                    f"moe cost: {moe_flops/1e9:,.3f} G"
                )
                scale += 1
            

        results.append((tau, total_flops, total_moe_flops, total_trans_flops))

    print("\n=== FLOP summary ===")
    logging.info(f'Performance for {args.expert_index_switch}, {args.final_path_save}, {args.path_file_moe}')
    for tau, tot, moe, trans in results:
        print(f"τ {tau} : {tot/1e9:,.3f}  GFLOPs, moe: {moe/1e9:,.3f} GFLOPs, trans: {trans/1e9:,.3f} GFLOPs")



@torch.no_grad()
def benchmark_expert_costs_from_wrapper(wrapper, dummy_labels, si, num_scales, cond_BD, next_token_map, f_hat, pn, cur_L, op_handlers):
    model  = wrapper.tc.model

    for b in wrapper.tc.model.blocks:
        b.ffn.experts.forward_mode = 'masked'

    # 1. locate MoE blocks and their experts sub-modules ------------------
    moe_names = find_module_names(model, lambda _, m: isinstance(m, MoELayer))
    experts_module_names = {}
    for moe in moe_names:
        ex = find_module_names(
            get_module_by_name(model, moe),
            lambda _, m: isinstance(m, (ExecuteAllExperts, CustomKernelExperts))
        )
        assert len(ex) == 1
        experts_module_names[moe] = f"{moe}.{ex[0]}"


    # 3. capture real inputs --------------------------------------------
    expert_paths = list(experts_module_names.values())
    experts_inputs, _, hooks = add_save_activations_hook(model, expert_paths)
    _ = wrapper(dummy_labels, si, num_scales, cond_BD, next_token_map, f_hat, pn, cur_L)

    # 4. FLOP price per token-expert ------------------------------------
    expert_costs = {}
    for moe_name, ex_path in experts_module_names.items():
        experts_mod  = get_module_by_name(model, ex_path)
        tokens, rout = experts_inputs[ex_path]
        dev, dt = rout.device, rout.dtype
        one_hot = torch.zeros((1, experts_mod.num_experts), device=dev, dtype=dt)
        one_hot[0, 0] = 1.0
        dummy_inp = (tokens[:1], one_hot)

        with torch.cuda.amp.autocast(dtype=dt):
            analysis = (FlopCountAnalysis(experts_mod, dummy_inp)
                     .set_op_handle(**op_handlers))
        flops = analysis.total()

        if isinstance(experts_mod, (ExecuteAllExperts, CustomKernelExperts)):
            flops /= experts_mod.num_experts

        expert_costs[moe_name] = flops
        break 

    for b in wrapper.tc.model.blocks:
        b.ffn.experts.forward_mode = 'triton_atomic'

    remove_hooks(hooks)
    return expert_costs





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
    tc = TrainingContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    count_flops_for_all_taus(args, tc)


def main():
    args = OmegaConf.merge(
        arg_util.get_default_args(),
        OmegaConf.from_cli(),
    )
    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        level=logging.INFO,
        force=True,
    )
    count_flops_for_all_taus(args)



if __name__ == "__main__":
    main()
import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.dynamic_sparsification.relufication import train as relufication
from train import train
from utils import generate_run_name, submit_job


def load_env_variables(file_path):
    with open(file_path, 'r') as env_file:
        for line in env_file:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

def main():
    # ════════════════════════ submitit setup ════════════════════════ #
    current_directory = os.getcwd()
    os.system('source /DMoE-VAR/user.env')
    load_env_variables('/DMoE-VAR/user.env')
    job_name = 'effbench'
    account = None
    qos = None
    partition = 'gpu_a100'
    timeout = 15*60
    gpus_per_task = 1
    gpu_type = ''
    cpus_per_gpu = None
    mem_per_gpu = '16G'

    # ════════════════════════ experiment settings ════════════════════════ #

    common_args = get_default_args()
    exp_ids = [1]
    common_args.runs_dir = Path(os.environ['RUNS_DIR'])
    common_args.dataset = 'TINYIMAGENET_PATH'
    common_args.dataset_args = {}
    common_args.dataset_args.variant = 'deit3_rrc'
    common_args.mixup_alpha = None
    common_args.cutmix_alpha = None
    common_args.mixup_smoothing = 0.1
    common_args.batch_size = 128
    common_args.loss_type = 'ce'
    common_args.loss_args = {}
    common_args.optimizer_class = 'adam'
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = 0.001
    common_args.optimizer_args.weight_decay = 0.0
    common_args.scheduler_class = 'linear'
    common_args.scheduler_args = {}
    common_args.epochs = 10
    common_args.scheduler_args.num_warmup_steps  = common_args.epochs * 1/50
    common_args.clip_grad_norm = 1.0
    
    common_args.eval_points = 20
    common_args.use_wandb = False
    common_args.mixed_precision = None

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # # ════════════════════════ dsti moe split settings ════════════════════════ #
    dsti_gpus_per_task = 1
    expert_split_args = deepcopy(common_args)
    expert_split_args.model_class = 'dsti_expert_split'
    expert_split_args.epochs = 1
    expert_split_args.batch_size = 64
    expert_split_args.model_args = {}
    expert_split_args.model_args.expert_size = 8
    expert_split_args.model_args.experts_class = 'execute_all'
    expert_split_args.activation = None
    expert_split_args.final_path_save = 'base_moe'

    # # ════════════════════════ dsti moe split ════════════════════════ #
    base_split_exp_names = []
    for exp_id in exp_ids:
        args = deepcopy(expert_split_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        args.base_on = exp_name
        base_run_name = f'{exp_id}

    exp_names.append(exp_name)
    base_split_exp_names.append(exp_names[-1])
    display_names.append(f'DSTI expert split')

    # ════════════════════════ dsti router training model settings ════════════════════════ #
    dsti_gpus_per_task = 1

    dsti_routing_args = deepcopy(common_args)
    dsti_routing_args.model_class = 'dsti_router'
    dsti_routing_args.router_loss_type = 'mse'
    dsti_routing_args.epochs = 1
    dsti_routing_args.batch_size = 256
    dsti_routing_args.optimizer_args.lr = 0.001
    dsti_routing_args.model_args = {}
    dsti_routing_args.model_args.depth = 2
    dsti_routing_args.model_args.width = 128
    dsti_routing_args.model_args.activation = 'gelu'
    dsti_routing_args.model_args.output_activation = 'abs'
    dsti_routing_args.dsti_router_labels_layer = 'output'
    dsti_routing_args.dsti_router_labels_norm = 2
    dsti_routing_args.eval_points = 4
    dsti_routing_args.mixed_precision = None
    final_path_save = 'relu_data_0.1'
    path_file_ft = 'DMoE-VAR/shared/results/effbench_runs/relu_sparse_ft_0/final.pth'
    path_file_moe = 'DMoE-VAR/shared/results/effbench_runs/relu_moe_0.1_e128/final.pth'

    # # ════════════════════════ dsti router training ════════════════════════ #
    base_routed_dsti_exp_names = []
    dsti_routing_args.dsti_tau_to_eval = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.81839, 0.81302, 0.78686]]
    dsti_routing_args.batch_size_eff = 128
    dsti_routing_args.dsti_tau_as_list = True
    dsti_routing_args.use_router = True
    dsti_routing_args.fid = True
    dsti_routing_args.debug = True
    dsti_routing_args.final_path_save = final_path_save
    dsti_routing_args.path_file_ft = path_file_ft
    dsti_routing_args.path_file_moe = path_file_moe
    dsti_routing_args.expert_index_switch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dsti_routing_args.model_experts_size = 128
    dsti_routing_args.relu_index_switch = [9]

    args = deepcopy(dsti_routing_args)
    relufication(args)

if __name__ == '__main__':
    main()

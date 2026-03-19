import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.dynamic_sparsification.expert_split import train as dsti_expert_split
from methods.dynamic_sparsification.rep_distill_train import train as mha_distill
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.dynamic_sparsification.train_routers import train as dsti_train_routers
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
    partition = 'gpu_h100'
    timeout = 24*60 
    gpus_per_task = 1
    gpu_type = ''
    cpus_per_gpu = None
    mem_per_gpu = '16G'

    executor = submitit.AutoExecutor(folder=os.environ['LOGS_DIR'])
    executor.update_parameters(
        stderr_to_stdout=True,
        timeout_min=timeout,
        slurm_job_name=job_name,
        slurm_account=account,
        slurm_qos=qos,
        slurm_partition=partition,
        slurm_ntasks_per_node=1,
        slurm_cpus_per_gpu=cpus_per_gpu,
        slurm_mem_per_gpu=mem_per_gpu,
    )

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
    expert_split_args.model_args.expert_size = 32   
    expert_split_args.model_args.experts_class = 'custom_kernel'
    expert_split_args.activation = None
    expert_split_args.final_path_save = 'base_moe'

    # # ════════════════════════ dsti moe split ════════════════════════ #
    base_split_exp_names = []
    for exp_id in exp_ids:
        args = deepcopy(expert_split_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        args.base_on = exp_name
        base_run_name = f'{exp_id}'

    exp_names.append(exp_name)
    base_split_exp_names.append(exp_names[-1])
    display_names.append(f'DSTI expert split')

    # ════════════════════════ dsti router training model settings ════════════════════════ #

    # dsti_gpus_per_task = 4
    dsti_gpus_per_task = 1

    dsti_routing_args = deepcopy(common_args)
    dsti_routing_args.model_class = 'dsti_router'
    dsti_routing_args.router_loss_type = 'mse'
    dsti_routing_args.epochs = 2
    dsti_routing_args.batch_size = 256
    dsti_routing_args.optimizer_args.lr = 0.001
    dsti_routing_args.model_args = {}
    dsti_routing_args.model_args.depth = 2
    dsti_routing_args.model_args.width = 128
    dsti_routing_args.model_args.activation = 'gelu'
    dsti_routing_args.model_args.output_activation = 'abs'
    dsti_routing_args.dsti_router_labels_layer = 'output'
    dsti_routing_args.dsti_router_labels_norm = 2
    dsti_routing_args.dsti_tau_to_eval = [0.9, 0.925, 0.93, 0.935, 0.94, 0.945,0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0]
    dsti_routing_args.dsti_expert_selection_mode = 'dynk_max'
    dsti_routing_args.eval_points = 4
    dsti_routing_args.mixed_precision = None
    dsti_routing_args.fid = False
    dsti_routing_args.debug = False
    dsti_routing_args.use_router = False

    final_path_save = [
        'relu_var_d16',
        'relu_var_d20',
    ]

    path_file_ft = [
        'VAR-main/local_output/ar-ckpt-d16-best.pth',
        'VAR-main/local_output/ar-ckpt-d20-best.pth',

    ]

    path_file_moe = [
        'DMoE-VAR/shared/results/effbench_runs/relu_moe_0_e128_var_d16/final.pth',
        'DMoE-VAR/shared/results/effbench_runs/relu_moe_0_e128_var_d20/final.pth',
    ]
    depth_list = [16, 20]


    # ════════════════════════ dsti router training ════════════════════════ #
    base_routed_dsti_exp_names = []
    for base_on_exp_name in base_split_exp_names:
        for exp_id in exp_ids:
            for i in range(len(final_path_save)):

                dsti_routing_args.path_file_moe = path_file_moe[i]
                dsti_routing_args.final_path_save = final_path_save[i]
                dsti_routing_args.activation = 'relu'
                dsti_routing_args.path_file_ft = path_file_ft[i]
                dsti_routing_args.var_d = depth_list[i]

                args = deepcopy(dsti_routing_args)
                args.exp_id = exp_id
                args.base_on = base_on_exp_name
                exp_name, run_name = generate_run_name(args)
                base_run_name = f'{base_on_exp_name}_{exp_id}'
                executor.update_parameters(slurm_additional_parameters={})
                job = submit_job(executor, dsti_train_routers, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
                jobs.append(job)
                run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_routed_dsti_exp_names.append(exp_names[-1])
        display_names.append(f'DSTI')

    # ═════════════════════════════════════════════════════════ #

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")

if __name__ == '__main__':
    main()

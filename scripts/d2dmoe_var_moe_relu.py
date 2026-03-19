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
    os.system('source DMoE-VAR/user.env')
    load_env_variables('DMoE-VAR/user.env')
    job_name = 'effbench'
    account = None
    qos = None
    partition = 'gpu_a100'
    timeout = 3*60

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
    expert_split_args.model_args.expert_size = 128
    expert_split_args.model_args.experts_class = 'custom_kernel'
    expert_split_args.activation = 'relu'
    
    final_path_save = [
        'relu_moe_0_e128_var_d16',
        'relu_moe_0_e128_var_d20',
    ]
    path_file = [
        '/home/jvincenti/VAR-main/local_output/ar-ckpt-d16-best.pth',
        '/home/jvincenti/VAR-main/local_output/ar-ckpt-d20-best.pth',
    ]

    depth_list = [16, 20]

    # # ════════════════════════ dsti moe split ════════════════════════ #
    base_split_exp_names = []
    for exp_id in exp_ids:
        for i in range(len(final_path_save)):
            expert_split_args.var_d = depth_list[i]
            expert_split_args.path_file = path_file[i]
            expert_split_args.final_path_save = final_path_save[i]
            args = deepcopy(expert_split_args)
            args.exp_id = exp_id
            exp_name, run_name = generate_run_name(args)
            args.base_on = exp_name
            base_run_name = f'{exp_id}'
            job = submit_job(executor, dsti_expert_split, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)

    exp_names.append(exp_name)
    base_split_exp_names.append(exp_names[-1])
    display_names.append(f'DSTI expert split')


if __name__ == '__main__':
    main()

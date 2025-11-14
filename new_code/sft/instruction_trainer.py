import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import BASE_PATH
sys.path.append(BASE_PATH)

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import time
import json
from tqdm import tqdm
import argparse

from instruction_prepare_data import get_examples
from model.modeling import get_model, save_adapter, load_adapter
from instruction_dataloader import get_dataset
import logging
import wandb
from util.utils import setup, count_parameters, get_wsd_scheduler, training_step

# 配置日志，同时输出到屏幕和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('log.txt', mode='w'),
        logging.StreamHandler()
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instruction (SFT) training for CompressLLM using args (no config.json)."
    )

    # 基本运行参数
    parser.add_argument(
        '--work_dir',
        type=str,
        required=True,
        help='Directory for this experiment (for loading pretrain adapter & saving SFT adapter).'
    )
    parser.add_argument(
        '--port',
        type=str,
        default='14527',
        help='port for DDP training'
    )

    # 模型 & 指令数据
    parser.add_argument(
        '--model_id',
        type=str,
        required=True,
        help='Base model id/path, e.g. LLM-Research/Llama-3.2-1B-Instruct'
    )
    parser.add_argument(
        '--instruction_dataset_repo',
        type=str,
        required=True,
        help='Instruction / QA dataset repo, e.g. mrqa-workshop/mrqa'
    )
    parser.add_argument(
        '--samples_num',
        type=int,
        required=True,
        help='Number of instruction training examples to sample.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for shuffling instruction data.'
    )

    # 压缩相关（需要和预训练阶段保持一致）
    parser.add_argument(
        '--chunk_size',
        type=int,
        required=True,
        help='Context chunk size used in CompressLLM (must match pretrain config).'
    )
    parser.add_argument(
        '--mem_size',
        type=int,
        required=True,
        help='Number of memory tokens per chunk (must match pretrain config).'
    )
    parser.add_argument(
        '--compress_ratio',
        type=int,
        required=True,
        help='Integer r such that mem_size * r == chunk_size.'
    )

    # 训练超参（SFT）
    parser.add_argument(
        '--batch_size_per_device',
        type=int,
        default=1,
        help='Per-GPU batch size.'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=8,
        help='Gradient accumulation steps.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-5,
        help='Learning rate for SFT.'
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Gradient clipping norm.'
    )
    parser.add_argument(
        '--log_step',
        type=int,
        default=50,
        help='Log every N optimizer steps.'
    )
    parser.add_argument(
        '--save_step',
        type=int,
        default=1000,
        help='Save adapter every N optimizer steps.'
    )

    return parser.parse_args()


# Training process
def train(rank, args, world_size):

    if rank == 0:
        wandb.init(project="local-experiment", entity="1762204162-", mode="disabled")

    # DDP 初始化
    setup(rank, world_size, args.port)
    torch.cuda.set_device(rank)

    # -------- 构造 sft_training_config / sft_task_config / data_config（替代 config.json） --------
    # SFT 训练超参
    sft_training_config = {
        "model_id": args.model_id,
        "batch_size_per_device": args.batch_size_per_device,
        "device_count": world_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "log_step": args.log_step,
        "save_step": args.save_step,
    }
    total_batch_size = (
        args.batch_size_per_device * world_size * args.gradient_accumulation_steps
    )
    sft_training_config["total_batch_size"] = total_batch_size

    # SFT 任务配置（CompressLLM 的 QA Task）
    sft_task_config = {
        "task_type": "Compress",   # instruction_dataloader / get_model 里用
        "chunk_size": args.chunk_size,
        "mem_size": args.mem_size,
        "compress_ratio": args.compress_ratio,
        "is_pretrain": False,
        "is_sft": True,
        "use_ae_loss": False,
        "use_lm_loss": True,
        "use_pe": False,
    }

    # 指令数据配置：instruction_prepare_data.get_examples
    output_dir = os.path.join(args.work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    data_config = {
        "model_id": args.model_id,
        "instruction_dataset_repo": args.instruction_dataset_repo,
        "samples_num": args.samples_num,
        "output_dir": output_dir,
        "seed": args.seed,
    }

    # 用于存盘记录
    config = {
        "sft_training_config": sft_training_config,
        "sft_task_config": sft_task_config,
        "data_config": data_config,
    }

    # -------- 准备数据（会缓存到 work_dir/output 下） --------
    train_examples, eval_examples = get_examples(**data_config)

    # 计算总 step 数
    training_steps = len(train_examples) // sft_training_config["total_batch_size"]

    # 丢掉尾部不能整除 batch 的样本
    train_examples = train_examples[: training_steps * sft_training_config["total_batch_size"]]
    if rank == 0:
        logging.info(
            f"[INFO] total_examples:{len(train_examples)} | training_steps:{training_steps}"
        )

    # 按 GPU 切分数据（连续切片）
    example_num_per_gpu = len(train_examples) // sft_training_config["device_count"]
    start_index = rank * example_num_per_gpu
    end_index = start_index + example_num_per_gpu
    train_examples = train_examples[start_index:end_index]

    logging.info(
        f"[INFO] rank{rank} training examples[{start_index}:{end_index}] "
        f"| example_nums:{len(train_examples)} | training_steps:{training_steps}"
    )

    # -------- 构建模型 & 加载预训练 adapter --------
    model = get_model(sft_training_config["model_id"], sft_task_config, rank)
    # 这里默认从同一个 work_dir 下的 output/adapter.pt 加载预训练好的 adapter
    pretrain_adapter_path = os.path.join(args.work_dir, "output", "adapter.pt")
    if os.path.exists(pretrain_adapter_path):
        model = load_adapter(model, save_path_and_name=pretrain_adapter_path, log=False)
        if rank == 0:
            logging.info(f"[INFO] Loaded pretrain adapter from {pretrain_adapter_path}")
    else:
        if rank == 0:
            logging.warning(f"[WARN] Pretrain adapter not found at {pretrain_adapter_path}, "
                            f"starting SFT from base model weights.")

    # 查看可训练参数
    if rank == 0:
        count_parameters(model, config)

    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # -------- DataLoader --------
    dataset = get_dataset(
        sft_task_config["task_type"],
        train_examples,
        sft_training_config["batch_size_per_device"]
    )
    loader = DataLoader(dataset, batch_size=None)

    # -------- 优化器 & scheduler --------
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=sft_training_config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scheduler = get_wsd_scheduler(optimizer, training_steps)

    accumulation_steps = sft_training_config["gradient_accumulation_steps"]
    step_num = 0

    optimizer.zero_grad()
    ddp_model.train()

    info_list = []
    start_time = time.time()

    for epoch in range(1):

        def save():
            if rank != 0:
                return
            # 保存 instruction SFT 的训练 log + 配置 + adapter
            with open(os.path.join(args.work_dir, "output/instruction_info.json"), "w") as f:
                json.dump(info_list, f, indent=4)

            with open(os.path.join(args.work_dir, "output/config.json"), "w") as f:
                json.dump(config, f, indent=4)

            save_adapter(
                ddp_model.module,
                save_path_and_name=os.path.join(
                    args.work_dir, "output/instruction_adapter.pt"
                ),
            )

        if rank == 0:
            progress_bar = tqdm(total=training_steps * accumulation_steps)

        for inputs in loader:
            step_num += 1

            if step_num % accumulation_steps == 0:
                loss_info = training_step(ddp_model, inputs, rank, accumulation_steps)
            else:
                with ddp_model.no_sync():
                    loss_info = training_step(
                        ddp_model, inputs, rank, accumulation_steps
                    )

            info_list.append(
                {
                    "run_time(hours)": (time.time() - start_time) / 3600,
                    "total_steps": training_steps,
                    "steps": step_num / accumulation_steps,
                    "training_loss": loss_info,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            if rank == 0:
                wandb.log(
                    {
                        "run_time(hours)": (time.time() - start_time) / 3600,
                        "total_steps": training_steps,
                        "steps": step_num / accumulation_steps,
                        "training_loss": loss_info,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

            if step_num % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    ddp_model.parameters(), sft_training_config["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step_num % (sft_training_config["log_step"] * accumulation_steps) == 0:
                if rank == 0:
                    logging.info(info_list[-1])

            if step_num % (sft_training_config["save_step"] * accumulation_steps) == 0:
                save()

            if rank == 0:
                progress_bar.update(1)

        if rank == 0:
            progress_bar.close()
        save()


# Launch multi-process training
if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()

    mp.spawn(
        train,
        args=(args, world_size),
        nprocs=world_size,
        join=True,
    )

"""
示例启动命令（假设 2 卡 SFT，预训练 adapter 已在 work_dir/output/adapter.pt）：

CUDA_VISIBLE_DEVICES=0,1 python instruction_trainer.py \
  --work_dir /home/syt/project/compressor_500/new_code/experiment/llama32_1b_500to1 \
  --port 12314 \
  --model_id /home/syt/project/Cram/model/model_scope_model/LLM-Research/Llama-3.2-1B-Instruct \
  --instruction_dataset_repo /home/syt/project/compressor_500/data/mrqa-workshop___mrqa \
  --samples_num 50000 \
  --seed 0 \
  --chunk_size 500 \
  --mem_size 1 \
  --compress_ratio 500 \
  --batch_size_per_device 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --max_grad_norm 1.0 \
  --log_step 10 \
  --save_step 100
"""

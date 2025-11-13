import random
import sys
import os

# 当前文件：.../new_code/pretrain/pre_trainer.py
PRETRAIN_DIR = os.path.dirname(os.path.abspath(__file__))   # .../new_code/pretrain
BASE_PATH = os.path.dirname(PRETRAIN_DIR)                   # .../new_code

# 把 new_code 加到 sys.path，方便 import model, util 等模块
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.multiprocessing as mp
import time
import json
from tqdm import tqdm
from transformers.models.llama.configuration_llama import LlamaConfig
import argparse

from pre_prepare_data import get_examples
from model.modeling import get_model, save_adapter, load_adapter
from pre_dataloader import get_dataset

import logging
import wandb

from util.utils import get_wsd_scheduler, training_step, setup, count_parameters

# 配置日志，同时输出到屏幕和文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("log.txt", mode="w"),
        logging.StreamHandler(),
    ],
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ICAE / 500xCompressor style pretraining with DPL (no config.json)."
    )

    # 基本运行参数
    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help="Experiment root dir for saving model & logs (e.g. ../experiment/llama32_1b_500to1)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="14527",
        help="port for DDP training",
    )

    # 模型 & 数据路径
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Base LLM model path / HF id, e.g. meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        required=True,
        help="Pretraining dataset repo/path, e.g. DKYoon/SlimPajama-6B or local path",
    )
    parser.add_argument(
        "--instruction_dataset_repo",
        type=str,
        default=None,
        help="(Optional) SFT / QA dataset path, not used in pretrain but kept for logging.",
    )

    # 数据准备超参（和你 pre_prepare_data 对齐）
    parser.add_argument(
        "--samples_num",
        type=int,
        default=320000,
        help="Number of training examples (plus 1000 eval examples).",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=510,
        help="Minimum token length (after tokenization) to keep in pre_prepare_data.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=2040,
        help="Maximum token length (after tokenization) to keep in pre_prepare_data.",
    )
    parser.add_argument(
        "--data_output_dir",
        type=str,
        default="output",
        help="Where pre_prepare_data saves *.pt (relative to pretrain/). "
             "Should match the --output_dir you used in pre_prepare_data.py.",
    )

    # 压缩相关超参（DPL / ICAE / 500x）
    parser.add_argument(
        "--chunk_size",
        type=int,
        required=True,
        help="Token count of a context chunk L (e.g. 500 or 496).",
    )
    parser.add_argument(
        "--mem_size",
        type=int,
        required=True,
        help="Number of memory tokens |M| (e.g. 1 or 16).",
    )
    parser.add_argument(
        "--compress_ratio",
        type=int,
        required=True,
        help="Integer r such that mem_size * r == chunk_size (approximate compression ratio).",
    )

    # 训练超参（不再从 config.json 读）
    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=1,
        help="Per-GPU batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=2.0,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--log_step",
        type=int,
        default=100,
        help="Log every N optimizer steps.",
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=1000,
        help="Save adapter every N optimizer steps.",
    )

    return parser.parse_args()


class CompressDataset(IterableDataset):
    def __init__(self, examples, batch_size):
        super(CompressDataset, self).__init__()
        self.examples = examples
        self.batch_size = batch_size

    def __iter__(self):
        input_ids = []
        ae_targets = []
        lm_targets = []
        for example in self.examples:
            input_ids.append(example["inputs"])
            ae_targets.append(example["ae_target"])
            lm_targets.append(example["lm_target"])

            if len(input_ids) == self.batch_size:
                yield {
                    "input_ids": torch.stack(input_ids),
                    "ae_targets": torch.stack(ae_targets),
                    "lm_targets": torch.stack(lm_targets),
                }
                input_ids = []
                ae_targets = []
                lm_targets = []


def get_dataset(task_type, examples, batch_size):
    if task_type == "Compress":
        return CompressDataset(examples, batch_size)
    raise Exception(f"Don't exist [{task_type}] task.")


# Training process
def train(rank, args, world_size):

    if rank == 0:
        # 这里默认关掉 wandb，如果你要开监控，把 mode 改成 "online"
        wandb.init(project="local-experiment", entity="1762204162-", mode="disabled")

    setup(rank, world_size, args.port)
    torch.cuda.set_device(rank)

    # --------- 构造 training_config / task_config / data_config（替代原来的 config.json） ---------
    training_config = {
        "model_id": args.model_id,
        "chunk_size": args.chunk_size,
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
    training_config["total_batch_size"] = total_batch_size

    task_config = {
        "is_pretrain": True,
        "is_sft": False,
        "chunk_size": args.chunk_size,
        "mem_size": args.mem_size,
        "compress_ratio": args.compress_ratio,
        "task_type": "Compress",
        "use_pe": False,
        "use_ae_loss": True,  # 预训练阶段 AE+LM
        "use_lm_loss": True,
    }

    data_config = {
        "dataset_repo": args.dataset_repo,
        "samples_num": args.samples_num,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "model_id": args.model_id,
        "instruction_dataset_repo": args.instruction_dataset_repo,
        "output_dir": args.data_output_dir,
    }

    # 和原来 config.json 里一样的 sanity check
    assert (
        world_size == training_config["device_count"]
    ), "device_count wrong (must equal number of visible GPUs)"
    assert training_config["total_batch_size"] == (
        training_config["batch_size_per_device"]
        * training_config["device_count"]
        * training_config["gradient_accumulation_steps"]
    ), "total_batch_size mismatch"
    assert training_config["chunk_size"] == task_config["chunk_size"], "chunk_size mismatch"
    assert (
        task_config["mem_size"] * task_config["compress_ratio"] == task_config["chunk_size"]
    ), "mem_size * compress_ratio must equal chunk_size"

    # 用来保存配置（方便复现），不再要求 config.json 作为输入
    config = {
        "pretrain_training_config": training_config,
        "pretrain_task_config": task_config,
        "data_config": data_config,
    }

    # --------- 准备数据（会复用你之前跑好的 output/*.pt）---------
    train_examples, eval_examples = get_examples(**data_config)

    # 计算总 step 数（和原始 UPL 一样：320k / 16 = 20k steps）
    training_steps = len(train_examples) // training_config["total_batch_size"]

    # 丢掉尾部不能整除 batch 的样本
    train_examples = train_examples[: training_steps * training_config["total_batch_size"]]
    if rank == 0:
        logging.info(
            f"[INFO] total_examples:{len(train_examples)} | training_steps:{training_steps}"
        )

    indices = list(range(len(train_examples)))

    device_count = training_config["device_count"]
    # 交错划分到不同 rank，保证长度分布接近
    train_examples = train_examples[rank::device_count]

    logging.info(
        f"[INFO] rank{rank} training examples: "
        f"{indices[rank::device_count][:4]} ... {indices[rank::device_count][-4:]} "
        f"| example_nums:{len(train_examples)} | training_steps:{training_steps}"
    )

    # --------- 构建模型（ICAE / 500x 实现都在 get_model 里） ---------
    model = get_model(training_config["model_id"], task_config, rank)

    # 查看可训练参数
    if rank == 0:
        count_parameters(model, config)

    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # DataLoader
    dataset = get_dataset(task_config["task_type"], train_examples, training_config["batch_size_per_device"])
    loader = DataLoader(dataset, batch_size=None)

    # 优化器 & scheduler
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=training_config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scheduler = get_wsd_scheduler(optimizer, training_steps)
    accumulation_steps = training_config["gradient_accumulation_steps"]
    step_num = 0

    optimizer.zero_grad()
    ddp_model.train()

    info_list = []
    start_time = time.time()

    # 创建保存目录（实验输出在 work_dir/output/ 下）
    exp_output_dir = os.path.join(args.work_dir, "output")
    os.makedirs(exp_output_dir, exist_ok=True)

    def save():
        if rank != 0:
            return
        # 保存训练日志 & 配置 & LoRA/adapter
        with open(os.path.join(exp_output_dir, "info.json"), "w") as f:
            json.dump(info_list, f, indent=4)

        with open(os.path.join(exp_output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        save_adapter(
            ddp_model.module,
            save_path_and_name=os.path.join(exp_output_dir, "adapter.pt"),
        )

    for epoch in range(1):  # 单 epoch，全部样本跑完
        if rank == 0:
            progress_bar = tqdm(total=training_steps * accumulation_steps)

        for inputs in loader:
            step_num += 1

            if step_num % accumulation_steps == 0:
                loss = training_step(ddp_model, inputs, rank, accumulation_steps)
            else:
                with ddp_model.no_sync():
                    loss = training_step(ddp_model, inputs, rank, accumulation_steps)

            info = {
                "run_time(hours)": (time.time() - start_time) / 3600,
                "total_steps": training_steps,
                "steps": step_num / accumulation_steps,
                "training_loss": loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            info_list.append(info)

            if rank == 0:
                wandb.log(info)

            if step_num % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    ddp_model.parameters(), training_config["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step_num % (training_config["log_step"] * accumulation_steps) == 0:
                if rank == 0:
                    logging.info(info_list[-1])

            if step_num % (training_config["save_step"] * accumulation_steps) == 0:
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

    # 仅用于确保 work_dir/output 存在（实验输出）
    exp_output_dir = os.path.join(args.work_dir, "output")
    os.makedirs(exp_output_dir, exist_ok=True)

    mp.spawn(
        train,
        args=(args, world_size),
        nprocs=world_size,
        join=True,
    )



"""
cd pretrain

CUDA_VISIBLE_DEVICES=0,1,2 python pre_trainer.py \
  --work_dir '../experiment/llama32_1b_500to1' \
  --port 14529 \
  --model_id /home/syt/project/Cram/model/model_scope_model/LLM-Research/Llama-3.2-1B-Instruct \
  --dataset_repo /home/syt/project/compressor_500/data/DKYoon___slim_pajama-6_b \
  --instruction_dataset_repo /home/syt/project/compressor_500/data/mrqa-workshop___mrqa \
  --samples_num 320000 \
  --min_len 510 \
  --max_len 2040 \
  --data_output_dir output \
  --chunk_size 500 \
  --mem_size 1 \
  --compress_ratio 500 \
  --batch_size_per_device 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_grad_norm 2.0 \
  --log_step 100 \
  --save_step 1000

cd pretrain

CUDA_VISIBLE_DEVICES=0,1 python pre_trainer.py \
  --work_dir '../experiment/qwen3_4b_500to1' \
  --port 14530 \
  --model_id /home/syt/project/model/qwen3/Qwen/Qwen3-4B-Instruct-2507 \
  --dataset_repo /home/syt/project/compressor_500/data/DKYoon___slim_pajama-6_b \
  --instruction_dataset_repo /home/syt/project/compressor_500/data/mrqa-workshop___mrqa \
  --samples_num 320000 \
  --min_len 510 \
  --max_len 2040 \
  --data_output_dir output \
  --chunk_size 500 \
  --mem_size 1 \
  --compress_ratio 500 \
  --batch_size_per_device 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_grad_norm 2.0 \
  --log_step 100 \
  --save_step 1000

"""
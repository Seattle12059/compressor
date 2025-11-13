from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
import os
import random
from tqdm import tqdm
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare AE/LM pretraining data for EPL/ICAE style training."
    )

    # 原来从 config['pretrain_training_config']['model_id'] 里拿的
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HF model id used for tokenization, e.g. meta-llama/Llama-3.2-1B",
    )

    # 原来 config['data_config']['dataset_repo']
    parser.add_argument(
        "--dataset_repo",
        type=str,
        required=True,
        help="HuggingFace dataset repo, e.g. cerebras/SlimPajama-627B",
    )

    # 原来 config['data_config']['samples_num']
    parser.add_argument(
        "--samples_num",
        type=int,
        required=True,
        help="Number of training samples to generate (plus 1000 eval samples).",
    )

    # 原来 config['data_config']['min_len'], 'max_len'
    parser.add_argument(
        "--min_len",
        type=int,
        required=True,
        help="Minimum token length (after tokenization) of a kept example.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        required=True,
        help="Maximum token length (after tokenization) of a kept example.",
    )

    # 代码里虽然没用到，但保持接口兼容
    parser.add_argument(
        "--instruction_dataset_repo",
        type=str,
        default=None,
        help="(Optional) extra instruction dataset repo; kept for compatibility.",
    )

    # 原来在 main 里写死的 output_dir = 'output'
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to store cached long_text.json and train/eval *.pt files.",
    )

    return parser.parse_args()


def get_long_text_list(dataset_repo, output_dir, min_len, max_len):
    """
    先按字符长度粗筛，避免每次都完整遍历 streaming dataset。
    结果缓存成 long_text.json。
    """
    cache_path = os.path.join(output_dir, "long_text.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            long_text_list = json.load(f)
        return long_text_list

    dataset = load_dataset(dataset_repo, split="train", streaming=True)

    long_text_list = []
    for example in tqdm(dataset, desc="Scanning raw dataset for long texts"):
        text = example["text"]
        # 1 token ≈ 2~6 chars，先用粗略范围过滤
        if min_len * 2 <= len(text) <= max_len * 6:
            long_text_list.append(text)

    os.makedirs(output_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(long_text_list, f, ensure_ascii=False)

    return long_text_list


def get_examples(
    model_id,
    dataset_repo,
    samples_num,
    min_len,
    max_len,
    instruction_dataset_repo,  # 仍然保留这个参数以兼容原 config 结构
    output_dir,
):
    model_name = model_id.split("/")[-1]
    train_data_name = os.path.join(
        output_dir,
        f"train_{model_name}_{samples_num}samples_{min_len}-{max_len}len.pt",
    )
    eval_data_name = os.path.join(
        output_dir,
        f"eval_{model_name}_{samples_num}samples_{min_len}-{max_len}len.pt",
    )

    if os.path.exists(train_data_name) and os.path.exists(eval_data_name):
        print(f"[INFO] Loading cached tensors from {output_dir} ...")
        return torch.load(train_data_name), torch.load(eval_data_name)

    print(
        f"[INFO] Preparing data\n"
        f"       train_file = {train_data_name}\n"
        f"       eval_file  = {eval_data_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    long_text_list = get_long_text_list(dataset_repo, output_dir, min_len, max_len)

    examples = []
    for text in tqdm(long_text_list, desc="Tokenizing & constructing examples"):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]

        # 严格按 token 数过滤
        if len(ids) < min_len:
            continue
        if len(ids) > max_len:
            continue

        # 一半做上下文，一半做 completion
        last_start = len(ids) // 2

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id is None，这个模型不适合当前脚本，请检查。")

        # 如果没有 bos，就不要强行加，直接用纯上下文
        if bos_id is not None:
            inputs = [bos_id] + ids[:last_start]
        else:
            inputs = ids[:last_start]

        ae_target = inputs + [eos_id]
        lm_target = ids[last_start:] + [eos_id]

        inputs = torch.LongTensor(inputs)
        ae_target = torch.LongTensor(ae_target)
        lm_target = torch.LongTensor(lm_target)

        examples.append(
            {"inputs": inputs, "ae_target": ae_target, "lm_target": lm_target}
        )

        if len(examples) == samples_num + 2000:
            break

    if len(examples) <= 1000:
        raise RuntimeError(
            f"Not enough examples collected ({len(examples)}), "
            f"try lowering min_len/max_len or samples_num."
        )

    train_examples = examples[1000:]
    eval_examples = examples[:1000]

    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_examples, train_data_name)
    torch.save(eval_examples, eval_data_name)

    print(
        f"[INFO] Done. Train: {len(train_examples)} examples, "
        f"Eval: {len(eval_examples)} examples."
    )

    return train_examples, eval_examples


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_examples, eval_examples = get_examples(
        model_id=args.model_id,
        dataset_repo=args.dataset_repo,
        samples_num=args.samples_num,
        min_len=args.min_len,
        max_len=args.max_len,
        instruction_dataset_repo=args.instruction_dataset_repo,
        output_dir=args.output_dir,
    )


"""
cd pretrain

python pre_prepare_data.py \
  --model_id /home/syt/project/Cram/model/model_scope_model/LLM-Research/Llama-3.2-1B-Instruct \
  --dataset_repo /home/syt/project/compressor_500/data/DKYoon___slim_pajama-6_b \
  --samples_num 320000 \
  --min_len 510 \
  --max_len 2040 \
  --output_dir output   已经成功了


python pre_prepare_data.py \
  --model_id /home/syt/project/model/qwen3/Qwen/Qwen3-4B-Instruct-2507 \
  --dataset_repo /home/syt/project/compressor_500/data/DKYoon___slim_pajama-6_b \
  --samples_num 320000 \
  --min_len 510 \
  --max_len 2040 \
  --output_dir output
  
  

"""
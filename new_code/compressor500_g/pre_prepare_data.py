from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os
import random
from tqdm import tqdm
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare regeneration data for 500xCompressor-style pretraining."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HF model name or local path (used only for tokenizer).",
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        required=True,
        help="HuggingFace dataset repo or local dataset path (must have a 'text' field).",
    )
    parser.add_argument(
        "--samples_num",
        type=int,
        required=True,
        help="Number of training examples to collect (exclude eval). "
             "Script will actually collect samples_num + 1000 to hold out eval.",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        required=True,
        help="Minimum token length (after tokenization) of each example.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        required=True,
        help="Maximum token length (after tokenization) of each example.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed .pt files and long_text.json cache.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling examples.",
    )
    args = parser.parse_args()
    return args


def get_long_text_list(dataset_repo, output_dir, min_len, max_len):
    """
    和原 pre_prepare_data 一样：
    先按字符长度粗筛，避免每次都完整遍历 streaming dataset。
    结果缓存成 long_text.json。
    """
    cache_path = os.path.join(output_dir, "long_text_500x.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            long_text_list = json.load(f)
        print(f"[INFO] Loaded cached long_text_list from {cache_path}, "
              f"size = {len(long_text_list)}")
        return long_text_list

    print(f"[INFO] Scanning raw dataset from {dataset_repo} (streaming) ...")
    dataset = load_dataset(dataset_repo, split="train", streaming=True)

    long_text_list = []
    for example in tqdm(dataset, desc="Scanning for long texts"):
        # 这里假定数据里有 'text' 字段，和你原来的一致
        text = example["text"]
        # 1 token ≈ 2~6 chars，先用粗略范围过滤
        if min_len * 2 <= len(text) <= max_len * 6:
            long_text_list.append(text)

    os.makedirs(output_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(long_text_list, f, ensure_ascii=False)

    print(f"[INFO] Saved long_text_list to {cache_path}, size = {len(long_text_list)}")
    return long_text_list


def get_500x_examples(
    model_id,
    dataset_repo,
    samples_num,
    min_len,
    max_len,
    output_dir,
    seed=42,
):
    """
    为 500xCompressor 预训练做数据：
    每个样本：
      - inputs:         原始 context 的 token id 序列（不含 BOS/EOS）
      - regen_targets:  [BOS] + context_ids + [EOS]
    只需要这两项，后面模型用它做 regeneration 即可。
    """
    model_name = model_id.split("/")[-1]

    train_data_name = os.path.join(
        output_dir,
        f"train_500x_{model_name}_{samples_num}samples_{min_len}-{max_len}len.pt",
    )
    eval_data_name = os.path.join(
        output_dir,
        f"eval_500x_{model_name}_{samples_num}samples_{min_len}-{max_len}len.pt",
    )

    # 如果已经有缓存，直接加载
    if os.path.exists(train_data_name) and os.path.exists(eval_data_name):
        print(f"[INFO] Found cached train/eval data:")
        print(f"       train_file = {train_data_name}")
        print(f"       eval_file  = {eval_data_name}")
        train_examples = torch.load(train_data_name)
        eval_examples = torch.load(eval_data_name)
        return train_examples, eval_examples

    print(
        f"[INFO] Preparing 500x pretrain data\n"
        f"       model_id  = {model_id}\n"
        f"       dataset   = {dataset_repo}\n"
        f"       samples   = {samples_num} (train) + 1000 (eval)\n"
        f"       min_len   = {min_len}\n"
        f"       max_len   = {max_len}\n"
        f"       output    = {output_dir}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    if eos_id is None:
        raise ValueError(
            "tokenizer.eos_token_id is None，这个模型不适合当前脚本，请检查。"
        )

    # 先粗筛一批长文本
    long_text_list = get_long_text_list(dataset_repo, output_dir, min_len, max_len)

    # 随机打乱，保证每次顺序不一样
    random.seed(seed)
    random.shuffle(long_text_list)

    examples = []
    minn = 10 ** 9
    maxn = 0

    # 需要 samples_num + 1000 个样本（多出来的 1000 个做 eval）
    target_total = samples_num + 1000

    for text in tqdm(long_text_list, desc="Tokenizing & constructing 500x examples"):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]

        # 严格按 token 数过滤
        if len(ids) < min_len:
            continue
        if len(ids) > max_len:
            continue

        # encoder 输入：原始 context（不加 BOS/EOS）
        inputs = ids

        # decoder 重建目标： [BOS] + ids + [EOS]
        if bos_id is not None:
            regen_targets = [bos_id] + ids + [eos_id]
        else:
            regen_targets = ids + [eos_id]

        minn = min(minn, len(ids))
        maxn = max(maxn, len(ids))

        example = {
            "inputs": torch.LongTensor(inputs),
            "regen_targets": torch.LongTensor(regen_targets),
        }
        examples.append(example)

        if len(examples) >= target_total:
            break

    if len(examples) <= 1000:
        raise RuntimeError(
            f"Not enough examples collected ({len(examples)}), "
            f"try lowering min_len/max_len or samples_num."
        )

    print(
        f"[INFO] Collected {len(examples)} examples. "
        f"Token length in [{minn}, {maxn}]"
    )

    # 前 1000 个做 eval，其余做 train（和之前脚本一致）
    eval_examples = examples[:1000]
    train_examples = examples[1000:]

    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_examples, train_data_name)
    torch.save(eval_examples, eval_data_name)

    print(
        f"[INFO] Done. Train: {len(train_examples)} examples\n"
        f"              Eval:  {len(eval_examples)} examples\n"
        f"       Saved to:\n"
        f"         {train_data_name}\n"
        f"         {eval_data_name}"
    )

    return train_examples, eval_examples


def main():
    args = parse_args()
    get_500x_examples(
        model_id=args.model_id,
        dataset_repo=args.dataset_repo,
        samples_num=args.samples_num,
        min_len=args.min_len,
        max_len=args.max_len,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

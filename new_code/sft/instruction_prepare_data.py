from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch import nn
import os
import random
from tqdm import tqdm
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare instruction (QA) data for SFT from MRQA-style datasets (args-based)."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HF model id or local path used for tokenization, e.g. LLM-Research/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--instruction_dataset_repo",
        type=str,
        required=True,
        help="Instruction / QA dataset repo, e.g. mrqa-workshop/mrqa",
    )
    parser.add_argument(
        "--samples_num",
        type=int,
        required=True,
        help="Number of training examples to sample from train split.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to cache JSON and save train/eval *.pt files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling training examples.",
    )
    return parser.parse_args()


def get_examples_list(instruction_dataset_repo, split, output_dir):
    """
    把 MRQA 这种多 split 数据合并起来，并缓存为 json：
    - train: 只用 train split
    - test: 合并 test + validation
    """
    instruction_dataset_repo_name = instruction_dataset_repo.split("/")[-1]
    cache_path = os.path.join(
        output_dir, f"{instruction_dataset_repo_name}_{split}_instruction_dataset.json"
    )

    # cache long text for preventing full dataset traversal on each preparation.
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            examples_list = json.load(f)
        return examples_list

    examples_list = []
    if split == "train":
        dataset = load_dataset(instruction_dataset_repo, split=split, streaming=True)
        for example in tqdm(dataset, desc=f"Processing {split} examples"):
            examples_list.append(example)
    else:
        # test 模式下，合并 test + validation
        dataset = load_dataset(instruction_dataset_repo, split="test", streaming=True)
        for example in tqdm(dataset, desc="Processing test examples"):
            examples_list.append(example)
        dataset = load_dataset(
            instruction_dataset_repo, split="validation", streaming=True
        )
        for example in tqdm(dataset, desc="Processing validation examples"):
            examples_list.append(example)

    os.makedirs(output_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(examples_list, f, ensure_ascii=False)

    return examples_list


def get_ids(instruction_dataset_repo_name, examples_list, tokenizer, split):
    """
    把原始 MRQA example 转成：
    - input_ids: context 部分 ([BOS] + "### Context:" + context)
    - lm_targets:
        train: question + answer（让模型学会在上下文 mem 上回答）
        test:  只有 question（用来评估时喂给模型）
    - instruction_target:
        train: question 部分 mask 掉(-100)，只对 answer 计算 loss
        test: 不返回这个字段
    """
    examples = []
    minn = 999999
    maxn = 0

    for example in tqdm(examples_list, desc="Tokenizing examples"):
        # 取一个答案用于训练即可
        answer_text = example["answers"][0]

        context = tokenizer(example["context"], add_special_tokens=False)["input_ids"]
        prompt = tokenizer(example["question"], add_special_tokens=False)["input_ids"]
        answer = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

        context_ids = (
            [tokenizer.bos_token_id]
            + tokenizer("### Context:\n", add_special_tokens=False)["input_ids"]
            + context
        )
        question_ids = (
            tokenizer("\n### Question:\n", add_special_tokens=False)["input_ids"]
            + prompt
            + tokenizer("\n### Answer:\n", add_special_tokens=False)["input_ids"]
        )
        answer_ids = answer + [tokenizer.eos_token_id]

        tot_len = len(context_ids) + len(question_ids) + len(answer_ids)
        minn = min(minn, tot_len)
        maxn = max(maxn, tot_len)

        # 生成 instruction_target：只对 answer 部分计算 loss
        instruction_target = [-100 for _ in question_ids] + [x for x in answer_ids]
        # 对齐 decoder 输出长度（因为 decoder 输入从 question 开始，context 走的是 mem）
        instruction_target = instruction_target[1:]

        inputs = torch.LongTensor(context_ids)
        if split == "train":
            lm_target = torch.LongTensor(question_ids + answer_ids)
        else:
            lm_target = torch.LongTensor(question_ids)

        instruction_target = torch.LongTensor(instruction_target)

        if split == "test":
            examples.append({"input_ids": inputs, "lm_targets": lm_target})
        else:
            examples.append(
                {
                    "input_ids": inputs,
                    "lm_targets": lm_target,
                    "instruction_target": instruction_target,
                }
            )

    print(f"len range: [{minn}:{maxn}]")
    return examples


def get_examples(
    model_id,
    instruction_dataset_repo,
    samples_num,
    output_dir="output",
    seed=0,
):
    model_name = model_id.split("/")[-1]
    instruction_dataset_repo_name = instruction_dataset_repo.split("/")[-1]

    train_data_name = os.path.join(
        output_dir,
        f"{instruction_dataset_repo_name}_train_{model_name}_{samples_num}samples_instruction.pt",
    )
    eval_data_name = os.path.join(
        output_dir,
        f"{instruction_dataset_repo_name}_eval_{model_name}_{samples_num}samples_instruction.pt",
    )

    print(f"in:train_data_name:{train_data_name}")
    if os.path.exists(train_data_name) and os.path.exists(eval_data_name):
        print("loading data...")
        return torch.load(train_data_name), torch.load(eval_data_name)

    print(f"preparing data : train_data_name:{train_data_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_examples_list = get_examples_list(
        instruction_dataset_repo, split="train", output_dir=output_dir
    )
    test_examples_list = get_examples_list(
        instruction_dataset_repo, split="test", output_dir=output_dir
    )

    random.seed(seed)
    random.shuffle(train_examples_list)
    train_examples_list = train_examples_list[:samples_num]

    train_data = get_ids(
        instruction_dataset_repo_name,
        train_examples_list,
        tokenizer,
        split="train",
    )
    test_data = get_ids(
        instruction_dataset_repo_name,
        test_examples_list,
        tokenizer,
        split="test",
    )

    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_data, train_data_name)
    torch.save(test_data, eval_data_name)

    return train_data, test_data


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_examples, eval_examples = get_examples(
        model_id=args.model_id,
        instruction_dataset_repo=args.instruction_dataset_repo,
        samples_num=args.samples_num,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(f"#train examples: {len(train_examples)}")
    print(train_examples[50] if len(train_examples) > 50 else train_examples[0])
    print(f"#eval examples: {len(eval_examples)}")
    print(eval_examples[50] if len(eval_examples) > 50 else eval_examples[0])

"""
示例命令：

python instruction_prepare_data.py \
  --model_id /home/syt/project/Cram/model/model_scope_model/LLM-Research/Llama-3.2-1B-Instruct \
  --instruction_dataset_repo /home/syt/project/compressor_500/data/mrqa-workshop___mrqa \
  --samples_num 50000 \
  --output_dir output_instruction_llama32_1b

"""

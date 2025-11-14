import os
import torch
from model.modeling import load_model_with_adapter


# ----------------------
# Utilities
# ----------------------

def get_device(model):
    return next(model.parameters()).device


def build_task_config(chunk_size=500, mem_size=1, compress_ratio=500):
    return {
        "task_type": "Compress",
        "chunk_size": chunk_size,
        "mem_size": mem_size,
        "compress_ratio": compress_ratio,
        "is_pretrain": False,
        "is_sft": True,
        "use_pe": True,
        "use_ae_loss": True,
        "use_lm_loss": True,
    }


def load_compress_model(model_id, adapter_path, chunk_size, mem_size, compress_ratio):
    task_config = build_task_config(chunk_size, mem_size, compress_ratio)
    print(f"Loading base: {model_id}")
    print(f"Loading adapter: {adapter_path}")

    model = load_model_with_adapter(
        model_id=model_id,
        task_config=task_config,
        rank=0,
        save_path_and_name=adapter_path,
        log=True,
    )
    model.eval()
    tokenizer = model.tokenizer
    return model, tokenizer


# ----------------------
# 1) 压缩 + 自由生成解压 (ae_inference)
# ----------------------

def ae_free_reconstruct(model, tokenizer, context_text):
    device = get_device(model)

    # Tokenize
    ctx_ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]
    if tokenizer.bos_token_id is not None:
        ctx_ids = [tokenizer.bos_token_id] + ctx_ids

    input_ids = torch.LongTensor(ctx_ids).unsqueeze(0).to(device)

    # Compress
    with torch.no_grad():
        compress_ids, compress_token, _ = model.compress({"input_ids": input_ids})

    print("\n=== Memory Token (Compressed Representation) ===")
    print(compress_token)
    print()

    # Free generate
    with torch.no_grad():
        gen_ids = model.ae_inference({"input_ids": input_ids})

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Compute free-generation token accuracy
    target = tokenizer(context_text, add_special_tokens=False)["input_ids"]
    if tokenizer.bos_token_id:
        target = [tokenizer.bos_token_id] + target

    # align prediction length
    pred = gen_ids[: len(target)]
    target = target[: len(pred)]

    correct = sum(int(a == b) for a, b in zip(pred, target))
    acc = correct / len(target)

    return gen_text, acc, compress_token


# ----------------------
# 2) 压缩 + teacher forcing 解压
# ----------------------

def ae_teacher_reconstruct(model, tokenizer, context_text):
    device = get_device(model)

    # tokenize context
    ctx_ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]
    if tokenizer.bos_token_id:
        ctx_ids = [tokenizer.bos_token_id] + ctx_ids

    # AE targets include EOS
    if tokenizer.eos_token_id:
        ae_targets_ids = ctx_ids + [tokenizer.eos_token_id]
    else:
        ae_targets_ids = ctx_ids

    input_ids = torch.LongTensor(ctx_ids).unsqueeze(0).to(device)
    ae_targets = torch.LongTensor(ae_targets_ids).unsqueeze(0).to(device)

    # Compress
    with torch.no_grad():
        compress_ids, compress_token, _ = model.compress({"input_ids": input_ids})

    # Teacher forcing decode
    with torch.no_grad():
        embeds = model.decoder.model.embed_tokens(ae_targets)
        bsz, seqlen, h = embeds.shape

        ae_tok = model.special_tokens[0:1].unsqueeze(0).expand(bsz, 1, h)
        ae_emb = torch.cat([compress_token, ae_tok, embeds[:, :-1, :]], dim=1)

        pos = torch.arange(0, seqlen, device=device).unsqueeze(0)
        pos_ids = torch.cat([compress_ids, pos], dim=1)

        out = model.decoder(inputs_embeds=ae_emb, position_ids=pos_ids)
        logits = out.logits[:, compress_token.size(1):]  # keep AE part only

        pred_ids = logits.argmax(dim=-1)[0].tolist()

    # teacher-forcing full reconstruction
    recon_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

    # token accuracy
    target = ae_targets_ids[: len(pred_ids)]
    correct = sum(int(a == b) for a, b in zip(pred_ids, target))
    acc = correct / len(target)

    return recon_text, acc


# ----------------------
# Main
# ----------------------

def main():
    model_id = "/home/syt/project/Cram/model/model_scope_model/LLM-Research/Llama-3.2-1B-Instruct"
    work_dir = "/home/syt/project/compressor_500/new_code/experiment/llama32_1b_500to1"

    adapter_path = os.path.join(work_dir, "output", "instruction_adapter.pt")
    # 或者用预训练版本：
    # adapter_path = os.path.join(work_dir, "output", "adapter.pt")

    model, tokenizer = load_compress_model(
        model_id=model_id,
        adapter_path=adapter_path,
        chunk_size=500,
        mem_size=1,
        compress_ratio=500,
    )

    # 测试文本
    context = (
        "We show that every reciprocity sheaf  gives rise to a cycle (pre)module in  the sense of Rost over a perfect field.  Over a perfect field of positive characteristic, we show that the first cohomology group of a logarithmic de RhamWitt sheaf has a partial cycle module  structure. As a consequence, we show  that Kato complexes of logarithmic de  Rham-Witt sheaves satisfy functoriality properties similar to Rost’s cycle  complexes."
        " and# show that the groupcity groupaf on1 rise to a Gal sheaviouslyst. a1 category of Kost and the field group. We a perfect field, characteristic characteristic, we show that the cycle cohomology group of the recipromic cycle Rham coey-Staf is a natural is structure. structure. a consequence, we show that that theummer’s are logarithmic de R Rham sheitt sheaves are aiality.. to thoseost’s. module module."
    )

    # ----------------------
    # AE Free Reconstruction
    # ----------------------
    free_text, free_acc, mem_token = ae_free_reconstruct(model, tokenizer, context)

    print("=== Free AE Reconstruction ===")
    print(free_text)
    print(f"\nFree-generation Token Accuracy: {free_acc*100:.2f}%\n")

    # ----------------------
    # AE Teacher Forcing Reconstruction
    # ----------------------
    tf_text, tf_acc = ae_teacher_reconstruct(model, tokenizer, context)

    print("=== Teacher-Forcing AE Reconstruction ===")
    print(tf_text)
    print(f"\nTeacher-forcing Token Accuracy: {tf_acc*100:.2f}%\n")


if __name__ == "__main__":
    main()

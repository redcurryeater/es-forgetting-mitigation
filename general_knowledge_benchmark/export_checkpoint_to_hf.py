"""Convert a vLLM-named ES checkpoint (`pytorch_model.pth`) into an HF model dir.

The ES script saves engine 0's weights via vLLM's parameter names, which fuse
attention as `qkv_proj` and MLP as `gate_up_proj`. HF Qwen2 expects them split
back into q/k/v and gate/up. This script:
  1. loads the base HF model (architecture + tokenizer),
  2. loads the .pth state dict,
  3. for each vLLM key, either copies it through or splits it into HF keys,
  4. saves a fresh HF directory ready for `lm_eval --model vllm`.

Usage:
  python export_checkpoint_to_hf.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --ckpt /path/to/final_model_iteration_150/pytorch_model.pth \
    --out  /path/to/final_hf_dir
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def split_qkv(qkv_weight, num_heads, num_kv_heads, head_dim):
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q, k, v = torch.split(qkv_weight, [q_size, kv_size, kv_size], dim=0)
    return q, k, v


def split_gate_up(gate_up_weight, intermediate_size):
    gate, up = torch.split(gate_up_weight, [intermediate_size, intermediate_size], dim=0)
    return gate, up


def convert(vllm_sd, cfg):
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_heads)
    intermediate_size = cfg.intermediate_size

    hf_sd = {}
    for k, v in vllm_sd.items():
        if k.startswith("__"):  # skip lora bookkeeping if present
            continue
        if "qkv_proj" in k:
            base = k.replace("qkv_proj", "")
            q, kk, vv = split_qkv(v, num_heads, num_kv_heads, head_dim)
            hf_sd[base + "q_proj" + k.split("qkv_proj")[1]] = q
            hf_sd[base + "k_proj" + k.split("qkv_proj")[1]] = kk
            hf_sd[base + "v_proj" + k.split("qkv_proj")[1]] = vv
        elif "gate_up_proj" in k:
            gate, up = split_gate_up(v, intermediate_size)
            hf_sd[k.replace("gate_up_proj", "gate_proj")] = gate
            hf_sd[k.replace("gate_up_proj", "up_proj")] = up
        else:
            hf_sd[k] = v
    return hf_sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True,
                    help="HF model id used for the run (e.g. Qwen/Qwen2.5-1.5B-Instruct).")
    ap.add_argument("--ckpt", required=True,
                    help="Path to the saved pytorch_model.pth (vLLM-named state dict).")
    ap.add_argument("--out", required=True,
                    help="Output directory for the HF-format model.")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"[load] base model {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(args.base_model)

    print(f"[load] checkpoint {args.ckpt}")
    vllm_sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)

    print("[convert] splitting fused qkv_proj / gate_up_proj")
    hf_sd = convert(vllm_sd, model.config)

    # cast to model dtype
    hf_sd = {k: v.to(dtype) for k, v in hf_sd.items()}

    print("[load_state_dict] strict=False (tolerate tied lm_head)")
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    # Qwen2.5 ties lm_head.weight to embed_tokens.weight; missing lm_head is OK.
    real_missing = [m for m in missing if m != "lm_head.weight"]
    if real_missing:
        raise RuntimeError(f"Missing keys after conversion: {real_missing[:10]}...")
    if unexpected:
        raise RuntimeError(f"Unexpected keys after conversion: {unexpected[:10]}...")

    os.makedirs(args.out, exist_ok=True)
    print(f"[save] {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print("[done]")


if __name__ == "__main__":
    main()

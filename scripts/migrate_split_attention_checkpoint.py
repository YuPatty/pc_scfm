"""Migrate a checkpoint trained with the old shared `attention.*` weights
in MambAttentionBlock to the new split `time_attention.*` / `freq_attention.*`
weights, so training can warm-start instead of starting from scratch.

Usage:
    python3 scripts/migrate_split_attention_checkpoint.py \
        --input ../runs/ecg_baseline_wander/checkpoint/<exp_name>/<model_name>/best_pcc_model.pt \
        --output ../runs/ecg_baseline_wander/checkpoint/<exp_name>/<model_name>/best_pcc_model_split_attn.pt

The output checkpoint has both branches initialized identically to the old
shared attention weights; they will diverge as training continues.
"""
import argparse
import re

import torch


def migrate_state_dict(state_dict):
    new_state = {}
    migrated = 0
    for key, value in state_dict.items():
        m = re.search(r"(^|\.)attention\.(.+)$", key)
        if m is None:
            new_state[key] = value
            continue
        prefix = key[: m.start(2) - len("attention.")]
        suffix = m.group(2)
        new_state[f"{prefix}time_attention.{suffix}"] = value.clone()
        new_state[f"{prefix}freq_attention.{suffix}"] = value.clone()
        migrated += 1
    print(f"Migrated {migrated} attention.* tensors into time_attention.*/freq_attention.* pairs.")
    return new_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the old checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Path to write the migrated checkpoint (.pt)")
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt["state_dict"] = migrate_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict) and all(
        hasattr(v, "shape") or isinstance(v, dict) for v in ckpt.values()
    ):
        ckpt = migrate_state_dict(ckpt)
    else:
        raise ValueError(
            "Unrecognized checkpoint format; expected a raw state_dict or a "
            "dict with a 'state_dict' key. Inspect the checkpoint manually."
        )
    torch.save(ckpt, args.output)
    print(f"Wrote migrated checkpoint to {args.output}")


if __name__ == "__main__":
    main()

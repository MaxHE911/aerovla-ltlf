"""Project-owned training entry for idea 3 (logic-conditioned SFT).

The baseline training script (``AeroVLA/src/train_aerovla.py``) hardcodes its data split, output
directory and step budget at module level and has no CLI.  We must not modify baseline files, so this
entry re-creates the *same* training recipe (identical LoRA config, identical collator, identical
Trainer arguments except the ones we parameterise) while importing the baseline's dataset class
unchanged.

Everything that differs from the baseline run is explicit and logged:
``SPLIT_JSON`` (the Stage-A json produced by ``ops/make_posttrain_data.py``), ``OUTPUT_DIR``,
``MAX_STEPS`` and the batch sizes.

Usage (server, from the repository root):

    python ops/train_aerovla_ltlf.py \
        --split work/posttrain/sft_stageA_mcm.json \
        --out checkpoints/aero_vla_ltlf_stageA \
        --micro-batch 2 --accum 8 --max-steps 2500 --epochs 1

Smoke test (a few steps, no checkpoint of value):

    python ops/train_aerovla_ltlf.py --split ... --out work/posttrain/smoke \
        --micro-batch 1 --accum 1 --max-steps 8 --workers 0
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LoRA SFT of AeroVLA on logic-conditioned instructions")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="repository root (contains AeroVLA/)")
    ap.add_argument("--model-path", default=None, help="default <repo>/AeroVLA/openvla-7b")
    ap.add_argument("--data-root", default=None, help="default <repo>/AeroVLA/dataset_raw")
    ap.add_argument("--split", required=True, help="Stage-A json produced by ops/make_posttrain_data.py")
    ap.add_argument("--out", required=True, help="output checkpoint dir")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument("--save-total-limit", type=int, default=5)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--entropy-monitor", type=int, default=0,
                    help="if >0, also log the mean predictive entropy (nats) over the action tokens "
                         "every N optimizer steps; entropy collapsing to ~0 is the earliest sign of "
                         "the mode collapse observed on 2026-11-27 (loss 6.2 -> 0.8 while the policy "
                         "started emitting one constant action)")
    args = ap.parse_args(argv)

    repo = os.path.abspath(args.repo)
    aerovla = os.path.join(repo, "AeroVLA")
    src = os.path.join(aerovla, "src")
    if not os.path.isdir(src):
        raise SystemExit("AeroVLA/src not found under %s" % repo)
    sys.path.insert(0, src)

    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, TrainingArguments

    from aerovla_dataset import AeroVLADataset  # baseline class, imported unchanged

    model_path = args.model_path or os.path.join(aerovla, "openvla-7b")
    data_root = args.data_root or os.path.join(aerovla, "dataset_raw")
    split = args.split if os.path.isabs(args.split) else os.path.join(repo, args.split)
    out = args.out if os.path.isabs(args.out) else os.path.join(repo, args.out)

    print("[ltlf-sft] repo       =", repo)
    print("[ltlf-sft] model_path =", model_path)
    print("[ltlf-sft] data_root  =", data_root)
    print("[ltlf-sft] split      =", split)
    print("[ltlf-sft] out        =", out)
    print("[ltlf-sft] micro_bs=%d accum=%d lr=%g epochs=%g max_steps=%d"
          % (args.micro_batch, args.accum, args.lr, args.epochs, args.max_steps))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    print("[ltlf-sft] using native bf16 (LoRA)")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["projector"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("[ltlf-sft] trainable %s / %s (%.4f%%)" % (f"{trainable:,}", f"{total:,}", 100.0 * trainable / total))

    dataset = AeroVLADataset(data_root, split, tokenizer, processor)

    def data_collator(features):
        pixel_values = torch.stack([f["image"] for f in features]).to(torch.bfloat16)
        texts = [f["text"] for f in features]
        prompts = [f["prompt_only"] for f in features]
        batch = tokenizer(texts, return_tensors="pt", padding="longest", truncation=True, max_length=2048)
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        labels = input_ids.clone()
        for i, prompt in enumerate(prompts):
            prompt_len = len(tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0])
            labels[i, :prompt_len] = -100
        labels[attention_mask == 0] = -100
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    training_args = TrainingArguments(
        output_dir=out,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        weight_decay=0.03,
        warmup_ratio=0.03,
        max_grad_norm=1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        dataloader_num_workers=args.workers,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        fp16=False,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    class EntropyTrainer(transformers.Trainer):
        """Trainer that additionally reports the predictive entropy over the action tokens.

        Motivation (measured 2026-11-27): 125 optimizer steps at lr 2e-4 drove the loss from 6.20 to
        0.82 while the closed-loop policy degenerated into emitting one constant action
        (136/139 steps identical).  The training loss therefore does not track policy quality, but the
        entropy of the action-token distribution does: it collapses before the behaviour does.  Set
        ``--entropy-monitor N`` to log it every N optimizer steps and stop early when it flattens.
        """

        def __init__(self, *a, every: int = 0, **kw):
            super().__init__(*a, **kw)
            self.every = every
            self._logits = None
            self._labels = None

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            out = model(**inputs)
            logits = out[1] if isinstance(out, tuple) else getattr(out, "logits", None)
            if logits is not None and self.every:
                self._logits = logits.detach()
                self._labels = inputs.get("labels")
            return out[0] if isinstance(out, tuple) else out.loss

        def log(self, logs):
            if self.every and self._logits is not None and self._labels is not None:
                if self.state.global_step % self.every == 0:
                    try:
                        mask = self._labels != -100
                        if bool(mask.any()):
                            lg = self._logits[mask].float()
                            logp = torch.log_softmax(lg, dim=-1)
                            ent = -(logp.exp() * logp).sum(-1).mean().item()
                            logs = dict(logs)
                            logs["action_token_entropy_nats"] = round(float(ent), 4)
                    except Exception:
                        pass
                self._logits = None
                self._labels = None
            super().log(logs)

    trainer = EntropyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        every=args.entropy_monitor,
    )
    print("[ltlf-sft] starting training")
    trainer.train()
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print("[ltlf-sft] finished ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


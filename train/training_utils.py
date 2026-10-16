from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llama_chat import (
    DEFAULT_SYSTEM_PROMPT,
    build_training_text,
    configure_tokenizer,
    encode_training_example,
)

PROTOCOL_VERSION = "llama3.1-system-user-assistant-v1"

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SCAFFOLD_CORE_BEGIN = "<CRYSTAL_SCAFFOLD_CORE_V4>"
SCAFFOLD_FULL_BEGIN = "<CRYSTAL_SCAFFOLD_FULL_V4>"
PREVIOUS_STAGE = {"parse": None, "scaffold": "parse", "cif": "scaffold"}
CKPT_DIRNAME = "ckpt"

DEFAULT_PARSE_CAPS = {
    "dual_symmetry_summary": 2500,
    "formula_z_closure": 3000,
    "cell_archetype": 2500,
    "lattice_bin_correction": 2500,
    "coord_profile_parse": 2500,
    "scaffold_core_parse": 2500,
    "failure_bank_diagnose": 3000,
}


@dataclass
class TrainConfig:
    stage_name: str
    max_seq_length: int
    num_train_epochs: float
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    model_name: str = "meta-llama/Llama-3.1-8B"
    train_file: str = ""
    val_file: Optional[str] = None
    pipeline_root: str = "."
    output_dir: str = ""
    adapter_dir: str = ""
    resume_mode: str = "base"
    resume_adapter_path: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    load_in_4bit: bool = True
    dtype: Optional[str] = None
    lora_rank: int = 16
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.05
    per_device_eval_batch_size: int = 4
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 0
    save_total_limit: int = 3
    early_stopping_patience: int = 0
    seed: int = 3407
    report_to: str = "none"
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    gradient_checkpointing: str = "unsloth"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    length_audit_samples: int = 128
    max_steps: int = 0
    dataloader_num_workers: int = 4
    group_by_length: bool = True
    parse_task_caps: Optional[str] = None
    cif_context: str = "full"

    def finalize(self) -> "TrainConfig":
        if self.lora_alpha is None:
            self.lora_alpha = int(self.lora_rank) * 2
        if not self.output_dir:
            self.output_dir = os.path.join(self.pipeline_root, CKPT_DIRNAME, self.stage_name)
        if not self.adapter_dir:
            self.adapter_dir = os.path.join(self.output_dir, "adapter")
        self.resume_mode = str(self.resume_mode or "base").lower()
        if self.resume_mode not in {"base", "previous", "path"}:
            raise ValueError(f"Unsupported resume_mode={self.resume_mode!r}")
        if self.resume_adapter_path:
            self.resume_mode = "path"
        if self.resume_mode == "previous":
            prev = PREVIOUS_STAGE.get(self.stage_name)
            if prev is None:
                self.resume_mode = "base"
                self.resume_adapter_path = None
            else:
                self.resume_adapter_path = os.path.join(self.pipeline_root, CKPT_DIRNAME, prev, "adapter")
        if self.resume_mode == "base":
            self.resume_adapter_path = None
        if self.resume_mode == "path" and not self.resume_adapter_path:
            raise ValueError("--resume-mode path requires --resume-adapter-path")
        for name in ("max_seq_length", "lora_rank", "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps", "save_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.warmup_ratio <= 1 or self.warmup_steps < 0:
            raise ValueError("Invalid warmup settings")
        if self.eval_steps < 0 or self.max_steps < 0 or self.early_stopping_patience < 0:
            raise ValueError("Evaluation, optimizer-step cap and patience must be nonnegative")
        if self.eval_steps and self.save_steps % self.eval_steps:
            raise ValueError("save_steps must be a multiple of eval_steps")
        return self


def add_common_train_args(parser: argparse.ArgumentParser, *, stage_name: str, default_train_file: str) -> None:
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--train-file", default=default_train_file)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--pipeline-root", default=".")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--resume-mode", default=None, choices=["base", "previous", "path"])
    parser.add_argument("--resume-adapter-path", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--gradient-checkpointing", default="unsloth")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--length-audit-samples", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--no-group-by-length", action="store_false", dest="group_by_length")
    parser.add_argument("--group-by-length", action="store_true", dest="group_by_length")
    parser.set_defaults(group_by_length=True, cif_context="full")
    if stage_name == "cif":
        parser.add_argument("--cif-context", choices=["full", "core", "none"], default="full")
    parser.add_argument(
        "--parse-task-caps",
        default=None,
        help="JSON dict task->cap used only for parse-stage downsampling.",
    )


def config_from_args(args: argparse.Namespace, *, stage_name: str, stage_defaults: Dict[str, Any]) -> TrainConfig:
    output_dir = args.output_dir or os.path.join(args.pipeline_root, CKPT_DIRNAME, stage_name)
    cfg = TrainConfig(
        stage_name=stage_name,
        model_name=args.model_name,
        train_file=args.train_file,
        val_file=args.val_file,
        pipeline_root=args.pipeline_root,
        output_dir=output_dir,
        adapter_dir=args.adapter_dir or os.path.join(output_dir, "adapter"),
        resume_mode=args.resume_mode or stage_defaults.get("resume_mode", "base"),
        resume_adapter_path=args.resume_adapter_path,
        resume_checkpoint=args.resume_checkpoint,
        max_seq_length=args.max_seq_length if args.max_seq_length is not None else stage_defaults["max_seq_length"],
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs is not None else stage_defaults["num_train_epochs"],
        learning_rate=args.learning_rate if args.learning_rate is not None else stage_defaults["learning_rate"],
        per_device_train_batch_size=args.per_device_train_batch_size if args.per_device_train_batch_size is not None else stage_defaults["per_device_train_batch_size"],
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else stage_defaults["gradient_accumulation_steps"],
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        length_audit_samples=args.length_audit_samples,
        max_steps=args.max_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        group_by_length=args.group_by_length,
        parse_task_caps=args.parse_task_caps,
        cif_context=args.cif_context,
    )
    return cfg.finalize()


def lazy_import_training_stack() -> Dict[str, Any]:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    import torch
    from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed
    from peft import PeftModel
    return {
        "FastLanguageModel": FastLanguageModel,
        "is_bfloat16_supported": is_bfloat16_supported,
        "torch": torch,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "set_seed": set_seed,
        "PeftModel": PeftModel,
    }


def load_model_and_tokenizer(cfg: TrainConfig) -> Tuple[Any, Any]:
    stack = lazy_import_training_stack()
    stack["set_seed"](cfg.seed)
    FastLanguageModel = stack["FastLanguageModel"]
    dtype = getattr(stack["torch"], cfg.dtype) if cfg.dtype else None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=dtype,
        load_in_4bit=cfg.load_in_4bit,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
    )
    configure_tokenizer(tokenizer, padding_side="right")
    if cfg.resume_adapter_path:
        model = stack["PeftModel"].from_pretrained(
            model, cfg.resume_adapter_path, is_trainable=True,
        )
        model = FastLanguageModel.patch_peft_model(model, cfg.gradient_checkpointing)
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_rank,
            target_modules=DEFAULT_TARGET_MODULES,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            random_state=cfg.seed,
            use_rslora=False,
            init_lora_weights=True,
            loftq_config=None,
        )
    return model, tokenizer


def load_table(path: str, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path_obj.suffix.lower() == ".jsonl":
        rows = []
        with open(path_obj, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path_obj)
    required = {"instruction", "input", "output"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"No training examples in {path}")
    df = df.copy()
    for col in ["instruction", "input", "output"]:
        df[col] = df[col].fillna("").astype(str)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("Sample limit must be positive")
        df = df.head(max_rows).copy()
    return df


def validate_stage_table(df: pd.DataFrame, cfg: TrainConfig) -> None:
    if df.empty:
        raise ValueError("Stage dataset is empty")
    if cfg.stage_name == "parse":
        if "task" in df.columns:
            allowed = {
                "dual_symmetry_summary",
                "formula_z_closure",
                "cell_archetype",
                "lattice_bin_correction",
                "coord_profile_parse",
                "scaffold_core_parse",
                "failure_bank_diagnose",
            }
            bad = sorted(set(df["task"].astype(str)) - allowed)
            if bad:
                raise ValueError(f"Unexpected parse tasks: {bad}")
    elif cfg.stage_name == "scaffold":
        if not df["output"].str.contains(SCAFFOLD_CORE_BEGIN, regex=False).all():
            raise ValueError("Scaffold stage expects outputs tagged with <CRYSTAL_SCAFFOLD_CORE_V4>.")
    elif cfg.stage_name == "cif":
        context_tag = {"full": SCAFFOLD_FULL_BEGIN, "core": SCAFFOLD_CORE_BEGIN, "none": None}[cfg.cif_context]
        if context_tag and not df["input"].str.contains(context_tag, regex=False).all():
            raise ValueError(f"CIF context {cfg.cif_context} requires {context_tag} in every input")
        if cfg.cif_context == "none" and any(df["input"].str.contains(tag, regex=False).any() for tag in (SCAFFOLD_FULL_BEGIN, SCAFFOLD_CORE_BEGIN)):
            raise ValueError("Direct CIF inputs must not contain scaffolds")
        if cfg.cif_context == "core" and df["input"].str.contains(SCAFFOLD_FULL_BEGIN, regex=False).any():
            raise ValueError("Core-only CIF inputs must not contain compiled scaffolds")
        if not df["output"].str.contains("_atom_site_type_symbol", regex=False).all():
            raise ValueError("CIF stage outputs do not look like CIF text.")


def summarize_stage_dataset(df: pd.DataFrame, cfg: TrainConfig) -> None:
    print(f"[INFO] stage={cfg.stage_name} rows={len(df)}")
    if cfg.stage_name == "parse" and "task" in df.columns:
        vc = df["task"].value_counts().to_dict()
        print("[INFO] parse task distribution=" + json.dumps(vc, ensure_ascii=False))
    if cfg.stage_name in {"scaffold", "cif"} and "condition_sg_number" in df.columns:
        top = df["condition_sg_number"].value_counts().head(10).to_dict()
        print("[INFO] top condition_sg_number=" + json.dumps(top, ensure_ascii=False))


def _safe_parse_caps(text: Optional[str]) -> Dict[str, int]:
    caps = dict(DEFAULT_PARSE_CAPS)
    if not text:
        return caps
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) - set(caps):
        raise ValueError("--parse-task-caps must map known task names to nonnegative integers")
    for key, value in obj.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid task cap: {key}={value!r}")
        caps[key] = value
    return caps


def optimize_parse_dataset(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    if cfg.stage_name != "parse" or "task" not in df.columns:
        return df

    caps = _safe_parse_caps(cfg.parse_task_caps)
    df = df.copy()
    out_parts = []
    for task, cap in caps.items():
        part = df[df["task"].astype(str) == task].copy()
        if len(part) == 0 or cap <= 0:
            continue
        part = part.drop_duplicates(subset=["instruction", "input", "output"], keep="first")
        if "material_id" in part.columns:
            part = part.sort_values(["material_id", "task"]).groupby("material_id", as_index=False).head(2)
        if len(part) > cap:
            part = part.head(cap).copy()
        out_parts.append(part)

    if not out_parts:
        raise ValueError("Parse caps removed every training example")

    out = pd.concat(out_parts, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=["instruction", "input", "output"], keep="first").reset_index(drop=True)
    print(f"[INFO] parse warm-up reduced from {len(df)} -> {len(out)} rows")
    return out


class PromptResponseDataset:
    def __init__(self, records: Sequence[Dict[str, str]], tokenizer: Any, *, max_seq_length: int) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch
        ex = self.records[idx]
        try:
            encoded = encode_training_example(
                self.tokenizer, ex["instruction"], ex["input"], ex["output"],
                max_length=self.max_seq_length,
            )
        except ValueError as exc:
            raise ValueError(f"Training row {idx}: {exc}") from exc
        return {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}


class Seq2SeqStyleCollator:
    def __init__(self, tokenizer: Any):
        self.pad_id = int(tokenizer.pad_token_id)

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        max_len = max(int(f["input_ids"].shape[0]) for f in features)
        batch_input_ids, batch_attention, batch_labels = [], [], []
        for feat in features:
            seq_len = int(feat["input_ids"].shape[0])
            pad_len = max_len - seq_len
            batch_input_ids.append(torch.nn.functional.pad(feat["input_ids"], (0, pad_len), value=self.pad_id))
            batch_attention.append(torch.nn.functional.pad(feat["attention_mask"], (0, pad_len), value=0))
            batch_labels.append(torch.nn.functional.pad(feat["labels"], (0, pad_len), value=-100))
        return {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attention, dim=0),
            "labels": torch.stack(batch_labels, dim=0),
        }


def audit_sequence_lengths(df: pd.DataFrame, tokenizer: Any, cfg: TrainConfig, *, title: str) -> None:
    n = min(len(df), max(int(cfg.length_audit_samples), 1))
    if n == 0:
        return
    sample = df.head(n)
    lens: List[int] = []
    for _, row in sample.iterrows():
        text = build_training_text(str(row["instruction"]), str(row["input"]), str(row["output"]))
        toks = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        lens.append(len(toks))
    lens = sorted(lens)

    def pct(q: float) -> int:
        idx = min(len(lens) - 1, max(0, int(round((len(lens) - 1) * q))))
        return int(lens[idx])

    trunc_frac = sum(1 for x in lens if x > cfg.max_seq_length) / max(len(lens), 1)
    print(
        f"[INFO] {title} token lengths: p50={pct(0.50)} p90={pct(0.90)} p95={pct(0.95)} "
        f"p99={pct(0.99)} max={max(lens)} limit={cfg.max_seq_length} trunc_frac={trunc_frac:.3%}"
    )
    if trunc_frac:
        raise ValueError(f"{title}: {trunc_frac:.3%} of audited records exceed the sequence limit; no target is silently truncated.")


def build_trainer(cfg: TrainConfig, model: Any, tokenizer: Any, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> Any:
    stack = lazy_import_training_stack()
    torch = stack["torch"]
    Trainer = stack["Trainer"]
    TrainingArguments = stack["TrainingArguments"]
    EarlyStoppingCallback = stack.get("EarlyStoppingCallback")

    train_records = train_df[["instruction", "input", "output"]].to_dict(orient="records")
    eval_records = None if val_df is None else val_df[["instruction", "input", "output"]].to_dict(orient="records")

    train_dataset = PromptResponseDataset(
        train_records,
        tokenizer,
        max_seq_length=cfg.max_seq_length,
    )
    eval_dataset = None
    if eval_records is not None and len(eval_records) > 0:
        eval_dataset = PromptResponseDataset(
            eval_records,
            tokenizer,
            max_seq_length=cfg.max_seq_length,
        )

    bf16 = cfg.dtype == "bfloat16" if cfg.dtype else bool(stack["is_bfloat16_supported"]())
    use_eval = eval_dataset is not None and cfg.eval_steps and cfg.eval_steps > 0

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        warmup_ratio=0.0 if cfg.warmup_steps > 0 else cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        fp16=not bf16 and cfg.dtype != "float32",
        bf16=bf16,
        logging_steps=cfg.logging_steps,
        optim=cfg.optim,
        report_to=cfg.report_to,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=cfg.eval_steps if use_eval else None,
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        load_best_model_at_end=bool(use_eval),
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False if use_eval else None,
        max_steps=cfg.max_steps if cfg.max_steps and cfg.max_steps > 0 else -1,
        dataloader_num_workers=cfg.dataloader_num_workers,
        group_by_length=cfg.group_by_length,
    )

    callbacks = []
    if use_eval and cfg.early_stopping_patience > 0 and EarlyStoppingCallback is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Seq2SeqStyleCollator(tokenizer),
        callbacks=callbacks,
    )
    return trainer


def save_training_artifacts(cfg: TrainConfig, model: Any, tokenizer: Any, *, train_rows: int, val_rows: int, trainer: Any = None) -> None:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    manifest = asdict(cfg)
    manifest.update({
        "instruction_protocol": PROTOCOL_VERSION,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "response_only_loss": True,
        "overlength_policy": "error",
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "per_device_effective_batch_size": int(cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps),
    })
    manifest["software_versions"] = {}
    for package in ("torch", "transformers", "peft", "unsloth", "bitsandbytes", "pandas"):
        try:
            manifest["software_versions"][package] = version(package)
        except PackageNotFoundError:
            manifest["software_versions"][package] = None
    if trainer is not None:
        Path(cfg.adapter_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(cfg.adapter_dir)
        tokenizer.save_pretrained(cfg.adapter_dir)
        trainer.save_state()
        manifest["optimizer_updates"] = int(trainer.state.global_step)
        manifest["completed_epochs"] = trainer.state.epoch
        manifest["world_size"] = int(trainer.args.world_size)
        manifest["global_batch_size"] = int(cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * trainer.args.world_size)
        protocol = {
            "instruction_protocol": PROTOCOL_VERSION,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "stage": cfg.stage_name,
        }
        if cfg.stage_name == "cif":
            protocol["cif_context"] = cfg.cif_context
        with open(Path(cfg.adapter_dir) / "adapter_protocol.json", "w", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, ensure_ascii=False)
    with open(Path(cfg.output_dir) / "train_config.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def run_training(cfg: TrainConfig) -> None:
    cfg.finalize()

    train_df = load_table(cfg.train_file, max_rows=cfg.max_train_samples)
    if cfg.stage_name == "parse":
        train_df = optimize_parse_dataset(train_df, cfg)
        if cfg.max_train_samples is not None and len(train_df) > cfg.max_train_samples:
            train_df = train_df.head(cfg.max_train_samples).copy()

    val_df = load_table(cfg.val_file, max_rows=cfg.max_eval_samples) if cfg.val_file else None

    validate_stage_table(train_df, cfg)
    summarize_stage_dataset(train_df, cfg)
    if val_df is not None:
        validate_stage_table(val_df, cfg)

    if cfg.resume_adapter_path and not Path(cfg.resume_adapter_path).exists():
        raise FileNotFoundError(
            f"resume adapter not found: {cfg.resume_adapter_path}. Run the previous stage first or pass --resume-mode base."
        )

    if cfg.resume_adapter_path:
        adapter_file = Path(cfg.resume_adapter_path) / "adapter_config.json"
        if not adapter_file.is_file():
            raise FileNotFoundError(f"Missing adapter configuration: {adapter_file}")
        adapter_config = json.loads(adapter_file.read_text(encoding="utf-8"))
        predecessor_manifest = Path(cfg.resume_adapter_path) / "adapter_protocol.json"
        predecessor = json.loads(predecessor_manifest.read_text(encoding="utf-8")) if predecessor_manifest.is_file() else {}
        if predecessor.get("instruction_protocol") != PROTOCOL_VERSION or predecessor.get("system_prompt") != DEFAULT_SYSTEM_PROMPT:
            raise ValueError("Inherited adapter lacks this release Llama 3.1 protocol marker; train the predecessor stage with this release before continuing.")
        for key, expected in (("r", cfg.lora_rank), ("lora_alpha", cfg.lora_alpha), ("lora_dropout", cfg.lora_dropout)):
            if adapter_config.get(key) != expected:
                raise ValueError(f"Inherited adapter {key}={adapter_config.get(key)} disagrees with requested {expected}")
        if set(adapter_config.get("target_modules", [])) != set(DEFAULT_TARGET_MODULES):
            raise ValueError("Inherited adapter target modules disagree with this training configuration")
    if cfg.resume_checkpoint:
        checkpoint = Path(cfg.resume_checkpoint)
        previous_manifest = checkpoint.parent / "train_config.json"
        if not checkpoint.is_dir() or not previous_manifest.is_file():
            raise ValueError("--resume-checkpoint needs a checkpoint and its release train_config.json.")
        saved = json.loads(previous_manifest.read_text(encoding="utf-8"))
        if saved.get("instruction_protocol") != PROTOCOL_VERSION or saved.get("stage_name") != cfg.stage_name:
            raise ValueError("Checkpoint protocol/stage differs from this run.")
    if val_df is None and cfg.eval_steps > 0:
        warnings.warn("No --val-file: validation and early stopping are disabled; this is not the reported protocol.")
    model, tokenizer = load_model_and_tokenizer(cfg)
    if hasattr(model, "print_trainable_parameters"):
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    audit_sequence_lengths(train_df, tokenizer, cfg, title=f"{cfg.stage_name}/train")
    if val_df is not None and len(val_df) > 0:
        audit_sequence_lengths(val_df, tokenizer, cfg, title=f"{cfg.stage_name}/val")

    trainer = build_trainer(cfg, model, tokenizer, train_df, val_df)
    save_training_artifacts(cfg, model, tokenizer, train_rows=len(train_df), val_rows=0 if val_df is None else len(val_df))
    trainer.train(resume_from_checkpoint=cfg.resume_checkpoint)
    save_training_artifacts(cfg, model, tokenizer, train_rows=len(train_df), val_rows=0 if val_df is None else len(val_df), trainer=trainer)
    print(f"[OK] finished training stage={cfg.stage_name}")
    print(f"[OK] adapter saved to {cfg.adapter_dir}")


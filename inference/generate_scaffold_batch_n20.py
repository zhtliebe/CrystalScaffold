from __future__ import annotations


import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from inference_utils import (
    GenerationConfig,
    ModelRunner,
    material_prompt_input,
    maybe_compile_generated_core,
)

from inference_utils import schema, save_run_metadata


def _first_existing(d: Dict[str, str], keys: List[str]) -> str:
    for key in keys:
        if key in d and str(d[key]).strip() != "":
            return str(d[key]).strip()
    return ""


def _load_pairs(input_csv: str, first_n: Optional[int]) -> List[Tuple[str, int, int]]:
    pairs: List[Tuple[str, int, int]] = []
    with open(input_csv, "r", encoding="utf-8", newline="") as handle:
        for source_row, row in enumerate(csv.DictReader(handle)):
            if first_n is not None and source_row >= first_n:
                break
            composition = _first_existing(row, ["composition", "pretty_formula", "formula", "reduced_formula"])
            sg_text = _first_existing(row, ["condition_sg_number", "spacegroup.number", "space_group", "sg_number", "spacegroup"])
            if not composition or not sg_text:
                raise ValueError(f"Missing condition in source row {source_row}")
            pairs.append((composition, int(float(sg_text)), source_row))
    return pairs


def _chunked(xs, n: int):
    for i in range(0, len(xs), max(1, n)):
        yield xs[i:i + max(1, n)]


def _batch_generate_scaffolds(runner: ModelRunner, compositions_and_sgs: List[Tuple[str, int, int]]) -> List[List[str]]:
    return runner.generate_batch(schema.SCAFFOLD_CORE_INSTRUCTION,
        [material_prompt_input(composition, sg) for composition, sg, source_row in compositions_and_sgs])


def _generate_many_from_responses(
    responses: List[str],
    composition: str,
    space_group: int,
    num_scaffolds: int,
) -> List[Dict]:
    rows: List[Dict] = []
    for i in range(num_scaffolds):
        resp = responses[i] if i < len(responses) else ""
        if resp:
            core_text, full_text, ok, errors, fields = maybe_compile_generated_core(
                resp,
                formula=composition,
                condition_sg_number=space_group,
            )
        else:
            core_text, full_text, ok, errors, fields = "", "", False, ["generation_missing"], {}
        rows.append({
            "scaffold_index": i,
            "raw_response": str(resp).strip(),
            "scaffold_core": str(core_text).strip(),
            "scaffold_full": str(full_text).strip(),
            "valid_scaffold": bool(ok),
            "scaffold_errors": list(errors),
            "fields": fields,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate N scaffold attempts per original target row using the N=1 scaffold logic."
    )
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--adapter-path", default="ckpt/scaffold/adapter")
    parser.add_argument("--input-csv", required=True, help="CSV containing composition + space_group columns.")
    parser.add_argument("--output-csv", required=True, help="Output scaffold CSV path.")
    parser.add_argument("--first-n", type=int, default=None, help="Only process the first N source rows.")
    parser.add_argument("--num-scaffolds", type=int, default=20, help="Number of scaffold attempts per target.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of target rows to generate in parallel.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    pairs = _load_pairs(args.input_csv, args.first_n)
    if not pairs:
        raise ValueError(f"No valid (composition, space_group) pairs found in {args.input_csv}")

    cfg = GenerationConfig(
        model_name=args.model_name,
        adapter_path=args.adapter_path, adapter_stage="scaffold",
        num_return_sequences=args.num_scaffolds,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.03,
        do_sample=True,
        max_seq_length=4096,
    )
    runner = ModelRunner(cfg)

    output_rows: List[Dict[str, str]] = []
    total = len(pairs)
    chunk_idx = 0
    total_chunks = (total + max(1, args.batch_size) - 1) // max(1, args.batch_size)

    for pair_chunk in _chunked(pairs, args.batch_size):
        chunk_idx += 1
        start = (chunk_idx - 1) * max(1, args.batch_size) + 1
        end = min(start + len(pair_chunk) - 1, total)
        print(
            f"[INFO] chunk {chunk_idx}/{total_chunks} generating scaffolds for pairs {start}-{end} / {total}",
            flush=True,
        )
        grouped_responses = _batch_generate_scaffolds(runner, pair_chunk)
        for (composition, space_group, source_row), responses in zip(pair_chunk, grouped_responses):
            rows = _generate_many_from_responses(responses, composition, space_group, args.num_scaffolds)
            for row in rows:
                output_rows.append({
                    "source_row": str(source_row),
                    "generation_index": str(row["scaffold_index"]),
                    "composition": composition,
                    "space_group": str(space_group),
                    "scaffold_index": str(row["scaffold_index"]),
                    "raw_response": row["raw_response"],
                    "scaffold_core": row["scaffold_core"],
                    "scaffold_full": row["scaffold_full"],
                    "valid_scaffold": str(bool(row["valid_scaffold"])),
                    "scaffold_errors": json.dumps(row["scaffold_errors"], ensure_ascii=False),
                })

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_row",
        "generation_index",
        "composition",
        "space_group",
        "scaffold_index",
        "raw_response",
        "scaffold_core",
        "scaffold_full",
        "valid_scaffold",
        "scaffold_errors",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    save_run_metadata(args.output_csv, args, cfg)
    print(f"[OK] wrote {len(output_rows)} scaffold rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()

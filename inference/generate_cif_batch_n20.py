from __future__ import annotations


import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from inference_utils import (
    GenerationConfig,
    ModelRunner,
    material_prompt_input,
    maybe_canonicalize_generated_cif,
    verify_cif_against_scaffold,
)

from inference_utils import schema, save_run_metadata


def _first_existing(d: Dict[str, str], keys: List[str]) -> str:
    for key in keys:
        if key in d and str(d[key]).strip() != "":
            return str(d[key]).strip()
    return ""


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "item"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def _chunked(xs, n: int):
    for i in range(0, len(xs), max(1, n)):
        yield xs[i:i + max(1, n)]


def _load_rows(input_csv: str, first_n: Optional[int], cif_context: str = "full") -> List[Dict[str, str]]:
    rows_out: List[Dict[str, str]] = []
    with open(input_csv, "r", encoding="utf-8", newline="") as handle:
        for input_row, row in enumerate(csv.DictReader(handle)):
            source_row = int(row.get("source_row", input_row))
            if first_n is not None and source_row >= first_n:
                continue
            composition = _first_existing(row, ["composition", "pretty_formula", "formula", "reduced_formula"])
            sg_text = _first_existing(row, ["condition_sg_number", "spacegroup.number", "space_group", "sg_number", "spacegroup"])
            if not composition or not sg_text:
                raise ValueError(f"Missing condition in source row {source_row}")
            generation_index = _first_existing(row, ["generation_index", "scaffold_index"]) or "0"
            rows_out.append({"source_row": str(source_row), "generation_index": generation_index,
                "scaffold_index": generation_index, "composition": composition,
                "space_group": str(int(float(sg_text))),
                "scaffold_full": _first_existing(row, ["scaffold_core", "SCAFFOLD_core"]) if cif_context == "core" else _first_existing(row, ["scaffold_full", "SCAFFOLD_full", "scaffold"])})
    return rows_out


def _batch_generate_cifs(runner: ModelRunner, rows: List[Dict[str, str]], direct: bool = False) -> List[str]:
    active = [index for index, row in enumerate(rows) if direct or row["scaffold_full"]]
    inputs = [material_prompt_input(rows[index]["composition"], int(rows[index]["space_group"])) +
        ("" if direct else "\n\n" + rows[index]["scaffold_full"]) for index in active]
    generated = runner.generate_batch(schema.BASELINE_INSTRUCTION if direct else schema.CIF_WITH_SCAFFOLD_INSTRUCTION, inputs)
    responses = [""] * len(rows)
    for index, group in zip(active, generated):
        responses[index] = group[0]
    return responses


def _evaluate_single_response(
    composition: str,
    space_group: int,
    scaffold_full: str,
    resp: str,
    direct: bool = False,
) -> Dict:
    cif_text, canonicalized, can_err = maybe_canonicalize_generated_cif(resp)
    verify = {} if direct else verify_cif_against_scaffold(
        cif_text,
        scaffold_full,
        formula=composition,
        condition_sg_number=space_group,
    )
    return {
        "raw_response": str(resp).strip(),
        "canonicalized": canonicalized,
        "canonicalize_error": can_err,
        "cif": cif_text,
        **verify,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one CIF per scaffold row using the N=1 CIF logic."
    )
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--adapter-path", default="ckpt/cif/adapter")
    parser.add_argument("--input-csv", required=True, help="Scaffold CSV containing composition + space_group + scaffold_full.")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated CIF files.")
    parser.add_argument("--output-csv", required=True, help="Summary CSV containing generated CIF paths and scores.")
    parser.add_argument("--first-n", type=int, default=None, help="Only process rows belonging to the first N source rows.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of scaffold rows to generate in parallel.")
    parser.add_argument("--cif-context", choices=["full", "core", "none"], default="full")
    parser.add_argument("--direct", action="store_true", help="Sample direct CIF baseline from condition rows.")
    parser.add_argument("--num-samples", type=int, default=20, help="Attempts per source row for --direct; scaffold mode uses input attempt indices.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.direct:
        args.cif_context = "none"
    args.direct = args.cif_context == "none"

    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    rows_in = _load_rows(args.input_csv, args.first_n, args.cif_context)
    if args.direct:
        if args.num_samples < 1:
            parser.error("--num-samples must be positive")
        rows_in = [{**row, "generation_index": str(index), "scaffold_index": str(index)}
            for row in rows_in for index in range(args.num_samples)]
    if not rows_in:
        raise ValueError(f"No valid rows found in {args.input_csv}. Need composition + space_group (+ scaffold_full preferred).")

    cfg = GenerationConfig(
        model_name=args.model_name,
        adapter_path=args.adapter_path, adapter_stage="cif", cif_context=args.cif_context,
        num_return_sequences=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.05,
        do_sample=True,
        max_seq_length=4096,
    )
    runner = ModelRunner(cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, str]] = []
    total = len(rows_in)
    chunk_idx = 0
    total_chunks = (total + max(1, args.batch_size) - 1) // max(1, args.batch_size)

    for row_chunk in _chunked(rows_in, args.batch_size):
        chunk_idx += 1
        start = (chunk_idx - 1) * max(1, args.batch_size) + 1
        end = min(start + len(row_chunk) - 1, total)
        print(f"[INFO] chunk {chunk_idx}/{total_chunks} generating CIFs for rows {start}-{end} / {total}", flush=True)

        responses = _batch_generate_cifs(runner, row_chunk, direct=args.direct)

        for row, resp in zip(row_chunk, responses):
            composition = row["composition"]
            space_group = int(row["space_group"])
            scaffold_index = int(row["scaffold_index"])
            scaffold_full = row["scaffold_full"]

            best = _evaluate_single_response(
                composition=composition,
                space_group=space_group,
                scaffold_full=scaffold_full,
                resp=resp,
                direct=args.direct,
            )

            pair_dir = out_dir / f"row{row['source_row']}__{_sanitize_name(composition)}__sg{space_group}"
            cif_path = pair_dir / f"scaffold_{scaffold_index:02d}.cif"

            if str(best.get("cif", "")).strip():
                _write_text(cif_path, str(best.get("cif", "")).strip() + "\n")
            summary_rows.append({
                "source_row": row["source_row"],
                "generation_index": row["generation_index"],
                "cif": str(best.get("cif", "")) if best else "",
                "composition": composition,
                "space_group": str(space_group),
                "scaffold_index": str(scaffold_index),
                "scaffold_full": scaffold_full,
                "cif_path": str(cif_path) if str(best.get("cif", "")).strip() else "",
                "valid_cif": "" if args.direct else str(bool(best.get("valid_cif", False))),
                "canonicalized": str(bool(best.get("canonicalized", False))),
                "canonicalize_error": str(best.get("canonicalize_error", "") or ""),
                "consistency_score": str(best.get("consistency_score", "")),
                "collapse_penalty": str(best.get("collapse_penalty", "")),
                "verify_errors": json.dumps(best.get("verify_errors", []), ensure_ascii=False),
                "raw_response": str(best.get("raw_response", "")).strip(),
            })

    fieldnames = [
        "source_row",
        "generation_index",
        "cif",
        "composition",
        "space_group",
        "scaffold_index",
        "scaffold_full",
        "cif_path",
        "valid_cif",
        "canonicalized",
        "canonicalize_error",
        "consistency_score",
        "collapse_penalty",
        "verify_errors",
        "raw_response",
    ]
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    save_run_metadata(args.output_csv, args, cfg)
    print(f"[OK] wrote {len(summary_rows)} CIF rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
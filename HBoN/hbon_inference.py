from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from inference_utils import GenerationConfig, ModelRunner, material_prompt_input, schema, save_run_metadata
from generate_scaffold_bon import load_rows, read_pool, seed_sampling, release_runner
from search_and_repair import admitted_scaffolds, select_best_cif


def allocate_scaffolds(candidates: List[Dict[str, Any]], mode: str, top_scaffolds: int) -> List[Dict[str, Any]]:
    if mode == "flat":
        return candidates[:top_scaffolds]
    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)[:top_scaffolds]


def main() -> None:
    parser = argparse.ArgumentParser(description="Flat or hierarchical Best-of-N using raw sampled cores and the original G/F scores.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-jsonl", default=None)
    parser.add_argument("--scaffold-pool", required=True, help="Raw JSONL pool saved by generate_scaffold_bon.py; reuse the same file for Flat and HBoN.")
    parser.add_argument("--search-mode", choices=["flat", "hbon"], default="hbon")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--cif-adapter", default="ckpt/cif/adapter")
    parser.add_argument("--num-scaffold", type=int, default=16)
    parser.add_argument("--num-cif-per-scaffold", type=int, default=None)
    parser.add_argument("--top-scaffolds", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--first-n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    top = args.top_scaffolds if args.top_scaffolds is not None else (4 if args.search_mode == "hbon" else 16)
    per = args.num_cif_per_scaffold if args.num_cif_per_scaffold is not None else (4 if args.search_mode == "hbon" else 1)
    if min(args.num_scaffold, top, per) <= 0 or top > args.num_scaffold:
        parser.error("Positive candidate counts and top-scaffolds <= num-scaffold are required.")
    rows = load_rows(args.input_csv, args.first_n)
    pool = read_pool(args.scaffold_pool, rows, args.num_scaffold)
    seed_sampling(args.seed)
    cfg = GenerationConfig(model_name=args.model_name, adapter_path=args.cif_adapter, adapter_stage="cif", cif_context="full",
        num_return_sequences=per, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, max_new_tokens=args.max_new_tokens, repetition_penalty=1.05)
    runner = ModelRunner(cfg)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_row", "generation_index", "composition", "space_group", "best_scaffold_full",
        "cif", "cif_path", "best_score", "valid_cif", "canonicalized", "canonicalize_error", "verifier_errors", "n_raw_scaffold",
        "n_scaffold_candidates", "n_scaffolds_kept", "n_cif_attempts", "n_cif_candidates", "search_mode", "all_candidates_fail_verifier", "selected_invalid", "scaffold_admission"]
    candidate_handle = None
    if args.candidate_jsonl:
        Path(args.candidate_jsonl).parent.mkdir(parents=True, exist_ok=True)
        candidate_handle = open(args.candidate_jsonl, "w", encoding="utf-8")
    try:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in pool:
                formula, sg = row["composition"], row["space_group"]
                admission = []
                candidates = admitted_scaffolds(row["responses"], formula, sg, diagnostics=admission)
                kept = allocate_scaffolds(candidates, args.search_mode, top)
                joint: List[Dict[str, Any]] = []
                attempts = []
                for scaffold in kept:
                    full = scaffold["scaffold_full"]
                    responses = runner.generate(schema.CIF_WITH_SCAFFOLD_INSTRUCTION,
                        material_prompt_input(formula, sg) + "\n\n" + full)
                    responses = list(responses[:per]) + [""] * max(0, per - len(responses))
                    selected = select_best_cif(responses, full, formula, sg)
                    for attempt in selected["attempts"]:
                        attempts.append({**attempt, "raw_scaffold_index": scaffold["raw_index"]})
                    for candidate in selected["all_candidates"]:
                        joint.append({**candidate, "scaffold_full": full,
                            "raw_scaffold_index": scaffold["raw_index"], "cif_index": candidate["candidate_index"]})
                best = max(joint, key=lambda candidate: candidate["score"]) if joint else {}
                cif = best.get("cif", "")
                cif_path = outdir / f"row{row['source_row']}.cif"
                if cif:
                    cif_path.write_text(cif.rstrip() + "\n", encoding="utf-8")
                writer.writerow({"source_row": row["source_row"], "generation_index": 0,
                    "composition": formula, "space_group": sg,
                    "best_scaffold_full": best.get("scaffold_full", ""), "cif": cif,
                    "cif_path": str(cif_path) if cif else "", "best_score": best.get("score", ""),
                    "valid_cif": best.get("detail", {}).get("valid", False),
                    "canonicalized": best.get("canonicalized", False),
                    "canonicalize_error": best.get("canonicalize_error", ""),
                    "verifier_errors": json.dumps(best.get("detail", {"errors": ["no_candidates"]}).get("errors", [])),
                    "n_raw_scaffold": len(row["responses"]), "n_scaffold_candidates": len(candidates),
                    "n_scaffolds_kept": len(kept), "n_cif_attempts": len(kept) * per,
                    "n_cif_candidates": len(joint), "search_mode": args.search_mode,
                    "all_candidates_fail_verifier": bool(joint) and not any(candidate["detail"].get("valid", False) for candidate in joint),
                    "selected_invalid": bool(best) and not best["detail"].get("valid", False),
                    "scaffold_admission": json.dumps(admission)})
                handle.flush()
                if candidate_handle:
                    candidate_handle.write(json.dumps({"source_row": row["source_row"],
                        "composition": formula, "space_group": sg, "scaffolds": candidates, "scaffold_admission": admission,
                        "selected_raw_scaffold_indices": [candidate["raw_index"] for candidate in kept],
                        "attempts": attempts, "selected": best}, ensure_ascii=False) + "\n")
                    candidate_handle.flush()
    finally:
        if candidate_handle:
            candidate_handle.close()
        release_runner(runner)
    args.top_scaffolds = top
    args.num_cif_per_scaffold = per
    save_run_metadata(str(output), args, cfg)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

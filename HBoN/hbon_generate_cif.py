from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from generate_scaffold_bon import release_runner, sample_pool, seed_sampling
from hbon_inference import allocate_scaffolds
from inference_utils import GenerationConfig, ModelRunner, material_prompt_input, save_run_metadata, schema
from search_and_repair import admitted_scaffolds, select_best_cif


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one final CIF for one composition and space group using Flat-BoN or HBoN.")
    parser.add_argument("--composition", required=True)
    parser.add_argument("--space-group", type=int, required=True)
    parser.add_argument("--output-cif", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--search-mode", choices=["flat", "hbon"], default="hbon")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--scaffold-adapter-path", "--scaffold-adapter", dest="scaffold_adapter_path", default="ckpt/scaffold/adapter")
    parser.add_argument("--cif-adapter-path", "--cif-adapter", dest="cif_adapter_path", default="ckpt/cif/adapter")
    parser.add_argument("--num-scaffold", type=int, default=16)
    parser.add_argument("--top-scaffolds", type=int, default=None)
    parser.add_argument("--num-cif-per-scaffold", type=int, default=None)
    parser.add_argument("--scaffold-temperature", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0, help="CIF sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    args.composition = args.composition.strip()
    if not args.composition:
        parser.error("--composition must not be empty")
    if not 1 <= args.space_group <= 230:
        parser.error("--space-group must be between 1 and 230")
    top = args.top_scaffolds if args.top_scaffolds is not None else min(args.num_scaffold, 4 if args.search_mode == "hbon" else 16)
    per = args.num_cif_per_scaffold if args.num_cif_per_scaffold is not None else (4 if args.search_mode == "hbon" else 1)
    if args.num_scaffold < 1 or top < 1 or per < 1 or top > args.num_scaffold:
        parser.error("Candidate counts must be positive, and top-scaffolds must not exceed num-scaffold")
    if args.scaffold_temperature <= 0 or args.temperature <= 0:
        parser.error("Sampling temperatures must be positive")
    if not 0 < args.top_p <= 1 or args.top_k < 0 or args.max_new_tokens < 1:
        parser.error("Require 0 < top-p <= 1, top-k >= 0, and max-new-tokens >= 1")

    scaffold_cfg = GenerationConfig(
        model_name=args.model_name,
        adapter_path=args.scaffold_adapter_path,
        adapter_stage="scaffold",
        num_return_sequences=args.num_scaffold,
        temperature=args.scaffold_temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.03,
    )
    cif_cfg = GenerationConfig(
        model_name=args.model_name,
        adapter_path=args.cif_adapter_path,
        adapter_stage="cif",
        cif_context="full",
        num_return_sequences=per,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.05,
    )

    pool = sample_pool(
        [{"source_row": 0, "composition": args.composition, "space_group": args.space_group}],
        model_name=args.model_name,
        adapter_path=args.scaffold_adapter_path,
        seed=args.seed,
        count=args.num_scaffold,
        temperature=args.scaffold_temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )
    raw = pool[0]
    admission: List[Dict[str, Any]] = []
    candidates = admitted_scaffolds(raw["responses"], args.composition, args.space_group, diagnostics=admission)
    kept = allocate_scaffolds(candidates, args.search_mode, top)
    joint: List[Dict[str, Any]] = []
    attempts: List[Dict[str, Any]] = []

    if kept:
        seed_sampling(args.seed)
        runner = ModelRunner(cif_cfg)
        try:
            for scaffold in kept:
                full = scaffold["scaffold_full"]
                responses = runner.generate(
                    schema.CIF_WITH_SCAFFOLD_INSTRUCTION,
                    material_prompt_input(args.composition, args.space_group) + "\n\n" + full,
                )
                responses = list(responses[:per]) + [""] * max(0, per - len(responses))
                selected = select_best_cif(responses, full, args.composition, args.space_group)
                for attempt in selected["attempts"]:
                    attempts.append({**attempt, "raw_scaffold_index": scaffold["raw_index"]})
                for candidate in selected["all_candidates"]:
                    joint.append({
                        **candidate,
                        "scaffold_full": full,
                        "raw_scaffold_index": scaffold["raw_index"],
                        "cif_index": candidate["candidate_index"],
                    })
        finally:
            release_runner(runner)

    best = max(joint, key=lambda candidate: candidate["score"]) if joint else None
    cif = best["cif"] if best else ""
    report: Dict[str, Any] = {
        "composition": args.composition,
        "space_group": args.space_group,
        "search_mode": args.search_mode,
        "n_raw_scaffold": len(raw["responses"]),
        "n_observed_scaffold": raw["observed_responses"],
        "n_scaffold_candidates": len(candidates),
        "n_scaffolds_kept": len(kept),
        "n_cif_attempts": len(kept) * per,
        "n_cif_candidates": len(joint),
        "all_candidates_fail_verifier": bool(joint) and not any(c["detail"].get("valid", False) for c in joint),
        "selected_invalid": bool(best) and not best["detail"].get("valid", False),
        "status": "generated" if cif.strip() else "no_cif_candidate",
        "best": best,
        "scaffold_admission": admission,
        "selected_raw_scaffold_indices": [scaffold["raw_index"] for scaffold in kept],
        "attempts": attempts,
    }
    if args.output_json:
        _write_json(args.output_json, report)
    if not cif.strip():
        raise RuntimeError("No CIF candidate was produced; inspect --output-json for scaffold/CIF diagnostics.")

    target = Path(args.output_cif)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(cif.rstrip() + "\n", encoding="utf-8")
    args.top_scaffolds = top
    args.num_cif_per_scaffold = per
    save_run_metadata(str(target), args, {"scaffold": asdict(scaffold_cfg), "cif": asdict(cif_cfg)})
    print(json.dumps({
        "output_cif": str(target),
        "search_mode": args.search_mode,
        "n_scaffolds_kept": len(kept),
        "n_cif_candidates": len(joint),
        "best_score": best["score"],
        "valid_cif": best["detail"].get("valid", False),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

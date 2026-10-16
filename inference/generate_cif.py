from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Dict, List

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


def _write_text(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def _load_scaffold_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one V4 CIF from composition + space group + scaffold_full.")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--adapter-path", default="ckpt/cif/adapter")
    parser.add_argument("--composition", required=True)
    parser.add_argument("--space-group", type=int, required=True)
    parser.add_argument("--scaffold-full-file", "--scaffold-file", dest="scaffold_full_file", default=None, help="Scaffold context text matching --cif-context full/core.")
    parser.add_argument("--cif-context", choices=["full", "core", "none"], default="full")
    parser.add_argument("--direct", action="store_true", help="Generate the direct CIF baseline without scaffold conditioning.")
    parser.add_argument("--num-samples", type=int, choices=[1], default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-cif", required=True)
    parser.add_argument("--output-json", default=None)
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

    if args.greedy:
        args.num_samples = 1

    if not args.direct and not args.scaffold_full_file:
        parser.error("--scaffold-full-file is required unless --direct is selected")
    if args.direct and args.scaffold_full_file:
        parser.error("--direct and --scaffold-full-file are mutually exclusive")
    scaffold_full = "" if args.direct else _load_scaffold_text(args.scaffold_full_file)
    if not args.direct and not scaffold_full:
        raise ValueError("Empty compiled scaffold; no CIF was sampled.")
    input_text = material_prompt_input(args.composition, args.space_group)
    if not args.direct:
        input_text += "\n\n" + scaffold_full

    cfg = GenerationConfig(
        model_name=args.model_name,
        adapter_path=args.adapter_path, adapter_stage="cif", cif_context=args.cif_context,
        num_return_sequences=args.num_samples,
        temperature=0.0 if args.greedy else args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.05,
        do_sample=not args.greedy,
        max_seq_length=4096,
    )
    runner = ModelRunner(cfg)
    raw = runner.generate(schema.BASELINE_INSTRUCTION if args.direct else schema.CIF_WITH_SCAFFOLD_INSTRUCTION, input_text)

    rows: List[Dict] = []
    for i, resp in enumerate(raw):
        cif_text, canonicalized, can_err = maybe_canonicalize_generated_cif(resp)
        verify = {} if args.direct else verify_cif_against_scaffold(
            cif_text,
            scaffold_full,
            formula=args.composition,
            condition_sg_number=args.space_group,
        )
        rows.append({
            "rank": i,
            "canonicalized": canonicalized,
            "canonicalize_error": can_err,
            "cif": cif_text,
            **verify,
        })

    best = rows[0] if rows else None

    if best is None:
        raise RuntimeError("No CIF candidate generated.")

    _write_text(args.output_cif, best["cif"].strip() + "\n")
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"best": best, "all": rows}, f, indent=2, ensure_ascii=False)

    save_run_metadata(args.output_cif, args, cfg)
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

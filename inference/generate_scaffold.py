from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from inference_utils import GenerationConfig, ModelRunner, material_prompt_input, maybe_compile_generated_core, schema, save_run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one V4 scaffold core and compile its context.")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--adapter-path", default="ckpt/scaffold/adapter")
    parser.add_argument("--composition", required=True)
    parser.add_argument("--space-group", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-scaffold-full-txt", default=None)
    args = parser.parse_args()
    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = GenerationConfig(model_name=args.model_name, adapter_path=args.adapter_path, adapter_stage="scaffold",
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_new_tokens=args.max_new_tokens, repetition_penalty=1.03)
    responses = ModelRunner(cfg).generate(schema.SCAFFOLD_CORE_INSTRUCTION,
        material_prompt_input(args.composition, args.space_group))
    core, full, ok, errors, fields = maybe_compile_generated_core(responses[0],
        formula=args.composition, condition_sg_number=args.space_group)
    row: Dict = {"rank": 0, "raw_response": responses[0], "scaffold_core": core,
        "scaffold_full": full, "valid_scaffold": ok, "scaffold_errors": errors, "fields": fields}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_scaffold_full_txt:
        target = Path(args.output_scaffold_full_txt)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(full + "\n", encoding="utf-8")
    save_run_metadata(args.output_json, args, cfg)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

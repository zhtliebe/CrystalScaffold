from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import json
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llama_chat import DEFAULT_SYSTEM_PROMPT, build_prompt, configure_tokenizer, extract_assistant, stop_token_ids
from preprocess_v4 import schema
from preprocess_v4.cif_utils import (
    build_prompt_input, canonicalize_cif_text, parse_cif_text, crystal_system_from_sg, length_pattern, angle_pattern,
    ratio_bin, c_over_a_bin_family, angle_bin, vol_per_atom_bin, metric_family,
    position_family, stack_family, compile_full_scaffold, parse_scaffold_text,
)


@dataclass
class GenerationConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    adapter_path: Optional[str] = None
    adapter_stage: Optional[str] = None
    cif_context: Optional[str] = None
    load_in_4bit: bool = True
    dtype: Optional[str] = None
    max_seq_length: int = 4096
    max_new_tokens: int = 1800
    temperature: float = 1.0
    top_p: float = 0.92
    top_k: int = 20
    num_return_sequences: int = 1
    repetition_penalty: float = 1.05
    do_sample: bool = True
    device: Optional[str] = None


def material_prompt_input(composition: str, space_group_number: int) -> str:
    return build_prompt_input(composition, int(space_group_number))


def extract_scaffold_core_from_response(text: str) -> str:
    if schema.SCAFFOLD_CORE_BEGIN in text and schema.SCAFFOLD_CORE_END in text:
        start = text.index(schema.SCAFFOLD_CORE_BEGIN)
        end = text.index(schema.SCAFFOLD_CORE_END, start) + len(schema.SCAFFOLD_CORE_END)
        return text[start:end].strip()
    return str(text).strip()


def extract_cif_from_response(text: str) -> str:
    text = str(text).strip()
    if "data_" in text:
        text = text[text.index("data_"):]
    closing_fence = re.search(r"(?m)^\s*```[ \t]*$", text)
    if closing_fence:
        text = text[:closing_fence.start()]
    return text.strip()


def _sync_padding_to_model(model: Any, tokenizer: Any) -> None:
    visited = set()
    queue = [model]
    while queue:
        obj = queue.pop(0)
        if obj is None or id(obj) in visited:
            continue
        visited.add(id(obj))
        for name in ("config", "generation_config"):
            config = getattr(obj, name, None)
            if config is not None:
                config.pad_token_id = tokenizer.pad_token_id
                config.eos_token_id = stop_token_ids(tokenizer)
        queue.extend(getattr(obj, name, None) for name in ("base_model", "model"))


def validate_adapter_protocol(adapter_path: str, expected_stage: Optional[str] = None, expected_cif_context: Optional[str] = None) -> Dict[str, Any]:
    path = Path(adapter_path)
    marker = path / "adapter_protocol.json"
    if not path.is_dir() or not marker.is_file():
        raise ValueError(
            f"Adapter directory {adapter_path!r} lacks adapter_protocol.json. "
            "Use an adapter trained by the migrated training entry points; "
            "historical adapters do not establish compatibility with this instruction protocol."
        )
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if (metadata.get("instruction_protocol") != "llama3.1-system-user-assistant-v1"
            or metadata.get("system_prompt") != DEFAULT_SYSTEM_PROMPT):
        raise ValueError("Adapter instruction protocol or system prompt differs from this release.")
    if expected_stage and metadata.get("stage") != expected_stage:
        raise ValueError(f"Expected {expected_stage!r} adapter stage; got {metadata.get('stage')!r}.")
    if expected_cif_context and metadata.get("cif_context") != expected_cif_context:
        raise ValueError(f"Expected CIF context {expected_cif_context!r}; got {metadata.get('cif_context')!r}.")
    return metadata


def save_run_metadata(output_path: str, args: Any, generation: Any) -> None:
    target = Path(str(output_path) + ".meta.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"instruction_protocol": "llama3.1-system-user-assistant-v1",
        "system_prompt": DEFAULT_SYSTEM_PROMPT, "arguments": vars(args),
        "resolved_generation": asdict(generation) if isinstance(generation, GenerationConfig) else generation}
    pool_path = getattr(args, "scaffold_pool", None)
    if pool_path:
        digest = hashlib.sha256()
        with open(pool_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        with open(pool_path, encoding="utf-8") as handle:
            header = json.loads(next(line for line in handle if line.strip()))
        payload["scaffold_pool"] = {"sha256": digest.hexdigest(), "metadata": header.get("metadata", {})}
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class ModelRunner:
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.backend = None

    def load(self) -> None:
        if self.cfg.adapter_path:
            validate_adapter_protocol(self.cfg.adapter_path, self.cfg.adapter_stage, self.cfg.cif_context)
        import torch
        has_cuda = torch.cuda.is_available()
        FastLanguageModel = None
        if has_cuda:
            try:
                from unsloth import FastLanguageModel
            except ImportError:
                pass
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        dtype = getattr(torch, self.cfg.dtype) if self.cfg.dtype else None
        if FastLanguageModel is not None:
            self.backend = "unsloth"
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.cfg.model_name, max_seq_length=self.cfg.max_seq_length,
                dtype=dtype, load_in_4bit=self.cfg.load_in_4bit,
            )
        else:
            self.backend = "transformers"
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
            model_kwargs = {"low_cpu_mem_usage": True}
            if has_cuda:
                model_kwargs["device_map"] = self.cfg.device or "auto"
                model_kwargs["torch_dtype"] = dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                if self.cfg.load_in_4bit:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
                        bnb_4bit_use_double_quant=True,
                    )
            else:
                model_kwargs["torch_dtype"] = dtype or torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **model_kwargs)
        configure_tokenizer(self.tokenizer, padding_side="left")
        if self.cfg.adapter_path:
            self.model = PeftModel.from_pretrained(self.model, self.cfg.adapter_path)
        _sync_padding_to_model(self.model, self.tokenizer)
        if FastLanguageModel is not None:
            FastLanguageModel.for_inference(self.model)
        self.model.eval()

    def _model_device(self):
        return next(self.model.parameters()).device

    def generate_batch(self, instruction: str, input_texts: Sequence[str]) -> List[List[str]]:
        if not input_texts:
            return []
        if self.model is None or self.tokenizer is None:
            self.load()
        if self.cfg.num_return_sequences < 1 or self.cfg.max_new_tokens < 1:
            raise ValueError("Generation counts and token budgets must be positive.")
        if not self.cfg.do_sample and self.cfg.num_return_sequences != 1:
            raise ValueError("Greedy decoding requires one return sequence.")
        prompts = [build_prompt(instruction, text) for text in input_texts]
        toks = self.tokenizer(prompts, return_tensors="pt", padding=True,
                              truncation=False, add_special_tokens=False)
        prompt_length = int(toks["input_ids"].shape[-1])
        if prompt_length + self.cfg.max_new_tokens > self.cfg.max_seq_length:
            raise ValueError(
                f"Prompt ({prompt_length}) plus requested generation ({self.cfg.max_new_tokens}) "
                f"exceeds max_seq_length={self.cfg.max_seq_length}; no content was truncated."
            )
        toks = {key: value.to(self._model_device()) for key, value in toks.items()}
        kwargs = dict(max_new_tokens=self.cfg.max_new_tokens,
                      num_return_sequences=self.cfg.num_return_sequences,
                      repetition_penalty=self.cfg.repetition_penalty,
                      do_sample=self.cfg.do_sample, use_cache=True,
                      pad_token_id=self.tokenizer.pad_token_id,
                      eos_token_id=stop_token_ids(self.tokenizer))
        if self.cfg.do_sample:
            kwargs.update(temperature=self.cfg.temperature, top_p=self.cfg.top_p, top_k=self.cfg.top_k)
        import torch
        with torch.inference_mode():
            outputs = self.model.generate(**toks, **kwargs)
        decoded = [extract_assistant(self.tokenizer, output[prompt_length:]) for output in outputs]
        n = self.cfg.num_return_sequences
        return [decoded[i * n:(i + 1) * n] for i in range(len(prompts))]

    def generate(self, instruction: str, input_text: str) -> List[str]:
        return self.generate_batch(instruction, [input_text])[0]


def scaffold_quality_checks_v4(scaffold_text: str, *, formula: str, condition_sg_number: int, expect_full: bool = False) -> Tuple[bool, List[str], Dict[str, str]]:
    errors: List[str] = []
    fields = dict(parse_scaffold_text(scaffold_text))
    if not fields:
        return False, ["empty_scaffold"], {}

    if fields.get("formula", "") != formula:
        errors.append("formula_mismatch")
    if str(fields.get("condition_sg_number", "")) != str(int(condition_sg_number)):
        errors.append("condition_sg_mismatch")
    expected_cs = crystal_system_from_sg(int(condition_sg_number))
    if fields.get("condition_crystal_system", "") not in {"", expected_cs}:
        errors.append("crystal_system_mismatch")

    required_core = [
        "formula", "reduced_ratio", "condition_sg_number", "condition_crystal_system",
        "target_cell_mode", "output_Z_raw", "metric_family", "length_pattern",
        "angle_pattern", "vol_per_atom_bin", "position_family", "stack_family",
    ]
    for key in required_core:
        if key not in fields or fields[key] == "":
            errors.append(f"missing_{key}")

    if fields.get("target_cell_mode", "") not in {"", "primitive"}:
        errors.append("bad_target_cell_mode")

    if expect_full:
        for key in ["site_schema", "n_listed_sites", "formula_sum_expected", "geometry_signature_v4", "prototype_bucket_v4"]:
            if key not in fields or fields[key] == "":
                errors.append(f"missing_{key}")

    return len(errors) == 0, errors, fields


def maybe_compile_generated_core(response_text: str, *, formula: str, condition_sg_number: int) -> Tuple[str, str, bool, List[str], Dict[str, str]]:
    core_text = extract_scaffold_core_from_response(response_text)
    ok, errors, fields = scaffold_quality_checks_v4(core_text, formula=formula, condition_sg_number=condition_sg_number, expect_full=False)
    full_text = ""
    if ok:
        try:
            full_text = compile_full_scaffold(core_text, formula=formula)
            ok2, errors2, full_fields = scaffold_quality_checks_v4(full_text, formula=formula, condition_sg_number=condition_sg_number, expect_full=True)
            ok = ok and ok2
            errors = list(dict.fromkeys(errors + errors2))
            fields = full_fields
        except Exception as e:
            ok = False
            errors.append(f"compile_full_failed:{e}")
    return core_text, full_text, ok, errors, fields


def _looks_degenerate_cif(cif_text: str) -> bool:
    low = str(cif_text)
    if "data_unknown" in low:
        return True
    if "_chemical_formula_structural unknown" in low:
        return True
    if "_chemical_formula_sum ''" in low or '_chemical_formula_sum ""' in low:
        return True
    if "_cell_length_a 0.000000" in low and "_cell_length_b 0.000000" in low and "_cell_length_c 0.000000" in low:
        return True
    return False


def maybe_canonicalize_generated_cif(text: str) -> Tuple[str, bool, Optional[str]]:
    cif_text = extract_cif_from_response(text)
    if not cif_text:
        return "", False, "empty_response"
    try:
        can = canonicalize_cif_text(cif_text)
        if _looks_degenerate_cif(can):
            return can, False, "degenerate_canonicalized_cif"
        return can, True, None
    except Exception as e:
        return cif_text, False, str(e)


def _counts_to_schema(counts: Dict[str, int]) -> str:
    return "|".join(f"{el}:{cnt}" for el, cnt in counts.items())


def _site_schema_counts(rec) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for site in rec.atom_sites:
        counts[site.element] = counts.get(site.element, 0) + 1
    return counts


def _b_over_a_bin_if_needed(rec) -> str:
    if length_pattern(rec) == "ALL_DIFF":
        return ratio_bin(rec.cell_b / max(rec.cell_a, 1e-8))
    return ""


def collapse_penalty(rec, scaffold_fields: Dict[str, str]) -> int:
    penalty = 0
    exp_pos = scaffold_fields.get("position_family", "")
    exp_stack = scaffold_fields.get("stack_family", "")
    cand_pos = position_family(rec)
    cand_stack = stack_family(rec)

    if exp_pos == "GENERAL_DOMINANT" and cand_pos in {"HIGH_SPECIAL", "MIXED_SPECIAL"}:
        penalty += 2
    if exp_pos == "MIXED_SPECIAL" and cand_pos == "HIGH_SPECIAL":
        penalty += 1
    if exp_stack == "NO_LAYER" and cand_stack.startswith("LAYERED"):
        penalty += 2
    if exp_stack == "SKEW_DIAGONAL" and cand_stack != "SKEW_DIAGONAL":
        penalty += 1
    return penalty


def verify_cif_against_scaffold(cif_text: str, scaffold_full_text: str, *, formula: str, condition_sg_number: int) -> Dict[str, Any]:
    scaffold_fields = dict(parse_scaffold_text(scaffold_full_text))
    out: Dict[str, Any] = {
        "valid_cif": False,
        "verify_errors": [],
        "consistency_score": -999,
        "collapse_penalty": 999,
    }
    try:
        rec = parse_cif_text(cif_text)
    except Exception as e:
        out["verify_errors"] = [f"parse_failed:{e}"]
        return out

    errors: List[str] = []
    if not rec.atom_sites:
        errors.append("empty_atom_sites")
    if float(rec.cell_a) <= 0 or float(rec.cell_b) <= 0 or float(rec.cell_c) <= 0:
        errors.append("zero_lattice")
    if str((rec.formula_structural or "").strip()) != str(formula).strip():
        errors.append("formula_structural_mismatch")
    if int(rec.z) != int(scaffold_fields.get("output_Z_raw", "1")):
        errors.append("output_Z_raw_mismatch")
    if len(rec.atom_sites) != int(scaffold_fields.get("n_listed_sites", str(len(rec.atom_sites)))):
        errors.append("n_listed_sites_mismatch")
    site_schema = _counts_to_schema(_site_schema_counts(rec))
    if site_schema != scaffold_fields.get("site_schema", site_schema):
        errors.append("site_schema_mismatch")
    if metric_family(rec) != scaffold_fields.get("metric_family", metric_family(rec)):
        errors.append("metric_family_mismatch")
    if length_pattern(rec) != scaffold_fields.get("length_pattern", length_pattern(rec)):
        errors.append("length_pattern_mismatch")
    if angle_pattern(rec) != scaffold_fields.get("angle_pattern", angle_pattern(rec)):
        errors.append("angle_pattern_mismatch")
    if "eq_angle_bin" in scaffold_fields:
        ap = angle_pattern(rec)
        if ap == "ALPHA_EQ_BETA_NE_GAMMA":
            cand_eq = angle_bin(rec.alpha)
        elif ap == "BETA_EQ_GAMMA_NE_ALPHA":
            cand_eq = angle_bin(rec.beta)
        elif ap == "ALPHA_EQ_GAMMA_NE_BETA":
            cand_eq = angle_bin(rec.alpha)
        else:
            cand_eq = ""
        if cand_eq != scaffold_fields.get("eq_angle_bin", cand_eq):
            errors.append("eq_angle_bin_mismatch")
    if "gamma_bin" in scaffold_fields and angle_bin(rec.gamma) != scaffold_fields.get("gamma_bin", angle_bin(rec.gamma)):
        errors.append("gamma_bin_mismatch")
    if "b_over_a_bin" in scaffold_fields and _b_over_a_bin_if_needed(rec) != scaffold_fields.get("b_over_a_bin", _b_over_a_bin_if_needed(rec)):
        errors.append("b_over_a_bin_mismatch")
    if "c_over_a_bin_family" in scaffold_fields and c_over_a_bin_family(rec) != scaffold_fields.get("c_over_a_bin_family", c_over_a_bin_family(rec)):
        errors.append("c_over_a_bin_family_mismatch")
    if vol_per_atom_bin(rec) != scaffold_fields.get("vol_per_atom_bin", vol_per_atom_bin(rec)):
        errors.append("vol_per_atom_bin_mismatch")
    if position_family(rec) != scaffold_fields.get("position_family", position_family(rec)):
        errors.append("position_family_mismatch")
    if stack_family(rec) != scaffold_fields.get("stack_family", stack_family(rec)):
        errors.append("stack_family_mismatch")

    penalty = collapse_penalty(rec, scaffold_fields)
    score = 100
    hard = {
        "empty_atom_sites", "zero_lattice", "formula_structural_mismatch", "output_Z_raw_mismatch",
        "n_listed_sites_mismatch", "site_schema_mismatch",
    }
    score -= 12 * sum(1 for e in errors if e in hard)
    score -= 5 * sum(1 for e in errors if e not in hard)
    score -= 6 * penalty

    out.update({
        "valid_cif": len(errors) == 0,
        "verify_errors": errors,
        "consistency_score": score,
        "collapse_penalty": penalty,
        "cif_fields": {
            "formula": rec.formula_structural,
            "output_Z_raw": int(rec.z),
            "n_listed_sites": len(rec.atom_sites),
            "site_schema": site_schema,
            "metric_family": metric_family(rec),
            "length_pattern": length_pattern(rec),
            "angle_pattern": angle_pattern(rec),
            "eq_angle_bin": angle_bin(rec.alpha) if angle_pattern(rec) in {"ALPHA_EQ_BETA_NE_GAMMA","ALPHA_EQ_GAMMA_NE_BETA"} else (angle_bin(rec.beta) if angle_pattern(rec) == "BETA_EQ_GAMMA_NE_ALPHA" else ""),
            "gamma_bin": angle_bin(rec.gamma),
            "b_over_a_bin": _b_over_a_bin_if_needed(rec),
            "c_over_a_bin_family": c_over_a_bin_family(rec),
            "vol_per_atom_bin": vol_per_atom_bin(rec),
            "position_family": position_family(rec),
            "stack_family": stack_family(rec),
        },
    })
    return out



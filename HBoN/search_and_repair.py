from __future__ import annotations
from pathlib import Path

from typing import Any, Dict, List, Sequence
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from crystal_verifier import cif_score, scaffold_score
from inference_utils import maybe_compile_generated_core, maybe_canonicalize_generated_cif


def admitted_scaffolds(responses: Sequence[str], formula: str, sg: int, diagnostics=None) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for index, response in enumerate(responses):
        core, full, ok, errors, fields = maybe_compile_generated_core(response,
            formula=formula, condition_sg_number=sg)
        if not ok or not full:
            if diagnostics is not None:
                diagnostics.append({"raw_index": index, "admitted": False, "errors": errors or ["empty_compiled_context"]})
            continue
        if full in seen:
            if diagnostics is not None:
                diagnostics.append({"raw_index": index, "admitted": False, "errors": ["duplicate_context"]})
            continue
        seen.add(full)
        try:
            detail = scaffold_score(full, formula, sg)
        except Exception as error:
            detail = {"score": -1e9, "valid": False, "errors": [f"scoring_failed:{error}"]}
        candidates.append({"raw_index": index, "scaffold_core": core, "scaffold_full": full,
            "score": detail["score"], "detail": detail})
        if diagnostics is not None:
            diagnostics.append({"raw_index": index, "admitted": True, "errors": []})
    return candidates


def select_best_scaffold(scaffold_candidates: Sequence[str], formula: str, sg: int) -> Dict[str, Any]:
    candidates = admitted_scaffolds(scaffold_candidates, formula, sg)
    if not candidates:
        return {"scaffold_full": "", "score": -1e9,
            "detail": {"errors": ["no_candidates"]}, "all_candidates": []}
    best = dict(max(candidates, key=lambda row: row["score"]))
    best["all_candidates"] = candidates
    return best


def select_best_cif(cif_candidates: Sequence[str], scaffold_full: str, formula: str, sg: int) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    attempts: List[Dict[str, Any]] = []
    seen = {}
    for index, text in enumerate(cif_candidates):
        if not str(text).strip():
            attempts.append({"candidate_index": index, "raw_response": str(text),
                "score": -1e9, "detail": {"valid": False, "errors": ["empty_response"]}})
            continue
        extracted, canonicalized, canonicalize_error = maybe_canonicalize_generated_cif(text)
        try:
            detail = cif_score(extracted, scaffold_full, formula, sg)
        except Exception as error:
            detail = {"score": -1e9, "valid": False, "errors": [f"scoring_failed:{error}"]}
        cif = detail.get("canonical_cif", extracted)
        duplicate_of = seen.get(cif)
        attempts.append({"candidate_index": index, "raw_response": text, "score": detail["score"],
            "detail": detail, "duplicate_of": duplicate_of, "canonicalized": canonicalized,
            "canonicalize_error": canonicalize_error})
        if cif in seen:
            continue
        seen[cif] = index
        candidates.append({"candidate_index": index, "cif": cif,
            "score": detail["score"], "detail": detail, "actions": [],
            "canonicalized": canonicalized, "canonicalize_error": canonicalize_error})
    if not candidates:
        return {"cif": "", "score": -1e9, "detail": {"errors": ["no_candidates"]},
            "all_candidates": [], "attempts": attempts,
            "all_candidates_fail_verifier": False, "selected_invalid": False}
    best = dict(max(candidates, key=lambda row: row["score"]))
    best.update(all_candidates=candidates, attempts=attempts,
        all_candidates_fail_verifier=not any(candidate["detail"].get("valid", False) for candidate in candidates),
        selected_invalid=not best["detail"].get("valid", False))
    return best

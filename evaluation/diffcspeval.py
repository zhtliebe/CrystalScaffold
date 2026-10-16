from __future__ import annotations

# Copyright (c) 2023 Luis M. Antunes (adapted benchmark portions).
# See THIRD_PARTY_NOTICES.md.

import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

from validity import smact_validity_from_structure, structure_validity
from crystextmatch import prepare_generated_groups


def is_valid(struct):
    return smact_validity_from_structure(struct) and structure_validity(struct)


def is_sensible(
    cif_str: str,
    length_lo: float = 0.5,
    length_hi: float = 1000.0,
    angle_lo: float = 10.0,
    angle_hi: float = 170.0,
) -> bool:
    cell_length_pattern = re.compile(r"_cell_length_[abc]\s+([\d\.]+)")
    cell_angle_pattern = re.compile(
        r"_cell_angle_(alpha|beta|gamma)\s+([\d\.]+)"
    )

    cell_lengths = cell_length_pattern.findall(cif_str)
    for length_str in cell_lengths:
        length = float(length_str)
        if length < length_lo or length > length_hi:
            return False

    cell_angles = cell_angle_pattern.findall(cif_str)
    for _, angle_str in cell_angles:
        angle = float(angle_str)
        if angle < angle_lo or angle > angle_hi:
            return False

    return True


def _normalize_cif_cell(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value)
    if not text.strip():
        return None
    return text


def _parse_reference(cif: str) -> Structure:
    
    return Structure.from_str(cif, fmt="cif")


def _evaluate_generated_against_reference(
    generated_cif: Optional[str],
    reference_structure: Structure,
    matcher: StructureMatcher,
    length_lo: float,
    length_hi: float,
    angle_lo: float,
    angle_hi: float,
) -> Tuple[bool, Optional[float], str]:
    if generated_cif is None:
        return False, None, "missing_generated_cif"

    try:
        if not is_sensible(
            generated_cif,
            length_lo=length_lo,
            length_hi=length_hi,
            angle_lo=angle_lo,
            angle_hi=angle_hi,
        ):
            return False, None, "not_sensible"
    except Exception as exc:
        return False, None, f"sensibility_error:{type(exc).__name__}"

    try:
        pred = Structure.from_str(generated_cif, fmt="cif")
    except Exception as exc:
        return False, None, f"generated_parse_error:{type(exc).__name__}"

    try:
        if not is_valid(pred):
            return False, None, "invalid_generated_structure"
    except Exception as exc:
        
        
        return False, None, f"validity_error:{type(exc).__name__}"

    try:
        rms_result = matcher.get_rms_dist(pred, reference_structure)
        if rms_result is None:
            return False, None, "structure_unmatched"
        rms_dist = float(rms_result[0])
        return True, rms_dist, "matched"
    except Exception as exc:
        return False, None, f"matcher_error:{type(exc).__name__}"


def load_csv(path: Path, cif_column: str, encoding: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        
        if encoding.lower().replace("_", "-") not in {"utf-8", "utf-8-sig"}:
            raise
        df = pd.read_csv(path, encoding="utf-8-sig")

    if cif_column not in df.columns:
        raise ValueError(
            f"CSV {path} does not contain required column {cif_column!r}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def resolve_n_pairs(n_gen: int, n_ref: int, first_n: Optional[int]) -> int:
    if first_n is not None:
        if first_n <= 0:
            raise ValueError("--first-n must be a positive integer.")
        if first_n > n_gen:
            raise ValueError(
                f"--first-n={first_n}, but generated CSV has only {n_gen} rows."
            )
        if first_n > n_ref:
            raise ValueError(
                f"--first-n={first_n}, but reference CSV has only {n_ref} rows."
            )
        return first_n

    if n_gen != n_ref:
        raise ValueError(
            "Generated and reference CSV row counts differ "
            f"({n_gen} vs {n_ref}). Use --first-n N to explicitly evaluate "
            "the first N row-aligned pairs, or provide equal-length CSVs."
        )
    return n_gen


def evaluate_groups(reference_df, groups, cif_column="cif", num_gens=1, first_n=None,
                    ltol=0.3, stol=0.5, angle_tol=10.0,
                    length_lo=0.5, length_hi=1000.0, angle_lo=10.0, angle_hi=170.0):
    if num_gens < 1 or (first_n is not None and first_n < 1):
        raise ValueError("Candidate count and subset size must be positive")
    n = len(reference_df) if first_n is None else first_n
    if not 0 < n <= len(reference_df):
        raise ValueError("Reference subset is empty or exceeds reference length")
    if any(i < 0 or i >= len(reference_df) for i in groups):
        raise ValueError("Generated source_row is outside the reference ledger")
    matcher = StructureMatcher(stol=stol, ltol=ltol, angle_tol=angle_tol)
    details = []
    for i in range(n):
        reference_cif = _normalize_cif_cell(reference_df.iloc[i][cif_column])
        if reference_cif is None:
            raise ValueError(f"Reference row {i} has an empty CIF")
        try:
            reference = _parse_reference(reference_cif)
        except Exception as exc:
            raise ValueError(f"Reference row {i} is invalid: {exc}") from exc
        candidates = groups.get(i, [])
        records = []
        for j in range(num_gens):
            cif = _normalize_cif_cell(candidates[j]) if j < len(candidates) else None
            matched, rms, status = _evaluate_generated_against_reference(
                cif, reference, matcher, length_lo, length_hi, angle_lo, angle_hi)
            if matched and not math.isfinite(rms):
                matched, rms, status = False, None, "nonfinite_rms"
            records.append({"generation_index": j, "matched": matched, "rms": rms, "status": status})
        matched_rms = [r["rms"] for r in records if r["matched"]]
        details.append({"source_row": i, "matched": bool(matched_rms),
                        "best_rms": min(matched_rms) if matched_rms else None,
                        "candidate_details": json.dumps(records)})
    count = sum(r["matched"] for r in details)
    errors = [r["best_rms"] for r in details if r["matched"]]
    summary = {"n_test_entries": n, "num_gens": num_gens, "matched_count": count,
               "match_rate": count / n, "match_rate_percent": 100 * count / n,
               "mean_best_rms_matched": float(np.mean(errors)) if errors else float("nan"),
               "ltol": ltol, "stol": stol, "angle_tol": angle_tol,
               "matcher_direction": "generated -> reference", "validity_gate": True}
    return summary, pd.DataFrame(details)


def build_parser():
    p = argparse.ArgumentParser(description="Formal Crystal-Scaffold generated-first recovery evaluation.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--generated-csv", type=Path)
    source.add_argument("--generated-dir", type=Path)
    p.add_argument("--reference-csv", required=True, type=Path)
    p.add_argument("--cif-column", default="cif")
    p.add_argument("--reference-cif-column", default=None)
    p.add_argument("--source-row-column", default="source_row")
    p.add_argument("--generation-column", default="generation_index")
    p.add_argument("--num-gens", type=int, default=1)
    p.add_argument("--first-n", type=int, default=None)
    p.add_argument("--details-csv", type=Path, default=Path("match_details.csv"))
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--length-lo", type=float, default=0.5)
    p.add_argument("--length-hi", type=float, default=1000.0)
    p.add_argument("--angle-lo", type=float, default=10.0)
    p.add_argument("--angle-hi", type=float, default=170.0)
    p.add_argument("--stol", type=float, default=0.5)
    p.add_argument("--ltol", type=float, default=0.3)
    p.add_argument("--angle-tol", type=float, default=10.0)
    return p


def run_evaluation(args):
    if args.num_gens < 1 or (args.first_n is not None and args.first_n < 1):
        raise ValueError("Candidate count and subset size must be positive")
    reference_column = args.reference_cif_column or args.cif_column
    references = load_csv(args.reference_csv, reference_column, args.encoding)
    if args.generated_csv:
        generated = load_csv(args.generated_csv, args.cif_column, args.encoding)
        if args.num_gens > 1 and args.source_row_column in generated and args.generation_column not in generated:
            raise ValueError("N>1 indexed evaluation requires an explicit generation_index column")
        if args.source_row_column not in generated:
            resolve_n_pairs(len(generated), len(references) * args.num_gens,
                            None if args.first_n is None else args.first_n * args.num_gens)
    groups = prepare_generated_groups(
        references, generated_csv=args.generated_csv, generated_dir=args.generated_dir,
        pred_cif_col=args.cif_column, source_row_col=args.source_row_column,
        generation_col=args.generation_column, num_gens=args.num_gens)
    summary, details = evaluate_groups(
        references, groups, cif_column=reference_column, num_gens=args.num_gens,
        first_n=args.first_n, ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol,
        length_lo=args.length_lo, length_hi=args.length_hi, angle_lo=args.angle_lo, angle_hi=args.angle_hi)
    if args.details_csv:
        args.details_csv.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(args.details_csv, index=False)
    print(json.dumps(summary, indent=2))
    return summary, details


def main():
    run_evaluation(build_parser().parse_args())


if __name__ == "__main__":
    main()

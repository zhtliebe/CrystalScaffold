#!/usr/bin/env python3



import argparse
import math
import re
import statistics
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


CRYSTEXT_PYMATGEN_VERSION = "2025.6.14"
LTOL = 0.3
STOL = 0.5
ANGLE_TOL = 10.0


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def read_gt(gt_csv, gt_cif_col="cif"):
    df = pd.read_csv(gt_csv)
    if gt_cif_col not in df.columns:
        raise ValueError(
            f"Ground-truth CSV has no column {gt_cif_col!r}. "
            f"Available columns: {df.columns.tolist()}"
        )
    return df


def read_generated_csv(
    generated_csv,
    pred_cif_col="cif",
    source_row_col="source_row",
    generation_col="generation_index",
    max_candidates=None,
):
    df = pd.read_csv(generated_csv)
    if pred_cif_col not in df.columns:
        raise ValueError(
            f"Generated CSV has no column {pred_cif_col!r}. "
            f"Available columns: {df.columns.tolist()}"
        )

    groups = defaultdict(list)

    if source_row_col in df.columns:
        sort_cols = [source_row_col]
        if generation_col in df.columns:
            sort_cols.append(generation_col)
        df = df.sort_values(sort_cols, kind="stable")

        indexed = {}
        for _, row in df.iterrows():
            row_id = int(row[source_row_col])
            if row_id < 0 or row_id != row[source_row_col]:
                raise ValueError("source_row must be a nonnegative integer")
            cif = row[pred_cif_col]
            cif = "" if pd.isna(cif) else str(cif)
            if generation_col in df.columns:
                gen = int(row[generation_col])
                if gen < 0 or gen != row[generation_col] or (row_id, gen) in indexed:
                    raise ValueError("generation_index must be unique and zero-based per source_row")
                indexed[row_id, gen] = cif
            else:
                groups[row_id].append(cif)
        for (row_id, gen), cif in indexed.items():
            if max_candidates is not None and gen >= max_candidates:
                groups.setdefault(row_id, [])
                continue
            if len(groups[row_id]) <= gen:
                groups[row_id].extend([""] * (gen + 1 - len(groups[row_id])))
            groups[row_id][gen] = cif
        return dict(groups), True

    
    raw = ["" if pd.isna(x) else str(x) for x in df[pred_cif_col].tolist()]
    return raw, False


ROW_RE = re.compile(r"(?:^|[_-])row[_-]?(\d+)(?:[_-]|$)", re.IGNORECASE)
GEN_RE = re.compile(
    r"(?:__|[_-]gen(?:eration)?[_-]?)(\d+)(?:[_-]|$)",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(r"^(\d+)$")


def infer_folder_mapping(cif_dir, gt_df, max_candidates=None):
    root = Path(cif_dir)
    files = sorted(root.rglob("*.cif"))
    if not root.is_dir():
        raise FileNotFoundError(root)

    material_to_row = {}
    if "material_id" in gt_df.columns:
        material_to_row = {
            str(mid): i for i, mid in enumerate(gt_df["material_id"].tolist())
        }

    tmp = defaultdict(list)
    unresolved = []

    for fp in files:
        stem = fp.stem
        row_id = None
        gen_id = 0
        material_generation = False

        m = ROW_RE.search(stem)
        if not m:
            for component in fp.relative_to(root).parts[:-1]:
                m = ROW_RE.search(component)
                if m:
                    attempt = re.fullmatch(r"scaffold_(\d+)", stem)
                    if attempt:
                        gen_id = int(attempt.group(1))
                    break
        if m:
            row_id = int(m.group(1))
        elif NUMERIC_RE.fullmatch(stem):
            row_id = int(stem)
        else:
            
            if "__" in stem:
                prefix, suffix = stem.rsplit("__", 1)
                if prefix in material_to_row:
                    row_id = material_to_row[prefix]
                    if suffix.isdigit():
                        gen_id = max(0, int(suffix) - 1)
                        material_generation = True
            elif stem in material_to_row:
                row_id = material_to_row[stem]

        gm = GEN_RE.search(stem)
        if gm and not material_generation:
            gen_id = int(gm.group(1))

        if row_id is None:
            unresolved.append(str(fp))
            continue

        text = fp.read_text(encoding="utf-8", errors="replace")
        tmp[row_id].append((gen_id, fp.name, text))

    if unresolved:
        preview = "\n".join(unresolved[:10])
        raise ValueError(
            "Could not align some CIF filenames to test.csv rows.\n"
            "Use rowN.cif / rowN_genM.cif / N.cif / material_id.cif / "
            "material_id__M.cif.\nExamples:\n" + preview
        )

    groups = {}
    for row_id, items in tmp.items():
        items.sort(key=lambda x: (x[0], x[1]))
        if max_candidates is not None:
            items = [x for x in items if x[0] < max_candidates]
        if not items:
            groups[row_id] = []
            continue
        groups[row_id] = [""] * (max(x[0] for x in items) + 1)
        seen = set()
        for gen, _, cif in items:
            if gen in seen:
                raise ValueError(f"Duplicate target/candidate index: {row_id}/{gen}")
            seen.add(gen)
            groups[row_id][gen] = cif

    return groups


def prepare_generated_groups(
    gt_df,
    generated_csv=None,
    generated_dir=None,
    pred_cif_col="cif",
    source_row_col="source_row",
    generation_col="generation_index",
    num_gens=1,
):
    if generated_csv:
        data, has_source_row = read_generated_csv(
            generated_csv,
            pred_cif_col=pred_cif_col,
            source_row_col=source_row_col,
            generation_col=generation_col,
            max_candidates=num_gens,
        )

        if has_source_row:
            return data

        raw = data

        if num_gens == 1:
            
            return {i: [cif] for i, cif in enumerate(raw)}

        
        
        if len(raw) % num_gens != 0:
            raise ValueError(
                f"Generated CSV has no {source_row_col!r} column and "
                f"{len(raw)} rows is not divisible by --num_gens={num_gens}. "
                "Add source_row/generation_index for unambiguous N>1 evaluation."
            )

        n_entries = len(raw) // num_gens
        return {
            i: raw[i * num_gens:(i + 1) * num_gens]
            for i in range(n_entries)
        }

    return infer_folder_mapping(generated_dir, gt_df, max_candidates=num_gens)


def evaluate(
    gt_df,
    pred_groups,
    gt_cif_col="cif",
    num_gens=1,
    first_n=None,
):
    matcher = StructureMatcher(
        stol=STOL,
        ltol=LTOL,
        angle_tol=ANGLE_TOL,
    )

    if num_gens < 1 or (first_n is not None and first_n < 1):
        raise ValueError("num_gens and first_n must be positive")
    n_eval = len(gt_df) if first_n is None else min(first_n, len(gt_df))
    if n_eval == 0:
        raise ValueError("Reference CSV is empty")
    if any(i < 0 or i >= len(gt_df) for i in pred_groups):
        raise ValueError("Generated source_row is outside the reference ledger")
    rows = []

    for row_id in range(n_eval):
        gt_cif = gt_df.iloc[row_id][gt_cif_col]

        try:
            gt_struct = Structure.from_str(str(gt_cif), fmt="cif")
        except Exception as e:
            raise RuntimeError(
                f"Ground-truth CIF failed to parse at test.csv row {row_id}: {e}"
            ) from e

        candidates = pred_groups.get(row_id, [])[:num_gens]

        parse_ok_count = 0
        match_rms_values = []
        candidate_records = []

        for gen_idx, pred_cif in enumerate(candidates):
            parse_ok = False
            matched = False
            rms = None
            error = ""

            try:
                pred_struct = Structure.from_str(str(pred_cif), fmt="cif")
                parse_ok = True
                parse_ok_count += 1

                
                
                rms_result = matcher.get_rms_dist(gt_struct, pred_struct)

                if rms_result is not None and math.isfinite(float(rms_result[0])):
                    matched = True
                    rms = float(rms_result[0])
                    match_rms_values.append(rms)

            except Exception as e:
                error = f"{type(e).__name__}: {e}"

            candidate_records.append(
                {
                    "generation_index": gen_idx,
                    "parse_ok": parse_ok,
                    "matched": matched,
                    "rms": rms,
                    "error": error,
                }
            )

        entry_matched = len(match_rms_values) > 0
        best_rms = min(match_rms_values) if entry_matched else None

        rows.append(
            {
                "source_row": row_id,
                "material_id": (
                    gt_df.iloc[row_id]["material_id"]
                    if "material_id" in gt_df.columns else ""
                ),
                "num_candidates_available": len(pred_groups.get(row_id, [])),
                "num_candidates_used": len(candidates),
                "parse_ok_candidates": parse_ok_count,
                "matched": entry_matched,
                "matched_flag": int(entry_matched),
                "best_rms": best_rms,
                "candidate_details": repr(candidate_records),
            }
        )

    details = pd.DataFrame(rows)
    matched_mask = details["matched"].astype(bool)

    matched_count = int(matched_mask.sum())
    match_rate = matched_count / n_eval if n_eval else float("nan")

    matched_rms = details.loc[matched_mask, "best_rms"].astype(float).tolist()
    mean_rms = statistics.mean(matched_rms) if matched_rms else float("nan")

    summary = {
        "n_test_entries": n_eval,
        "num_gens": num_gens,
        "matched_count": matched_count,
        "match_rate": match_rate,
        "match_rate_percent": 100.0 * match_rate,
        "mean_best_rms_matched": mean_rms,
        "ltol": LTOL,
        "stol": STOL,
        "angle_tol": ANGLE_TOL,
        "matcher_direction": "ground_truth -> generated",
        "pymatgen_version": get_version("pymatgen"),
        "crystext_project_pymatgen_version": CRYSTEXT_PYMATGEN_VERSION,
    }

    return summary, details


def main():
    ap = argparse.ArgumentParser(
        description="CrysText paper/project-aligned Structure Match Rate evaluator."
    )
    ap.add_argument("--reference-csv", dest="gt_csv", required=True, help="CrysText MP-20 test.csv")
    ap.add_argument("--gt_cif_col", default="cif")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--generated-csv", dest="generated_csv")
    src.add_argument("--generated_dir")

    ap.add_argument("--pred_cif_col", default="cif")
    ap.add_argument("--source_row_col", default="source_row")
    ap.add_argument("--generation_col", default="generation_index")

    ap.add_argument(
        "--num_gens",
        type=int,
        default=1,
        help="CrysText N. Use 1 for Table-1 N=1, or 20 for N=20.",
    )
    ap.add_argument(
        "--first-n",
        dest="first_n",
        type=int,
        default=None,
        help=(
            "Optional subset size. Omit for the full test.csv. "
            "For a paper-level MP-20 result, omit this and evaluate all 9046 rows."
        ),
    )
    ap.add_argument(
        "--details_csv",
        default="crystext_match_details.csv",
    )
    args = ap.parse_args()

    if args.num_gens < 1:
        raise ValueError("--num_gens must be >= 1")

    gt_df = read_gt(args.gt_csv, args.gt_cif_col)

    pred_groups = prepare_generated_groups(
        gt_df=gt_df,
        generated_csv=args.generated_csv,
        generated_dir=args.generated_dir,
        pred_cif_col=args.pred_cif_col,
        source_row_col=args.source_row_col,
        generation_col=args.generation_col,
        num_gens=args.num_gens,
    )

    summary, details = evaluate(
        gt_df=gt_df,
        pred_groups=pred_groups,
        gt_cif_col=args.gt_cif_col,
        num_gens=args.num_gens,
        first_n=args.first_n,
    )

    details_path = Path(args.details_csv)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(details_path, index=False, encoding="utf-8-sig")

    print("\n=== CrysText Structure Match Evaluation ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if summary["pymatgen_version"] != CRYSTEXT_PYMATGEN_VERSION:
        print(
            "\nWARNING: CrysText project requirements.txt pins "
            f"pymatgen=={CRYSTEXT_PYMATGEN_VERSION}, but the current environment "
            f"has pymatgen=={summary['pymatgen_version']}. "
            "For the strictest paper/project reproduction, install the pinned version."
        )

    print(f"\nPer-entry details saved to: {args.details_csv}")
    print("matched_flag: 1 = matched, 0 = unmatched (including missing/failed outputs)")
    print(f"Sum of matched_flag (m): {int(details['matched_flag'].sum())}")


if __name__ == "__main__":
    main()


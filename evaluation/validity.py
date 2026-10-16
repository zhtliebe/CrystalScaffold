#!/usr/bin/env python3
# Copyright (c) 2021 Tian Xie, Xiang Fu; 2023 Rui Jiao; 2023 Luis M. Antunes (benchmark portions).
# See THIRD_PARTY_NOTICES.md.


import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import smact
from smact.screening import pauling_test
from pymatgen.core import Structure


def smact_validity_from_structure(structure, use_pauling_test=True, include_alloys=True):
    atom_types = [str(specie) for specie in structure.species]
    elem_counter = Counter(atom_types)
    elems = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    comp, elem_counts = list(zip(*elems))

    elem_counts = np.array(elem_counts)
    elem_counts = elem_counts / np.gcd.reduce(elem_counts)
    count = tuple(elem_counts.astype("int").tolist())

    elem_symbols = tuple(comp)
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]

    
    if len(set(elem_symbols)) == 1:
        return True

    
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)

    
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False

    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        cn_e, _ = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold
        )
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_ok = pauling_test(ox_states, electronegs)
                except TypeError:
                    
                    electroneg_ok = True
            else:
                electroneg_ok = True
            if electroneg_ok:
                return True

    return False


def structure_validity(structure, cutoff=0.5):
    dist_mat = structure.distance_matrix
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.0)
    )
    if dist_mat.min() < cutoff or structure.volume < 0.1:
        return False
    return True


def load_cifs_from_csv(path, cif_column="cif", first_n=None):
    df = pd.read_csv(path)
    if cif_column not in df.columns:
        raise ValueError(
            f"CSV has no column {cif_column!r}; columns={df.columns.tolist()}"
        )
    if first_n is not None:
        df = df.iloc[:first_n]
    out = []
    for i, x in enumerate(df[cif_column].tolist()):
        out.append((i, "" if pd.isna(x) else str(x)))
    return out


def load_cifs_from_dir(path, first_n=None):
    files = sorted(Path(path).rglob("*.cif"))
    if first_n is not None:
        files = files[:first_n]
    out = []
    for i, fp in enumerate(files):
        out.append((str(fp), fp.read_text(encoding="utf-8", errors="replace")))
    return out


def evaluate(samples, protocol):
    rows = []
    parseable_rows = []

    for sample_id, cif in samples:
        parse_ok = False
        struct_valid = False
        comp_valid = False
        err = ""
        try:
            structure = Structure.from_str(cif, fmt="cif")
            parse_ok = True
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        if parse_ok:
            try:
                struct_valid = bool(structure_validity(structure))
            except Exception as e:
                err += f" structural:{type(e).__name__}: {e}"
            try:
                comp_valid = bool(smact_validity_from_structure(structure))
            except Exception as e:
                err += f" composition:{type(e).__name__}: {e}"

        row = {
            "sample_id": sample_id,
            "parse_ok": parse_ok,
            "structure_valid": struct_valid,
            "composition_valid": comp_valid,
            "both_valid": bool(struct_valid and comp_valid),
            "error": err,
        }
        rows.append(row)
        if parse_ok:
            parseable_rows.append(row)

    if protocol in {"crystext_table4", "mat2seq_validity"}:
        denom_rows = rows
        denominator_definition = "all generated samples"
    elif protocol == "crystal_text_llm":
        
        
        denom_rows = parseable_rows
        denominator_definition = "parseable generated samples only"
    else:
        raise ValueError(protocol)

    denom = len(denom_rows)
    if denom == 0:
        raise RuntimeError("No samples available in validity denominator")

    n_struct = sum(r["structure_valid"] for r in denom_rows)
    n_comp = sum(r["composition_valid"] for r in denom_rows)
    n_both = sum(r["both_valid"] for r in denom_rows)

    summary = {
        "protocol": protocol,
        "input_samples": len(rows),
        "parseable_samples": len(parseable_rows),
        "validity_denominator": denom,
        "denominator_definition": denominator_definition,
        "structure_valid_count": n_struct,
        "structure_validity": n_struct / denom,
        "structure_validity_percent": 100.0 * n_struct / denom,
        "composition_valid_count": n_comp,
        "composition_validity": n_comp / denom,
        "composition_validity_percent": 100.0 * n_comp / denom,
        "both_valid_count": n_both,
        "both_validity": n_both / denom,
        "both_validity_percent": 100.0 * n_both / denom,
    }
    return summary, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--generated_csv")
    src.add_argument("--generated_dir")
    ap.add_argument("--cif_column", default="cif")
    ap.add_argument("--first_n", type=int, default=None)
    ap.add_argument("--expected-count", type=int, default=None)
    ap.add_argument(
        "--protocol",
        choices=["crystext_table4", "mat2seq_validity", "crystal_text_llm"],
        default="crystext_table4",
    )
    ap.add_argument("--details_csv", default="validity_details.csv")
    args = ap.parse_args()

    if args.first_n is not None and args.first_n < 1:
        raise ValueError("--first_n must be positive")

    if args.generated_csv:
        samples = load_cifs_from_csv(
            args.generated_csv, args.cif_column, args.first_n
        )
    else:
        samples = load_cifs_from_dir(args.generated_dir, args.first_n)

    if args.expected_count is not None and len(samples) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} ledger rows, found {len(samples)}")
    if args.expected_count is not None and args.generated_csv:
        ledger = pd.read_csv(args.generated_csv)
        if args.first_n is not None:
            ledger = ledger.iloc[:args.first_n]
        if "source_row" in ledger and sorted(ledger["source_row"].tolist()) != list(range(len(samples))):
            raise ValueError("Formal validity needs one ledger row per original target")

    summary, details = evaluate(samples, args.protocol)
    Path(args.details_csv).parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.details_csv, index=False)

    print("=== Validity summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"details_csv: {args.details_csv}")


if __name__ == "__main__":
    main()

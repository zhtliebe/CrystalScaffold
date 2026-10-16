#!/usr/bin/env python3
"""
Compute MLIP-predicted decomposition energy / energy above hull for generated CIFs.

This version integrates:
  1) CrystalTextLLM/CDVAE-style basic validity filtering before relaxation:
     - parseability with pymatgen;
     - structural validity: min interatomic distance >= cutoff and cell volume >= 0.1 A^3;
     - compositional validity: SMACT charge-neutral oxidation-state check, optionally with Pauling test.
  2) MatLLMSearch-style fixed reference hull:
     - loads a local 2023-02-07-ppd-mp.pkl.gz PatchedPhaseDiagram file;
     - generated entries are NOT added to the reference hull.
  3) MaterialsProject2020Compatibility for the generated MLIP entry.

Outputs are MLIP-predicted approximate Ed/E_hull values, not DFT E_hull.
By default, the reported MatLLMSearch-style thresholds are Ed < 0.0, Ed < 0.03, and Ed < 0.1 eV/atom.

Download the MatLLMSearch phase diagram file once, then run:
  wget -O data/2023-02-07-ppd-mp.pkl.gz https://figshare.com/ndownloader/files/48241624

Example:
  python compute_mlip_ehull_with_validity_ppd.py \
    --input "模型生成的CIF.csv" \
    --output chgnet_ehull_validity.csv \
    --calculator chgnet \
    --ppd-mp-path data/2023-02-07-ppd-mp.pkl.gz \
    --steps 100
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import math
import os
import pickle
import tempfile
import traceback
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import smact
from smact.screening import pauling_test
from tqdm import tqdm

from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet

DEFAULT_PPD_MP_URL = "https://figshare.com/ndownloader/files/48241624"


CIF_COLUMN_CANDIDATES = [
    "cif_content", "cif", "generated_cif", "cif_str", "cif_string", "content"
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV containing generated CIF strings.")
    p.add_argument("--output", required=True, help="Output CSV path.")
    p.add_argument("--cif-col", default="auto", help="CIF column name, or 'auto'. Default: auto")
    p.add_argument("--calculator", choices=["m3gnet", "chgnet", "both"], default="chgnet")
    p.add_argument("--ppd-mp-path", default="data/2023-02-07-ppd-mp.pkl.gz", help="Local MatLLMSearch/Matbench Discovery patched phase diagram file.")
    p.add_argument("--download-ppd-mp", action="store_true", help="Download 2023-02-07-ppd-mp.pkl.gz to --ppd-mp-path if the file is missing.")
    p.add_argument("--ppd-url", default=DEFAULT_PPD_MP_URL, help="Download URL for 2023-02-07-ppd-mp.pkl.gz.")
    p.add_argument("--limit", type=int, default=None, help="Only evaluate first N rows.")
    p.add_argument("--fmax", type=float, default=0.05, help="Relaxation force convergence, eV/A.")
    p.add_argument("--steps", type=int, default=300, help="Maximum relaxation steps.")
    p.add_argument(
        "--ed-thresholds",
        default="0.0,0.03,0.1",
        help=(
            "Comma-separated Ed thresholds in eV/atom. Default follows MatLLMSearch: "
            "0.0,0.03,0.1, corresponding to Ed < 0.0, Ed < 0.03, Ed < 0.1."
        ),
    )

    # Basic validity options.
    p.add_argument("--skip-basic-validity", action="store_true", help="Do not filter by structural/composition validity before relaxation.")
    p.add_argument("--min-distance-cutoff", type=float, default=0.5, help="Minimum allowed interatomic distance in Angstrom. Default: 0.5")
    p.add_argument("--no-pauling-test", action="store_true", help="Disable SMACT Pauling electronegativity test.")
    p.add_argument("--exclude-alloys", action="store_true", help="Do not automatically accept all-metal compositions as composition-valid.")
    p.add_argument("--allow-single-site", action="store_true", help="Allow one-site structures to enter MLIP relaxation. Default: skip them, following CrystalTextLLM code.")

    # Compatibility options.
    p.add_argument("--no-mp2020-compat", action="store_true", help="Do not apply MaterialsProject2020Compatibility to generated entries.")
    p.add_argument("--strict-compat", action="store_true", help="If MP2020 compatibility processing drops/fails a generated entry, mark it as error instead of falling back.")
    return p.parse_args()


def parse_ed_thresholds(threshold_text: str) -> list[float]:
    """Parse comma-separated Ed thresholds and return sorted unique values."""
    thresholds: list[float] = []
    for part in str(threshold_text).split(","):
        part = part.strip()
        if not part:
            continue
        thresholds.append(float(part))
    if not thresholds:
        raise ValueError("At least one Ed threshold is required.")
    return sorted(set(thresholds))


def threshold_to_column_suffix(threshold: float) -> str:
    """Create a filename/CSV-safe suffix, e.g. 0.03 -> 0p03, -0.01 -> m0p01."""
    text = f"{threshold:g}"
    text = text.replace("-", "m").replace(".", "p")
    return text


def threshold_to_label(threshold: float) -> str:
    """Create a stable human-readable threshold label for summary output."""
    if abs(threshold) < 1e-12:
        return "0.0"
    return f"{threshold:g}"


def infer_cif_column(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Requested CIF column '{requested}' not found. Columns: {list(df.columns)}")
        return requested

    for col in CIF_COLUMN_CANDIDATES:
        if col in df.columns:
            return col

    for col in df.columns:
        series = df[col].dropna().astype(str).head(20)
        if len(series) and series.str.contains(r"_cell_length|data_", regex=True).any():
            return col

    raise ValueError(f"Could not infer CIF column. Please pass --cif-col explicitly. Columns: {list(df.columns)}")


def reduced_counts_from_structure(structure: Structure) -> tuple[list[str], np.ndarray]:
    comp = structure.composition.element_composition
    items = sorted(comp.items(), key=lambda kv: kv[0].Z)
    symbols = [el.symbol for el, _amt in items]
    counts_float = np.array([float(amt) for _el, amt in items], dtype=float)
    counts = np.rint(counts_float).astype(int)
    if np.any(counts < 1):
        counts[counts < 1] = 1
    gcd = int(np.gcd.reduce(counts)) if len(counts) else 1
    gcd = max(gcd, 1)
    return symbols, counts // gcd


def neutral_ratios_compatible(ox_states: tuple[int, ...], stoichs: list[tuple[int, ...]], threshold: int) -> tuple[bool, Any]:
    """Support both old SMACT API and SMACT 4.x API for neutral_ratios."""
    result = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (bool, np.bool_)):
        return bool(result[0]), result[1]
    ratios = result
    return bool(ratios), ratios


def smact_validity_from_symbols(
    elem_symbols: list[str] | tuple[str, ...],
    counts: np.ndarray,
    use_pauling_test: bool = True,
    include_alloys: bool = True,
) -> tuple[bool, str]:
    elem_symbols = tuple(str(x) for x in elem_symbols)
    if len(elem_symbols) == 0:
        return False, "empty_composition"
    if len(set(elem_symbols)) == 1:
        return True, ""

    if include_alloys:
        try:
            if all(elem_s in smact.metals for elem_s in elem_symbols):
                return True, ""
        except Exception as exc:
            return False, f"alloy_check_failed: {type(exc).__name__}: {exc}"

    try:
        space = smact.element_dictionary(elem_symbols)
        smact_elems = [e[1] for e in space.items()]
        electronegs = [e.pauling_eneg for e in smact_elems]
        ox_combos = [tuple(e.oxidation_states) for e in smact_elems]
    except Exception as exc:
        return False, f"smact_element_lookup_failed: {type(exc).__name__}: {exc}"

    if any(len(ox) == 0 for ox in ox_combos):
        return False, "missing_oxidation_states"

    threshold = int(np.max(counts)) if len(counts) else 1
    stoichs = [(int(c),) for c in counts]

    neutral_ratio_errors = 0
    charge_neutral_combos = 0
    pauling_type_errors = 0
    pauling_value_errors = 0
    pauling_other_errors = 0

    for ox_states in itertools.product(*ox_combos):
        try:
            cn_exists, _ratios = neutral_ratios_compatible(
                tuple(int(x) for x in ox_states), stoichs=stoichs, threshold=threshold
            )
        except Exception:
            neutral_ratio_errors += 1
            continue

        if not cn_exists:
            continue
        charge_neutral_combos += 1

        if use_pauling_test:
            try:
                electroneg_ok = pauling_test(ox_states, electronegs, symbols=elem_symbols)
            except TypeError:
                # Original CDVAE/Gruver behavior: missing EN/signature issues should not reject only for this.
                pauling_type_errors += 1
                electroneg_ok = True
            except ValueError:
                pauling_value_errors += 1
                electroneg_ok = False
            except Exception:
                pauling_other_errors += 1
                electroneg_ok = False
        else:
            electroneg_ok = True

        if electroneg_ok:
            return True, ""

    if charge_neutral_combos == 0:
        return False, f"no_charge_neutral_oxidation_state_combo; neutral_ratio_errors={neutral_ratio_errors}"

    return False, (
        "charge_neutral_but_failed_pauling"
        f"; charge_neutral_combos={charge_neutral_combos}"
        f"; pauling_type_errors={pauling_type_errors}"
        f"; pauling_value_errors={pauling_value_errors}"
        f"; pauling_other_errors={pauling_other_errors}"
        f"; neutral_ratio_errors={neutral_ratio_errors}"
    )


def structural_validity(structure: Structure, cutoff: float = 0.5) -> tuple[bool, float, float, str]:
    try:
        volume = float(structure.volume)
        if not math.isfinite(volume) or volume < 0.1:
            return False, float("nan"), volume, "volume_lt_0.1_A3"

        dist_mat = np.array(structure.distance_matrix, dtype=float)
        if dist_mat.shape[0] <= 1:
            return True, float("nan"), volume, ""

        dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
        min_dist = float(np.nanmin(dist_mat))
        if not math.isfinite(min_dist):
            return False, min_dist, volume, "min_distance_not_finite"
        if min_dist < cutoff:
            return False, min_dist, volume, f"min_distance_lt_{cutoff}_A"
        return True, min_dist, volume, ""
    except Exception as exc:
        return False, float("nan"), float("nan"), f"structure_eval_failed: {type(exc).__name__}: {exc}"


def evaluate_basic_validity(
    structure: Structure,
    cutoff: float,
    use_pauling_test: bool,
    include_alloys: bool,
) -> dict[str, Any]:
    struct_valid, min_dist, volume, struct_error = structural_validity(structure, cutoff=cutoff)
    try:
        symbols, counts = reduced_counts_from_structure(structure)
        comp_valid, comp_error = smact_validity_from_symbols(
            symbols, counts, use_pauling_test=use_pauling_test, include_alloys=include_alloys
        )
    except Exception as exc:
        comp_valid = False
        comp_error = f"composition_eval_failed: {type(exc).__name__}: {exc}"

    valid = bool(struct_valid and comp_valid)
    reasons: list[str] = []
    if not struct_valid:
        reasons.append(struct_error or "struct_invalid")
    if not comp_valid:
        reasons.append(comp_error or "composition_invalid_by_smact")

    return {
        "volume_A3": volume,
        "min_distance_A": min_dist,
        "struct_valid": bool(struct_valid),
        "struct_error": struct_error,
        "comp_valid": bool(comp_valid),
        "comp_error": comp_error,
        "basic_valid": valid,
        "basic_invalid_reason": "valid" if valid else "; ".join(reasons),
    }


def make_m3gnet_relaxer():
    from m3gnet.models import Relaxer
    return Relaxer()


def relax_m3gnet(structure: Structure, relaxer: Any, fmax: float = 0.05, steps: int = 300) -> tuple[Structure, float]:
    result = relaxer.relax(structure, fmax=fmax, steps=steps)
    relaxed = result["final_structure"]
    energy = float(result["trajectory"].energies[-1])
    return relaxed, energy


def make_chgnet_relaxer():
    try:
        from chgnet.model.dynamics import StructOptimizer
    except Exception:
        from chgnet.model import StructOptimizer
    return StructOptimizer()


def relax_chgnet(structure: Structure, relaxer: Any, fmax: float = 0.05, steps: int = 300) -> tuple[Structure, float]:
    result = relaxer.relax(structure, fmax=fmax, steps=steps, verbose=False)
    relaxed = result["final_structure"]
    energy = float(result["trajectory"].energies[-1])
    return relaxed, energy


class FixedPPDHull:
    """Load the fixed 2023-02-07-ppd-mp PatchedPhaseDiagram from disk."""

    def __init__(self, ppd_path: str | os.PathLike[str], download_if_missing: bool = False, url: str = DEFAULT_PPD_MP_URL):
        self.ppd_path = Path(ppd_path)
        if not self.ppd_path.exists():
            if download_if_missing:
                self._download(url, self.ppd_path)
            else:
                raise RuntimeError(
                    f"Missing patched phase diagram file: {self.ppd_path}\n"
                    "Download it once, for example:\n"
                    f"  mkdir -p {self.ppd_path.parent}\n"
                    f"  wget -O {self.ppd_path} {url}\n"
                    "or rerun this script with --download-ppd-mp."
                )

        self.phase_diagram = self._load_phase_diagram(self.ppd_path)

    @staticmethod
    def _download(url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".part")
        print(f"Downloading fixed MP patched phase diagram from {url}")
        print(f"Saving to {path}")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rb") as fh:
                return pickle.load(fh)
        with open(path, "rb") as fh:
            return pickle.load(fh)

    @classmethod
    def _extract_phase_diagram(cls, obj: Any) -> Any:
        # Expected case: the pickle is directly a PatchedPhaseDiagram-like object.
        if hasattr(obj, "get_e_above_hull"):
            return obj

        # Defensive support for dict wrappers used by some data releases.
        if isinstance(obj, dict):
            for key in (
                "patched_phase_diagram",
                "phase_diagram",
                "ppd_mp",
                "ppd",
                "mp_patched_phase_diagram",
            ):
                if key in obj and hasattr(obj[key], "get_e_above_hull"):
                    return obj[key]
            for value in obj.values():
                if hasattr(value, "get_e_above_hull"):
                    return value

        # Defensive support if the file contains entries instead of a pre-built PPD.
        if isinstance(obj, (list, tuple)):
            print("Loaded a list/tuple rather than a PatchedPhaseDiagram; constructing PatchedPhaseDiagram now.")
            return PatchedPhaseDiagram(list(obj), verbose=True)

        raise TypeError(
            "Could not find a PatchedPhaseDiagram-like object in the supplied file. "
            f"Loaded object type: {type(obj).__name__}"
        )

    @classmethod
    def _load_phase_diagram(cls, path: Path) -> Any:
        print(f"Loading fixed MP patched phase diagram: {path}", flush=True)
        obj = cls._load_pickle(path)
        ppd = cls._extract_phase_diagram(obj)
        print(f"Loaded phase diagram object: {type(ppd).__name__}", flush=True)
        return ppd

    def get_e_above_hull(self, entry: ComputedStructureEntry) -> float:
        try:
            return float(self.phase_diagram.get_e_above_hull(entry, allow_negative=True))
        except TypeError:
            return float(self.phase_diagram.get_e_above_hull(entry))


def make_generated_entry(
    structure: Structure,
    total_energy: float,
    entry_id: str,
    apply_mp2020_compat: bool = True,
    strict_compat: bool = False,
) -> tuple[ComputedStructureEntry, str]:
    """
    Create a ComputedStructureEntry for the generated relaxed structure.

    When apply_mp2020_compat=True, emulate CrystalTextLLM's generate_CSE approach:
      - write MPRelaxSet files to determine GGA/GGA+U metadata and a clean POSCAR structure;
      - create ComputedStructureEntry;
      - apply MaterialsProject2020Compatibility(check_potcar=False).
    """
    if not apply_mp2020_compat:
        entry = ComputedStructureEntry(
            structure=structure,
            energy=float(total_energy),
            correction=0.0,
            parameters={"hubbards": {}, "is_hubbard": False, "run_type": "GGA"},
            entry_id=entry_id,
        )
        return entry, "not_applied"

    try:
        vasp_set = MPRelaxSet(structure)
        with tempfile.TemporaryDirectory() as tmpdirname:
            vasp_set.write_input(tmpdirname, potcar_spec=True)
            poscar = Poscar.from_file(os.path.join(tmpdirname, "POSCAR"))
            incar = Incar.from_file(os.path.join(tmpdirname, "INCAR"))
            clean_structure = Structure.from_file(os.path.join(tmpdirname, "POSCAR"))

        hubbards: dict[str, float] = {}
        if "LDAUU" in incar:
            hubbards = dict(zip(poscar.site_symbols, incar["LDAUU"]))

        is_hubbard = bool(incar.get("LDAU", False) and sum(float(v) for v in hubbards.values()) > 0)
        parameters = {
            "hubbards": hubbards,
            "is_hubbard": is_hubbard,
            "run_type": "GGA+U" if is_hubbard else "GGA",
        }

        cse = ComputedStructureEntry(
            structure=clean_structure,
            energy=float(total_energy),
            correction=0.0,
            parameters=parameters,
            entry_id=entry_id,
        )

        compat = MaterialsProject2020Compatibility(check_potcar=False)
        processed = compat.process_entries([cse], clean=True)
        if processed:
            return processed[0], "applied"

        if strict_compat:
            raise RuntimeError("MaterialsProject2020Compatibility returned no processed entry")
        return cse, "fallback_unprocessed_empty_result"

    except Exception as exc:
        if strict_compat:
            raise
        entry = ComputedStructureEntry(
            structure=structure,
            energy=float(total_energy),
            correction=0.0,
            parameters={"hubbards": {}, "is_hubbard": False, "run_type": "GGA"},
            entry_id=entry_id,
        )
        return entry, f"fallback_unprocessed_after_error: {type(exc).__name__}: {exc}"


def compute_ehull_for_method(
    structure: Structure,
    name: str,
    method: str,
    relaxer: Any,
    fixed_hull: FixedPPDHull,
    fmax: float,
    steps: int,
    apply_mp2020_compat: bool,
    strict_compat: bool,
) -> dict[str, Any]:
    if method == "m3gnet":
        relaxed, total_energy = relax_m3gnet(structure, relaxer, fmax=fmax, steps=steps)
    elif method == "chgnet":
        relaxed, total_energy = relax_chgnet(structure, relaxer, fmax=fmax, steps=steps)
    else:
        raise ValueError(method)

    gen_entry, compat_status = make_generated_entry(
        structure=relaxed,
        total_energy=total_energy,
        entry_id=f"generated_{method}_{name}",
        apply_mp2020_compat=apply_mp2020_compat,
        strict_compat=strict_compat,
    )

    # Important: generated entry is NOT included in the fixed reference hull.
    ehull = fixed_hull.get_e_above_hull(gen_entry)

    return {
        f"{method}_relaxed_formula": relaxed.composition.reduced_formula,
        f"{method}_num_sites": len(relaxed),
        f"{method}_energy_total_eV": float(total_energy),
        f"{method}_energy_per_atom_eV": float(total_energy) / len(relaxed),
        f"{method}_mp2020_compat_status": compat_status,
        f"{method}_ehull_eV_per_atom": float(ehull),
        f"{method}_ed_eV_per_atom": float(ehull),
    }


def print_rate(label: str, numerator: int, denominator: int) -> None:
    pct = 100.0 * numerator / denominator if denominator else 0.0
    print(f"{label}: {numerator}/{denominator} = {pct:.2f}%")


def main() -> None:
    args = parse_args()
    ed_thresholds = parse_ed_thresholds(args.ed_thresholds)
    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    cif_col = infer_cif_column(df, args.cif_col)

    calculators = [args.calculator] if args.calculator != "both" else ["m3gnet", "chgnet"]
    relaxers: dict[str, Any] = {}
    for calc in calculators:
        print(f"Loading {calc} relaxer...", flush=True)
        relaxers[calc] = make_m3gnet_relaxer() if calc == "m3gnet" else make_chgnet_relaxer()

    fixed_hull = FixedPPDHull(args.ppd_mp_path, download_if_missing=args.download_ppd_mp, url=args.ppd_url)

    records: list[dict[str, Any]] = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        cif_text = row[cif_col]
        base: dict[str, Any] = {
            "row_index": int(idx),
            "compound_name": row.get("compound_name", ""),
            "parse_ok": False,
            "input_formula": "",
            "input_num_sites": np.nan,
            "volume_A3": np.nan,
            "min_distance_A": np.nan,
            "struct_valid": False,
            "struct_error": "",
            "comp_valid": False,
            "comp_error": "",
            "basic_valid": False,
            "basic_invalid_reason": "",
            "skipped_before_relaxation": False,
            "skip_reason": "",
        }

        if pd.isna(cif_text) or not str(cif_text).strip():
            base.update({
                "parse_ok": False,
                "basic_invalid_reason": "parse_failed: empty_cif",
                "skipped_before_relaxation": True,
                "skip_reason": "parse_failed: empty_cif",
            })
            records.append(base)
            continue

        try:
            structure = Structure.from_str(str(cif_text), fmt="cif")
            base.update({
                "parse_ok": True,
                "input_formula": structure.composition.reduced_formula,
                "input_num_sites": int(len(structure)),
            })
        except Exception as exc:
            msg = f"parse_failed: {type(exc).__name__}: {exc}"
            base.update({
                "parse_ok": False,
                "basic_invalid_reason": msg,
                "skipped_before_relaxation": True,
                "skip_reason": msg,
            })
            records.append(base)
            continue

        validity = evaluate_basic_validity(
            structure,
            cutoff=args.min_distance_cutoff,
            use_pauling_test=not args.no_pauling_test,
            include_alloys=not args.exclude_alloys,
        )
        base.update(validity)

        if len(structure) <= 1 and not args.allow_single_site:
            base["skipped_before_relaxation"] = True
            base["skip_reason"] = "single_site_structure"
            records.append(base)
            continue

        if not args.skip_basic_validity and not base["basic_valid"]:
            base["skipped_before_relaxation"] = True
            base["skip_reason"] = f"basic_validity_failed: {base['basic_invalid_reason']}"
            records.append(base)
            continue

        for calc in calculators:
            try:
                base.update(compute_ehull_for_method(
                    structure=structure,
                    name=str(row.get("compound_name", idx)),
                    method=calc,
                    relaxer=relaxers[calc],
                    fixed_hull=fixed_hull,
                    fmax=args.fmax,
                    steps=args.steps,
                    apply_mp2020_compat=not args.no_mp2020_compat,
                    strict_compat=args.strict_compat,
                ))
            except Exception as exc:
                base[f"{calc}_error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()

        records.append(base)

    out = pd.DataFrame(records)

    # MatLLMSearch-style Ed thresholds use strict inequalities: Ed < threshold.
    # Default thresholds are Ed < 0.0, Ed < 0.03, and Ed < 0.1 eV/atom.
    for calc in calculators:
        col = f"{calc}_ed_eV_per_atom"
        if col not in out.columns:
            col = f"{calc}_ehull_eV_per_atom"
        if col in out.columns:
            for threshold in ed_thresholds:
                suffix = threshold_to_column_suffix(threshold)
                out[f"{calc}_ed_lt_{suffix}"] = out[col] < threshold

    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved detailed results to: {args.output}")

    n_all = len(out)
    n_parse = int(out["parse_ok"].sum()) if "parse_ok" in out.columns else 0
    n_struct = int(out["struct_valid"].sum()) if "struct_valid" in out.columns else 0
    n_comp = int(out["comp_valid"].sum()) if "comp_valid" in out.columns else 0
    n_basic = int(out["basic_valid"].sum()) if "basic_valid" in out.columns else 0

    print("\nSummary:")
    print(f"Input rows: {n_all}")
    print(f"CIF column: {cif_col}")
    print_rate("Parsed rows", n_parse, n_all)
    print_rate("struct_valid", n_struct, n_all)
    print_rate("comp_valid", n_comp, n_all)
    print_rate("basic_valid", n_basic, n_all)
    print(f"Skipped before relaxation: {int(out['skipped_before_relaxation'].sum())}")

    for calc in calculators:
        col = f"{calc}_ed_eV_per_atom"
        if col not in out.columns:
            col = f"{calc}_ehull_eV_per_atom"
        if col in out.columns:
            evaluated = out[col].dropna()
            print(f"\n{calc.upper()} evaluated: {len(evaluated)}")
            if len(evaluated):
                print(f"Mean Ed: {evaluated.mean():.6f} eV/atom")
                print(f"Median Ed: {evaluated.median():.6f} eV/atom")
                for threshold in ed_thresholds:
                    count = int((evaluated < threshold).sum())
                    label = threshold_to_label(threshold)
                    print_rate(f"{calc.upper()} Ed < {label} eV/atom among evaluated", count, len(evaluated))
                    print_rate(f"Overall {calc.upper()} Ed < {label} eV/atom over all rows", count, n_all)

    if "skip_reason" in out.columns:
        print("\nTop skip reasons:")
        print(out["skip_reason"].replace("", np.nan).value_counts(dropna=True).head(20).to_string())


if __name__ == "__main__":
    main()

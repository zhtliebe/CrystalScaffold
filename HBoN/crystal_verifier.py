from __future__ import annotations


from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess_v4.cif_utils import (  
    CIFRecord,
    parse_cif_text,
    canonicalize_cif_text,
    record_to_canonical_cif,
    parse_scaffold_text,
    species_schema_from_record,
    species_schema_from_formula,
    n_listed_sites_from_formula,
    crystal_system_from_sg,
    length_pattern,
    angle_pattern,
    ratio_bin,
    c_over_a_bin_family,
    angle_bin,
    vol_per_atom_bin,
    metric_family,
    position_family,
    stack_family,
    formula_sum_expected,
)


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return default


COVALENT_RADIUS: Dict[str, float] = {
    "H":0.31,"He":0.28,"Li":1.28,"Be":0.96,"B":0.84,"C":0.76,"N":0.71,"O":0.66,"F":0.57,"Ne":0.58,
    "Na":1.66,"Mg":1.41,"Al":1.21,"Si":1.11,"P":1.07,"S":1.05,"Cl":1.02,"Ar":1.06,"K":2.03,"Ca":1.76,
    "Sc":1.70,"Ti":1.60,"V":1.53,"Cr":1.39,"Mn":1.39,"Fe":1.32,"Co":1.26,"Ni":1.24,"Cu":1.32,"Zn":1.22,
    "Ga":1.22,"Ge":1.20,"As":1.19,"Se":1.20,"Br":1.20,"Kr":1.16,"Rb":2.20,"Sr":1.95,"Y":1.90,"Zr":1.75,
    "Nb":1.64,"Mo":1.54,"Tc":1.47,"Ru":1.46,"Rh":1.42,"Pd":1.39,"Ag":1.45,"Cd":1.44,"In":1.42,"Sn":1.39,
    "Sb":1.39,"Te":1.38,"I":1.39,"Xe":1.40,"Cs":2.44,"Ba":2.15,"La":2.07,"Ce":2.04,"Pr":2.03,"Nd":2.01,
    "Pm":1.99,"Sm":1.98,"Eu":1.98,"Gd":1.96,"Tb":1.94,"Dy":1.92,"Ho":1.92,"Er":1.89,"Tm":1.90,"Yb":1.87,
    "Lu":1.87,"Hf":1.75,"Ta":1.70,"W":1.62,"Re":1.51,"Os":1.44,"Ir":1.41,"Pt":1.36,"Au":1.36,"Hg":1.32,
    "Tl":1.45,"Pb":1.46,"Bi":1.48,"Po":1.40,"At":1.50,"Rn":1.50,"Fr":2.60,"Ra":2.21,"Ac":2.15,"Th":2.06,
    "Pa":2.00,"U":1.96,"Np":1.90,"Pu":1.87,
}



def expected_site_schema(formula: str, z: int) -> str:
    return species_schema_from_formula(str(formula), int(z))


def expected_n_sites(formula: str, z: int) -> int:
    return n_listed_sites_from_formula(str(formula), int(z))


def scaffold_fields(scaffold_text: str) -> Dict[str, str]:
    return dict(parse_scaffold_text(str(scaffold_text)))


def rec_field_dict(rec: CIFRecord) -> OrderedDict[str, str]:
    lp = length_pattern(rec)
    ap = angle_pattern(rec)
    bbin = ratio_bin(rec.cell_b / max(rec.cell_a, 1e-8)) if lp == "ALL_DIFF" else ""
    if ap == "ALPHA_EQ_BETA_NE_GAMMA":
        eqbin = angle_bin(rec.alpha)
    elif ap == "BETA_EQ_GAMMA_NE_ALPHA":
        eqbin = angle_bin(rec.beta)
    elif ap == "ALPHA_EQ_GAMMA_NE_BETA":
        eqbin = angle_bin(rec.alpha)
    else:
        eqbin = ""
    return OrderedDict([
        ("formula", rec.formula_structural),
        ("output_Z_raw", str(int(rec.z))),
        ("n_listed_sites", str(len(rec.atom_sites))),
        ("site_schema", species_schema_from_record(rec)),
        ("metric_family", metric_family(rec)),
        ("length_pattern", lp),
        ("angle_pattern", ap),
        ("eq_angle_bin", eqbin),
        ("gamma_bin", angle_bin(rec.gamma)),
        ("b_over_a_bin", bbin),
        ("c_over_a_bin_family", c_over_a_bin_family(rec) if lp != "A_EQ_B_EQ_C" else ""),
        ("vol_per_atom_bin", vol_per_atom_bin(rec)),
        ("position_family", position_family(rec)),
        ("stack_family", stack_family(rec)),
    ])


def cell_volume(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> float:
    ar, br, gr = map(math.radians, [alpha, beta, gamma])
    term = 1 + 2 * math.cos(ar) * math.cos(br) * math.cos(gr) - math.cos(ar) ** 2 - math.cos(br) ** 2 - math.cos(gr) ** 2
    return float(a * b * c * math.sqrt(max(term, 0.0)))


def lattice_matrix(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:

    ar, br, gr = map(math.radians, [alpha, beta, gamma])
    va = np.array([a, 0.0, 0.0], dtype=float)
    vb = np.array([b * math.cos(gr), b * math.sin(gr), 0.0], dtype=float)
    cx = c * math.cos(br)
    sy = math.sin(gr)
    if abs(sy) < 1e-8:
        cy = 0.0
    else:
        cy = c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / sy
    cz2 = max(c * c - cx * cx - cy * cy, 0.0)
    vc = np.array([cx, cy, math.sqrt(cz2)], dtype=float)
    return np.vstack([va, vb, vc])


def record_lattice_matrix(rec: CIFRecord) -> np.ndarray:
    return lattice_matrix(rec.cell_a, rec.cell_b, rec.cell_c, rec.alpha, rec.beta, rec.gamma)


def periodic_diff_frac(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    d = np.asarray(f1, dtype=float) - np.asarray(f2, dtype=float)
    return d - np.round(d)


def distance_statistics(rec: CIFRecord) -> Dict[str, Any]:
    n = len(rec.atom_sites)
    if n < 2:
        return {"min_distance": float("inf"), "collision_count_0p8": 0, "collision_count_1p0": 0, "radius_collision_count": 0, "pairs": []}
    lat = record_lattice_matrix(rec)
    min_d = float("inf")
    c08 = c10 = rc = 0
    pairs: List[Tuple[int, int, float, float]] = []
    coords = [np.array([s.fract_x, s.fract_y, s.fract_z], dtype=float) for s in rec.atom_sites]
    for i in range(n):
        for j in range(i + 1, n):
            dfrac = periodic_diff_frac(coords[i], coords[j])
            dcart = float(np.linalg.norm(dfrac @ lat))
            min_d = min(min_d, dcart)
            if dcart < 0.8:
                c08 += 1
            if dcart < 1.0:
                c10 += 1
            ri = COVALENT_RADIUS.get(rec.atom_sites[i].element, 0.75)
            rj = COVALENT_RADIUS.get(rec.atom_sites[j].element, 0.75)
            
            thresh = max(0.65, min(1.25, 0.52 * (ri + rj)))
            if dcart < thresh:
                rc += 1
            pairs.append((i, j, dcart, thresh))
    return {
        "min_distance": min_d,
        "collision_count_0p8": c08,
        "collision_count_1p0": c10,
        "radius_collision_count": rc,
        "pairs": sorted(pairs, key=lambda x: x[2])[:10],
    }


def crystal_system_geometry_penalty(rec: CIFRecord, sg: int) -> Tuple[float, List[str]]:


    cs = crystal_system_from_sg(int(sg))
    errs: List[str] = []
    p = 0.0
    a, b, c, al, be, ga = rec.cell_a, rec.cell_b, rec.cell_c, rec.alpha, rec.beta, rec.gamma
    eq_ab = abs(a - b) / max((a + b) / 2, 1e-8)
    eq_ac = abs(a - c) / max((a + c) / 2, 1e-8)
    def angle_dev(x: float, tgt: float) -> float:
        return abs(float(x) - tgt)
    if cs == "cubic":
        if max(eq_ab, eq_ac) > 0.03:
            p += 4; errs.append("cubic_lengths_not_equal")
        if max(angle_dev(al,90), angle_dev(be,90), angle_dev(ga,90)) > 2:
            p += 4; errs.append("cubic_angles_not_90")
    elif cs == "tetragonal":
        if eq_ab > 0.03:
            p += 3; errs.append("tetragonal_a_b_not_equal")
        if max(angle_dev(al,90), angle_dev(be,90), angle_dev(ga,90)) > 2:
            p += 3; errs.append("tetragonal_angles_not_90")
    elif cs == "orthorhombic":
        if max(angle_dev(al,90), angle_dev(be,90), angle_dev(ga,90)) > 2:
            p += 3; errs.append("orthorhombic_angles_not_90")
    elif cs == "hexagonal":
        if eq_ab > 0.04:
            p += 3; errs.append("hexagonal_a_b_not_equal")
        if max(angle_dev(al,90), angle_dev(be,90), angle_dev(ga,120)) > 3:
            p += 3; errs.append("hexagonal_angles_not_90_90_120")
    elif cs == "trigonal":
        
        hex_like = eq_ab <= 0.06 and max(angle_dev(al,90), angle_dev(be,90), angle_dev(ga,120)) <= 5
        rhom_like = max(eq_ab, eq_ac) <= 0.06 and max(abs(al-be), abs(al-ga), abs(be-ga)) <= 5
        if not (hex_like or rhom_like):
            p += 2; errs.append("trigonal_not_hex_or_rhombo_like")
    elif cs == "monoclinic":
        
        pass
    return p, errs


def scaffold_required_errors(fields: Dict[str, str], formula: str, sg: int, expect_full: bool = True) -> List[str]:
    errors: List[str] = []
    if fields.get("formula", "") != str(formula):
        errors.append("formula_mismatch")
    if str(fields.get("condition_sg_number", "")) != str(int(sg)):
        errors.append("condition_sg_mismatch")
    expected_cs = crystal_system_from_sg(int(sg))
    if fields.get("condition_crystal_system", "") not in {"", expected_cs}:
        errors.append("crystal_system_mismatch")
    if fields.get("target_cell_mode", "") not in {"", "primitive"}:
        errors.append("bad_target_cell_mode")
    required = [
        "formula", "reduced_ratio", "condition_sg_number", "condition_crystal_system",
        "target_cell_mode", "output_Z_raw", "metric_family", "length_pattern", "angle_pattern",
        "vol_per_atom_bin", "position_family", "stack_family",
    ]
    if expect_full:
        required += ["site_schema", "n_listed_sites", "formula_sum_expected", "geometry_signature_v4", "prototype_bucket_v4"]
    for key in required:
        if not fields.get(key, ""):
            errors.append(f"missing_{key}")
    return errors


def unquote_formula_sum(value: str) -> str:
    value = str(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def scaffold_score(scaffold_text: str, formula: str, sg: int) -> Dict[str, Any]:
    fields = scaffold_fields(scaffold_text)
    errors = scaffold_required_errors(fields, formula, int(sg), expect_full=True)
    score = 100.0
    score -= 20.0 * sum(e.startswith("missing_") for e in errors)
    score -= 25.0 * sum(e in {"formula_mismatch", "condition_sg_mismatch"} for e in errors)
    score -= 8.0 * sum(e not in {"formula_mismatch", "condition_sg_mismatch"} and not e.startswith("missing_") for e in errors)

    
    z = safe_int(fields.get("output_Z_raw", 0), 0)
    if z <= 0 or z > 16:
        score -= 15; errors.append("bad_output_Z_raw_range")
    if z > 0:
        exp_schema = expected_site_schema(formula, z)
        exp_n = expected_n_sites(formula, z)
        exp_sum = formula_sum_expected(formula, z)
        if fields.get("site_schema", "") != exp_schema:
            score -= 18; errors.append("site_schema_closure_mismatch")
        if safe_int(fields.get("n_listed_sites", 0), -1) != exp_n:
            score -= 18; errors.append("n_listed_sites_closure_mismatch")
        if unquote_formula_sum(fields.get("formula_sum_expected", "")) != exp_sum:
            score -= 8; errors.append("formula_sum_closure_mismatch")

    
    cs = crystal_system_from_sg(int(sg))
    lp = fields.get("length_pattern", "")
    ap = fields.get("angle_pattern", "")
    if cs == "cubic":
        if lp != "A_EQ_B_EQ_C": score -= 10; errors.append("cubic_bad_length_pattern")
        if ap != "ALPHA_EQ_BETA_EQ_GAMMA": score -= 10; errors.append("cubic_bad_angle_pattern")
    elif cs in {"tetragonal", "hexagonal"}:
        if lp not in {"A_EQ_B_NE_C", "A_EQ_B_EQ_C"}: score -= 8; errors.append(f"{cs}_bad_length_pattern")
    if cs == "hexagonal":
        gb = fields.get("gamma_bin", "")
        if gb and gb not in {"105_120", "120_135"}: score -= 8; errors.append("hexagonal_bad_gamma_bin")
    if fields.get("position_family", "") not in {"GENERAL_DOMINANT", "MIXED_SPECIAL", "HIGH_SPECIAL"}:
        score -= 5; errors.append("unknown_position_family")
    if fields.get("stack_family", "") == "":
        score -= 5; errors.append("missing_stack_family")

    return {"score": score, "valid": len(errors) == 0, "errors": list(dict.fromkeys(errors)), "fields": fields}


def cif_scaffold_consistency_errors(rec: CIFRecord, scaffold_full_text: str, formula: str) -> List[str]:
    fields = scaffold_fields(scaffold_full_text)
    rf = rec_field_dict(rec)
    errors: List[str] = []
    if str(rec.formula_structural).strip() != str(formula).strip(): errors.append("formula_structural_mismatch")
    if str(int(rec.z)) != str(fields.get("output_Z_raw", int(rec.z))): errors.append("output_Z_raw_mismatch")
    if str(len(rec.atom_sites)) != str(fields.get("n_listed_sites", len(rec.atom_sites))): errors.append("n_listed_sites_mismatch")
    if species_schema_from_record(rec) != fields.get("site_schema", species_schema_from_record(rec)): errors.append("site_schema_mismatch")
    for key in ["metric_family", "length_pattern", "angle_pattern", "eq_angle_bin", "gamma_bin", "b_over_a_bin", "c_over_a_bin_family", "vol_per_atom_bin", "position_family", "stack_family"]:
        if key in fields and str(fields.get(key, "")) != str(rf.get(key, "")):
            if not (fields.get(key, "") == "" and rf.get(key, "") == ""):
                errors.append(f"{key}_mismatch")
    return errors


def cif_score(cif_text: str, scaffold_full_text: str, formula: str, sg: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"score": -1e9, "valid": False, "errors": [], "components": {}}
    try:
        can = canonicalize_cif_text(str(cif_text))
        rec = parse_cif_text(can)
    except Exception as e:
        out["errors"] = [f"parse_failed:{e}"]
        return out

    errors: List[str] = []
    score = 100.0
    components: Dict[str, Any] = {}

    if not rec.atom_sites:
        errors.append("empty_atom_sites"); score -= 80
    if min(rec.cell_a, rec.cell_b, rec.cell_c) <= 0:
        errors.append("zero_lattice"); score -= 80
    if str(rec.formula_structural).strip() != str(formula).strip():
        errors.append("formula_structural_mismatch"); score -= 40

    
    cons_errors = cif_scaffold_consistency_errors(rec, scaffold_full_text, formula)
    hard_cons = {"formula_structural_mismatch", "output_Z_raw_mismatch", "n_listed_sites_mismatch", "site_schema_mismatch"}
    score -= 14.0 * sum(e in hard_cons for e in cons_errors)
    score -= 4.0 * sum(e not in hard_cons for e in cons_errors)
    errors.extend(cons_errors)

    vpa = rec.volume / max(len(rec.atom_sites), 1)
    components["volume_per_atom"] = vpa
    if not (3.0 <= vpa <= 80.0):
        errors.append("implausible_volume_per_atom"); score -= 20
    elif not (5.0 <= vpa <= 55.0):
        score -= 5

    dstat = distance_statistics(rec)
    components.update({k: v for k, v in dstat.items() if k != "pairs"})
    min_d = dstat["min_distance"]
    if min_d < 0.60:
        score -= 70; errors.append("severe_atom_collision_lt_0p6")
    elif min_d < 0.80:
        score -= 45; errors.append("atom_collision_lt_0p8")
    elif min_d < 1.00:
        score -= 18; errors.append("short_contact_lt_1p0")
    score -= 18.0 * dstat["collision_count_0p8"]
    score -= 6.0 * max(0, dstat["collision_count_1p0"] - dstat["collision_count_0p8"])
    score -= 8.0 * dstat["radius_collision_count"]

    geo_penalty, geo_errors = crystal_system_geometry_penalty(rec, int(sg))
    score -= 3.0 * geo_penalty
    errors.extend(geo_errors)

    
    coords = [(round(s.fract_x % 1, 4), round(s.fract_y % 1, 4), round(s.fract_z % 1, 4)) for s in rec.atom_sites]
    dup = len(coords) - len(set(coords))
    if dup > 0:
        score -= 20.0 * dup; errors.append("duplicate_fractional_sites")
    if len(rec.atom_sites) > 0:
        special_like = sum(1 for xyz in coords for x in xyz if any(abs(x - t) < 1e-4 for t in [0, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 5/6, 1]))
        frac_special = special_like / (3 * len(rec.atom_sites))
        components["special_coord_fraction"] = frac_special
        
        sf = scaffold_fields(scaffold_full_text)
        if frac_special > 0.95 and sf.get("position_family") != "HIGH_SPECIAL":
            score -= 15; errors.append("special_position_collapse")

    out.update({
        "score": float(score),
        "valid": (len([e for e in errors if e in {"parse_failed", "empty_atom_sites", "zero_lattice", "formula_structural_mismatch", "severe_atom_collision_lt_0p6", "atom_collision_lt_0p8"}]) == 0),
        "errors": list(dict.fromkeys(errors)),
        "components": components,
        "canonical_cif": record_to_canonical_cif(rec),
        "fields": rec_field_dict(rec),
    })
    return out



from __future__ import annotations

import argparse
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from training_utils import add_common_train_args, config_from_args, run_training

STAGE_DEFAULTS = {
    "resume_mode": "base",
    "max_seq_length": 512,
    "num_train_epochs": 1.0,
    "learning_rate": 1.0e-4,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 1,
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage I parser-oriented warm-up for V4 Core->Full generation.")
    add_common_train_args(
        parser,
        stage_name="parse",
        default_train_file="data/parse_curriculum_train_v4.csv",
    )
    parser.set_defaults(
        logging_steps=20,
        save_steps=400,
        eval_steps=200,
        early_stopping_patience=0,
        max_steps=800,
        max_train_samples=12000,
        max_eval_samples=2000,
        dataloader_num_workers=4,
        group_by_length=True,
        parse_task_caps='{"dual_symmetry_summary":2500,"formula_z_closure":3000,"cell_archetype":2500,"lattice_bin_correction":2500,"coord_profile_parse":2500,"scaffold_core_parse":2500,"failure_bank_diagnose":3000}',
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    cfg = config_from_args(args, stage_name="parse", stage_defaults=STAGE_DEFAULTS)
    run_training(cfg)

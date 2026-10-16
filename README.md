# Crystal-Scaffold

Official implementation of "Crystal-Scaffold: Deterministic Scaffold Compilation for Text-Conditioned CIF Generation".

The repository provides code for data preprocessing, three-stage instruction fine-tuning, crystal structure generation, and Best-of-N inference.

## Environment Setup

**Requirements:**
- Linux
- Python 3.11
- NVIDIA GPU with CUDA support
- Llama-3.1-8B (`meta-llama/Llama-3.1-8B`)

Training was conducted on an NVIDIA RTX 4090 GPU using BF16 precision.

### Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
```

M3GNet-based energy evaluation requires a separate environment.

## Dataset

The experiments use the MP-20 dataset with the following splits:

| Dataset | Samples |
|---|---:|
| Training | 27,136 |
| Validation | 9,047 |
| Test | 9,046 |

Organize the datasets as follows:

```text
data/
└── mp20/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Each CSV file must contain four columns:

| Column | Description |
|---|---|
| `material_id` | Material identifier |
| `pretty_formula` | Chemical formula |
| `spacegroup.number` | Space-group number |
| `cif` | CIF structure text |

Preserve the original dataset splits and row order.

## Project Structure

```text
Crystal-Scaffold/
├── preprocess/    # Data preprocessing
├── train/         # Three-stage fine-tuning
├── inference/         # Crystal structure generation
├── HBoN/             # Best-of-N inference
├── requirements.txt
```

## Models and Resources

The base model is `meta-llama/Llama-3.1-8B`. Model weights, LoRA adapters, datasets, and external reference data must be obtained separately.

All commands should be executed from the project root directory. The `--model-name` argument accepts either a Hugging Face model identifier or a local model directory.

"""Convert a trained LabelSemanticsNER .pth checkpoint into a self-contained model directory.

The output directory contains:
- All tokenizer/config files copied from the base HuggingFace model directory
- ``pytorch_model.bin``: the full LabelSemanticsNER state_dict (with both
  ``token_encoder.*`` and ``label_encoder.*`` weights)

It can then be used with ``tests/batch_test.py`` like:

    python tests/batch_test.py \
        --model-path <output_dir> \
        --checkpoint-path <output_dir>/pytorch_model.bin

(``tests/batch_test.py`` currently only accepts ``.pth`` for ``--checkpoint-path``;
either rename ``pytorch_model.bin`` to ``model.pth`` or relax the suffix check.)
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEPENDENCY_ERROR = None

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    DEPENDENCY_ERROR = exc


TOKENIZER_AND_CONFIG_FILES = (
    "config.json",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a trained .pth checkpoint into a self-contained HF-style model directory.",
    )
    parser.add_argument(
        "--base-model-path",
        default="/home/chinese-roberta-wwm-ext",
        help="Original HuggingFace model directory to copy tokenizer/config files from.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Specific .pth checkpoint to convert (overrides --checkpoint-dir).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/english_contrastive",
        help="Directory to scan for the best .pth when --checkpoint-path is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/best_model",
        help="Where to write the converted model directory.",
    )
    parser.add_argument(
        "--overwrite",
        default=True,
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def find_best_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("model_epoch*_f1*.pth"))
    if not checkpoints:
        checkpoints = sorted(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    def score(path):
        if "f1" in path.stem:
            try:
                return float(path.stem.rsplit("f1", 1)[1])
            except ValueError:
                pass
        return path.stat().st_mtime

    return max(checkpoints, key=score)


def resolve_checkpoint(args):
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else find_best_checkpoint(args.checkpoint_dir)
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if checkpoint_path.suffix != ".pth":
        raise ValueError(f"Expected a .pth checkpoint, got: {checkpoint_path}")
    return checkpoint_path


def load_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def copy_base_files(base_dir, output_dir):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    copied = []
    for filename in TOKENIZER_AND_CONFIG_FILES:
        src = base_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
            copied.append(filename)
    return copied


def prepare_output_dir(output_dir, overwrite):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists: {output_dir} (use --overwrite to replace)"
            )
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def summarize_state_dict(state_dict):
    token_keys = sum(1 for key in state_dict if key.startswith("token_encoder."))
    label_keys = sum(1 for key in state_dict if key.startswith("label_encoder."))
    other_keys = len(state_dict) - token_keys - label_keys
    return token_keys, label_keys, other_keys


def main():
    args = parse_args()
    if DEPENDENCY_ERROR is not None:
        print(f"Missing dependency: {DEPENDENCY_ERROR.name}", file=sys.stderr)
        print("Install dependencies with: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1)

    base_dir = Path(args.base_model_path)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base model directory does not exist: {base_dir}")

    checkpoint_path = resolve_checkpoint(args)
    print(f"Selected checkpoint: {checkpoint_path}", flush=True)

    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    print(f"Output directory: {output_dir}", flush=True)

    print(f"Copying tokenizer/config files from {base_dir}", flush=True)
    copied = copy_base_files(base_dir, output_dir)
    if not copied:
        raise RuntimeError(
            f"No tokenizer/config files were found in {base_dir}; "
            f"expected at least one of: {', '.join(TOKENIZER_AND_CONFIG_FILES)}"
        )
    print(f"  copied: {', '.join(copied)}", flush=True)

    print(f"Loading state dict from {checkpoint_path}", flush=True)
    state_dict = load_state_dict(checkpoint_path)
    token_keys, label_keys, other_keys = summarize_state_dict(state_dict)
    print(
        f"  total tensors: {len(state_dict)}  "
        f"token_encoder: {token_keys}  label_encoder: {label_keys}  other: {other_keys}",
        flush=True,
    )

    bin_path = output_dir / "pytorch_model.bin"
    torch.save(state_dict, bin_path)
    size_mb = bin_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {bin_path}  ({size_mb:.2f} MB)", flush=True)

    print("\nUsage example:")
    print(
        f"  python tests/batch_test.py \\\n"
        f"      --model-path {output_dir} \\\n"
        f"      --checkpoint-path {bin_path}",
        flush=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

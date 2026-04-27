import argparse
import json
import sys
import time
from pathlib import Path

DEPENDENCY_ERROR = None

try:
    import torch
    from transformers import AutoTokenizer

    from label_semantics.labels import build_tag_maps, load_label_descriptions
    from label_semantics.metrics import get_entities
    from label_semantics.model import LabelSemanticsNER
except ModuleNotFoundError as exc:
    torch = None
    AutoTokenizer = None
    build_tag_maps = None
    load_label_descriptions = None
    get_entities = None
    LabelSemanticsNER = None
    DEPENDENCY_ERROR = exc


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-test company extraction with a trained .pth checkpoint.")
    parser.add_argument("--input-file", default="train.jsonl")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Base HuggingFace model name or local model path used to build the checkpoint architecture.",
    )
    parser.add_argument("--label-file", default="excel_ner_data/labels.json")
    parser.add_argument("--checkpoint-path", default=None, help="Generated trained checkpoint .pth file to load.")
    parser.add_argument("--checkpoint-dir", default="outputs/pretrain", help="Find best generated .pth here when --checkpoint-path is omitted.")
    parser.add_argument("--output-file", default="outputs/batch_test_predictions.jsonl")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def find_best_checkpoint(checkpoint_dir):
    checkpoints = sorted(Path(checkpoint_dir).glob("model_epoch*_f1*.pth"))
    if not checkpoints:
        checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    def score(path):
        if "f1" in path.stem:
            return float(path.stem.rsplit("f1", 1)[1])
        return path.stat().st_mtime

    return max(checkpoints, key=score)


def resolve_checkpoint(args):
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else find_best_checkpoint(args.checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if checkpoint_path.suffix != ".pth":
        raise ValueError(f"Expected a .pth checkpoint, got: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint_state_dict(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return rows


def encode_query(query, tokenizer, max_length, device):
    chars = list(query)
    inputs = tokenizer(
        chars,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    word_ids = inputs.word_ids(batch_index=0)
    model_inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "token_type_ids": inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"])).to(device),
    }
    return model_inputs, word_ids


def predict_companies(query, model, tokenizer, id2tag, max_length, device):
    model_inputs, word_ids = encode_query(query, tokenizer, max_length, device)
    with torch.no_grad():
        _, pred_ids = model(**model_inputs, use_label_cache=True)

    tags = ["O"] * len(query)
    for pred_id, word_id in zip(pred_ids[0].detach().cpu().tolist(), word_ids):
        if word_id is None or word_id >= len(tags):
            continue
        tags[word_id] = id2tag[pred_id]

    companies = []
    for entity_type, start, end in get_entities(tags):
        if entity_type == "C":
            companies.append(query[start:end + 1])
    return companies


def normalize_companies(companies):
    return sorted({str(company).strip() for company in companies if str(company).strip()})


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    if DEPENDENCY_ERROR is not None:
        print(f"Missing dependency: {DEPENDENCY_ERROR.name}", file=sys.stderr)
        print("Install dependencies with: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = resolve_checkpoint(args)

    print(f"Using device: {device}", flush=True)
    print(f"Loading labels from {args.label_file}", flush=True)
    label_descriptions = load_label_descriptions(args.label_file)
    tag2id, id2tag = build_tag_maps(label_descriptions)

    print(f"Loading tokenizer from {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    print(f"Loading model from {args.model_path}", flush=True)
    model = LabelSemanticsNER(args.model_path, tag2id, label_descriptions).to(device)
    print(f"Loading .pth checkpoint from {checkpoint_path}", flush=True)
    state_dict = load_checkpoint_state_dict(checkpoint_path, device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing checkpoint keys: {len(missing_keys)}", flush=True)
    if unexpected_keys:
        print(f"Unexpected checkpoint keys: {len(unexpected_keys)}", flush=True)
    model.eval()

    rows = load_jsonl(args.input_file, args.limit)
    print(f"Loaded {len(rows)} rows from {args.input_file}", flush=True)

    for row in rows[:args.warmup]:
        predict_companies(row.get("ori_query", ""), model, tokenizer, id2tag, args.max_length, device)
    sync_if_cuda(device)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0
    total_ms = 0.0
    max_ms = 0.0

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for index, row in enumerate(rows, start=1):
            query = row.get("ori_query", "")
            gold = normalize_companies(row.get("label", {}).get("entities", {}).get("companies", []))

            sync_if_cuda(device)
            start = time.perf_counter()
            predicted = normalize_companies(
                predict_companies(query, model, tokenizer, id2tag, args.max_length, device)
            )
            sync_if_cuda(device)
            elapsed_ms = (time.perf_counter() - start) * 1000

            is_correct = predicted == gold
            correct += int(is_correct)
            total += 1
            total_ms += elapsed_ms
            max_ms = max(max_ms, elapsed_ms)

            output_file.write(json.dumps({
                "ori_query": query,
                "gold_companies": gold,
                "pred_companies": predicted,
                "correct": is_correct,
                "inference_ms": round(elapsed_ms, 3),
            }, ensure_ascii=False) + "\n")

            if index % 100 == 0:
                accuracy = correct / total if total else 0.0
                avg_ms = total_ms / total if total else 0.0
                print(
                    f"processed={index} accuracy={accuracy:.6f} avg_ms={avg_ms:.3f}",
                    flush=True,
                )

    accuracy = correct / total if total else 0.0
    avg_ms = total_ms / total if total else 0.0
    print(f"Output: {output_path}", flush=True)
    print(
        f"total={total} correct={correct} accuracy={accuracy:.6f} "
        f"avg_ms={avg_ms:.3f} max_ms={max_ms:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

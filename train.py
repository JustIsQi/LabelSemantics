import argparse
from pathlib import Path

from label_semantics.training import TrainConfig, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train a label-semantics NER model.")
    parser.add_argument("--model-path", required=True, help="HuggingFace model name or local model path.")
    parser.add_argument("--data-dir", default="excel_ner_data", help="Directory containing BIO train/dev/test files.")
    parser.add_argument("--label-file", default="excel_ner_data/labels.json", help="Label description JSON.")
    parser.add_argument("--train-file", default="train.txt")
    parser.add_argument("--dev-file", default="dev.txt")
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--checkpoint-path", default=None, help="Optional pretrained checkpoint to finetune from.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sep", default="\t")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        model_name_or_path=args.model_path,
        data_dir=Path(args.data_dir),
        label_file=Path(args.label_file),
        train_file=args.train_file,
        dev_file=args.dev_file,
        test_file=args.test_file,
        output_dir=Path(args.output_dir),
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
        batch_size=args.batch_size,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        sep=args.sep,
        device=args.device,
    )
    train(config)


if __name__ == "__main__":
    main()

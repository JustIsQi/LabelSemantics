import argparse
from pathlib import Path

from .training import TrainConfig, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train a label-semantics NER model.")
    parser.add_argument("--model-path", required=True, help="HuggingFace model name or local model path.")
    parser.add_argument("--data-dir", default="data/excel_ner_data", help="Directory containing BIO train/dev/test files.")
    parser.add_argument("--label-file", default="data/excel_ner_data/labels.json", help="Label description JSON.")
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
    parser.add_argument(
        "--augment-role-prob",
        type=float,
        default=0.0,
        help=(
            "Per-sentence probability for the BROKER/C role-suffix data augmenter "
            "(see label_semantics.data.augment_role_suffix). 0.0 disables it. "
            "Recommend 0.3-0.5 to teach the model that '<broker/company> + "
            "<role/title/verb>' closes the entity boundary."
        ),
    )
    parser.add_argument("--augment-role-seed", type=int, default=0)
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.0,
        help="Weight for the supervised contrastive auxiliary loss. 0.0 disables it.",
    )
    parser.add_argument(
        "--contrastive-temperature",
        type=float,
        default=0.1,
        help="Temperature for the supervised contrastive auxiliary loss.",
    )
    parser.add_argument(
        "--contrastive-label",
        default="C",
        help="Entity label to focus the contrastive auxiliary loss on.",
    )
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
        augment_role_prob=args.augment_role_prob,
        augment_role_seed=args.augment_role_seed,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_label=args.contrastive_label,
    )
    train(config)


if __name__ == "__main__":
    main()

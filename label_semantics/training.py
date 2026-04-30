from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .constants import IGNORE_INDEX
from .data import (
    augment_role_suffix,
    build_dataloader,
    encode_bio_examples,
    read_bio_file,
    shuffle_features,
)
from .labels import build_tag_maps, load_label_descriptions
from .metrics import entity_f1_breakdown
from .model import LabelSemanticsNER


@dataclass
class TrainConfig:
    model_name_or_path: str
    data_dir: Path
    label_file: Path
    train_file: str = "train.txt"
    dev_file: str = "dev.txt"
    test_file: str | None = None
    output_dir: Path = Path("outputs")
    checkpoint_path: Path | None = None
    batch_size: int = 16
    max_length: int = 128
    epochs: int = 100
    learning_rate: float = 1e-5
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    sep: str = "\t"
    device: str | None = None
    # Per-sentence probability for ``augment_role_suffix``. 0.0 disables it
    # (default, preserves legacy behaviour). 0.3-0.5 is a reasonable range
    # for real training; the augmenter teaches the model that
    # "<broker/company> + <role/title/verb>" closes the entity boundary,
    # which fixes overshoot like "中信证券分析师明明" -> BROKER.
    augment_role_prob: float = 0.0
    augment_role_seed: int = 0
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.1
    contrastive_label: str = "C"


def log(message):
    print(message, flush=True)


def load_features(
    config,
    tokenizer,
    tag2id,
    filename,
    augment_role_prob=0.0,
):
    tokens, labels = read_bio_file(config.data_dir / filename, sep=config.sep)
    if augment_role_prob > 0.0:
        import random

        rng = random.Random(config.augment_role_seed)
        extra_tokens, extra_labels = augment_role_suffix(
            tokens, labels, prob=augment_role_prob, rng=rng,
        )
        log(
            f"augment_role_suffix: original={len(tokens)} extra={len(extra_tokens)} "
            f"prob={augment_role_prob}"
        )
        tokens = tokens + extra_tokens
        labels = labels + extra_labels
    return encode_bio_examples(tokens, labels, tokenizer, tag2id, config.max_length)


def evaluate(model, dataloader, id2tag, device):
    model.reset_label_cache()
    model.eval()
    predictions, targets = [], []
    english_masks = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, labels, english_mask = (
                item.to(device) for item in batch
            )
            _, pred_ids = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                use_label_cache=True,
            )
            predictions.extend(pred_ids.cpu().tolist())
            targets.extend(labels.cpu().tolist())
            english_masks.extend(english_mask.cpu().tolist())
    return entity_f1_breakdown(predictions, targets, id2tag, english_masks)


def _entity_type_for_tag(tag):
    if tag == "O":
        return "O"
    if "-" in tag:
        return tag.split("-", 1)[1]
    return tag


def supervised_contrastive_loss(
    token_embeddings,
    labels,
    english_mask,
    label_representation,
    id2tag,
    target_label="C",
    temperature=0.1,
):
    if temperature <= 0:
        raise ValueError("contrastive_temperature must be > 0")

    flat_embeddings = token_embeddings.reshape(-1, token_embeddings.shape[-1])
    flat_labels = labels.reshape(-1)
    flat_english_mask = english_mask.reshape(-1).bool()
    valid_mask = flat_labels.ne(IGNORE_INDEX) & flat_english_mask
    if not valid_mask.any():
        return token_embeddings.new_zeros(())

    valid_embeddings = flat_embeddings[valid_mask]
    valid_labels = flat_labels[valid_mask]
    valid_types = torch.tensor(
        [
            1 if _entity_type_for_tag(id2tag[int(label_id)]) == target_label else 0
            for label_id in valid_labels.detach().cpu().tolist()
        ],
        device=token_embeddings.device,
        dtype=torch.long,
    )
    anchor_mask = valid_types.eq(1)
    if not anchor_mask.any():
        return token_embeddings.new_zeros(())

    label_types = torch.tensor(
        [
            1 if _entity_type_for_tag(id2tag[index]) == target_label else 0
            for index in range(label_representation.shape[0])
        ],
        device=token_embeddings.device,
        dtype=torch.long,
    )
    candidate_embeddings = torch.cat([valid_embeddings, label_representation], dim=0)
    candidate_types = torch.cat([valid_types, label_types], dim=0)
    candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
    anchor_embeddings = candidate_embeddings[: valid_embeddings.shape[0]][anchor_mask]
    logits = torch.matmul(anchor_embeddings, candidate_embeddings.transpose(0, 1)) / temperature

    self_mask = torch.zeros_like(logits, dtype=torch.bool)
    anchor_indices = torch.nonzero(anchor_mask, as_tuple=False).squeeze(1)
    row_indices = torch.arange(anchor_indices.shape[0], device=token_embeddings.device)
    self_mask[row_indices, anchor_indices] = True
    logits = logits.masked_fill(self_mask, float("-inf"))

    positive_mask = candidate_types.unsqueeze(0).eq(1) & ~self_mask
    has_positive = positive_mask.any(dim=1)
    if not has_positive.any():
        return token_embeddings.new_zeros(())

    logits = logits[has_positive]
    positive_mask = positive_mask[has_positive]
    log_denominator = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
    log_positive = torch.logsumexp(positive_logits, dim=1)
    return -(log_positive - log_denominator).mean()


def format_metrics(prefix, metrics):
    c_metrics = metrics.get("by_type", {}).get("C", {"precision": 0.0, "recall": 0.0, "f1": 0.0})
    english_c = metrics.get("english_C", {"precision": 0.0, "recall": 0.0, "f1": 0.0})
    return (
        f"{prefix} precision={metrics['precision']:.6f} recall={metrics['recall']:.6f} "
        f"f1={metrics['f1']:.6f} c_f1={c_metrics['f1']:.6f} "
        f"english_c_f1={english_c['f1']:.6f}"
    )


def train(config):
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"Using device: {device}")
    log(f"Loading labels from {config.label_file}")
    label_descriptions = load_label_descriptions(config.label_file)
    tag2id, id2tag = build_tag_maps(label_descriptions)
    log(f"Loading tokenizer from {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)

    log(f"Encoding train data: {config.data_dir / config.train_file}")
    train_features = load_features(
        config, tokenizer, tag2id, config.train_file,
        augment_role_prob=config.augment_role_prob,
    )
    train_features = shuffle_features(train_features)
    log(f"Shuffled train features: num_examples={train_features['input_ids'].size(0)}")
    log(f"Encoding dev data: {config.data_dir / config.dev_file}")
    dev_features = load_features(config, tokenizer, tag2id, config.dev_file)
    train_loader = build_dataloader(train_features, config.batch_size, shuffle=True)
    dev_loader = build_dataloader(dev_features, config.batch_size)
    log(f"Built dataloaders: train_steps={len(train_loader)} dev_steps={len(dev_loader)}")

    log(f"Loading model from {config.model_name_or_path}")
    model = LabelSemanticsNER(config.model_name_or_path, tag2id, label_descriptions).to(device)
    if config.checkpoint_path:
        log(f"Loading checkpoint from {config.checkpoint_path}")
        model.load_state_dict(torch.load(config.checkpoint_path, map_location=device), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=len(train_loader) * config.epochs,
    )
    loss_function = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    for epoch in range(config.epochs):
        log(f"Starting epoch {epoch}")
        model.train()
        total_loss = 0.0
        steps = 0
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels, english_mask = (
                item.to(device) for item in batch
            )
            model_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                use_label_cache=False,
                return_representations=config.contrastive_weight > 0.0,
            )
            if config.contrastive_weight > 0.0:
                logits, _, token_embeddings, label_representation = model_outputs
            else:
                logits, _ = model_outputs
                token_embeddings = None
                label_representation = None

            ce_loss = loss_function(logits.reshape(-1, len(tag2id)), labels.reshape(-1))
            contrastive = logits.new_zeros(())
            if config.contrastive_weight > 0.0:
                contrastive = supervised_contrastive_loss(
                    token_embeddings,
                    labels,
                    english_mask,
                    label_representation,
                    id2tag,
                    target_label=config.contrastive_label,
                    temperature=config.contrastive_temperature,
                )
            loss = ce_loss + config.contrastive_weight * contrastive
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            steps += 1
            if step % 30 == 0:
                log(
                    f"epoch={epoch} step={step} train_loss={total_loss / steps:.6f} "
                    f"ce_loss={ce_loss.item():.6f} contrastive_loss={contrastive.item():.6f}"
                )

        metrics = evaluate(model, dev_loader, id2tag, device)
        log(f"epoch={epoch} train_loss={total_loss / max(steps, 1):.6f} " + format_metrics("dev", metrics))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), config.output_dir / f"model_epoch{epoch}_f1{best_f1:.6f}.pth")

    if config.test_file:
        test_features = load_features(config, tokenizer, tag2id, config.test_file)
        test_loader = build_dataloader(test_features, config.batch_size)
        test_metrics = evaluate(model, test_loader, id2tag, device)
        log(format_metrics("test", test_metrics))

    return best_f1

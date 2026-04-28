from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from .constants import IGNORE_INDEX


def read_bio_file(path, sep="\t"):
    tokens, labels = [], []
    current_tokens, current_labels = [], []
    with Path(path).open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                current_tokens, current_labels = [], []
                continue

            parts = line.split(sep)
            if len(parts) != 2:
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                current_tokens, current_labels = [], []
                continue

            current_tokens.append(parts[0])
            current_labels.append(parts[1])

    if current_tokens:
        tokens.append(current_tokens)
        labels.append(current_labels)
    return tokens, labels


def encode_bio_examples(tokens, labels, tokenizer, tag2id, max_length):
    encoded = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for token_row, label_row in zip(tokens, labels):
        inputs = tokenizer(
            token_row,
            is_split_into_words=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        word_ids = inputs.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(IGNORE_INDEX)
            elif word_idx != previous_word_idx:
                aligned_labels.append(tag2id[label_row[word_idx]])
            else:
                aligned_labels.append(IGNORE_INDEX)
            previous_word_idx = word_idx

        encoded["input_ids"].append(inputs["input_ids"])
        encoded["token_type_ids"].append(inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"])))
        encoded["attention_mask"].append(inputs["attention_mask"])
        encoded["labels"].append(aligned_labels)

    return {
        "input_ids": torch.cat(encoded["input_ids"], dim=0),
        "token_type_ids": torch.cat(encoded["token_type_ids"], dim=0),
        "attention_mask": torch.cat(encoded["attention_mask"], dim=0),
        "labels": torch.tensor(encoded["labels"], dtype=torch.long),
    }


def shuffle_features(features, generator=None):
    num_examples = features["input_ids"].size(0)
    perm = torch.randperm(num_examples, generator=generator)
    return {key: tensor[perm] for key, tensor in features.items()}


def build_dataset(features):
    return TensorDataset(
        features["input_ids"],
        features["attention_mask"],
        features["token_type_ids"],
        features["labels"],
    )


def build_dataloader(features, batch_size, shuffle=False):
    dataset = build_dataset(features)
    sampler = RandomSampler(dataset) if shuffle else None
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

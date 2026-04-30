import random
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from .constants import IGNORE_INDEX


# Tail context inserted right after a BROKER/C entity by ``augment_role_suffix``.
# All inserted characters are tagged ``O`` so the model learns that
# "<broker> + <role/person/verb>" closes the entity boundary -- the original
# training set has *zero* role-word occurrences, which is why a query like
# "中信证券分析师明明撰写..." gets the analyst name folded into BROKER.
_BROKER_ROLE_SUFFIX_POOL = (
    "分析师", "研究员", "首席", "首席分析师", "策略师", "策略分析师",
    "研究主管", "研究总监", "团队", "固收团队", "TMT 团队", "电新团队",
    "研究所", "研究院", "研究部", "营业部", "资管部", "投行部",
    "分析师明明", "分析师张三", "首席李四", "研究员王五", "首席分析师赵六",
    "团队最新", "首席观点", "分析师认为", "研究员指出",
    "发布", "披露", "推荐", "覆盖", "评级",
)
_COMPANY_ROLE_SUFFIX_POOL = (
    "董事长", "总经理", "总裁", "副总裁", "董秘", "CEO", "CFO", "CTO",
    "发言人", "创始人",
    "董事长李四", "总经理张三", "CEO 王五", "CFO 赵六",
    "发布", "披露", "表示", "回应", "声明", "签约", "收购", "并购",
    "子公司", "控股股东", "关联方", "董事会", "股东大会",
    "招股书", "年报", "半年报", "季报", "中报", "财报",
    "业绩说明会", "电话会议", "调研纪要",
)
_ROLE_SUFFIX_POOLS = {
    "BROKER": _BROKER_ROLE_SUFFIX_POOL,
    "C": _COMPANY_ROLE_SUFFIX_POOL,
}

_ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")


def _iter_entities(label_row):
    i, n = 0, len(label_row)
    while i < n:
        tag = label_row[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            end = i
            j = i + 1
            i_tag = f"I-{etype}"
            while j < n and label_row[j] == i_tag:
                end = j
                j += 1
            yield etype, start, end
            i = j
        else:
            i += 1


def augment_role_suffix(tokens, labels, prob=0.3, rng=None, max_examples=None):
    """Generate extra training examples by appending role/title/verb words
    (tagged ``O``) right after BROKER/C entities.

    Insertion is only done when the entity is followed by an ``O`` token (or
    end-of-sentence), so we never split a neighbouring entity. Inserted words
    are taken from ``_ROLE_SUFFIX_POOLS``.

    Returns ``(extra_tokens, extra_labels)`` -- new examples to be appended
    to the original dataset. The original dataset is not modified.
    """
    if prob <= 0.0:
        return [], []

    rng = rng or random.Random(0)
    extra_tokens, extra_labels = [], []

    for token_row, label_row in zip(tokens, labels):
        if rng.random() >= prob:
            continue

        entities = [
            (etype, start, end)
            for etype, start, end in _iter_entities(label_row)
            if etype in _ROLE_SUFFIX_POOLS
        ]
        if not entities:
            continue

        new_tokens = list(token_row)
        new_labels = list(label_row)
        modified = False

        # Walk right-to-left so insertion indices remain valid.
        for etype, _start, end in reversed(entities):
            insert_at = end + 1
            # Only insert when the slot right after the entity is O or EOS,
            # to avoid corrupting an adjacent entity.
            if insert_at < len(new_labels) and new_labels[insert_at] != "O":
                continue
            suffix = rng.choice(_ROLE_SUFFIX_POOLS[etype])
            insert_chars = list(suffix)
            insert_labels = ["O"] * len(insert_chars)
            new_tokens[insert_at:insert_at] = insert_chars
            new_labels[insert_at:insert_at] = insert_labels
            modified = True

        if modified:
            extra_tokens.append(new_tokens)
            extra_labels.append(new_labels)
            if max_examples is not None and len(extra_tokens) >= max_examples:
                break

    return extra_tokens, extra_labels


def has_ascii_alpha(text):
    return bool(_ASCII_ALPHA_RE.search(text))


def is_ascii_alpha_token(token):
    return len(token) == 1 and token.isascii() and token.isalpha()


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
        "english_mask": [],
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
        english_mask = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(IGNORE_INDEX)
                english_mask.append(0)
            elif word_idx != previous_word_idx:
                aligned_labels.append(tag2id[label_row[word_idx]])
                english_mask.append(int(is_ascii_alpha_token(token_row[word_idx])))
            else:
                aligned_labels.append(IGNORE_INDEX)
                english_mask.append(0)
            previous_word_idx = word_idx

        encoded["input_ids"].append(inputs["input_ids"])
        encoded["token_type_ids"].append(inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"])))
        encoded["attention_mask"].append(inputs["attention_mask"])
        encoded["labels"].append(aligned_labels)
        encoded["english_mask"].append(english_mask)

    return {
        "input_ids": torch.cat(encoded["input_ids"], dim=0),
        "token_type_ids": torch.cat(encoded["token_type_ids"], dim=0),
        "attention_mask": torch.cat(encoded["attention_mask"], dim=0),
        "labels": torch.tensor(encoded["labels"], dtype=torch.long),
        "english_mask": torch.tensor(encoded["english_mask"], dtype=torch.bool),
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
        features["english_mask"],
    )


def build_dataloader(features, batch_size, shuffle=False):
    dataset = build_dataset(features)
    sampler = RandomSampler(dataset) if shuffle else None
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

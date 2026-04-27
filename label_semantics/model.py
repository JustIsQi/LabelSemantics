import inspect

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .constants import DEFAULT_BIO_PREFIX_TEXT, DEFAULT_OUTSIDE_LABEL_TEXT


class LabelSemanticsNER(nn.Module):
    def __init__(
        self,
        model_name_or_path,
        tag2id,
        label_descriptions,
        bio_prefix_text=None,
        outside_label_text=DEFAULT_OUTSIDE_LABEL_TEXT,
    ):
        super().__init__()
        self.token_encoder = AutoModel.from_pretrained(model_name_or_path)
        self.label_encoder = AutoModel.from_pretrained(model_name_or_path)
        self.accepts_token_type_ids = "token_type_ids" in inspect.signature(self.token_encoder.forward).parameters
        self.label_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.tag2id = tag2id
        self.label_descriptions = label_descriptions
        self.bio_prefix_text = bio_prefix_text or DEFAULT_BIO_PREFIX_TEXT
        self.outside_label_text = outside_label_text
        self.register_buffer("cached_label_representation", torch.empty(0), persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def build_label_texts(self):
        label_texts = []
        for tag in sorted(self.tag2id, key=self.tag2id.get):
            if tag == "O":
                label_texts.append(self.outside_label_text)
                continue
            prefix, label = tag.split("-", 1)
            label_texts.append(self.bio_prefix_text[prefix] + self.label_descriptions[label])
        return label_texts

    def build_label_representation(self):
        inputs = self.label_tokenizer(
            self.build_label_texts(),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.label_encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    def reset_label_cache(self):
        self.cached_label_representation = torch.empty(0, device=self.device)

    def forward(self, input_ids, attention_mask, token_type_ids=None, use_label_cache=False):
        if use_label_cache and self.cached_label_representation.numel() > 0:
            label_representation = self.cached_label_representation
        else:
            label_representation = self.build_label_representation()
            if use_label_cache:
                self.cached_label_representation = label_representation.detach()

        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.accepts_token_type_ids and token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        outputs = self.token_encoder(**encoder_inputs)
        token_embeddings = outputs.last_hidden_state
        batch_size = token_embeddings.shape[0]
        label_count, hidden_size = label_representation.shape
        expanded_labels = label_representation.expand(batch_size, label_count, hidden_size)
        logits = torch.matmul(token_embeddings, expanded_labels.transpose(2, 1))
        predictions = torch.argmax(logits, dim=-1)
        return logits, predictions

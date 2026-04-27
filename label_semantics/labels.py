import json
from pathlib import Path


def load_label_descriptions(path):
    with Path(path).open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def build_tag_maps(label_descriptions):
    tag_set = []
    for label in label_descriptions:
        tag_set.extend([f"B-{label}", f"I-{label}"])
    tag_set.append("O")
    tag2id = {tag: index for index, tag in enumerate(tag_set)}
    id2tag = {index: tag for tag, index in tag2id.items()}
    return tag2id, id2tag


def save_label_descriptions(path, labels, descriptions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump({label: descriptions[label] for label in labels}, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

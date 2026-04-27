import argparse
import json
import random
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


DEFAULT_TYPE_TO_LABEL = {
    "机构-公司": "C",
    "机构-代码": "CODE",
    "行业/概念": "IND",
    "时间": "TIME",
}

DEFAULT_IGNORED_TYPES = {
    "产品",
    "技术",
    "机构-机构",
    "机构-政府",
    "机构-其他",
    "自然人",
}

DEFAULT_LABEL_DESCRIPTIONS = {
    "C": "公司",
    "CODE": "机构代码",
    "IND": "行业概念",
    "TIME": "时间",
}

MAIN_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


@dataclass
class ExcelConversionConfig:
    input_file: Path = Path("test_data_0413.xlsx")
    output_dir: Path = Path("excel_ner_data")
    query_column: str = "Query"
    entity_columns: tuple[str, ...] = ("company_entities", "other_entities")
    type_to_label: dict[str, str] | None = None
    ignored_types: set[str] | None = None
    label_descriptions: dict[str, str] | None = None
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    seed: int = 42
    max_report_items: int = 500

    def __post_init__(self):
        self.type_to_label = self.type_to_label or DEFAULT_TYPE_TO_LABEL
        self.ignored_types = self.ignored_types or DEFAULT_IGNORED_TYPES
        self.label_descriptions = self.label_descriptions or DEFAULT_LABEL_DESCRIPTIONS


def xlsx_text(element):
    return "".join(node.text or "" for node in element.findall(".//a:t", MAIN_NS))


def column_index(cell_ref):
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + ord(char.upper()) - ord("A") + 1
    return index - 1


def read_xlsx_rows(path):
    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [xlsx_text(item) for item in root.findall("a:si", MAIN_NS)]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("rel:Relationship", REL_NS)
        }

        all_rows = []
        for sheet in workbook.findall(".//a:sheet", MAIN_NS):
            target = rid_to_target[sheet.attrib[RID]]
            sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
            root = ET.fromstring(archive.read(sheet_path))
            for row in root.findall(".//a:sheetData/a:row", MAIN_NS):
                values = []
                for cell in row.findall("a:c", MAIN_NS):
                    index = column_index(cell.attrib.get("r", "A1"))
                    while len(values) <= index:
                        values.append("")
                    cell_type = cell.attrib.get("t")
                    value = cell.find("a:v", MAIN_NS)
                    inline = cell.find("a:is", MAIN_NS)
                    if cell_type == "s" and value is not None:
                        values[index] = shared_strings[int(value.text)]
                    elif cell_type == "inlineStr" and inline is not None:
                        values[index] = xlsx_text(inline)
                    elif value is not None:
                        values[index] = value.text or ""
                all_rows.append(values)
        return all_rows


def parse_entities(text):
    entities = []
    if not text:
        return entities
    for raw_line in str(text).splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        parts = [part.strip() for part in raw_line.split("|")]
        if len(parts) < 2 or not parts[1]:
            continue
        entities.append({"type": parts[0], "mention": parts[1], "raw": raw_line})
    return entities


def normalize_query_text(text):
    return str(text).replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()


def normalize_with_map(text):
    normalized_chars = []
    index_map = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        normalized = unicodedata.normalize("NFKC", char).casefold()
        for normalized_char in normalized:
            if normalized_char.isspace():
                continue
            normalized_chars.append(normalized_char)
            index_map.append(index)
    return "".join(normalized_chars), index_map


def find_all(haystack, needle):
    positions = []
    start = 0
    while needle:
        index = haystack.find(needle, start)
        if index < 0:
            break
        positions.append((index, index + len(needle)))
        start = index + len(needle)
    return positions


def find_mention_spans(query, mention):
    exact_spans = find_all(query, mention)
    if exact_spans:
        return exact_spans, "exact"

    normalized_query, index_map = normalize_with_map(query)
    normalized_mention, _ = normalize_with_map(mention)
    normalized_spans = []
    for start, end in find_all(normalized_query, normalized_mention):
        normalized_spans.append((index_map[start], index_map[end - 1] + 1))
    return normalized_spans, "normalized"


def apply_entities(query, entity_rows, report, type_to_label, ignored_types):
    labels = ["O"] * len(query)
    candidates = []

    for entity in entity_rows:
        original_type = entity["type"]
        if original_type in ignored_types:
            report["ignored"].append(entity)
            continue

        label = type_to_label.get(original_type)
        if not label:
            report["unknown_types"].append(entity)
            continue

        spans, match_mode = find_mention_spans(query, entity["mention"])
        if not spans:
            report["unmatched"].append(entity)
            continue

        for start, end in spans:
            candidates.append({
                "start": start,
                "end": end,
                "label": label,
                "mention": entity["mention"],
                "type": original_type,
                "match_mode": match_mode,
                "row": entity.get("row"),
                "query": entity.get("query"),
                "raw": entity.get("raw"),
            })
            report["matched_mentions"] += 1
            report["match_modes"][match_mode] = report["match_modes"].get(match_mode, 0) + 1

    candidates.sort(key=lambda item: (item["end"] - item["start"], -item["start"]), reverse=True)
    occupied = [False] * len(query)
    for item in candidates:
        start, end = item["start"], item["end"]
        if any(occupied[start:end]):
            report["overlaps"].append(item)
            continue
        labels[start] = f"B-{item['label']}"
        for index in range(start + 1, end):
            labels[index] = f"I-{item['label']}"
        for index in range(start, end):
            occupied[index] = True

    return labels


def write_bio(path, examples):
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for query, labels in examples:
            for char, label in zip(query, labels):
                output.write(f"{char}\t{label}\n")
            output.write("\n")


def compact_report(report, max_items):
    for key in ("ignored", "unmatched", "unknown_types", "overlaps"):
        items = report[key]
        report[f"{key}_count"] = len(items)
        report[key] = items[:max_items]
    report["max_report_items"] = max_items


def deduplicate_examples(examples):
    deduplicated = []
    seen = set()
    query_labels = defaultdict(set)
    for query, labels in examples:
        label_tuple = tuple(labels)
        key = (query, label_tuple)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append((query, labels))
        query_labels[query].add(label_tuple)
    conflicting_queries = sum(1 for labels in query_labels.values() if len(labels) > 1)
    return deduplicated, len(examples) - len(deduplicated), conflicting_queries


def split_examples_by_query(examples, train_ratio, dev_ratio, seed):
    grouped = defaultdict(list)
    for query, labels in examples:
        grouped[query].append((query, labels))

    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)

    total = len(examples)
    train_target = int(total * train_ratio)
    dev_target = int(total * dev_ratio)
    splits = {"train.txt": [], "dev.txt": [], "test.txt": []}

    for group in groups:
        if len(splits["train.txt"]) < train_target:
            split_name = "train.txt"
        elif len(splits["dev.txt"]) < dev_target:
            split_name = "dev.txt"
        else:
            split_name = "test.txt"
        splits[split_name].extend(group)

    return {"all.txt": examples, **splits}


def convert_excel_to_bio(config=None):
    config = config or ExcelConversionConfig()
    rows = read_xlsx_rows(config.input_file)
    if not rows:
        raise RuntimeError(f"No rows found in {config.input_file}")

    header = rows[0]
    data_rows = []
    for row_number, row in enumerate(rows[1:], start=2):
        padded = row + [""] * (len(header) - len(row))
        data_rows.append((row_number, dict(zip(header, padded))))

    report = {
        "input_file": str(config.input_file),
        "total_rows": len(data_rows),
        "written_examples": 0,
        "matched_mentions": 0,
        "match_modes": {},
        "ignored": [],
        "unmatched": [],
        "unknown_types": [],
        "overlaps": [],
    }

    examples = []
    for row_number, row in data_rows:
        query = normalize_query_text(row.get(config.query_column, ""))
        if not query:
            continue

        entity_rows = []
        for column in config.entity_columns:
            for entity in parse_entities(row.get(column, "")):
                entity["row"] = row_number
                entity["query"] = query
                entity["column"] = column
                entity_rows.append(entity)

        labels = apply_entities(query, entity_rows, report, config.type_to_label, config.ignored_types)
        examples.append((query, labels))

    original_example_count = len(examples)
    examples, duplicate_count, conflicting_query_count = deduplicate_examples(examples)
    splits = split_examples_by_query(examples, config.train_ratio, config.dev_ratio, config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, split_examples in splits.items():
        write_bio(config.output_dir / filename, split_examples)

    used_labels = sorted({label[2:] for _, labels in examples for label in labels if label != "O"})
    with (config.output_dir / "labels.json").open("w", encoding="utf-8", newline="\n") as output:
        json.dump({label: config.label_descriptions[label] for label in used_labels}, output, ensure_ascii=False, indent=2)
        output.write("\n")

    report["written_examples"] = len(examples)
    report["source_examples"] = original_example_count
    report["duplicate_examples_removed"] = duplicate_count
    report["conflicting_label_queries"] = conflicting_query_count
    report["splits"] = {filename: len(split_examples) for filename, split_examples in splits.items()}
    report["labels"] = used_labels
    compact_report(report, config.max_report_items)
    with (config.output_dir / "conversion_report.json").open("w", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(f"Wrote {len(examples)} examples to {config.output_dir}")
    print("Splits:", report["splits"])
    print("Labels:", ", ".join(used_labels))
    print("Unmatched annotations:", report["unmatched_count"])
    print("Overlapping annotations skipped:", report["overlaps_count"])
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Convert annotated Excel rows to BIO files.")
    parser.add_argument("--input-file", default="test_data_0413.xlsx")
    parser.add_argument("--output-dir", default="excel_ner_data")
    parser.add_argument("--query-column", default="Query")
    parser.add_argument("--entity-columns", default="company_entities,other_entities")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-report-items", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    convert_excel_to_bio(
        ExcelConversionConfig(
            input_file=Path(args.input_file),
            output_dir=Path(args.output_dir),
            query_column=args.query_column,
            entity_columns=tuple(column.strip() for column in args.entity_columns.split(",") if column.strip()),
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
            max_report_items=args.max_report_items,
        )
    )


if __name__ == "__main__":
    main()

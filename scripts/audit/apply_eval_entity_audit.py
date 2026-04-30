"""Apply high-confidence audit fixes to data/eval_entities.xlsx.

Inputs:
  - tests/batch_test_predictions.xlsx
  - data/securities_firm_src_org_names.json
  - data/eval_entities.xlsx

The script deliberately uses conservative rules.  It records every cell
change to data/audit/eval_entities_changes.json before saving the workbook.
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[2]
PRED_PATH = ROOT / "tests" / "batch_test_predictions.xlsx"
BROKER_LIST_PATH = ROOT / "data" / "securities_firm_src_org_names.json"
EVAL_PATH = ROOT / "data" / "eval_entities.xlsx"
OUT_DIR = ROOT / "data" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


EVAL_COLUMNS = {
    "companies": 2,
    "institutions": 3,
    "industries": 4,
    "products": 5,
}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    value = re.sub(r"\s+", " ", value.strip())
    return value.lower()


def split_cell(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = re.split(r"\s*(?:;|；|\n|\|)\s*", text)
    return [part.strip() for part in parts if part.strip()]


def split_prediction(value: object) -> list[str]:
    # Do not split on comma: English company names frequently contain commas.
    return split_cell(value)


def join_cell(values: Iterable[str]) -> str | None:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        key = normalize(item)
        if item and key not in seen:
            cleaned.append(item)
            seen.add(key)
    return "; ".join(cleaned) if cleaned else None


def contains_overlap(values: Iterable[str], item: str) -> bool:
    item_key = normalize(item)
    for value in values:
        value_key = normalize(value)
        if not value_key or not item_key:
            continue
        if value_key == item_key or value_key in item_key or item_key in value_key:
            return True
    return False


LEGAL_BROKER_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "有限公司",
)

BROKER_ALIASES = {
    "瑞信",
    "瑞银",
    "瑞银集团",
    "瑞银证券",
    "高盛",
    "高盛证券",
    "摩根大通",
    "摩根士丹利",
    "大摩",
    "小摩",
    "JP摩根",
    "JPMorgan",
    "JPMorgan Chase",
    "JPMorgan Chase & Co.",
    "Goldman Sachs",
    "Goldman Sachs Group",
    "Goldman Sachs Group, Inc.",
    "Morgan Stanley",
    "美银",
    "美银证券",
    "美银美林",
    "花旗",
    "花旗银行",
    "野村",
    "野村证券",
    "大和",
    "大和资本",
    "巴克莱",
    "巴克莱银行",
    "汇丰",
    "汇丰银行",
    "渣打",
    "渣打银行",
    "德意志",
    "德银",
    "法巴",
    "法兴",
    "法国兴业",
    "麦格理",
    "杰富瑞",
    "里昂",
    "里昂证券",
    "中信证券",
    "国泰君安",
    "国泰君安证券",
    "海通证券",
    "华泰证券",
    "中金",
    "中金公司",
    "广发证券",
    "招商证券",
    "申万宏源",
    "中信建投",
    "光大证券",
    "兴业证券",
    "东方证券",
    "国信证券",
    "安信证券",
    "方正证券",
    "银河证券",
    "国元证券",
    "国海证券",
    "财通证券",
    "长江证券",
    "东兴证券",
    "东吴证券",
    "西南证券",
    "华西证券",
    "国联证券",
    "恒泰证券",
    "华林证券",
    "太平洋证券",
    "红塔证券",
    "南京证券",
    "浙商证券",
    "中泰证券",
    "国金证券",
    "国投证券",
    "长城证券",
    "国泰海通",
    "西部证券",
    "恒投证券",
    "盈透证券",
}

AMBIGUOUS_BROKER_ALIASES = {
    "中信",
    "华泰",
    "招商",
    "海通",
    "广发",
    "国信",
    "国联",
    "光大",
    "兴业",
    "东方",
    "银河",
    "太平洋",
    "中金",
}

BROKER_CONTEXT = re.compile(
    r"(研报|研究报告|评级|目标价|分析师|发布|主办|组织|路演|纪要|会议|券商|证券|佣金|投行|卖方|策略)",
)

BROKER_NOISE = {
    "券商",
    "机构",
    "投行",
    "银行",
    "证券",
    "基金",
    "期货",
    "发布",
    "组织",
    "发",
    "绿",
    "卖",
    "买",
}

BROKEN_BROKER_MAP = {
    "摩根士丹": "摩根士丹利",
    "摩根士": "摩根士丹利",
    "摩根": None,
    "摩": None,
    "中信证券发": "中信证券",
    "招商证券发": "招商证券",
    "海通证券发": "海通证券",
    "华泰证券发": "华泰证券",
    "广发证券发": "广发证券",
    "国泰君安发": "国泰君安",
    "中信建投发": "中信建投",
    "中信建投组": "中信建投",
}

BROKEN_SUFFIXES = (
    "发布的",
    "发行的",
    "组织的",
    "主办的",
    "发布",
    "发行",
    "组织",
    "主办",
    "策略团队",
    "分析师",
    "研究员",
    "发",
    "组",
)


def build_broker_names() -> set[str]:
    with BROKER_LIST_PATH.open("r", encoding="utf-8") as input_file:
        raw_names = json.load(input_file)

    names: set[str] = set()
    for raw in raw_names:
        name = str(raw).strip()
        if not name:
            continue
        variants = {name}
        no_paren = re.sub(r"[\(（].*?[\)）]", "", name).strip()
        variants.add(no_paren)
        for suffix in LEGAL_BROKER_SUFFIXES:
            for variant in list(variants):
                if variant.endswith(suffix) and len(variant) > len(suffix) + 1:
                    variants.add(variant[: -len(suffix)].strip())
        for variant in variants:
            if len(variant) >= 2:
                names.add(normalize(variant))
    names.update(normalize(alias) for alias in BROKER_ALIASES)
    return names


BROKER_NAMES = build_broker_names()


def clean_broker_mention(query: str, mention: str) -> str | None:
    mention = mention.strip()
    if not mention or mention in BROKER_NOISE:
        return None
    mapped = BROKEN_BROKER_MAP.get(mention)
    if mapped:
        return mapped if mapped in query else None
    if mention in {"摩", "摩根"}:
        for full in ("摩根大通", "摩根士丹利"):
            if full in query:
                return full
        return None
    for suffix in BROKEN_SUFFIXES:
        if mention.endswith(suffix) and len(mention) > len(suffix) + 1:
            mention = mention[: -len(suffix)].strip()
            break
    return mention if mention else None


def is_broker(name: str, query: str = "") -> bool:
    key = normalize(name)
    if key in BROKER_NAMES:
        if name in AMBIGUOUS_BROKER_ALIASES:
            return bool(BROKER_CONTEXT.search(query))
        return True
    if re.search(r"(证券|證券|期货)$", name, flags=re.IGNORECASE):
        return True
    if re.search(r"(证券|證券|期货)(股份有限公司|有限责任公司|有限公司|集团公司|集团|控股)?$", name):
        return True
    return False


INDUSTRY_SUFFIX_RE = re.compile(r"(行业|产业链|产业|板块|赛道|概念|链条)$")
INDUSTRY_KEYWORDS = {
    "半导体",
    "半导体行业",
    "半导体设备",
    "光伏",
    "光伏行业",
    "光伏产业链",
    "光伏组件",
    "新能源汽车",
    "新能源汽车行业",
    "新能源车",
    "新能源行业",
    "低空经济",
    "低空经济产业",
    "低空经济产业链",
    "云计算",
    "固态电池",
    "非银金融",
    "消费电子",
    "消费电子行业",
    "白酒",
    "白酒行业",
    "医药行业",
    "医疗行业",
    "机器人行业",
    "机器人产业",
    "房地产行业",
    "保险行业",
    "金融科技",
    "金融科技行业",
    "AI芯片",
    "AI服务器",
    "电动车",
    "电动汽车",
    "光通信行业",
    "高端机床行业",
    "有色金属行业",
    "固收市场",
}
INDUSTRY_NOISE = {
    "市场",
    "业务",
    "公司",
    "企业",
    "供应链",
    "金融",
    "汽车",
    "手机",
    "游戏",
    "零售",
    "银行",
    "科技行业",
    "金融行业",
    "消费行业",
    "互联",
    "银行板",
    "云计",
    "光伏企",
    "港股",
    "A股",
    "美股",
}


def is_industry_candidate(name: str, query: str, current: list[str]) -> bool:
    name = name.strip()
    if len(name) < 2 or name not in query:
        return False
    if name in INDUSTRY_NOISE:
        return False
    if name.endswith(("且", "投", "企", "板", "产")):
        return False
    if contains_overlap(current, name):
        return False
    if INDUSTRY_SUFFIX_RE.search(name):
        return True
    if name in INDUSTRY_KEYWORDS:
        return True
    return False


GENERIC_COMPANY_RE = re.compile(
    r"(A股|港股|美股|深交所|上交所|北交所|纳斯达克|纽交所|科创板|创业板|板块|行业公司|上市公司|中概股|企业)$",
)
COMPANY_NOISE = {
    "IO",
    "UMH",
    "Q2",
    "SK",
    "LM",
    "WB",
    "FIRST",
    "Bank",
    "Model",
    "iPhone",
    "金融科技",
    "上证50ETF",
}
COMPANY_BAD_TOKENS = re.compile(
    r"(财报|公告|报告|研报|新闻|季度|年度|半年|全年|业务|收入|费用|margin|filing|filings|proxy statement|breakdown|earnings call|10-k|10-q)",
    re.IGNORECASE,
)
COMPANY_SUFFIX_RE = re.compile(
    r"(股份|集团|控股|银行|药业|科技|超媒|汽车|音乐|电器|电子|能源|石油|地产|网络|旅游|股份有限公司|有限公司)$",
)
ENGLISH_COMPANY_SUFFIX_RE = re.compile(
    r"\b(Inc\.?|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|Limited|Holdings?|Bancorp|Bank|Financial|Oyj|SA|AG|PLC|LLP)\b",
    re.IGNORECASE,
)
EXPLICIT_COMPANY_ALLOW = {
    "京东",
    "Target",
    "Synchrony Financial",
    "Sanoma Oyj",
    "福耀美国",
    "携程网",
    "PERMA-FIX ENVIRONMENTAL SERVICES, INC.",
    "芒果超媒",
    "金钟股份",
    "珠江股份",
    "裕兴股份",
    "海创药业",
    "KeyCorp",
    "HDFC银行",
    "腾讯音乐娱乐",
}


def expand_english_company(query: str, mention: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9&.,' -]*", mention):
        return mention
    pattern = re.compile(re.escape(mention) + r"(?:[ A-Za-z&.,'-]+)")
    match = pattern.search(query)
    if not match:
        return mention
    expanded = match.group(0).strip(" ,，。的在")
    suffix_match = ENGLISH_COMPANY_SUFFIX_RE.search(expanded)
    if suffix_match:
        expanded = expanded[: suffix_match.end()].strip(" ,")
    else:
        expanded = mention
    if len(expanded) > len(mention) and len(expanded) <= 60:
        return expanded
    return mention


def is_company_candidate(name: str, query: str, current: list[str], other_values: list[str]) -> bool:
    if not name or name not in query:
        return False
    if name in COMPANY_NOISE or len(name) < 2:
        return False
    if is_broker(name, query):
        return False
    if GENERIC_COMPANY_RE.search(name):
        return False
    if COMPANY_BAD_TOKENS.search(name):
        return False
    if contains_overlap(other_values, name):
        return False
    if contains_overlap(current, name):
        return name in EXPLICIT_COMPANY_ALLOW and any(
            normalize(cur) in normalize(name) and len(name) > len(cur)
            for cur in current
        )
    return name in EXPLICIT_COMPANY_ALLOW


def add_change(changes: dict[int, dict[str, set[str]]], row_idx: int, column: str, value: str) -> None:
    changes[row_idx][column].add(value)


def main() -> None:
    pred = pd.read_excel(PRED_PATH, sheet_name="明细")
    eval_df = pd.read_excel(EVAL_PATH, sheet_name="Sheet1")
    if len(pred) != len(eval_df):
        raise RuntimeError(f"row count mismatch: {len(pred)} vs {len(eval_df)}")

    changes: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    removals: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for i in range(len(eval_df)):
        query = str(eval_df.iloc[i]["ori_query"]).strip()
        if str(pred.iloc[i]["原始Query"]).strip() != query:
            raise RuntimeError(f"query mismatch at row {i}")

        companies = split_cell(eval_df.iloc[i].get("companies"))
        institutions = split_cell(eval_df.iloc[i].get("institutions"))
        industries = split_cell(eval_df.iloc[i].get("industries"))
        products = split_cell(eval_df.iloc[i].get("products"))

        # Broker additions from model output.
        broker_mentions = []
        for col in ("BROKER/券商 - 预测", "BROKER/券商 - 多余"):
            broker_mentions.extend(split_prediction(pred.iloc[i].get(col)))
        for mention in broker_mentions:
            cleaned = clean_broker_mention(query, mention)
            if not cleaned:
                continue
            if cleaned not in query and normalize(cleaned) not in BROKER_NAMES:
                continue
            if is_broker(cleaned, query) and not contains_overlap(institutions, cleaned):
                add_change(changes, i, "institutions", cleaned)

        # Move broker-like entities out of companies.
        for company in companies:
            if is_broker(company, query):
                removals[i]["companies"].add(company)
                if not contains_overlap(institutions, company):
                    add_change(changes, i, "institutions", company)

        # Industry additions from high-confidence extra predictions.
        for mention in split_prediction(pred.iloc[i].get("IND/行业概念 - 多余")):
            if is_industry_candidate(mention, query, industries):
                add_change(changes, i, "industries", mention)

        # Conservative company additions.  Do not remove translated aliases.
        other_values = institutions + industries + products
        for mention in split_prediction(pred.iloc[i].get("C/公司 - 多余")):
            candidate = expand_english_company(query, mention)
            if is_company_candidate(candidate, query, companies, other_values):
                add_change(changes, i, "companies", candidate)

    wb = load_workbook(EVAL_PATH)
    ws = wb["Sheet1"]

    audit_records = []
    for row_idx in sorted(set(changes) | set(removals)):
        excel_row = row_idx + 2
        record = {
            "row_idx": row_idx,
            "excel_row": excel_row,
            "query": ws.cell(excel_row, 1).value,
            "columns": {},
        }
        for column, col_idx in EVAL_COLUMNS.items():
            original_values = split_cell(ws.cell(excel_row, col_idx).value)
            next_values = [v for v in original_values if v not in removals[row_idx].get(column, set())]
            next_values.extend(sorted(changes[row_idx].get(column, set())))
            new_value = join_cell(next_values)
            old_value = join_cell(original_values)
            if old_value != new_value:
                ws.cell(excel_row, col_idx).value = new_value
                record["columns"][column] = {
                    "old": old_value,
                    "new": new_value,
                    "add": sorted(changes[row_idx].get(column, set())),
                    "remove": sorted(removals[row_idx].get(column, set())),
                }
        if record["columns"]:
            audit_records.append(record)

    backup = EVAL_PATH.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    shutil.copy2(EVAL_PATH, backup)
    wb.save(EVAL_PATH)

    audit_path = OUT_DIR / "eval_entities_changes.json"
    with audit_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "backup": str(backup.relative_to(ROOT)),
                "changed_rows": len(audit_records),
                "records": audit_records,
            },
            output_file,
            ensure_ascii=False,
            indent=2,
        )
        output_file.write("\n")

    counts = defaultdict(int)
    for record in audit_records:
        for column, detail in record["columns"].items():
            counts[f"{column}.rows"] += 1
            counts[f"{column}.add"] += len(detail["add"])
            counts[f"{column}.remove"] += len(detail["remove"])

    print(f"backup: {backup.relative_to(ROOT)}")
    print(f"audit: {audit_path.relative_to(ROOT)}")
    print(f"changed_rows: {len(audit_records)}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()

"""根据 batch_test_predictions.xlsx + 券商白名单 + 启发式规则，
从模型预测的 "多余" / "遗漏" 中分离出：
  - HIGH_CONFIDENCE_ADD：高置信度需补标到 eval_entities.xlsx
  - NEEDS_LLM_REVIEW：需要 LLM Agent 审核
  - HIGH_CONFIDENCE_REMOVE：高置信度需从 eval_entities.xlsx 删除（错标）
  - SKIP：明显是模型预测错误，不动标注

输出三个 JSON 文件供后续 phase 使用。
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED_PATH = ROOT / "tests" / "batch_test_predictions.xlsx"
BROKER_LIST_PATH = ROOT / "data" / "securities_firm_src_org_names.json"
EVAL_PATH = ROOT / "data" / "eval_entities.xlsx"
OUT_DIR = ROOT / "data" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 加载券商白名单 + 扩展 ----------
with BROKER_LIST_PATH.open("r", encoding="utf-8") as f:
    broker_raw = json.load(f)

# 常见后缀（用于做"截断匹配"）
SUFFIXES = [
    "股份有限公司", "有限责任公司", "有限公司", "经纪有限公司",
    "研究中心", "研究所", "研究", "国际证券", "国际",
    "(中国)", "(亚洲)", "(香港)", "（中国）", "（亚洲）", "（香港）",
    "（境内）", "(境内)", "Securities", "Research", "Markets", "Capital",
]
TYPES = ["证券", "期货", "基金", "银行", "信托", "保险", "投资", "资管", "金控", "金融", "控股", "国际"]


def norm(s):
    s = s.strip()
    s = re.sub(r"[\(\（].*?[\)\）]", "", s)
    s = s.strip()
    return s


broker_set = set()
for b in broker_raw:
    broker_set.add(b)
    broker_set.add(b.lower())
    n = norm(b)
    if n:
        broker_set.add(n)
        broker_set.add(n.lower())
    # 截断后缀，得到核心名
    for suf in SUFFIXES:
        if n.endswith(suf) and len(n) > len(suf) + 1:
            core = n[: -len(suf)].strip()
            if core:
                broker_set.add(core)
                broker_set.add(core.lower())


# 业内常见简称/别名（手工补充行业知识）
EXTRA_BROKERS = {
    # 国际投行/银行简称
    "瑞信", "瑞银", "高盛", "摩根大通", "摩根士丹利", "大摩", "小摩", "美银", "花旗",
    "美林", "美林证券", "野村", "大和", "巴克莱", "汇丰", "渣打", "德意志", "德银",
    "法巴", "法兴", "法国兴业", "麦格理", "杰富瑞", "伯恩斯坦", "里昂", "里昂证券",
    "凯基", "凯基证券", "瑞穗", "三井住友", "瑞士信贷", "美银美林", "TD证券",
    "瑞银集团", "瑞银证券", "汇丰银行", "渣打银行", "巴克莱银行",
    "瑞士再保险", "巴黎银行", "兴业银行",
    # 中国头部券商及其常见简称
    "中信证券", "中信", "国泰君安", "海通证券", "海通", "华泰证券", "华泰",
    "中金", "中金公司", "中国国际金融", "中国国际金融股份有限公司",
    "广发证券", "广发", "招商证券", "招商", "申万宏源", "申万", "宏源",
    "中信建投", "建投", "光大证券", "光大", "兴业证券", "兴业", "东方证券", "东方",
    "国信证券", "国信", "安信证券", "安信", "方正证券", "方正",
    "银河证券", "银河", "国元证券", "国元", "国海证券", "国海",
    "财通证券", "财通", "长江证券", "长江", "东兴证券", "东兴",
    "东吴证券", "东吴", "西南证券", "西南", "华西证券", "华西",
    "国联证券", "国联", "恒泰证券", "恒泰", "华林证券", "华林",
    "太平洋", "太平洋证券", "红塔证券", "南京证券", "浙商证券", "浙商",
    "中泰证券", "中泰", "国金证券", "国金", "国投证券", "国投",
    "甬兴证券", "中原证券", "中原", "财信证券", "财信", "德邦证券", "德邦",
    "中银证券", "中银", "中航证券", "中航", "英大证券", "英大", "万联证券", "万联",
    "川财证券", "川财", "上海证券", "东海证券", "东海", "东莞证券", "东莞",
    "湘财证券", "湘财", "渤海证券", "渤海", "华创证券", "华创", "华安证券", "华安",
    "华龙证券", "华龙", "华福证券", "华福", "财达证券", "财达", "银泰证券", "银泰",
    "世纪证券", "中山证券", "中天证券", "诚通证券", "红塔", "甬兴",
    "粤开证券", "粤开", "麦高证券", "麦高", "五矿证券", "五矿",
    "联讯证券", "联讯", "联储证券", "中天国富", "华兴证券", "华兴",
    "中州证券", "东方汇财证券", "东方汇财证券国际控股有限公司", "海川证券", "海川证券公司",
    "天风证券", "天风",
    # 香港中资券商
    "海通国际", "招银国际", "中银国际", "中银国际证券", "中信国际", "工银国际",
    "建银国际", "农银国际", "国信(香港)", "广发香港", "光大新鸿基",
    "中信里昂", "里昂证券", "兴证国际", "兴证(香港)", "中泰国际", "中信(国际)",
    "复星恒利", "辉立证券", "辉立",
    # 第三方机构
    "彭博", "Bloomberg", "路透", "Reuters", "汤森路透", "Refinitiv",
    "惠誉", "穆迪", "Moody's", "标普", "S&P", "Standard & Poor",
    "IDC", "Gartner", "Forrester", "艾瑞", "艾瑞咨询",
    "麦肯锡", "波士顿咨询", "BCG", "罗兰贝格", "科尔尼",
    "安永", "EY", "PwC", "普华永道", "德勤", "Deloitte", "毕马威", "KPMG",
    "BIS", "国际清算银行", "世界银行", "IMF", "国际货币基金组织",
    # 期货/基金（在白名单中已大量包含，这里补充常见简称）
    "中信期货", "永安期货", "银河期货", "海通期货", "中信建投期货",
    # 监管/交易所（不算券商，但会被模型预测为机构）
    # 这些不放入 broker_set，留给 LLM 判断
}
broker_set.update(EXTRA_BROKERS)
broker_set.update(b.lower() for b in EXTRA_BROKERS)


def is_known_broker(name):
    n = name.strip()
    if not n:
        return False
    if n in broker_set or n.lower() in broker_set:
        return True
    n2 = norm(n)
    if n2 in broker_set or n2.lower() in broker_set:
        return True
    for suf in SUFFIXES:
        if n2.endswith(suf) and len(n2) > len(suf) + 1:
            core = n2[: -len(suf)].strip()
            if core in broker_set or core.lower() in broker_set:
                return True
    return False


# 显然不是券商的关键词（pure 噪音）
NOISE_TOKENS = {
    "券商", "头部券商", "机构", "投行", "银行", "证券", "期货", "基金",
    "卖", "买", "由", "组织", "组织的", "发", "发布", "撰写", "撰写的",
    "前", "目前", "当前", "最近", "最新",
    "公司", "上市公司", "股份", "集团",
    "深交", "上交", "港交", "北交", "纽交", "纳斯达克", "伦交",
    "深交所", "上交所", "港交所", "北交所", "伦敦证券", "伦敦证券交易所",
    "美交所", "纽交所",
}
# 形似分析师姓名（简短的纯中文 2-4 字 / 含某等）
ANALYST_PAT = re.compile(r"^[\u4e00-\u9fa5]{2,4}$")
KNOWN_ANALYST_FRAGMENTS = {"郭明錤", "张继强", "明明", "李伟", "李明", "朱悦",
                            "郭某", "张某", "李某", "王某", "明某",
                            "苏姿丰"}


# 典型券商名后缀模式
BROKER_LIKE = re.compile(
    r".+(证券|證券|期货|基金|银行|信托|保险|资管|证投|金控|金融|控股|"
    r"研究院|研究中心|研究所|集团|国际|国际证券|investments?|capital|securities|markets|research|partners|advisors)$",
    re.IGNORECASE,
)


# 模型常见破碎前缀/后缀
BROKEN_SUFFIX = ("发", "组织", "组织的", "策略团队", "首席分析师", "研究员", "分析师", "团队")
BROKEN_PARTIAL = {"摩根士丹", "摩根士", "中信证券发", "海通证券发", "招商证券发",
                   "中信建投发", "中信建投组", "中信证券组", "海通证券策略团队",
                   "美银证券组织的", "中信建投发", "摩根大通组织", "中信建投组织",
                   "摩根", "摩"}


def classify_broker_extra(query, mention, gold_brokers):
    """对 BROKER 多余实体做分类：
    Returns: ("ADD", reason) 或 ("REVIEW", reason) 或 ("SKIP", reason)
    """
    m = mention.strip()
    if not m:
        return "SKIP", "empty"

    if m in NOISE_TOKENS:
        return "SKIP", "noise_token"
    if m in KNOWN_ANALYST_FRAGMENTS:
        return "SKIP", "known_analyst"

    # 模型预测的破碎实体（如"摩根士丹"、"中信证券发"）
    # 检查是否是某个完整券商名的前缀
    if m in BROKEN_PARTIAL:
        # 如果 query 中含完整名（如"摩根士丹利"、"中信证券"），则补完整名
        for full in ["摩根士丹利", "中信证券", "海通证券", "招商证券",
                       "中信建投", "美银证券", "摩根大通", "摩根士丹利",
                       "海通证券", "天风证券", "华泰证券"]:
            if full in query and full not in gold_brokers:
                return "ADD", f"broken->{full}"
        return "SKIP", "broken_no_match"

    # 末尾带破碎后缀（如"中信证券发"、"美银证券组织的"）
    for bs in BROKEN_SUFFIX:
        if m.endswith(bs) and len(m) > len(bs) + 1:
            core = m[: -len(bs)]
            if is_known_broker(core) and core not in gold_brokers:
                return "ADD", f"broken_suffix->{core}"

    if is_known_broker(m):
        return "ADD", "in_whitelist"

    if BROKER_LIKE.search(m):
        # 但要排除"上市公司"、"分析师"等噪音
        if m in NOISE_TOKENS:
            return "SKIP", "noise"
        return "ADD", "broker_pattern"

    # 短中文（可能是分析师名/碎片）
    if ANALYST_PAT.match(m) and len(m) <= 3:
        return "SKIP", "short_chinese_likely_analyst"

    # 否则交给 LLM
    return "REVIEW", "uncertain"


# ---------- 处理预测数据 ----------
df = pd.read_excel(PRED_PATH)
print(f"加载 {len(df)} 行预测")

eval_df = pd.read_excel(EVAL_PATH)
print(f"加载 {len(eval_df)} 行评估数据")

# index 对齐：两份文件应该是行号对齐的
assert len(df) == len(eval_df), f"长度不一致 {len(df)} vs {len(eval_df)}"


def parse_list(val):
    if val is None or pd.isna(val):
        return []
    return [x.strip() for x in str(val).split("\n") if x.strip()]


broker_add_records = []      # [{row_idx, query, mention, reason}]
broker_skip_records = []
broker_review_records = []
broker_remove_records = []   # 标注的 broker 但 query 中没出现 / 不像券商

c_extra_review = []          # C/公司 多余 → 全部走 LLM 审核
c_missing_review = []
ind_extra_review = []
ind_missing_review = []


def to_corrected_mention(query, mention):
    """如果 mention 是 query 中某个完整词的破碎版本，返回完整版本。"""
    return mention


for i in range(len(df)):
    row = df.iloc[i]
    query = str(row["原始Query"]).strip()
    eval_row = eval_df.iloc[i]
    assert eval_row["ori_query"] == query, f"row {i} query 不一致"

    # ---------- BROKER 多余 ----------
    extras = parse_list(row.get("BROKER/券商 - 多余"))
    gold_brokers = set(parse_list(row.get("BROKER/券商 - 标注")))
    for ext in extras:
        verdict, reason = classify_broker_extra(query, ext, gold_brokers)
        rec = {"row_idx": i, "query": query, "mention": ext, "reason": reason,
               "current_institutions": str(eval_row.get("institutions", "") or "")}
        if verdict == "ADD":
            # 实际写入用规则化后的版本
            actual = ext
            if reason.startswith("broken->"):
                actual = reason.split("->", 1)[1]
            elif reason.startswith("broken_suffix->"):
                actual = reason.split("->", 1)[1]
            rec["mention_to_add"] = actual
            broker_add_records.append(rec)
        elif verdict == "REVIEW":
            broker_review_records.append(rec)
        else:
            broker_skip_records.append(rec)

    # ---------- BROKER 遗漏（标注里有但模型没预测出，且可能 query 里也没有）----------
    missing = parse_list(row.get("BROKER/券商 - 遗漏"))
    for m in missing:
        # 如果标注里的实体在 query 中根本不存在，肯定是错标
        if m not in query:
            broker_remove_records.append({
                "row_idx": i, "query": query, "mention": m,
                "reason": "标注实体在query中不存在",
                "current_institutions": str(eval_row.get("institutions", "") or "")
            })

    # ---------- C 多余 / 遗漏 ----------
    for ext in parse_list(row.get("C/公司 - 多余")):
        c_extra_review.append({"row_idx": i, "query": query, "mention": ext,
                                "current_companies": str(eval_row.get("companies", "") or "")})
    for m in parse_list(row.get("C/公司 - 遗漏")):
        if m not in query:
            c_missing_review.append({"row_idx": i, "query": query, "mention": m,
                                      "reason": "标注实体在query中不存在",
                                      "current_companies": str(eval_row.get("companies", "") or ""),
                                      "verdict_hint": "REMOVE"})
        else:
            c_missing_review.append({"row_idx": i, "query": query, "mention": m,
                                      "current_companies": str(eval_row.get("companies", "") or "")})

    # ---------- IND 多余 / 遗漏 ----------
    for ext in parse_list(row.get("IND/行业概念 - 多余")):
        ind_extra_review.append({"row_idx": i, "query": query, "mention": ext,
                                  "current_industries": str(eval_row.get("industries", "") or "")})
    for m in parse_list(row.get("IND/行业概念 - 遗漏")):
        if m not in query:
            ind_missing_review.append({"row_idx": i, "query": query, "mention": m,
                                        "reason": "标注实体在query中不存在",
                                        "current_industries": str(eval_row.get("industries", "") or ""),
                                        "verdict_hint": "REMOVE"})
        else:
            ind_missing_review.append({"row_idx": i, "query": query, "mention": m,
                                        "current_industries": str(eval_row.get("industries", "") or "")})


def dump(name, records):
    p = OUT_DIR / name
    with p.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  {name}: {len(records)}")


print("\n=== Phase 1 输出 ===")
dump("broker_auto_add.json", broker_add_records)
dump("broker_auto_skip.json", broker_skip_records)
dump("broker_review.json", broker_review_records)
dump("broker_auto_remove.json", broker_remove_records)
print("\n=== Phase 2 待审核 ===")
dump("c_extra_review.json", c_extra_review)
dump("c_missing_review.json", c_missing_review)
dump("ind_extra_review.json", ind_extra_review)
dump("ind_missing_review.json", ind_missing_review)

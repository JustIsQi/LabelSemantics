# LabelSemantics

复现论文 *Label Semantics for Few Shot Named Entity Recognition* 的工程化版本。核心思路保持不变：一个 Transformer encoder 编码句子 token，另一个 Transformer encoder 编码自然语言标签，再用 token 表示和标签表示的点积做 BIO 分类。

本次重构把模型、数据处理、训练循环和 Excel 转换都迁移到了 `label_semantics/` 包中，根目录只保留命令行入口和脚本。

## 代码结构

```text
.
├── train.py                         # 通用训练 CLI
├── convert_excel_to_bio.py          # Excel 标注转 BIO CLI
├── scripts/
│   └── train_full.sh                # 完整训练 shell：转换数据 + 训练 + 微调
├── label_semantics/
│   ├── data.py                      # BIO 读取、tokenizer 对齐、DataLoader
│   ├── excel_to_bio.py              # Excel 转 BIO 的可复用实现
│   ├── labels.py                    # 标签描述和 BIO id 映射
│   ├── metrics.py                   # 实体级 precision/recall/F1
│   ├── model.py                     # 双 encoder LabelSemanticsNER
│   └── training.py                  # 训练、验证、checkpoint
└── excel_ner_data/
    ├── labels.json
    ├── train.txt
    ├── dev.txt
    └── test.txt
```

## 训练数据格式

BIO 文件每行一个字符和标签，用 tab 分隔：

```text
茅	B-C
台	I-C
2	B-TIME
0	I-TIME
2	I-TIME
3	I-TIME
年	I-TIME

```

标签描述文件是 JSON，例如：

```json
{
  "C": "公司",
  "CODE": "机构代码",
  "IND": "行业概念",
  "TIME": "时间"
}
```

## Excel 转 BIO

默认读取 `test_data_0413.xlsx`，输出到 `excel_ner_data/`：

```bash
python convert_excel_to_bio.py
```

当前标签规则：

- `机构-公司` -> `C`
- `机构-代码` -> `CODE`
- `行业/概念` -> `IND`
- `时间` -> `TIME`
- 忽略：`产品`、`技术`、`机构-政府`、`机构-其他`、`自然人`

转换后会生成：

```text
excel_ner_data/
├── all.txt
├── train.txt
├── dev.txt
├── test.txt
├── labels.json
└── conversion_report.json
```

`conversion_report.json` 会记录未匹配标注、被忽略类型和重叠标注。

## 完整训练脚本

推荐直接使用：

```bash
bash scripts/train_full.sh
```

常用参数通过环境变量覆盖：

```bash
MODEL_PATH=/path/to/chinese-roberta-wwm-ext \
EXCEL_FILE=test_data_0413.xlsx \
DATA_DIR=excel_ner_data \
PRETRAIN_EPOCHS=20 \
FEWSHOT_EPOCHS=20 \
PRETRAIN_BATCH_SIZE=16 \
FEWSHOT_BATCH_SIZE=16 \
bash scripts/train_full.sh
```

脚本会依次执行：

1. `python convert_excel_to_bio.py`
2. `python train.py` 训练基础模型并保存到 `outputs/pretrain`
3. 自动选择最佳 F1 checkpoint
4. 再次 `python train.py --checkpoint-path ...` 微调并保存到 `outputs/fewshot`

## 单独训练

推荐使用通用入口：

```bash
python train.py \
  --model-path /root/workspace/berts/chinese-roberta-wwm-ext \
  --data-dir excel_ner_data \
  --label-file excel_ner_data/labels.json \
  --train-file train.txt \
  --dev-file dev.txt \
  --test-file test.txt \
  --output-dir outputs/pretrain \
  --batch-size 16 \
  --epochs 100 \
  --learning-rate 1e-5
```

从已有 checkpoint 微调：

```bash
python train.py \
  --model-path /root/workspace/berts/chinese-roberta-wwm-ext \
  --data-dir excel_ner_data \
  --label-file excel_ner_data/labels.json \
  --checkpoint-path outputs/pretrain/model_epoch0_f10.123456.pth \
  --output-dir outputs/fewshot \
  --batch-size 16 \
  --epochs 100 \
  --learning-rate 5e-5
```

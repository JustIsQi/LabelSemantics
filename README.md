# LabelSemantics

复现论文 *Label Semantics for Few Shot Named Entity Recognition* 的工程化版本。核心思路保持不变：一个 Transformer encoder 编码句子 token，另一个 Transformer encoder 编码自然语言标签，再用 token 表示和标签表示的点积做 BIO 分类。

模型、数据处理、训练循环和 Excel 转换都封装在 `label_semantics/` 包中。根目录只保留 README、依赖和子目录，所有 CLI 入口、测试脚本、数据和模型产物都各自独立成目录。

## 代码结构

```text
.
├── README.md
├── requirements.txt
├── label_semantics/                 # 核心库
│   ├── data.py                      # BIO 读取、tokenizer 对齐、DataLoader
│   ├── excel_to_bio.py              # Excel 转 BIO 的可复用实现
│   ├── labels.py                    # 标签描述和 BIO id 映射
│   ├── metrics.py                   # 实体级 precision/recall/F1
│   ├── model.py                     # 双 encoder LabelSemanticsNER
│   ├── training.py                  # 训练、验证、checkpoint
│   └── train.py                     # 训练 CLI（python -m label_semantics.train）
├── scripts/                         # 工具脚本与 shell 包装
│   ├── convert_excel_to_bio.py      # Excel 标注转 BIO CLI
│   ├── convert_checkpoint_to_model_dir.py  # 把 .pth checkpoint 打包为可分发模型目录
│   └── train.sh                     # 一键训练脚本
├── tests/                           # 评估 / 批量测试脚本
│   ├── batch_test.py                # 用 eval.jsonl 跑模型并输出 Excel 评估报告
│   ├── _fastapi_client.py           # FastAPI 测试脚本共用的 stdlib HTTP 客户端
│   ├── fastapi_single_test.py       # FastAPI 单条测试（输出 ms 级延迟）
│   ├── fastapi_batch_test.py        # FastAPI 批量回归（与 batch_test 指标对齐）
│   └── fastapi_stress_test.py       # FastAPI 并发压测（QPS + 延迟分布）
├── service/                         # FastAPI 推理服务（多 worker）
│   ├── app.py                       # ASGI app + 推理逻辑（与 batch_test 对齐）
│   └── __main__.py                  # python -m service 多 worker 启动器
├── data/                            # 原始与衍生数据
│   ├── eval.jsonl
│   ├── test_data_0413.xlsx
│   └── excel_ner_data/
│       ├── labels.json
│       ├── train.txt
│       ├── dev.txt
│       ├── test.txt
│       └── conversion_report.json
├── outputs/                         # 训练产物 / 转换后的模型目录（gitignored）
└── docs/                            # 论文 / 设计文档
    └── Label Semantics for Few Shot Named Entity Recognition.md
```

> 所有命令默认在项目根目录下执行。训练入口以模块形式调用 `python -m label_semantics.train`；`scripts/` 与 `tests/` 中的辅助脚本会自动把项目根注入 `sys.path`，import `label_semantics` 不需要额外设置。

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
  "TIME": "时间",
  "BROKER": "券商"
}
```

## Excel 转 BIO

默认读取 `data/test_data_0413.xlsx`，输出到 `data/excel_ner_data/`：

```bash
python scripts/convert_excel_to_bio.py
```

当前标签规则：

- `机构-公司` -> `C`
- `机构-代码` -> `CODE`
- `行业/概念` -> `IND`
- `券商` -> `BROKER`
- `时间` -> `TIME`
- 忽略：`产品`、`技术`、`机构-政府`、`机构-其他`、`自然人`

转换后会生成：

```text
data/excel_ner_data/
├── all.txt
├── train.txt
├── dev.txt
├── test.txt
├── labels.json
└── conversion_report.json
```

`conversion_report.json` 会记录未匹配标注、被忽略类型和重叠标注。

## 一键训练

```bash
bash scripts/train.sh
```

## 单独训练

推荐使用通用入口：

```bash
python -m label_semantics.train \
  --model-path /home/chinese-roberta-wwm-ext \
  --data-dir data/excel_ner_data \
  --label-file data/excel_ner_data/labels.json \
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
python -m label_semantics.train \
  --model-path /root/workspace/berts/chinese-roberta-wwm-ext \
  --data-dir data/excel_ner_data \
  --label-file data/excel_ner_data/labels.json \
  --checkpoint-path outputs/pretrain/model_epoch0_f10.123456.pth \
  --output-dir outputs/fewshot \
  --batch-size 16 \
  --epochs 100 \
  --learning-rate 5e-5
```

## 打包模型 & 批量评估

将训练得到的 `.pth` checkpoint 打包成自包含模型目录（含 tokenizer/config）：

```bash
python scripts/convert_checkpoint_to_model_dir.py \
  --base-model-path /home/chinese-roberta-wwm-ext \
  --checkpoint-dir outputs/pretrain \
  --output-dir outputs/best_model \
  --overwrite
```

然后用 `data/eval.jsonl` 进行批量评估，预测结果会写成一份方便人工复核的
Excel 文件（含「明细」和「汇总」两个 sheet，错误行整行高亮，多余/遗漏单元格
单独高亮）：

```bash
python tests/batch_test.py \
  --input-file data/eval.jsonl \
  --model-path outputs/best_model \
  --label-file data/excel_ner_data/labels.json \
  --output-file tests/batch_test_predictions.xlsx
```

## FastAPI 推理服务（多 worker）

`service/` 包提供一个 FastAPI 服务，推理逻辑直接对齐 `tests/batch_test.py`，
输入是 query，输出是各标签下的实体列表。每个 uvicorn worker 进程会在启动时
独立加载一份模型，因此 `--workers N` 可天然得到 N 路并行推理（注意单卡 GPU
显存会被多个 worker 共享，按需调整）。

一键启动：

```bash
LS_WORKERS=4 LS_DEVICE=cuda:0 bash scripts/serve.sh
```

或直接调用 launcher：

```bash
python -m service --host 0.0.0.0 --port 8000 --workers 4
```

也可以用裸 uvicorn（更灵活，比如想搭配 `--limit-concurrency` 等参数）：

```bash
LS_MODEL_PATH=outputs/best_model \
LS_LABEL_FILE=data/excel_ner_data/labels.json \
uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 配置（环境变量）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LS_MODEL_PATH` | `outputs/best_model` | 自包含模型目录 |
| `LS_LABEL_FILE` | `data/excel_ner_data/labels.json` | 标签描述 JSON |
| `LS_MAX_LENGTH` | `128` | tokenizer 最大长度 |
| `LS_DEVICE` | 自动（有 GPU 用 cuda） | torch 设备：`cuda` / `cuda:0` / `cpu` |
| `LS_WORKERS` | `1` | uvicorn worker 数（每个进程一份模型） |
| `LS_HOST` / `LS_PORT` | `0.0.0.0` / `8000` | 监听地址与端口 |
| `LS_WARMUP_QUERIES` | 内置示例 | `;` 分隔的预热 query；置空则跳过 |
| `LS_INCLUDE_TIME_REGEX` | `0` | 设为 `1` 时把正则抽取的 TIME 词合并到响应 |

### 接口

- `GET /health` — 检查模型是否就绪，返回当前进程 PID、设备、可用标签集。
- `GET /labels` — 返回当前服务加载的标签描述。
- `POST /predict` — 单条推理。

  请求体：

  ```json
  { "query": "茅台2023年的营收情况" }
  ```

  响应体：

  ```json
  {
    "query": "茅台2023年的营收情况",
    "entities": {
      "C": ["茅台"],
      "TIME": ["2023年"],
      "BROKER": [],
      "IND": [],
      "CODE": []
    },
    "inference_ms": 12.345,
    "timings": {
      "tokenize_ms": 0.42,
      "model_ms": 11.10,
      "decode_ms": 0.08,
      "inference_ms": 11.60,
      "server_handler_ms": 12.34
    }
  }
  ```

  另外服务端 ASGI 中间件会在响应头里注入 `X-Request-Total-Ms`，覆盖
  *请求解析 + pydantic 校验 + 路由处理 + 响应序列化* 的全流程耗时。
  结合上述字段就能定位时间花在哪一段（参见 `tests/fastapi_single_test.py`
  的注释）。

- `POST /predict_batch` — 一次提交多条 query（`{"queries": ["...", "..."]}`），
  返回与 `/predict` 同结构的结果数组。注意单 worker 内部串行执行，并发请通过
  增加 worker 数解决。

### 服务端的延迟优化

为了把单次请求的端到端时间压低，服务端做了几件事：

- 默认响应类切到 **ORJSONResponse**（`requirements.txt` 已加 `orjson`），
  比 stdlib `json` 快几倍，对小 dict 响应尤其明显。
- 路由直接返回 `dict`，不再用 `response_model=...`，避开 FastAPI 默认的
  **二次 pydantic 验证 + dump**（`PredictResponse` 等 model 仍然保留作客户端文档用）。
- 用纯 ASGI 中间件（不是 `BaseHTTPMiddleware`）注入 `X-Request-Total-Ms`，
  避免给请求加额外的流式包裹开销。
- 模型 forward 后用 `.cpu().tolist()` 隐式 sync CUDA，去掉了显式
  `torch.cuda.synchronize` 的多余调用。

客户端侧（`tests/_fastapi_client.py`）切到 `http.client.HTTPConnection` +
`Connection: keep-alive` + 每线程独立连接，因此重复请求不再每次重做 TCP 握手，
并发线程也不会因为共用 socket 而互相串行。

调用示例：

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"query":"中信证券2024年Q1对宁德时代的研报"}' | python -m json.tool
```

### 客户端测试脚本

`tests/` 下提供三个零额外依赖（仅用 stdlib `urllib`）的客户端脚本，按职责拆分，
共用 `tests/_fastapi_client.py` 里的薄 HTTP 客户端：

```bash
# 1) 单条测试：发送一次 /predict，打印实体结果 + 毫秒级延迟
python tests/fastapi_single_test.py "茅台2023年的营收"
python tests/fastapi_single_test.py --repeat 5 "比亚迪最近股价表现"

# 2) 批量回归：用 eval.jsonl 重放服务，输出 per-label recall + extras，
#    指标格式与 tests/batch_test.py 完全一致，便于 diff
python tests/fastapi_batch_test.py \
  --base-url http://127.0.0.1:8000 \
  --input-file data/eval.jsonl \
  --output-file tests/fastapi_batch_test_predictions.jsonl

# 3) 并发压测：验证多 worker 实际吞吐（QPS + 延迟分布 p50/p95/p99）
python tests/fastapi_stress_test.py --concurrency 8 --requests 1000
python tests/fastapi_stress_test.py \
  --concurrency 16 --input-file data/eval.jsonl --limit 500 --requests 2000
```

批量回归脚本会输出 `server_inference_ms`（服务端模型 forward 时间，由
`/predict` 响应回传）和 `client_total_ms`（客户端 wall time，含网络往返），
两者差就是网络 + 序列化开销。`fastapi_batch_test.py` 的预测仍以 jsonl
形式落盘，便于命令行做对比；in-process 的 `tests/batch_test.py` 现在
默认输出 Excel（方便人工复核），与 `fastapi_batch_test_predictions.jsonl`
schema 不同，但二者推理路径完全一致（`service/app.py` 复用 batch_test
的 `encode_query` / `predict_entities`）。

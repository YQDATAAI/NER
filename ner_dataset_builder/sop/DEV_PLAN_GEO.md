# 开发计划文档（DEV_PLAN）

## 0. 项目信息

- 项目名称：基于 Qwen3.5-4B 的 NER 数据集自动构建与 BIO 转换工程
- 项目路径：`/home/superuser/dev/NER/ner_dataset_builder`
- 数据路径：`/home/superuser/dev/NER/data`
- 模型路径：`/home/superuser/LLM_Model/Qwen3.5-4B`
- 运行环境：`/home/superuser/.conda/envs/dsbi/bin/python`

---

## 1. 项目背景与总体目标

本项目目标是将录井报告分页数据批量转换为可训练 NER 数据，支持后续微调 BERT 类模型。

### 1.1 业务目标

- 从 `page_text` 中抽取实体并生成结构化标注。
- 通过规则与审计机制，持续提升标注质量与稳定性。
- 形成“可回溯、可迭代、可复现”的训练数据流水线。

### 1.2 当前工程角色分工（已落地）

- LLM：根据 Prompt + Few-shots 输出实体 JSON。
- 解析器：忽略 LLM 原始 offset，按原文回溯重算索引并过滤幻觉。
- 规则层：对实体进行 `include_patterns/exclude_texts` 校验。
- 转换层：输出字符级 BIO 数据与人工复核文件。

### 1.3 当前实体范围（最新）

- `well_name`（井名）
- `block`（区块）
- `strat_unit`（层位名称）
- `lithology`（主岩性）
- `lithology_color`（岩性颜色）
- `lithology_texture`（岩性结构描述）

---

## 2. 工程目录结构（最新）

```text
ner_dataset_builder/
├── main.py
├── extract_bi_for_review.py
├── README_SOP.MD
├── Instruction.MD
├── DEV_PLAN.md
├── requirements.txt
├── configs/
│   ├── prompt_config.json
│   ├── few_shots.yaml
│   └── entity_rules.yaml
├── core/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model_engine.py
│   ├── prompt_builder.py
│   ├── parser.py
│   └── bio_converter.py
└── output/
    ├── train.bio
    ├── train.sentence_map.tsv
    ├── rules.audit.tsv
    ├── train.bi_only.tsv
    └── train.bi_spans.tsv
```

---

## 3. 配置规范（最新状态）

### 3.1 `configs/prompt_config.json`

- `system_instruction`：石油地质标注专家角色。
- `task_description`：已升级为 6 类实体。
- `template`：支持 few-shots 注入。
- `generation_config`：默认 `max_new_tokens=512`，可通过 CLI 覆盖。

### 3.2 `configs/few_shots.yaml`

- 支持动态加载。
- 已包含 6 类实体的样例（含颜色、结构）。

### 3.3 `configs/entity_rules.yaml`

- 支持规则过滤与空规则容错。
- 当前已支持按实体类型配置：
  - `include_patterns`
  - `exclude_texts`

---

## 4. 核心模块开发要求（同步到当前实现）

### 4.1 数据加载模块（`core/data_loader.py`）

- 读取目录中的 `.md` 文件（内容实际为 JSON 数组）。
- 提取每条 `page_text`，过滤空值。
- 输出结构包含：`source_file/source_path/page_number/block_index/text`。

### 4.2 模型引擎（`core/model_engine.py`）

- 使用 `AutoTokenizer` + `AutoModelForCausalLM`。
- 支持 `device_map="auto"`。
- 支持指定设备（用于多进程多卡 worker）。
- 自动选择 `bf16/fp16`。
- `generate_text(prompt, **kwargs)` 支持覆盖 generation 参数。

### 4.3 Prompt 构建器（`core/prompt_builder.py`）

- 读取 JSON 配置与 YAML few-shots。
- few-shots 字符串在初始化时缓存，减少重复构建开销。
- 模板注入 `system_instruction/task_description/few_shots_str/input_text`。

### 4.4 解析与纠错器（`core/parser.py`）

- 处理 markdown code fence JSON 包裹。
- 支持 `json.loads + ast.literal_eval` 容错解析。
- 忽略 LLM 输出偏移，按原文 `re.finditer` 回算准确 offset。
- 原文不存在实体字符串则丢弃。
- 应用实体规则并返回审计信息（过滤原因）。

### 4.5 BIO 转换器（`core/bio_converter.py`）

- 字符级拆分。
- 默认 `O`，实体起始 `B-<type>`，内部 `I-<type>`。
- 输出标准 BIO 文本。

### 4.6 主程序（`main.py`）

- 主链路：加载 -> Prompt -> 推理 -> 解析纠偏 -> 规则过滤 -> BIO 输出。
- 新增输出：
  - `train.sentence_map.tsv`
  - `rules.audit.tsv`
- 新增性能能力：
  - 多卡并行：`--num-workers --gpu-ids`
  - 文本截断：`--max-input-chars`
  - 生成长度控制：`--max-new-tokens`

### 4.7 复核导出（`extract_bi_for_review.py`）

- 输入 `train.bio + train.sentence_map.tsv`。
- 输出：
  - `train.bi_only.tsv`
  - `train.bi_spans.tsv`

---

## 5. 全流程命令（当前推荐）

```bash
/home/superuser/.conda/envs/dsbi/bin/python /home/superuser/dev/NER/ner_dataset_builder/main.py \
  --input-dir /home/superuser/dev/NER/data \
  --model-path /home/superuser/LLM_Model/Qwen3.5-4B \
  --prompt-config /home/superuser/dev/NER/ner_dataset_builder/configs/prompt_config.json \
  --few-shots /home/superuser/dev/NER/ner_dataset_builder/configs/few_shots.yaml \
  --rules-config /home/superuser/dev/NER/ner_dataset_builder/configs/entity_rules.yaml \
  --output-file /home/superuser/dev/NER/ner_dataset_builder/output/train.bio \
  --sentence-map-file /home/superuser/dev/NER/ner_dataset_builder/output/train.sentence_map.tsv \
  --rules-audit-file /home/superuser/dev/NER/ner_dataset_builder/output/rules.audit.tsv \
  --num-workers 5 \
  --gpu-ids 0,1,2,3,4 \
  --max-input-chars 3000 \
  --max-new-tokens 128 \
&& /home/superuser/.conda/envs/dsbi/bin/python /home/superuser/dev/NER/ner_dataset_builder/extract_bi_for_review.py \
  --input-file /home/superuser/dev/NER/ner_dataset_builder/output/train.bio \
  --bi-lines-output /home/superuser/dev/NER/ner_dataset_builder/output/train.bi_only.tsv \
  --spans-output /home/superuser/dev/NER/ner_dataset_builder/output/train.bi_spans.tsv \
  --sentence-map-file /home/superuser/dev/NER/ner_dataset_builder/output/train.sentence_map.tsv
```

---

## 6. 当前迭代闭环（质量控制）

### 6.1 首轮运行（可无规则启动）

- `entity_rules.yaml` 可为空或不存在，不影响初始化。
- 先拿到 `train.bi_spans.tsv` 作为首轮人工复核基线。

### 6.2 后续迭代顺序

1. 查看 `train.bi_spans.tsv`（识别误标/漏标模式）。
2. 查看 `rules.audit.tsv`（识别误杀/漏杀）。
3. 更新 `entity_rules.yaml`（正则与黑名单）。
4. 必要时更新 `few_shots.yaml` 或 `prompt_config.json`。
5. 重跑全流程并对比指标。

### 6.3 建议关注指标

- 各实体类型数量变化。
- `well_name` 等关键实体误标率。
- 规则审计中 `excluded_text/not_match_include_patterns` 的占比趋势。

---

## 7. 近期开发计划（下一阶段）

### 7.1 P0（稳定性）

- 增加并行 worker 异常检测与失败重试。
- 增加输出文件完整性校验（字段/行数/空值）。

### 7.2 P1（质量）

- 增加规则误杀样本回放机制。
- 增加实体级抽样报告（按类型随机抽样）。

### 7.3 P2（性能）

- 引入按 token 长度切片替代纯字符截断。
- 增加并行批处理统计（吞吐、平均时延、GPU 利用率日志）。

---

## 8. 交付与验收标准

- 一键命令可完整产出 5 类输出文件：
  - `train.bio`
  - `train.sentence_map.tsv`
  - `rules.audit.tsv`
  - `train.bi_only.tsv`
  - `train.bi_spans.tsv`
- 输出可被人工复核并支持规则闭环迭代。
- 并行模式可在多卡环境稳定运行。

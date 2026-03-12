# 开发计划文档（DEV_PLAN）

## 0. 项目信息

- 项目名称：基于 Chinese-MacBERT 的地质 NER 训练与 ONNX 推理工程
- 项目路径：`/home/superuser/dev/NER/ner_macbert_trainer`
- 上游数据路径：`/home/superuser/dev/NER/ner_dataset_builder/output`
- 输入数据主文件：`/home/superuser/dev/NER/ner_dataset_builder/output/train.bio`
- 底模路径：`/home/superuser/LLM_Model/chinese-macbert-large`
- 运行环境：`/home/superuser/.conda/envs/dsbi/bin/python`

---

## 1. 项目背景与总体目标

本项目负责把上游 BIO 数据训练成可部署 NER 模型，并导出 ONNX（FP32/INT8）用于批量推理。

### 1.1 业务目标

- 面向录井报告文本，输出稳定可用的实体识别结果。
- 对齐输出格式，支持下游复核与数据闭环。
- 形成“可复现、可诊断、可迭代”的模型优化流程。

### 1.2 当前实体范围

- `well_name`
- `block`
- `strat_unit`
- `lithology`
- `lithology_color`
- `lithology_texture`

### 1.3 当前优化主线

- 文档分组切分（减少随机切分偏差）
- 长序列窗口化训练与推理全覆盖（减少截断损失）
- 训练/推理一致剔除 table 块
- 纹理类专项增强（定向上采样、规则增强）
- 双模型融合（总体优先 + 纹理优先）

---

## 2. 工程目录结构（最新）

```text
ner_macbert_trainer/
├── main_train.py
├── export_onnx.py
├── inference_onnx.py
├── inference_onnx_dual.py
├── conf/
│   ├── training_args.yaml
│   ├── training_args.dual_base.yaml
│   └── training_args.dual_texture.yaml
├── core/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── metrics.py
├── output/
│   ├── checkpoints/
│   ├── saved_models/
│   ├── onnx_models/
│   ├── infer/
│   └── test_metrics.json
├── output_dual/
│   ├── base/
│   ├── texture/
│   └── fused/
└── sop/
    ├── Model_Optimize.md
    └── DEV_PLAN_MACBERT_NER.md
```

---

## 3. 配置规范

### 3.1 `conf/training_args.yaml`

- `data.train_bio_path`：训练 BIO 文件路径。
- `model.pretrained_model_path`：底模路径。
- `output.*`：checkpoints、saved model、onnx、metric 输出路径。
- `train.*`：学习率、batch size、epoch、warmup、logging、save policy。
- `runtime.*`：max_length、stride、分组切分、table剔除、上采样、后处理约束、词典。

### 3.2 双模型配置

- `training_args.dual_base.yaml`：总体 F1 优先（`train_texture_upsample=2`）。
- `training_args.dual_texture.yaml`：纹理召回优先（`train_texture_upsample=4`）。

---

## 4. 核心模块开发要求

### 4.1 数据处理（`core/dataset.py`）

- 解析 BIO 文件并构建句级样本。
- 支持剔除 `<table>...</table>` 块。
- 支持 overflow window 对齐（`return_overflowing_tokens + stride`）。
- 支持正样本上采样与纹理类定向上采样。

### 4.2 训练入口（`main_train.py`）

- 按 `source_file` 分组切分 train/eval/test。
- 兼容不同 transformers 版本参数。
- 支持可选 weighted loss trainer。
- 输出 `test_metrics.json` 并保存模型与标签映射。

### 4.3 导出（`export_onnx.py`）

- 从 `saved_models` 导出 `ner_macbert_fp32.onnx`。
- 执行 INT8 动态量化输出 `ner_macbert_int8.onnx`。

### 4.4 推理（`inference_onnx.py`）

- 批量读取 `data/*.md`（JSON数组）中的 `page_text`。
- 全覆盖窗口推理并输出：
  - `predict.bio`
  - `predict.sentence_map.tsv`
  - `predict.bi_only.tsv`
  - `predict.bi_spans.tsv`
- 支持 BIO 合法化、span 约束、纹理词典增强。

### 4.5 融合推理（`inference_onnx_dual.py`）

- 加载 base + texture 两个 ONNX。
- 规则融合：默认以 base 为主，仅在满足条件时吸收 texture 标签。
- 输出与单模型一致的四类文件。

---

## 5. 全流程命令（当前推荐）

### 5.1 单模型训练 + 导出 + 推理

```bash
cd /home/superuser/dev/NER/ner_macbert_trainer

torchrun --nproc_per_node=5 main_train.py --config conf/training_args.yaml

/home/superuser/.conda/envs/dsbi/bin/python export_onnx.py --config conf/training_args.yaml

/home/superuser/.conda/envs/dsbi/bin/python inference_onnx.py \
  --config conf/training_args.yaml \
  --input-dir /home/superuser/dev/NER/data
```

### 5.2 双模型训练 + 融合推理

```bash
cd /home/superuser/dev/NER/ner_macbert_trainer

torchrun --nproc_per_node=5 main_train.py --config conf/training_args.dual_base.yaml
/home/superuser/.conda/envs/dsbi/bin/python export_onnx.py --config conf/training_args.dual_base.yaml

torchrun --nproc_per_node=5 main_train.py --config conf/training_args.dual_texture.yaml
/home/superuser/.conda/envs/dsbi/bin/python export_onnx.py --config conf/training_args.dual_texture.yaml

/home/superuser/.conda/envs/dsbi/bin/python inference_onnx_dual.py \
  --base-config conf/training_args.dual_base.yaml \
  --texture-config conf/training_args.dual_texture.yaml \
  --input-dir /home/superuser/dev/NER/data
```

---

## 6. 与上游任务通讯交互规范（新增）

上游工程：`/home/superuser/dev/NER/ner_dataset_builder`

### 6.1 输入契约（上游 -> 本工程）

- 必须提供：
  - `train.bio`
  - `train.sentence_map.tsv`
  - 建议同步：`train.bi_only.tsv`、`train.bi_spans.tsv`
- 版本要求：
  - 每次上游重推理后，需要明确“数据版本号/日期/变更说明”。
  - 至少说明三点：实体范围是否变化、table策略是否变化、规则策略是否变化。

### 6.2 变更通知模板

上游交付时建议附带：

- 数据版本：`YYYYMMDD_xx`
- 变更摘要：例如“已去除 table 块；新增/删除实体类型；规则更新”
- 影响预期：例如“BI占比上升/下降”“某类样本显著减少”
- 产物路径：绝对路径列表

### 6.3 下游回传（本工程 -> 上游）

本工程每轮训练后回传：

- `output/test_metrics.json`
- 关键结论（总F1、各实体F1、风险项）
- 若发现数据异常，回传证据：
  - 标签分布统计
  - 类别缺失/稀疏告警
  - 截断风险统计

### 6.4 联调节奏

1. 上游提交新数据版本与变更说明。
2. 本工程先做数据健康检查（标签覆盖、O占比、table比例、长度分布）。
3. 通过后启动训练与推理；失败则回传问题单给上游。
4. 双方固定一个“可比基线窗口”，避免混淆归因。

---

## 7. 当前迭代闭环（训练侧）

1. 读取新语料并做健康检查。
2. 跑基线训练（固定配置）。
3. 分析 metric 与各类 F1。
4. 只改一个变量进行增量实验。
5. 更新 `sop/Model_Optimize.md` 记录结论与下一步。

---

## 8. 交付与验收标准

- 单模型链路可稳定输出：
  - `saved_models/`
  - `onnx_models/`
  - `test_metrics.json`
  - `infer` 四类对齐文件
- 双模型融合链路可稳定输出 `output_dual/fused/*`。
- 每轮实验有“配置快照 + 指标快照 + 结论归因”。
- 与上游交互遵循第 6 节规范，可追溯数据版本与变更影响。

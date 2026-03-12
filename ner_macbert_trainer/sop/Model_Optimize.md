# Model Optimize SOP

## 0. 文档目标与适用范围

本 SOP 只聚焦一件事：`ner_macbert_trainer` 的模型效果优化（尤其是 F1 提升与可解释迭代）。

- 项目路径：`/home/superuser/dev/NER/ner_macbert_trainer`
- 训练数据：`/home/superuser/dev/NER/ner_dataset_builder/output/train.bio`
- 复核数据：`/home/superuser/dev/NER/ner_dataset_builder/output/train.bi_only.tsv`、`/home/superuser/dev/NER/ner_dataset_builder/output/train.bi_spans.tsv`
- 预训练模型：`/home/superuser/LLM_Model/chinese-macbert-large`
- Python 环境：`/home/superuser/.conda/envs/dsbi/bin/python`

---

## 1. Metric 症状与当前结论

### 1.1 已观测到的核心症状

- 新语料重训后，`test_f1=0.7826`，表面显著提升，但该结果与旧轮次不可直接横比。
- 当前测试集中有效类别更集中，`lithology=0.0`，且未出现 `lithology_color/lithology_texture` 指标项，提示数据覆盖发生变化。
- 现阶段主要风险从“table噪声”转为“标签覆盖与类别代表性不足”。

### 1.2 关键实验结果（可复现）

- 新语料当前基线：`test_f1=0.7826`，`precision=0.75`，`recall=0.8182`，`test_f1_lithology=0.0`  
  文件：`/home/superuser/dev/NER/ner_macbert_trainer/output/test_metrics.json`
- `train_texture_upsample=2`：`test_f1=0.4063`，`texture_f1=0.0000`  
  文件：`/home/superuser/dev/NER/ner_macbert_trainer/output/test_metrics.up2.json`
- `train_texture_upsample=3`：`test_f1=0.3976`，`texture_f1=0.0909`  
  文件：`/home/superuser/dev/NER/ner_macbert_trainer/output/test_metrics.up3.json`
- `train_texture_upsample=4`：`test_f1=0.3947`，`texture_f1=0.2857`  
  文件：`/home/superuser/dev/NER/ner_macbert_trainer/output/test_metrics.up4.json`

结论：历史实验中的 `upsample=2/3/4` 结论仍可用于旧分布；对新语料应先做类别覆盖审计，再决定是否继续纹理专项增强。

---

## 2. 导致效果差的原因（含比例分配）

以下比例是基于当前实验与数据分析的工程归因占比（用于排查优先级，不是数学真值），总计 100%。

### 2.1 数据标签稀疏与类别极不平衡（55%）【最大原因】

- 最新统计：实体 token 占比约 `0.2979%`，`O` 约 `99.7021%`（比旧语料更稀疏）。
- 直接影响：模型天然偏向预测 `O`，少数类（尤其纹理类）学习信号弱。
- 复现统计命令（已执行）：

```bash
PYTHONPATH=/home/superuser/dev/NER/ner_macbert_trainer /home/superuser/.conda/envs/dsbi/bin/python - <<'PY'
import yaml
from core.dataset import parse_bio_file
cfg=yaml.safe_load(open('/home/superuser/dev/NER/ner_macbert_trainer/conf/training_args.yaml','r',encoding='utf-8'))
_,tags=parse_bio_file(cfg['data']['train_bio_path'])
all_tokens=sum(len(x) for x in tags)
entity_tokens=sum(sum(1 for t in row if t!='O') for row in tags)
print(entity_tokens/all_tokens)
PY
```

### 2.2 训练语料噪声高（含 table 块）（5%）

- 上游已优化并重新产出语料，当前 `table_removed_ratio=0.0`，说明训练语料中 table 块已基本消除。
- 该因素不再是主瓶颈，降为低权重监控项（只做巡检，不再作为主要提效抓手）。

### 2.3 长序列覆盖问题（历史问题，已工程修复）（15%）

- 在“单窗口截断”设定下，历史上约 `55.68%` 样本会被截断（max_length=256）。
- 直接影响：实体尾部丢失、标签对齐失真、召回受损。
- 现状：已通过 overflow window + stride 做系统修复（见第 3 节）。

### 2.4 类别覆盖不足与分布漂移（15%）

- 新语料标签分布显示，`B/I-lithology` 仅极少量出现，`lithology_color/lithology_texture` 未形成有效训练评估闭环。
- 直接影响：整体 F1 可能“看起来变高”，但对全实体体系的真实泛化能力下降。

### 2.5 预训练权重命名兼容与加载噪声（7%）

- 训练日志持续出现 `LayerNorm.beta/gamma` 与 `weight/bias` 的 missing/unexpected 提示。
- 虽不是唯一主因，但会增加训练不稳定性与排障成本。

### 2.6 纯超参层优化空间有限（3%）

- 单纯调 learning rate / epoch 可以微调，但难以穿透数据与系统瓶颈。
- 这也是本 SOP 强调“系统性提效优先于纯超参搜索”的原因。

---

## 3. 已实施的系统性优化（工程 + 依据 + 结果）

本节记录“做了什么、为什么做、为什么判定方向正确”。

### 3.1 长序列窗口化训练（避免截断损失）

- 实施内容：
  - 训练数据改为 `return_overflowing_tokens + stride`，不再只取首窗口。
  - 文件：`core/dataset.py`（`tokenize_and_align_labels_windows`、`NERDataset`）
  - 入口接线：`main_train.py`
- 判定依据：
  - 历史截断风险高（55.68%），必须优先修。
  - 该改造是“把丢失标签找回来”，属于正确系统修复，不依赖运气。

### 3.2 推理端全覆盖解码（与训练口径一致）

- 实施内容：
  - 推理由固定切块改为 overflow 窗口预测并回填全字符位。
  - 文件：`inference_onnx.py`
- 判定依据：
  - 训练与推理分块机制不一致会导致线上掉点。
  - 统一口径后，能稳定减少尾部漏检。

### 3.3 文档分组切分（按 source_file）

- 实施内容：
  - train/eval/test 改为按文档来源分组切分，减少随机句段切分偏差。
  - 文件：`main_train.py`（`split_dataset_by_group`）
- 判定依据：
  - 文档型任务中，随机句段切分易造成估计偏差。
  - 分组切分可提升评估稳定性与泛化可信度。

### 3.4 训练/推理一致地剔除 table 内容

- 实施内容：
  - 训练侧：`strip_table_blocks`（`core/dataset.py`）
  - 推理侧：`strip_table_blocks_from_text`（`inference_onnx.py`）
- 判定依据：
  - 该任务明确不处理表格信息。
  - 统一剔除是必要的数据口径治理，而非“调参技巧”。
  - 新语料统计已显示 `table_removed_ratio=0.0`，该措施已从“增益项”转为“守门项”。

### 3.5 BIO 合法化 + span 约束后处理

- 实施内容：
  - `normalize_bio_tags`、`apply_span_constraints`（`inference_onnx.py`）
- 判定依据：
  - 非法 `I-` 边界会直接伤害 span F1。
  - 该处理可减少格式性错误，属于低风险收益项。

### 3.6 纹理类专项增强（非全局加权）

- 实施内容：
  - 增加 `train_texture_upsample`（`main_train.py` + `core/dataset.py`）
  - 增加 `texture_lexicon` 规则增强（`inference_onnx.py`）
- 判定依据：
  - 目标是修复 `lithology_texture`，应做“定向增强”，而非全局损失加权。
  - 对比结果证明该方向有效：`texture_f1` 可由 `0` 提升至 `0.2857`。

### 3.7 双模型融合（总体优先 + 纹理优先）

- 实施内容：
  - 新增配置：
    - `conf/training_args.dual_base.yaml`（upsample=2）
    - `conf/training_args.dual_texture.yaml`（upsample=4）
  - 新增融合推理：`inference_onnx_dual.py`
  - 融合策略：仅在 base 为 `O` 或 texture 时接收 texture 结果。
- 判定依据：
  - 单模型难同时最优兼顾“总体F1”和“纹理F1”。
  - 融合可把 trade-off 从“二选一”变为“按实体类型分治”。

---

## 4. 为什么这些是“正确但短期涨幅不大”的系统性提效

- 数据本身极稀疏（实体仅约 0.2979%），决定了任何工程优化都很难出现跃迁式提升。
- 但这些优化解决的是结构性问题：
  - 覆盖问题（截断）；
  - 口径一致性问题（训练/推理）；
  - 分布偏差问题（切分方式）；
  - 任务噪声问题（table，当前已受控）；
  - 类别覆盖问题（当前主瓶颈）。
- 因此，即使单轮 metric 增益不大，也属于“长期正确路径”，在数据质量提高后会放大收益。

---

## 5. 标准化迭代流程（接手即用）

### 5.1 每轮固定步骤

1. 跑基线训练并记录 `test_metrics.json`。
2. 只改一个系统变量（如：texture_upsample、lexicon、约束规则）。
3. 重训并记录同口径指标文件（建议另存 `test_metrics.xxx.json`）。
4. 对比：
   - 总体：`test_f1 / precision / recall`
   - 弱类：`test_f1_lithology_texture`
5. 决策：
   - 若目标是总体 F1：优先保 precision。
   - 若目标是纹理修复：允许适度 precision 换 recall。

### 5.2 推荐命令模板

```bash
cd /home/superuser/dev/NER/ner_macbert_trainer

# 训练
torchrun --nproc_per_node=5 main_train.py --config conf/training_args.yaml

# 导出 ONNX
/home/superuser/.conda/envs/dsbi/bin/python export_onnx.py --config conf/training_args.yaml

# ONNX 推理
/home/superuser/.conda/envs/dsbi/bin/python inference_onnx.py \
  --config conf/training_args.yaml \
  --input-dir /home/superuser/dev/NER/data
```

### 5.3 双模型融合命令模板

```bash
cd /home/superuser/dev/NER/ner_macbert_trainer

# base
torchrun --nproc_per_node=5 main_train.py --config conf/training_args.dual_base.yaml
/home/superuser/.conda/envs/dsbi/bin/python export_onnx.py --config conf/training_args.dual_base.yaml

# texture
torchrun --nproc_per_node=5 main_train.py --config conf/training_args.dual_texture.yaml
/home/superuser/.conda/envs/dsbi/bin/python export_onnx.py --config conf/training_args.dual_texture.yaml

# fused inference
/home/superuser/.conda/envs/dsbi/bin/python inference_onnx_dual.py \
  --base-config conf/training_args.dual_base.yaml \
  --texture-config conf/training_args.dual_texture.yaml \
  --input-dir /home/superuser/dev/NER/data
```

---

## 6. 数据端优化后应期待的结果映射

当数据端改进（补齐各实体覆盖、平衡正负样本、持续清洗噪声）后，本 SOP 的系统优化会让收益更稳定体现为：

- `lithology/lithology_color/lithology_texture`：应从“缺失或低分”恢复到可评估、可优化状态。
- 整体 `precision`：在保持结构化后处理约束下逐步回升。
- 整体 `test_f1`：在多类覆盖完整前提下，逐轮提高并减少震荡。

---

## 7. 接手人第一天待办（Checklist）

- 确认路径与环境可用（第 0 节）。
- 先读取以下结果文件并建立当前基线：
  - `output/test_metrics.json`（最新新语料基线）
  - `output/test_metrics.up2.json`
  - `output/test_metrics.up3.json`
  - `output/test_metrics.up4.json`
  - `output_dual/base/test_metrics.train.json`
  - `output_dual/texture/test_metrics.train.json`
- 明确本轮目标是“总体 F1 优先”还是“纹理类优先”。
- 按第 5 节流程只改一个变量并复跑，保留指标快照与结论。
